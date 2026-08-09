"""
SNOMED CT offline utilities for Stages 5–6.

Uses RF2 Snapshot under data/SnomedCT_* / Snapshot/Terminology/.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# SNOMED concept / attribute typeIds (RF2)
# ---------------------------------------------------------------------------
IS_A = "116680003"
# Attribute conceptIds used as Relationship.typeId (outbound only in Stage 6).
# Inverses are excluded — they pull noise (e.g. drug → poisoning/overdose via
# has_causative_agent inbound).
ATTR_TYPE_IDS: Dict[str, str] = {
    # "cause_of" not an attribute FSN in RF2; map to "Due to" (42752001)
    "cause_of": "42752001",  # Due to
    "has_causative_agent": "246075003",  # outbound only (organism/agent of disease)
    "has_finding_site": "363698007",
    "has_associated_morphology": "116676008",
    "has_pathological_process": "370135005",
    "after": "255234002",  # post-procedure / sequela
    "associated_with": "47429007",
    "occurrence": "246454002",  # congenital / acquired timing
    "clinical_course": "263502005",  # acute / chronic / intermittent
}
# Prefer PT for display when available
TYPE_FSN = "900000000000003001"
TYPE_SYNONYM = "900000000000013009"

# Embedding cosine: retain if similarity ≥ 0.70; high-confidence tier ≥ 0.80
DEFAULT_MIN_COSINE_SIM = 0.70
DEFAULT_HIGH_CONF_COSINE_SIM = 0.80
DEFAULT_MAX_COSINE_DISTANCE = 1.0 - DEFAULT_MIN_COSINE_SIM  # 0.30
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNOMED_GLOB = "SnomedCT_*"
DEFAULT_SNOMED_DIR = REPO_ROOT / "data"


def find_snomed_root(data_dir: Optional[Path] = None) -> Path:
    """Locate offline SNOMED CT package under data/."""
    base = Path(data_dir or DEFAULT_SNOMED_DIR)
    candidates = sorted(base.glob(DEFAULT_SNOMED_GLOB), key=lambda p: p.name, reverse=True)
    for c in candidates:
        snap = c / "Snapshot" / "Terminology"
        if snap.is_dir() and any(snap.glob("sct2_Description_Snapshot*.txt")):
            return c
    raise FileNotFoundError(
        f"No SNOMED CT RF2 package found under {base}. "
        f"Expected folder like data/SnomedCT_.../Snapshot/Terminology/"
    )


def snomed_terminology_dir(snomed_root: Path) -> Path:
    return Path(snomed_root) / "Snapshot" / "Terminology"


def _find_file(term_dir: Path, prefix: str) -> Path:
    matches = list(term_dir.glob(f"{prefix}*.txt"))
    if not matches:
        raise FileNotFoundError(f"No file matching {prefix}* under {term_dir}")
    return matches[0]


def normalize_term(text: str) -> str:
    if text is None:
        return ""
    t = str(text).lower().strip()
    if t in ("nan", "none"):
        return ""
    t = re.sub(r"\(.*?\)", " ", t)  # drop (disorder) etc.
    t = re.sub(r"[^a-z0-9\s\-/+]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Similarity: MiniLM sentence embeddings (preferred) + char n-gram fallback
# ---------------------------------------------------------------------------
def _char_ngrams(text: str, n: int = 3) -> Counter:
    s = f" {normalize_term(text)} "
    if len(s) < n:
        return Counter([s])
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def cosine_similarity_text(a: str, b: str, n: int = 3) -> float:
    """Char 3-gram cosine (lexical fallback when MiniLM is unavailable)."""
    ca, cb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def cosine_distance_text(a: str, b: str) -> float:
    return 1.0 - cosine_similarity_text(a, b)


class TextEmbedder:
    """
    Local sentence-transformer encoder (default: all-MiniLM-L6-v2).

    Encodes terms once, L2-normalizes so cosine = dot product.
    Falls back to char n-gram cosine if sentence-transformers is missing.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        prefer_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.prefer_embeddings = prefer_embeddings
        self._model = None
        self._vectors: Dict[str, Any] = {}
        self.method = "char_ngram"
        self._load_attempted = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted or not self.prefer_embeddings:
            return False
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            print(f"Loading embedding model: {self.model_name} (local MiniLM)...")
            self._model = SentenceTransformer(self.model_name)
            self.method = "minilm"
            return True
        except Exception as exc:  # noqa: BLE001 — graceful offline fallback
            print(
                f"WARNING: MiniLM unavailable ({exc}). "
                "Falling back to char n-gram cosine."
            )
            self.method = "char_ngram"
            self._model = None
            return False

    def encode_many(self, texts: Sequence[str], show_progress: bool = True) -> None:
        """Encode unique non-empty strings; cache vectors."""
        uniq = []
        seen: Set[str] = set()
        for t in texts:
            key = (t or "").strip()
            if not key or key in seen or key in self._vectors:
                continue
            seen.add(key)
            uniq.append(key)
        if not uniq:
            return
        if not self._ensure_model():
            return
        import numpy as np

        embs = self._model.encode(
            uniq,
            batch_size=self.batch_size,
            show_progress_bar=show_progress and len(uniq) > 32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        for t, v in zip(uniq, embs):
            self._vectors[t] = np.asarray(v, dtype=np.float32)

    def similarity(self, a: str, b: str) -> float:
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return 0.0
        if self.method == "minilm" and a in self._vectors and b in self._vectors:
            import numpy as np

            return float(np.dot(self._vectors[a], self._vectors[b]))
        # encode on demand if model is up
        if self._ensure_model():
            missing = [t for t in (a, b) if t not in self._vectors]
            if missing:
                self.encode_many(missing, show_progress=False)
            if a in self._vectors and b in self._vectors:
                import numpy as np

                return float(np.dot(self._vectors[a], self._vectors[b]))
        return cosine_similarity_text(a, b)


def score_candidates(
    anchor: str,
    candidates: Sequence[Dict[str, Any]],
    sim_fn,
    min_similarity: float = DEFAULT_MIN_COSINE_SIM,
    high_conf_similarity: float = DEFAULT_HIGH_CONF_COSINE_SIM,
    method: str = "minilm",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Score candidates vs anchor (patient IE term).
    Return (all_scored, retained_only) where retained means sim ≥ min_similarity.
    """
    scored: List[Dict[str, Any]] = []
    retained: List[Dict[str, Any]] = []
    for c in candidates:
        term = c.get("term") or ""
        sim = float(sim_fn(anchor, term))
        # clip numerical noise outside [-1, 1]
        sim = max(-1.0, min(1.0, sim))
        dist = 1.0 - sim
        keep = sim >= min_similarity - 1e-9
        row = {
            **c,
            "cosine_similarity": round(sim, 4),
            "cosine_distance": round(dist, 4),
            "retained": keep,
            "high_confidence": sim >= high_conf_similarity - 1e-9,
            "similarity_method": method,
            "anchor_term": anchor,
        }
        scored.append(row)
        if keep:
            retained.append(row)
    return scored, retained


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
@dataclass
class SnomedIndex:
    """In-memory SNOMED lookup + relationships (active Snapshot rows only)."""

    snomed_root: Path
    # conceptId -> preferred / FSN display term
    concept_fsn: Dict[str, str] = field(default_factory=dict)
    concept_pt: Dict[str, str] = field(default_factory=dict)
    # normalized term -> list of conceptIds (active synonyms)
    term_to_concepts: Dict[str, List[str]] = field(default_factory=dict)
    # first-token / significant token -> list of normalized terms (for fuzzy)
    token_to_terms: Dict[str, List[str]] = field(default_factory=dict)
    # sourceId -> list of (typeId, destinationId)
    outgoing: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    # reverse: destinationId -> list of (typeId, sourceId) for "cause_of" inverse walks
    incoming: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    active_concepts: Set[str] = field(default_factory=set)

    def display_term(self, concept_id: str) -> str:
        return (
            self.concept_pt.get(concept_id)
            or self.concept_fsn.get(concept_id)
            or concept_id
        )


def build_snomed_index(
    snomed_root: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> SnomedIndex:
    root = Path(snomed_root) if snomed_root else find_snomed_root()
    term_dir = snomed_terminology_dir(root)
    cache_path = Path(cache_path) if cache_path else (
        REPO_ROOT / "data" / "snomed_index" / "snomed_index.pkl"
    )
    if cache_path.exists() and not force_rebuild:
        print(f"Loading SNOMED index cache → {cache_path}")
        with cache_path.open("rb") as f:
            idx: SnomedIndex = pickle.load(f)
        idx.snomed_root = root
        return idx

    print(f"Building SNOMED index from {root} (one-time, may take 1–3 min)...")
    idx = SnomedIndex(snomed_root=root)

    # Concepts
    concept_path = _find_file(term_dir, "sct2_Concept_Snapshot")
    concepts = pd.read_csv(
        concept_path, sep="\t", dtype=str, usecols=["id", "active"]
    )
    concepts = concepts[concepts["active"] == "1"]
    idx.active_concepts = set(concepts["id"].tolist())
    print(f"  Active concepts: {len(idx.active_concepts):,}")

    # Descriptions
    desc_path = _find_file(term_dir, "sct2_Description_Snapshot")
    term_map: Dict[str, List[str]] = defaultdict(list)
    fsn: Dict[str, str] = {}
    pt: Dict[str, str] = {}
    n_desc = 0
    for chunk in pd.read_csv(
        desc_path,
        sep="\t",
        dtype=str,
        usecols=["active", "conceptId", "typeId", "term"],
        chunksize=250_000,
    ):
        chunk = chunk[chunk["active"] == "1"]
        chunk = chunk[chunk["conceptId"].isin(idx.active_concepts)]
        for _, row in chunk.iterrows():
            cid = row["conceptId"]
            term = row["term"] or ""
            tid = row["typeId"]
            if tid == TYPE_FSN:
                fsn[cid] = term
            elif tid == TYPE_SYNONYM:
                # first synonym seen as provisional PT (FSN is more unique)
                if cid not in pt:
                    pt[cid] = term
            norm = normalize_term(term)
            if norm:
                term_map[norm].append(cid)
            n_desc += 1
    idx.concept_fsn = fsn
    idx.concept_pt = pt
    # Deduplicate concept lists per term
    idx.term_to_concepts = {k: list(dict.fromkeys(v)) for k, v in term_map.items()}
    # Token inverted index for fuzzy search (skip ultra-short tokens)
    tok_inv: Dict[str, List[str]] = defaultdict(list)
    for t_norm in idx.term_to_concepts:
        for tok in t_norm.split():
            if len(tok) < 3:
                continue
            tok_inv[tok].append(t_norm)
    # Cap posting lists to keep fuzzy search fast
    idx.token_to_terms = {
        k: v[:400] if len(v) > 400 else v for k, v in tok_inv.items()
    }
    print(f"  Descriptions indexed: {n_desc:,} | unique terms: {len(idx.term_to_concepts):,}")

    # Relationships (is_a + attribute types of interest)
    keep_types = {IS_A, *ATTR_TYPE_IDS.values()}
    rel_path = _find_file(term_dir, "sct2_Relationship_Snapshot")
    outgoing: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    incoming: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_rel = 0
    for chunk in pd.read_csv(
        rel_path,
        sep="\t",
        dtype=str,
        usecols=["active", "sourceId", "destinationId", "typeId"],
        chunksize=400_000,
    ):
        chunk = chunk[chunk["active"] == "1"]
        chunk = chunk[chunk["typeId"].isin(keep_types)]
        for _, row in chunk.iterrows():
            s, t, d = row["sourceId"], row["typeId"], row["destinationId"]
            outgoing[s].append((t, d))
            incoming[d].append((t, s))
            n_rel += 1
    idx.outgoing = dict(outgoing)
    idx.incoming = dict(incoming)
    print(f"  Relationships indexed: {n_rel:,}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached index → {cache_path}")
    return idx


# ---------------------------------------------------------------------------
# Entity extraction from patient records / IE JSON
# ---------------------------------------------------------------------------
def extract_entities_from_ie(extracted: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flatten IE JSON to mappable clinical entities."""
    entities: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def add(term: str, kind: str, status: str = "") -> None:
        term = (term or "").strip()
        if not term:
            return
        key = normalize_term(term)
        if not key or key in seen:
            return
        seen.add(key)
        entities.append({"term": term, "kind": kind, "status": status or ""})

    for s in extracted.get("symptoms") or []:
        if isinstance(s, dict):
            add(s.get("term", ""), "symptom", str(s.get("status", "")))

    for d in extracted.get("diagnoses_mentioned") or []:
        if isinstance(d, dict):
            add(d.get("term", ""), "diagnosis", str(d.get("certainty", "")))

    for p in extracted.get("procedures") or []:
        if isinstance(p, dict):
            add(p.get("name", ""), "procedure")

    for m in extracted.get("medications") or []:
        if isinstance(m, dict):
            add(m.get("name", ""), "medication", str(m.get("status", "")))

    for lab in extracted.get("labs") or []:
        if isinstance(lab, dict):
            add(lab.get("name", ""), "lab")

    for t in extracted.get("temporal") or []:
        if isinstance(t, dict):
            add(t.get("finding", ""), "temporal")

    return entities


def collect_entities_from_patient_records(
    export_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Walk patient_records/*/admissions/*/information_extraction.json
    → list of {patient_id, hadm_id, term, kind, status}.
    """
    export_dir = Path(export_dir)
    rows: List[Dict[str, Any]] = []
    for ie_path in sorted(export_dir.glob("patient_*/admissions/hadm_*/information_extraction.json")):
        patient_id = ""
        hadm_id = ""
        for p in ie_path.parts:
            if p.startswith("patient_"):
                patient_id = p.replace("patient_", "")
            if p.startswith("hadm_"):
                hadm_id = p.replace("hadm_", "")
        try:
            extracted = json.loads(ie_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for ent in extract_entities_from_ie(extracted):
            rows.append(
                {
                    "patient_id": patient_id,
                    "hadm_id": hadm_id,
                    **ent,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Stage 5 — entity → SNOMED mapping
# ---------------------------------------------------------------------------
_NEGATION_TOKENS = frozenset(
    {
        "no",
        "not",
        "non",
        "without",
        "absent",
        "absence",
        "negative",
        "denied",
        "denies",
        "never",
        "none",
    }
)


def _term_has_negation(norm: str) -> bool:
    return bool(set(norm.split()) & _NEGATION_TOKENS)


def _fsn_semantic_tag(fsn: Optional[str]) -> str:
    if not fsn:
        return ""
    m = re.search(r"\(([^)]+)\)\s*$", fsn)
    return (m.group(1) if m else "").lower()


def map_term_to_snomed(
    term: str,
    index: SnomedIndex,
    top_k: int = 3,
    min_fuzzy_score: float = 0.55,
) -> Dict[str, Any]:
    """
    Lexical map term → SNOMED concept.

    Strategy: exact normalized match → token overlap / char-ngram rank among
    candidate multi-word partials. Prefer clinical findings/disorders; skip
    polarity flips (e.g. map "bleeding" → not "No ... blood loss").
    """
    raw = (term or "").strip()
    norm = normalize_term(raw)
    empty = {
        "query_term": raw,
        "mapped": False,
        "concept_id": None,
        "preferred_term": None,
        "fsn": None,
        "match_method": None,
        "score": 0.0,
        "candidates": [],
    }
    if not norm:
        return empty

    # Exact
    if norm in index.term_to_concepts:
        cid = index.term_to_concepts[norm][0]
        return {
            "query_term": raw,
            "mapped": True,
            "concept_id": cid,
            "preferred_term": index.display_term(cid),
            "fsn": index.concept_fsn.get(cid),
            "match_method": "exact",
            "score": 1.0,
            "candidates": [],
        }

    # Token-set / substring: score candidates via token inverted index
    tokens = [t for t in norm.split() if len(t) >= 3]
    candidate_terms: Set[str] = set()
    for tok in tokens[:6]:
        for t_norm in index.token_to_terms.get(tok, [])[:200]:
            candidate_terms.add(t_norm)
    if not candidate_terms and tokens:
        for tok in tokens:
            for t_norm in index.token_to_terms.get(tok, [])[:50]:
                candidate_terms.add(t_norm)

    q_neg = _term_has_negation(norm)
    scored: List[Tuple[float, str, str]] = []
    q_tokens = set(norm.split())
    preferred_tags = {
        "finding",
        "disorder",
        "situation",
        "observable entity",
        "procedure",
        "substance",
        "product",
        "morphologic abnormality",
        "body structure",
    }

    for t_norm in candidate_terms:
        if _term_has_negation(t_norm) != q_neg:
            continue  # polarity flip
        t_tokens = set(t_norm.split())
        if not t_tokens:
            continue
        jacc = len(q_tokens & t_tokens) / max(len(q_tokens | t_tokens), 1)
        cos = cosine_similarity_text(norm, t_norm)
        score = 0.55 * cos + 0.45 * jacc
        if score < min_fuzzy_score:
            continue
        for cid in index.term_to_concepts.get(t_norm, [])[:2]:
            tag = _fsn_semantic_tag(index.concept_fsn.get(cid))
            # mild boost for clinical semantic tags; mild penalty for qualifiers/navigational
            adj = score
            if tag in preferred_tags:
                adj += 0.03
            if tag in ("qualifier value", "namespace concept", "linkage concept"):
                adj -= 0.08
            scored.append((adj, t_norm, cid))

    if not scored:
        # fallback: progressive truncations (still exact on a shorter phrase)
        for cut in (norm, " ".join(norm.split()[:4]), " ".join(norm.split()[:2])):
            if cut and cut in index.term_to_concepts:
                cid = index.term_to_concepts[cut][0]
                return {
                    "query_term": raw,
                    "mapped": True,
                    "concept_id": cid,
                    "preferred_term": index.display_term(cid),
                    "fsn": index.concept_fsn.get(cid),
                    "match_method": "prefix_exact",
                    "score": 0.9,
                    "candidates": [],
                }
        return empty

    scored.sort(key=lambda x: (-x[0], x[1]))
    seen: Set[str] = set()
    top: List[Tuple[float, str, str]] = []
    for sc, tn, cid in scored:
        if cid in seen:
            continue
        seen.add(cid)
        top.append((sc, tn, cid))
        if len(top) >= top_k:
            break

    best_sc, _best_tn, best_cid = top[0]
    return {
        "query_term": raw,
        "mapped": True,
        "concept_id": best_cid,
        "preferred_term": index.display_term(best_cid),
        "fsn": index.concept_fsn.get(best_cid),
        "match_method": "fuzzy",
        "score": float(best_sc),
        "candidates": [
            {
                "concept_id": cid,
                "term": index.display_term(cid),
                "score": float(sc),
            }
            for sc, _, cid in top[1:]
        ],
    }


def map_entities(
    entities: Sequence[Dict[str, Any]],
    index: SnomedIndex,
) -> List[Dict[str, Any]]:
    """Map a list of entity dicts (must have 'term'). Unique terms cached."""
    cache: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    for ent in entities:
        term = ent.get("term", "")
        key = normalize_term(term)
        if key not in cache:
            cache[key] = map_term_to_snomed(term, index)
        mapping = cache[key]
        out.append({**ent, "snomed": mapping})
    return out


# ---------------------------------------------------------------------------
# Stage 6 — ancestors / attribute targets + cosine distance filter
# ---------------------------------------------------------------------------
def is_a_parents(index: SnomedIndex, concept_id: str) -> List[str]:
    parents = []
    for type_id, dest in index.outgoing.get(concept_id, []):
        if type_id == IS_A:
            parents.append(dest)
    return parents


def is_a_ancestors_depth2(index: SnomedIndex, concept_id: str) -> List[Dict[str, Any]]:
    """Parent (depth 1) and grandparent (depth 2) via Is a."""
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for p in is_a_parents(index, concept_id):
        if p in seen:
            continue
        seen.add(p)
        results.append(
            {
                "concept_id": p,
                "term": index.display_term(p),
                "relation": "is_a",
                "depth": 1,
                "role": "ancestor",
            }
        )
        for gp in is_a_parents(index, p):
            if gp in seen:
                continue
            seen.add(gp)
            results.append(
                {
                    "concept_id": gp,
                    "term": index.display_term(gp),
                    "relation": "is_a",
                    "depth": 2,
                    "role": "ancestor",
                }
            )
    return results


def attribute_related_concepts(
    index: SnomedIndex,
    concept_id: str,
    include_inverse: bool = False,
) -> List[Dict[str, Any]]:
    """
    Collect **outbound** attribute destinations for configured relationship types,
    plus one Is-a parent of each destination (second hop / ancestor of the attribute target).

    Inverses are off by default: inbound has_causative_agent on drugs, for example,
    surfaces poisonings/overdoses rather than clinical coding context.
    """
    type_name = {v: k for k, v in ATTR_TYPE_IDS.items()}
    results: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()  # (relation, hop, concept_id)

    def add(rel: str, hop: str, cid: str, via: Optional[str] = None) -> None:
        key = (rel, hop, cid)
        if key in seen or cid == concept_id:
            return
        seen.add(key)
        results.append(
            {
                "concept_id": cid,
                "term": index.display_term(cid),
                "relation": rel,
                "hop": hop,
                "via_concept_id": via,
                "role": "attribute_target" if hop == "direct" else "attribute_target_ancestor",
            }
        )

    for type_id, dest in index.outgoing.get(concept_id, []):
        if type_id not in type_name:
            continue
        rel = type_name[type_id]
        add(rel, "direct", dest)
        for p in is_a_parents(index, dest)[:2]:
            add(rel, "ancestor_of_target", p, via=dest)

    if include_inverse:
        # Opt-in only: inbound edges of same attribute types
        for type_id, src in index.incoming.get(concept_id, []):
            if type_id not in type_name:
                continue
            rel = type_name[type_id] + "_inverse"
            add(rel, "direct", src)
            for p in is_a_parents(index, src)[:2]:
                add(rel, "ancestor_of_target", p, via=src)

    return results


def filter_by_min_similarity(
    anchor: str,
    candidates: Sequence[Dict[str, Any]],
    min_similarity: float = DEFAULT_MIN_COSINE_SIM,
    high_conf_similarity: float = DEFAULT_HIGH_CONF_COSINE_SIM,
    sim_fn=None,
    method: str = "char_ngram",
) -> List[Dict[str, Any]]:
    """Keep candidates with cosine similarity ≥ min_similarity vs *anchor* (IE term)."""
    if sim_fn is None:
        sim_fn = cosine_similarity_text
    _scored, retained = score_candidates(
        anchor,
        candidates,
        sim_fn,
        min_similarity=min_similarity,
        high_conf_similarity=high_conf_similarity,
        method=method,
    )
    return retained


# backwards-compatible alias
def filter_by_cosine_distance(
    mapped_term: str,
    candidates: Sequence[Dict[str, Any]],
    max_distance: float = DEFAULT_MAX_COSINE_DISTANCE,
) -> List[Dict[str, Any]]:
    return filter_by_min_similarity(
        mapped_term,
        candidates,
        min_similarity=1.0 - max_distance,
    )


def enrich_mapped_entity_with_context(
    mapped: Dict[str, Any],
    index: SnomedIndex,
    min_similarity: float = DEFAULT_MIN_COSINE_SIM,
    high_conf_similarity: float = DEFAULT_HIGH_CONF_COSINE_SIM,
    sim_fn=None,
    method: str = "char_ngram",
) -> Dict[str, Any]:
    """
    For one Stage-5 mapping: gather 2-level is_a ancestors + attribute relation
    targets (and their ancestors). Score vs **patient IE entity term** (not SNOMED PT).
    Retain if cosine similarity ≥ min_similarity (default 0.70).
    """
    if sim_fn is None:
        sim_fn = cosine_similarity_text

    snomed = mapped.get("snomed") or {}
    empty_ctx = {
        "ancestors_depth2_all": [],
        "attribute_relations_all": [],
        "retained_ancestors": [],
        "retained_attribute_relations": [],
        "retained": [],
        "min_cosine_similarity": min_similarity,
        "high_conf_similarity": high_conf_similarity,
        "max_cosine_distance": round(1.0 - min_similarity, 4),
        "similarity_method": method,
        "anchor_field": "ie_entity_term",
        "weights": None,
    }
    if not snomed.get("mapped") or not snomed.get("concept_id"):
        return {**mapped, "ontology_context": empty_ctx}

    cid = snomed["concept_id"]
    # Anchor on the original extracted entity from patient records
    anchor = (mapped.get("term") or "").strip()
    if not anchor:
        anchor = snomed.get("preferred_term") or snomed.get("fsn") or ""

    ancestors = is_a_ancestors_depth2(index, cid)
    attr_related = attribute_related_concepts(index, cid)

    anc_scored, retained_ancestors = score_candidates(
        anchor,
        ancestors,
        sim_fn,
        min_similarity=min_similarity,
        high_conf_similarity=high_conf_similarity,
        method=method,
    )
    attr_scored, retained_attrs = score_candidates(
        anchor,
        attr_related,
        sim_fn,
        min_similarity=min_similarity,
        high_conf_similarity=high_conf_similarity,
        method=method,
    )
    retained = retained_ancestors + retained_attrs

    return {
        **mapped,
        "ontology_context": {
            "mapped_concept_id": cid,
            "mapped_snomed_term": snomed.get("preferred_term") or snomed.get("fsn"),
            "anchor_term": anchor,
            "anchor_field": "ie_entity_term",
            "ancestors_depth2_all": anc_scored,
            "attribute_relations_all": attr_scored,
            "retained_ancestors": retained_ancestors,
            "retained_attribute_relations": retained_attrs,
            "retained": retained,
            "n_retained": len(retained),
            "n_high_confidence": sum(1 for r in retained if r.get("high_confidence")),
            "min_cosine_similarity": min_similarity,
            "high_conf_similarity": high_conf_similarity,
            "max_cosine_distance": round(1.0 - min_similarity, 4),
            "similarity_method": method,
            "attribute_types": ATTR_TYPE_IDS,
            "weights": None,  # future work
        },
    }


def run_stage05_mapping(
    entities: Sequence[Dict[str, Any]],
    index: SnomedIndex,
) -> Dict[str, Any]:
    mapped = map_entities(entities, index)
    n_mapped = sum(1 for m in mapped if (m.get("snomed") or {}).get("mapped"))
    return {
        "stage": 5,
        "description": "Entity → SNOMED CT mapping (offline RF2 lexical)",
        "generated_at": datetime.now().isoformat(),
        "n_entities": len(mapped),
        "n_mapped": n_mapped,
        "n_unmapped": len(mapped) - n_mapped,
        "results": mapped,
    }


def run_stage06_ancestors(
    stage05_payload: Dict[str, Any],
    index: SnomedIndex,
    min_similarity: float = DEFAULT_MIN_COSINE_SIM,
    high_conf_similarity: float = DEFAULT_HIGH_CONF_COSINE_SIM,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    use_embeddings: bool = True,
    max_distance: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Stage 6: is_a ×2 + attribute targets, filtered by embedding cosine vs IE term.

    - default: sentence-transformers all-MiniLM-L6-v2
    - retain if similarity ≥ min_similarity (default 0.70)
    - flag high_confidence if ≥ high_conf_similarity (default 0.80)
    - max_distance: optional legacy override (min_similarity = 1 - max_distance)
    """
    if max_distance is not None:
        min_similarity = 1.0 - max_distance

    rows = list(stage05_payload.get("results") or [])

    # Collect strings for a single batch encode
    texts_to_encode: List[str] = []
    prepared: List[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]] = []
    for row in rows:
        snomed = row.get("snomed") or {}
        if not snomed.get("mapped") or not snomed.get("concept_id"):
            prepared.append((row, [], []))
            continue
        cid = snomed["concept_id"]
        ancestors = is_a_ancestors_depth2(index, cid)
        attrs = attribute_related_concepts(index, cid)
        prepared.append((row, ancestors, attrs))
        anchor = (row.get("term") or "").strip() or snomed.get("preferred_term") or ""
        if anchor:
            texts_to_encode.append(anchor)
        for c in ancestors + attrs:
            if c.get("term"):
                texts_to_encode.append(c["term"])

    embedder = TextEmbedder(
        model_name=embedding_model,
        prefer_embeddings=use_embeddings,
    )
    if use_embeddings:
        embedder.encode_many(texts_to_encode, show_progress=True)
    method = embedder.method
    sim_fn = embedder.similarity

    results = []
    for row, ancestors, attrs in prepared:
        snomed = row.get("snomed") or {}
        if not snomed.get("mapped") or not snomed.get("concept_id"):
            results.append(
                enrich_mapped_entity_with_context(
                    row,
                    index,
                    min_similarity=min_similarity,
                    high_conf_similarity=high_conf_similarity,
                    sim_fn=sim_fn,
                    method=method,
                )
            )
            continue
        # reuse precomputed trees — inject via temporary re-score path
        anchor = (row.get("term") or "").strip()
        if not anchor:
            anchor = snomed.get("preferred_term") or snomed.get("fsn") or ""
        anc_scored, retained_ancestors = score_candidates(
            anchor,
            ancestors,
            sim_fn,
            min_similarity=min_similarity,
            high_conf_similarity=high_conf_similarity,
            method=method,
        )
        attr_scored, retained_attrs = score_candidates(
            anchor,
            attrs,
            sim_fn,
            min_similarity=min_similarity,
            high_conf_similarity=high_conf_similarity,
            method=method,
        )
        retained = retained_ancestors + retained_attrs
        results.append(
            {
                **row,
                "ontology_context": {
                    "mapped_concept_id": snomed["concept_id"],
                    "mapped_snomed_term": snomed.get("preferred_term") or snomed.get("fsn"),
                    "anchor_term": anchor,
                    "anchor_field": "ie_entity_term",
                    "ancestors_depth2_all": anc_scored,
                    "attribute_relations_all": attr_scored,
                    "retained_ancestors": retained_ancestors,
                    "retained_attribute_relations": retained_attrs,
                    "retained": retained,
                    "n_retained": len(retained),
                    "n_high_confidence": sum(
                        1 for r in retained if r.get("high_confidence")
                    ),
                    "min_cosine_similarity": min_similarity,
                    "high_conf_similarity": high_conf_similarity,
                    "max_cosine_distance": round(1.0 - min_similarity, 4),
                    "similarity_method": method,
                    "embedding_model": embedding_model if method == "minilm" else None,
                    "attribute_types": ATTR_TYPE_IDS,
                    "weights": None,
                },
            }
        )

    n_retained = sum(
        len((r.get("ontology_context") or {}).get("retained") or []) for r in results
    )
    n_high = sum(
        (r.get("ontology_context") or {}).get("n_high_confidence") or 0 for r in results
    )
    n_with_ret = sum(
        1
        for r in results
        if (r.get("ontology_context") or {}).get("retained")
    )
    return {
        "stage": 6,
        "description": (
            "SNOMED 2-level is_a ancestors + outbound attribute relations "
            f"({', '.join(ATTR_TYPE_IDS)}); "
            f"no inverse edges; retain MiniLM cosine similarity ≥ {min_similarity} "
            f"vs patient IE entity term (high-conf ≥ {high_conf_similarity})"
        ),
        "generated_at": datetime.now().isoformat(),
        "min_cosine_similarity": min_similarity,
        "high_conf_similarity": high_conf_similarity,
        "max_cosine_distance": round(1.0 - min_similarity, 4),
        "similarity_method": method,
        "embedding_model": embedding_model if method == "minilm" else None,
        "anchor_field": "ie_entity_term",
        "attribute_types": ATTR_TYPE_IDS,
        "weights": None,
        "n_entities": len(results),
        "n_entities_with_retained": n_with_ret,
        "n_retained_context_links": n_retained,
        "n_high_confidence_links": n_high,
        "results": results,
    }


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def export_mappings_to_patient_folders(
    payload: Dict[str, Any],
    export_dir: Path,
    filename: str = "snomed_mapping.json",
) -> int:
    """Group Stage 5/6 results by patient/hadm and write into admission folders."""
    export_dir = Path(export_dir)
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in payload.get("results") or []:
        pid, hid = str(row.get("patient_id")), str(row.get("hadm_id"))
        by_key[(pid, hid)].append(row)

    n = 0
    for (pid, hid), rows in by_key.items():
        adm = export_dir / f"patient_{pid}" / "admissions" / f"hadm_{hid}"
        if not adm.is_dir():
            continue
        out = {
            "stage": payload.get("stage"),
            "generated_at": payload.get("generated_at"),
            "patient_id": pid,
            "hadm_id": hid,
            "n_entities": len(rows),
            "entities": rows,
        }
        write_json(adm / filename, out)
        n += 1
    return n


def _slim_retained_entity(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Entity view containing only retained context links."""
    ctx = row.get("ontology_context") or {}
    retained = ctx.get("retained") or []
    if not retained:
        return None
    sn = row.get("snomed") or {}
    return {
        "term": row.get("term"),
        "kind": row.get("kind"),
        "status": row.get("status"),
        "snomed_concept_id": sn.get("concept_id"),
        "snomed_preferred_term": sn.get("preferred_term"),
        "anchor_term": ctx.get("anchor_term") or row.get("term"),
        "n_retained": len(retained),
        "n_high_confidence": sum(1 for r in retained if r.get("high_confidence")),
        "retained_ancestors": ctx.get("retained_ancestors") or [],
        "retained_attribute_relations": ctx.get("retained_attribute_relations") or [],
        "retained": retained,
    }


def format_retained_txt(out: Dict[str, Any]) -> str:
    """Human-readable retained summary for an admission."""
    lines = [
        f"Patient: {out.get('patient_id')}  HADM: {out.get('hadm_id')}",
        f"Generated: {out.get('generated_at')}",
        f"Min cosine similarity: {out.get('min_cosine_similarity')}",
        f"High-confidence: {out.get('high_conf_similarity')}",
        f"Method: {out.get('similarity_method')} ({out.get('embedding_model')})",
        f"Entities with retained: {out.get('n_entities_with_retained')} / "
        f"{out.get('n_entities')}  |  links: {out.get('n_retained_links')}",
        "",
    ]
    for ent in out.get("entities") or []:
        lines.append(
            f"• {ent.get('term')}  [{ent.get('kind')}]  "
            f"→ SNOMED: {ent.get('snomed_preferred_term')} ({ent.get('snomed_concept_id')})"
        )
        for r in ent.get("retained") or []:
            hc = " [HIGH]" if r.get("high_confidence") else ""
            lines.append(
                f"    - [{r.get('relation')}] {r.get('term')} "
                f"(id={r.get('concept_id')}, sim={r.get('cosine_similarity')}){hc}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_retained_to_patient_folders(
    payload: Dict[str, Any],
    export_dir: Path,
    json_name: str = "snomed_retained.json",
    txt_name: str = "snomed_retained.txt",
) -> int:
    """
    Write slim retained-only JSON + readable TXT into each admission folder.

    Path: patient_records/patient_<id>/admissions/hadm_<id>/snomed_retained.{json,txt}
    """
    export_dir = Path(export_dir)
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in payload.get("results") or []:
        pid, hid = str(row.get("patient_id")), str(row.get("hadm_id"))
        by_key[(pid, hid)].append(row)

    n = 0
    for (pid, hid), rows in by_key.items():
        adm = export_dir / f"patient_{pid}" / "admissions" / f"hadm_{hid}"
        if not adm.is_dir():
            continue
        slim = []
        for row in rows:
            s = _slim_retained_entity(row)
            if s:
                slim.append(s)
        n_links = sum(len(e.get("retained") or []) for e in slim)
        out = {
            "stage": 6,
            "description": "Retained ontology context only (MiniLM sim ≥ threshold)",
            "generated_at": payload.get("generated_at"),
            "patient_id": pid,
            "hadm_id": hid,
            "min_cosine_similarity": payload.get("min_cosine_similarity"),
            "high_conf_similarity": payload.get("high_conf_similarity"),
            "similarity_method": payload.get("similarity_method"),
            "embedding_model": payload.get("embedding_model"),
            "n_entities": len(rows),
            "n_entities_with_retained": len(slim),
            "n_retained_links": n_links,
            "entities": slim,
        }
        write_json(adm / json_name, out)
        (adm / txt_name).write_text(format_retained_txt(out), encoding="utf-8")
        n += 1
    return n
