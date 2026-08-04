"""
Shared SNOMED CT grounding + similarity utilities.

Used by:
  - snomed_similarity_app.py (interactive Streamlit exploration, at the project root)
  - stage_05_ontology_routing_agent.ipynb (batch pipeline, in this notebooks/ folder)

Both consumers import from here so the actual algorithm — search, graph
traversal, Wu-Palmer, and the big-gap neighborhood cutoff — only exists in
one place. Cached with functools.lru_cache (fine for a single process/kernel
run); callers that need a cache that survives a restart can wrap these with
their own persistence.
"""
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"
UMLS_VERSION = "current"
ROOT_SCTID = "138875005"  # "SNOMED CT Concept" — root of the entire is-a hierarchy

BIOPORTAL_BASE = "https://data.bioontology.org"

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

MAX_WORKERS = 8
MAX_NEIGHBORHOOD_CANDIDATES = 30  # hard cap on how many concepts explore_neighborhood ever returns
GAP_SEARCH_WINDOW = 15  # only look for the "big gap" within the first N ranks
GAP_OUTLIER_Z_THRESHOLD = 2.5  # modified z-score required to call a gap a genuine outlier

UMLS_API_KEY = ""  # set via configure()
BIOPORTAL_API_KEY = ""  # set via configure_bioportal()


def _load_env_file(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def load_umls_api_key(project_root: Path) -> str:
    """Read UMLS_API_KEY from <project_root>/.env, falling back to the OS environment."""
    env = _load_env_file(project_root / ".env")
    return env.get("UMLS_API_KEY") or os.environ.get("UMLS_API_KEY", "")


def configure(api_key: str):
    """Set the UMLS API key used by every function below."""
    global UMLS_API_KEY
    UMLS_API_KEY = api_key


def load_bioportal_api_key(project_root: Path) -> str:
    """Read BIOPORTAL_API_KEY from <project_root>/.env, falling back to the OS environment."""
    env = _load_env_file(project_root / ".env")
    return env.get("BIOPORTAL_API_KEY") or os.environ.get("BIOPORTAL_API_KEY", "")


def configure_bioportal(api_key: str):
    """Set the BioPortal API key used by search_snomed_bioportal()."""
    global BIOPORTAL_API_KEY
    BIOPORTAL_API_KEY = api_key


@lru_cache(maxsize=None)
def search_snomed_bioportal(term: str, n: int = 8):
    """Search SNOMED CT via BioPortal's REST API. Returns list of {sctid, name}.

    Unlike search_snomed() (UMLS), BioPortal returns SCTIDs directly — no CUI
    intermediate step needed. Note: empirically, BioPortal's relevance ranking
    is not more clinically-aware than UMLS's — tested side by side on
    "Vaginal hemorrhage", both rank the narrower "Neonatal vaginal hemorrhage"
    above the correct general concept, and BioPortal doesn't surface the
    correct concept in its top 8 at all. This is not a fix for that ranking
    problem by itself; graph traversal (parents/children) still goes through
    UMLS regardless of which search backend is used here."""
    if not BIOPORTAL_API_KEY:
        return []
    r = requests.get(
        f"{BIOPORTAL_BASE}/search",
        params={"q": term, "ontologies": "SNOMEDCT", "apikey": BIOPORTAL_API_KEY, "pagesize": n},
        timeout=15,
    )
    r.raise_for_status()
    results = []
    for item in r.json().get("collection", []):
        sctid = item.get("@id", "").split("/")[-1]
        name = item.get("prefLabel", "")
        if sctid and name:
            results.append({"sctid": sctid, "name": name})
    return results


# ── UMLS / SNOMED CT lookup ───────────────────────────────────────────────────
def _umls_search(term: str, search_type: str, n: int):
    r = requests.get(
        f"{UMLS_BASE}/search/{UMLS_VERSION}",
        params={
            "string": term,
            "sabs": "SNOMEDCT_US",
            "returnIdType": "concept",
            "searchType": search_type,
            "apiKey": UMLS_API_KEY,
            "pageSize": n,
        },
        timeout=15,
    )
    r.raise_for_status()
    results = []
    for item in r.json().get("result", {}).get("results", []):
        cui = item.get("ui", "")
        name = item.get("name", "")
        if cui and cui != "NONE":
            results.append({"cui": cui, "name": name})
    return results


@lru_cache(maxsize=None)
def search_snomed(term: str, n: int = 8):
    """Search SNOMED CT (US edition) for a term. Returns list of {cui, name}.

    Combines UMLS's default fuzzy word-based search with an exact-match query,
    so the literal term is always present even when fuzzy ranking would
    otherwise bump it out of the top N.
    """
    if not UMLS_API_KEY:
        return []
    fuzzy = _umls_search(term, "words", n)
    exact = _umls_search(term, "exact", 3)
    seen = {r["cui"] for r in exact}
    return exact + [r for r in fuzzy if r["cui"] not in seen]


@lru_cache(maxsize=None)
def get_sctid(cui: str):
    """Best-effort fetch of the real SNOMED CT concept ID (SCTID) for a CUI."""
    if not UMLS_API_KEY or not cui:
        return None
    r = requests.get(
        f"{UMLS_BASE}/content/{UMLS_VERSION}/CUI/{cui}/atoms",
        params={"sabs": "SNOMEDCT_US", "apiKey": UMLS_API_KEY, "pageSize": 5},
        timeout=15,
    )
    if r.status_code != 200:
        return None
    for atom in r.json().get("result", []):
        code_url = atom.get("code", "")
        code = code_url.split("/")[-1] if "/" in code_url else code_url
        if code and code.isdigit():
            return code
    return None


def _source_edges(sctid: str, relation: str):
    r = requests.get(
        f"{UMLS_BASE}/content/{UMLS_VERSION}/source/SNOMEDCT_US/{sctid}/{relation}",
        params={"apiKey": UMLS_API_KEY, "pageSize": 50},
        timeout=15,
    )
    time.sleep(0.05)
    if r.status_code != 200:
        return ()  # 404 = none (root/leaf), or a transient error
    return tuple(item.get("ui", "") for item in r.json().get("result", []) if item.get("ui"))


@lru_cache(maxsize=None)
def get_parents(sctid: str):
    """Immediate is-a parents of a SNOMED CT concept (SCTID). Cached — every
    other graph function is built on repeated calls to this one."""
    if not sctid:
        return ()
    return _source_edges(sctid, "parents")


@lru_cache(maxsize=None)
def get_children(sctid: str):
    """Immediate is-a children of a SNOMED CT concept (SCTID)."""
    if not sctid:
        return ()
    return _source_edges(sctid, "children")


@lru_cache(maxsize=None)
def get_concept_name(sctid: str):
    r = requests.get(
        f"{UMLS_BASE}/content/{UMLS_VERSION}/source/SNOMEDCT_US/{sctid}",
        params={"apiKey": UMLS_API_KEY},
        timeout=15,
    )
    time.sleep(0.05)
    if r.status_code != 200:
        return sctid
    return r.json().get("result", {}).get("name", sctid)


@lru_cache(maxsize=None)
def get_cui_for_sctid(sctid: str):
    """Reverse-lookup: CUI for a SCTID discovered via graph traversal (which
    only ever gives us source-asserted SCTIDs, not CUIs)."""
    r = requests.get(
        f"{UMLS_BASE}/search/{UMLS_VERSION}",
        params={
            "string": sctid,
            "sabs": "SNOMEDCT_US",
            "searchType": "exact",
            "inputType": "sourceUi",
            "apiKey": UMLS_API_KEY,
        },
        timeout=15,
    )
    time.sleep(0.05)
    if r.status_code != 200:
        return None
    results = r.json().get("result", {}).get("results", [])
    return results[0]["ui"] if results else None


# ── Graph-based similarity (Wu-Palmer over the is-a hierarchy) ───────────────
@lru_cache(maxsize=None)
def ancestors_with_dist(sctid: str, max_hops: int = 25):
    """BFS upward through is-a parents. Returns {ancestor_sctid: shortest hops
    from `sctid`}, including `sctid` itself at 0. SNOMED is a polyhierarchy
    (multiple parents), so this is a true shortest-path BFS, not a simple walk."""
    visited = {sctid: 0}
    frontier = {sctid}
    hops = 0
    while frontier and hops < max_hops:
        hops += 1
        next_frontier = set()
        for node in frontier:
            for parent in get_parents(node):
                if parent not in visited:
                    visited[parent] = hops
                    next_frontier.add(parent)
        frontier = next_frontier
    return visited


def depth_to_root(sctid: str) -> int:
    """Shortest-path distance from `sctid` up to the SNOMED root."""
    dists = ancestors_with_dist(sctid)
    if ROOT_SCTID in dists:
        return dists[ROOT_SCTID]
    return max(dists.values()) if dists else 0


@lru_cache(maxsize=None)
def wu_palmer(sctid_a: str, sctid_b: str):
    """Wu-Palmer similarity: 2*depth(LCS) / (depth(A) + depth(B)), where LCS is
    the deepest (most specific) shared ancestor and depth is shortest hops to
    the SNOMED root. Returns None if either SCTID is missing/unresolvable."""
    if not sctid_a or not sctid_b:
        return None
    if sctid_a == sctid_b:
        return 1.0

    anc_a = ancestors_with_dist(sctid_a)
    anc_b = ancestors_with_dist(sctid_b)
    common = set(anc_a) & set(anc_b)
    if not common:
        return None

    # LCS = the common ancestor with the greatest absolute depth (most specific);
    # ties broken by whichever candidate needs the fewest combined hops from A
    # and B. Ties are common — e.g. whenever one concept is a direct parent of
    # the other. Breaking ties with plain max() over a set picks whichever
    # candidate Python's (hash-randomized) set iteration happens to visit last,
    # making the score non-deterministic across process runs. Sorting first and
    # breaking ties by combined hop distance is both deterministic and the
    # semantically correct choice.
    common_sorted = sorted(common)
    lcs = min(common_sorted, key=lambda x: (-depth_to_root(x), anc_a[x] + anc_b[x]))
    depth_lcs = depth_to_root(lcs)

    # Depth of A and B *relative to this LCS* (hops to it plus its own depth),
    # not each concept's independently-shortest path to root. SNOMED is a
    # polyhierarchy, so those two numbers can differ — using each concept's own
    # independent depth let the LCS come out "deeper" than A or B themselves,
    # pushing the score above 1. Anchoring both sides to the same LCS path
    # guarantees depth_lcs <= depth_a and depth_lcs <= depth_b, so the ratio
    # stays in [0, 1] (mirrors how NLTK's wup_similarity handles WordNet's own
    # multiple-inheritance hypernym graph).
    depth_a = anc_a[lcs] + depth_lcs
    depth_b = anc_b[lcs] + depth_lcs
    if depth_a + depth_b == 0:
        return None

    return round(2 * depth_lcs / (depth_a + depth_b), 4)


@lru_cache(maxsize=None)
def wu_palmer_with_lcs(sctid_a: str, sctid_b: str):
    """Same computation as wu_palmer(), but also returns *which* concept was
    the shared ancestor and its depth -- used where the caller needs to
    explain *why* two concepts are related (e.g. Stage 6's cross-symptom
    routing), not just how related. Kept as a separate function rather than
    changing wu_palmer()'s return shape, since wu_palmer() is already relied
    on elsewhere (the app, Stage 5) expecting a bare float/None.

    Returns a dict {score, lcs_sctid, lcs_depth}, or None if no shared
    ancestor exists (e.g. the two concepts sit in unrelated branches of the
    hierarchy and only share the absolute SNOMED root -- which still counts
    as a shared ancestor, just at depth 0, giving score 0.0)."""
    if not sctid_a or not sctid_b:
        return None
    if sctid_a == sctid_b:
        d = depth_to_root(sctid_a)
        return {"score": 1.0, "lcs_sctid": sctid_a, "lcs_depth": d}

    anc_a = ancestors_with_dist(sctid_a)
    anc_b = ancestors_with_dist(sctid_b)
    common = set(anc_a) & set(anc_b)
    if not common:
        return None

    common_sorted = sorted(common)
    lcs = min(common_sorted, key=lambda x: (-depth_to_root(x), anc_a[x] + anc_b[x]))
    depth_lcs = depth_to_root(lcs)

    depth_a = anc_a[lcs] + depth_lcs
    depth_b = anc_b[lcs] + depth_lcs
    if depth_a + depth_b == 0:
        return None

    score = round(2 * depth_lcs / (depth_a + depth_b), 4)
    return {"score": score, "lcs_sctid": lcs, "lcs_depth": depth_lcs}


# SNOMED's defining attributes (as opposed to the is-a hierarchy) -- these
# capture *why* a concept is what it is, independent of where it sits in the
# is-a tree. Two concepts can be structurally distant (different is-a
# branches entirely -- e.g. a neoplasm vs. a metabolic disorder) while still
# sharing a real causal/anatomical link that only these attributes reveal
# (e.g. both having the same finding_site). This is exactly what Wu-Palmer's
# pure graph-distance measure cannot see on its own.
DEFINING_ATTRIBUTES = {
    "has_finding_site": "finding_site",
    "has_causative_agent": "causative_agent",
    "has_associated_morphology": "associated_morphology",
    "has_pathological_process": "pathological_process",
}

# cause_of/due_to are handled separately (see direct_causal_link below), not as
# part of DEFINING_ATTRIBUTES -- they're technically "associative relationships"
# in SNOMED's terminology, not formal defining attributes, and they need a
# different comparison: a DIRECT link between two specific concepts ("does A
# cause B?"), not "do A and B both reference the same third-party value?"
# (which is what shared_attributes() below checks for the four attributes
# above). Checking a full relations scan surfaced several other relation
# labels too (mapped_to, possibly_equivalent_to, same_as, referred_to_by,
# isa/inverse_isa) -- deliberately excluded: the isa-family duplicates what
# Wu-Palmer already measures via /parents /children, and the rest are
# terminology-maintenance or lexical-variant relationships, not clinical
# relatedness signals.
CAUSAL_RELATION_LABELS = {"cause_of", "due_to"}


@lru_cache(maxsize=None)
def get_defining_attributes(sctid: str):
    """Fetch a concept's defining attributes (finding_site, causative_agent,
    associated_morphology) via the same UMLS relations endpoint used
    elsewhere. Returns {"finding_site": {sctid, ...}, "causative_agent": {...},
    "associated_morphology": {...}} -- each value is a set of related SCTIDs
    (a concept can have more than one, e.g. multiple finding sites), empty
    sets for attributes the concept doesn't define."""
    result = {name: set() for name in DEFINING_ATTRIBUTES.values()}
    if not sctid:
        return result
    r = requests.get(
        f"{UMLS_BASE}/content/{UMLS_VERSION}/source/SNOMEDCT_US/{sctid}/relations",
        params={"apiKey": UMLS_API_KEY, "pageSize": 50},
        timeout=15,
    )
    time.sleep(0.05)
    if r.status_code != 200:
        return result
    for rel in r.json().get("result", []):
        label = rel.get("additionalRelationLabel", "")
        attr_name = DEFINING_ATTRIBUTES.get(label)
        if not attr_name:
            continue
        related_url = rel.get("relatedId", "")
        related_sctid = related_url.split("/")[-1] if "/" in related_url else related_url
        if related_sctid:
            result[attr_name].add(related_sctid)
    return result


def shared_attributes(sctid_a: str, sctid_b: str):
    """Compare two concepts' defining attributes. Returns
    {"finding_site": shared_sctid_or_None, "causative_agent": ..., "associated_morphology": ...,
    "pathological_process": ...} -- picks one shared value per attribute (if any) for display
    purposes; a concept pair can share more than one, but one example is enough to show *why*
    they're related."""
    attrs_a = get_defining_attributes(sctid_a)
    attrs_b = get_defining_attributes(sctid_b)
    result = {}
    for attr_name in DEFINING_ATTRIBUTES.values():
        shared = attrs_a[attr_name] & attrs_b[attr_name]
        result[attr_name] = sorted(shared)[0] if shared else None
    return result


@lru_cache(maxsize=None)
def get_causal_targets(sctid: str):
    """Fetch a concept's own cause_of/due_to relation targets (what it directly
    causes, or is directly caused by) via the same relations endpoint. Returns
    a set of related SCTIDs. Both directions are combined into one set since
    UMLS doesn't consistently populate both cause_of and due_to on both ends of
    the same causal fact -- checking one concept's own relation list can miss
    it, so direct_causal_link() checks both concepts' lists in both
    directions."""
    if not sctid:
        return set()
    r = requests.get(
        f"{UMLS_BASE}/content/{UMLS_VERSION}/source/SNOMEDCT_US/{sctid}/relations",
        params={"apiKey": UMLS_API_KEY, "pageSize": 50},
        timeout=15,
    )
    time.sleep(0.05)
    if r.status_code != 200:
        return set()
    targets = set()
    for rel in r.json().get("result", []):
        if rel.get("additionalRelationLabel", "") in CAUSAL_RELATION_LABELS:
            related_url = rel.get("relatedId", "")
            related_sctid = related_url.split("/")[-1] if "/" in related_url else related_url
            if related_sctid:
                targets.add(related_sctid)
    return targets


def direct_causal_link(sctid_a: str, sctid_b: str):
    """Does either concept directly cause (or arise from) the other? Checks
    both concepts' own cause_of/due_to targets for the other's SCTID, in both
    directions. Returns True if a direct causal link exists, False otherwise.

    This is a different kind of check from shared_attributes() -- it's not
    "do A and B reference the same third-party value", it's "does A appear in
    B's own causal relation list, or vice versa" -- a direct assertion between
    this specific pair, which is why it's tracked separately rather than
    folded into the same shared-value pattern as the four DEFINING_ATTRIBUTES."""
    if not sctid_a or not sctid_b:
        return False
    return sctid_b in get_causal_targets(sctid_a) or sctid_a in get_causal_targets(sctid_b)


@lru_cache(maxsize=None)
def get_semantic_tag(sctid: str):
    """Fetch a SNOMED concept's semantic tag -- the parenthesized category at
    the end of its Fully Specified Name, e.g. "Liver cell carcinoma (disorder)"
    -> "disorder", "Hepatocellular carcinoma (morphologic abnormality)" ->
    "morphologic abnormality". This is SNOMED's own top-level categorization,
    and it's the cleanest way to tell whether a grounding candidate is an
    actual clinical finding/disorder versus a building-block concept (a
    morphology, body structure, procedure, substance, etc.) that shouldn't be
    a standalone grounding target for a symptom/diagnosis term. Returns None
    if no FSN is found."""
    if not sctid:
        return None
    r = requests.get(
        f"{UMLS_BASE}/content/{UMLS_VERSION}/source/SNOMEDCT_US/{sctid}/atoms",
        params={"apiKey": UMLS_API_KEY, "pageSize": 20, "ttys": "FN"},
        timeout=15,
    )
    time.sleep(0.05)
    if r.status_code != 200:
        return None
    for atom in r.json().get("result", []):
        if atom.get("termType") == "FN":
            name = atom.get("name", "")
            if name.endswith(")") and "(" in name:
                return name.rsplit("(", 1)[1].rstrip(")")
    return None


# Semantic tags that represent something a patient can actually present with
# or be diagnosed with. Everything else (morphologic abnormality, body
# structure, procedure, substance, organism, qualifier value, observable
# entity, etc.) is a building-block concept, not a valid grounding target for
# a symptom/diagnosis term.
GROUNDABLE_SEMANTIC_TAGS = {"disorder", "finding"}


def explore_neighborhood(
    seed_sctid: str, max_hops: int, executor: ThreadPoolExecutor, max_candidates: int = MAX_NEIGHBORHOOD_CANDIDATES
):
    """BFS outward from `seed_sctid` via both is-a parents and children (so it
    picks up ancestors, descendants, and — via a shared parent — siblings).
    Returns {sctid: first_seen_hop} for every concept found, excluding the seed
    itself. Each hop's parent/child lookups fire concurrently (they're
    independent network calls) rather than one at a time.

    Stops as soon as `max_candidates` total concepts have been collected —
    some SNOMED branches (e.g. generic device/prosthesis hierarchies) fan out
    into hundreds of concepts within 2-3 hops, which makes both a UI list and
    the big-gap search unusably large. Within a hop, candidates are added in a
    fixed sorted-by-SCTID order (not raw set/dict order, which is
    hash-randomized) so the result is reproducible."""
    visited = {seed_sctid}
    frontier = {seed_sctid}
    found = {}
    for hop in range(1, max_hops + 1):
        nodes = list(frontier)
        parent_results = executor.map(get_parents, nodes)
        children_results = executor.map(get_children, nodes)
        next_frontier = set()
        for parents, children in zip(parent_results, children_results):
            for neighbor in list(parents) + list(children):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        ordered = sorted(next_frontier)
        remaining_capacity = max_candidates - len(found)
        for node in ordered[:remaining_capacity]:
            found[node] = hop
        if len(found) >= max_candidates:
            break
        frontier = next_frontier
        if not frontier:
            break
    return found


def biggest_gap_cutoff(scored: list, window: int = GAP_SEARCH_WINDOW, z_threshold: float = GAP_OUTLIER_Z_THRESHOLD):
    """Given [(sctid, score), ...] sorted descending by score, find the
    "natural" cluster boundary within the first `window` ranks.

    Rather than just taking the single largest consecutive-score gap, each
    gap is measured against the *typical* gap size within this same list —
    median gap size, and median absolute deviation (MAD) as a robust measure
    of spread — so what counts as "big" adapts to that specific
    neighborhood's own density instead of a fixed global number. A gap only
    counts as a genuine boundary if its modified z-score
    (0.6745*(gap-median)/MAD) clears `z_threshold`; if none do (e.g. scores
    decay smoothly with no real break), we fall back to the largest available
    gap but flag it as not statistically significant.

    Returns (cutoff_idx, gap_size, is_significant):
      - cutoff_idx: last index to KEEP (None if <2 items)
      - gap_size: the chosen gap's raw value
      - is_significant: True if the chosen gap is a genuine outlier vs. the
        rest of the window; False if nothing stood out and this is just a
        best-effort pick.
    """
    if len(scored) < 2:
        return None, 0.0, False

    limit = min(len(scored) - 1, window)
    gaps = [scored[i][1] - scored[i + 1][1] for i in range(limit)]

    if limit == 1:
        return 0, gaps[0], True  # only one possible cut point, nothing to compare it against

    median_gap = statistics.median(gaps)
    mad = statistics.median([abs(g - median_gap) for g in gaps])
    mad_safe = mad if mad > 1e-9 else 1e-9

    modified_z = [0.6745 * (g - median_gap) / mad_safe for g in gaps]
    outliers = [i for i, z in enumerate(modified_z) if z >= z_threshold]

    if outliers:
        best_idx = max(outliers, key=lambda i: gaps[i])
        return best_idx, gaps[best_idx], True

    best_idx = max(range(limit), key=lambda i: gaps[i])
    return best_idx, gaps[best_idx], False


# ── Embeddings ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def embed(text: str):
    r = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    r.raise_for_status()
    vec = r.json()["embeddings"][0]
    return np.array(vec, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
