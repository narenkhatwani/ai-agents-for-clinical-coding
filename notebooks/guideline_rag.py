"""Hybrid retrieval over the chunked ICD-10-CM Guidelines.

MiniLM vectors + BM25 -> reciprocal rank fusion -> optional cross-encoder rerank -> expand to parent rules.
Models are loaded from local folders (<models>/minilm, <models>/reranker) with plain transformers.

    python guideline_rag.py build --chunks guidelines_chunks_FY2026.json --models models --index guideline_index
    python guideline_rag.py query --index guideline_index --models models "bleeding after hysterectomy complication"
    python guideline_rag.py eval  --index guideline_index --models models --gold guideline_rag_gold.json
    python guideline_rag.py cases --index guideline_index --models models --admission patient_records/patient_x/admissions/hadm_y
"""
import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_dir, max_len=256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).eval()
        self.max_len = max_len

    def encode(self, texts, batch=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                out.append(self.torch.nn.functional.normalize(v, dim=1).numpy())
        return np.vstack(out).astype(np.float32)


class Reranker:
    def __init__(self, model_dir, max_len=512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.max_len = max_len

    def score(self, query, passages, batch=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(passages), batch):
                chunk = passages[i:i + batch]
                enc = self.tok([query] * len(chunk), chunk, padding=True, truncation="only_second",
                               max_length=self.max_len, return_tensors="pt")
                out.extend(self.model(**enc).logits[:, 0].tolist())
        return out


_STOP = frozenset("a an and are as at be by for from has have in is it its of on or that the to was were will with "
                  "this these those not may must any all can".split())


def _stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text):
    """Keeps code-like tokens whole (i.b.16, n99.820, excludes1) and also adds their parts."""
    toks = []
    for m in re.finditer(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*", text.lower()):
        t = m.group(0)
        parts = re.split(r"[.\-/]", t)
        for c in [t] + (parts if len(parts) > 1 else []):
            if c in _STOP or (len(c) < 2 and not c.isdigit()):
                continue
            toks.append(_stem(c))
    return toks


class _OwnBM25:
    backend = "own (rank_bm25 not installed)"

    def __init__(self, docs, k1=1.5, b=0.75):
        self.tfs = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query_tokens):
        s = np.zeros(len(self.tfs))
        for t in set(query_tokens):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class _RankBM25:
    backend = "rank_bm25"

    def __init__(self, docs):
        from rank_bm25 import BM25Okapi
        self.model = BM25Okapi(docs)

    def scores(self, query_tokens):
        return np.asarray(self.model.get_scores(query_tokens), dtype=float)


def make_bm25(docs):
    try:
        return _RankBM25(docs)
    except ImportError:
        return _OwnBM25(docs)


COLLECTION = "guideline_children"


def _chroma_collection(index_dir, create):
    import chromadb
    client = chromadb.PersistentClient(path=str(Path(index_dir) / "chroma"))
    if create:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        return client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    return client.get_collection(COLLECTION)


class GuidelineIndex:
    """Dense search runs in a Chroma collection (cosine); BM25 and the chunk/parent tables stay in memory."""

    def __init__(self, data, collection, embedder=None, reranker=None):
        self.children, self.parents = data["children"], data["parents"]
        self.fy = data.get("fy")
        self.col, self.embedder, self.reranker = collection, embedder, reranker
        self.pos = {c["id"]: i for i, c in enumerate(self.children)}
        self.bm25 = make_bm25([tokenize(c["text"]) for c in self.children])

    @classmethod
    def build(cls, chunks_path, embedder, out_dir):
        data = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        kids = data["children"]
        emb = embedder.encode([c["text"] for c in kids])
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chunks_path, out / "chunks.json")
        col = _chroma_collection(out, create=True)
        for i in range(0, len(kids), 200):
            part = kids[i:i + 200]
            col.add(ids=[c["id"] for c in part], embeddings=emb[i:i + 200].tolist(),
                    documents=[c["text"] for c in part],
                    metadatas=[{"parent_id": c["parent_id"], "node_id": c["node_id"], "fy": c["fy"] or 0,
                                "chapter": c["chapter"] or -1} for c in part])
        (out / "meta.json").write_text(json.dumps({"fy": data.get("fy"), "n_children": len(kids), "dim": int(emb.shape[1])}))
        return cls(data, col, embedder)

    @classmethod
    def load(cls, index_dir, embedder=None, reranker=None):
        p = Path(index_dir)
        data = json.loads((p / "chunks.json").read_text(encoding="utf-8"))
        return cls(data, _chroma_collection(p, create=False), embedder, reranker)

    def search(self, query, mode="hybrid+rerank", top_parents=5, k=20, rrf_k=60, rerank_pool=30,
               chapter=None, chapter_boost=0.0):
        """mode: dense | bm25 | hybrid | hybrid+rerank. chapter/chapter_boost raise that chapter's chunks softly."""
        lists = []
        if mode in ("dense", "hybrid", "hybrid+rerank"):
            res = self.col.query(query_embeddings=[self.embedder.encode([query])[0].tolist()], n_results=k)
            lists.append([self.pos[cid] for cid in res["ids"][0]])
        if mode in ("bm25", "hybrid", "hybrid+rerank"):
            bm = self.bm25.scores(tokenize(query))
            lists.append([i for i in np.argsort(-bm)[:k] if bm[i] > 0])
        fused = {}
        for lst in lists:
            for rank, i in enumerate(lst):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + rank + 1)
        if chapter is not None and chapter_boost:
            for i in fused:
                if self.children[i]["chapter"] == chapter:
                    fused[i] *= 1 + chapter_boost
        order = sorted(fused, key=lambda i: -fused[i])
        scores = fused
        if mode == "hybrid+rerank" and self.reranker and order:
            pool = order[:rerank_pool]
            sc = self.reranker.score(query, [self.children[i]["text"] for i in pool])
            order = [pool[j] for j in np.argsort(-np.array(sc))]
            scores = {i: s for i, s in zip(pool, sc)}
        out, seen = [], {}
        for i in order:
            c = self.children[i]
            pid = c["parent_id"]
            if pid in seen:
                seen[pid]["child_ids"].append(c["id"])
            elif len(out) < top_parents:
                par = self.parents[pid]
                seen[pid] = {"parent_id": pid, "node_id": par["node_id"], "score": float(scores[i]),
                             "child_ids": [c["id"]], "path": par["path_titles"], "refs": par["refs"],
                             "text": par["text"]}
                out.append(seen[pid])
        return out


def chapter_of(code, tabular):
    node, _ = tabular.resolve(code)
    if node is None:
        return None
    for n in tabular.ancestry(node):
        if n["key"].startswith("ch"):
            return int(n["key"][2:])
    return None


def build_query(case, max_evidence=3):
    parts = []
    if case.get("code"):
        parts.append(f"{case.get('code_title', '')} ({case['code']})".strip())
    if case.get("diagnosis"):
        parts.append(f"Diagnosis: {case['diagnosis']}")
    if case.get("procedures"):
        parts.append("Procedure: " + "; ".join(case["procedures"][:2]))
    if case.get("evidence"):
        parts.append("Evidence: " + "; ".join(case["evidence"][:max_evidence]))
    return ". ".join(parts)


_COMPLICATION_CUE = re.compile(r"\b(post[- ]?(operative|procedural|op|\w+ectomy|\w+otomy|\w+plasty)|s/p|status post|"
                               r"complication|iatrogenic|following (a |the )?(surgery|procedure|operation))\b", re.I)

CODING_QUERIES = {
    "complication": "documentation of complications of care: provider documents the cause-and-effect relationship "
                    "between the procedure and the condition, complication code",
    "unspecified": "use of unspecified codes versus a more specific code, sign or symptom code when a definitive "
                   "diagnosis is available",
    "etiology_manifestation": "code first the underlying condition, use additional code, etiology and "
                              "manifestation convention",
    "seventh_character": "7th character placeholder X required to complete the code",
}


def case_triggers(case, tabular=None):
    """Which coding-rule questions this case raises, judged from the code, its Tabular List notes and the case text."""
    text = " ".join([case.get("diagnosis") or "", *(case.get("evidence") or []), *(case.get("procedures") or [])])
    fired = []
    if _COMPLICATION_CUE.search(text):
        fired.append("complication")
    if re.search(r"unspecified|\bNOS\b", case.get("code_title", ""), re.I):
        fired.append("unspecified")
    rec = tabular.record(case["code"]) if tabular and case.get("code") else None
    if rec:
        if rec["notes"].get("codeFirst"):
            fired.append("etiology_manifestation")
        if rec["needs_7th"]:
            fired.append("seventh_character")
    return fired


def retrieve_for_case(index, case, tabular=None, per_query=2, mode="hybrid"):
    """Case-level retrieval: one query per triggered coding rule, plus the clinical query with a chapter boost."""
    chapter = chapter_of(case["code"], tabular) if tabular and case.get("code") else None
    jobs = [(t, CODING_QUERIES[t], None, 0.0) for t in case_triggers(case, tabular)]
    jobs.append(("chapter+case", build_query(case), chapter, 1.0))
    merged, seen = [], {}
    for label, q, ch, boost in jobs:
        for r in index.search(q, mode=mode, top_parents=per_query, chapter=ch, chapter_boost=boost):
            if r["parent_id"] in seen:
                seen[r["parent_id"]]["triggers"].append(label)
            else:
                seen[r["parent_id"]] = {**r, "triggers": [label]}
                merged.append(seen[r["parent_id"]])
    return merged


def format_context(results, max_chars=3500):
    blocks = [r["text"][:max_chars] for r in results]
    return "RELEVANT CODING GUIDELINES (retrieved; cite by the bracketed id):\n\n" + "\n\n".join(blocks)


def cases_from_stage8(admission_dir, include_supporting=False):
    """One case per differential row, from its first principal code. include_supporting=True also checks rows that Naren's
    pipeline mapped only to supporting codes (icd_status "supporting_only", e.g. R65.21 for septic shock), using the first
    code of the row's package; rows with no code at all ("unmapped") are still skipped."""
    adm = Path(admission_dir)
    d = json.loads((adm / "icd_coding.json").read_text(encoding="utf-8"))
    procs = []
    ie = adm / "information_extraction.json"
    if ie.exists():
        data = json.loads(ie.read_text(encoding="utf-8"))
        # Naren's extraction stores them under "procedures" as [{"name": ..., "result": ...}]. (Earlier versions of this function
        # read a key "procedures_this_stay" that does not exist, so no procedure ever reached the checker.)
        for p in data.get("procedures") or data.get("procedures_this_stay") or []:
            if isinstance(p, dict):
                name, result = str(p.get("name") or "").strip()[:90], str(p.get("result") or "").strip()[:40]
                procs.append(name if not result or result.lower() == "performed" else f"{name} ({result})")
            else:
                procs.append(str(p).strip()[:90])
    cases = []
    for row in d.get("differential", []):
        for pr in (row.get("icd10_principal") or [])[:1]:
            cases.append({"rank": row.get("rank"), "diagnosis": row.get("diagnosis"), "code": pr["code"],
                          "code_title": pr.get("title", ""), "evidence": row.get("supporting_evidence") or [],
                          "procedures": [p for p in procs if p]})
        if include_supporting and not row.get("icd10_principal") and row.get("icd_status") == "supporting_only":
            pkg = [p for p in (row.get("icd10_package") or []) if p.get("code")]
            if pkg:
                cases.append({"rank": row.get("rank"), "diagnosis": row.get("diagnosis"), "code": pkg[0]["code"],
                              "code_title": pkg[0].get("title", ""), "evidence": row.get("supporting_evidence") or [],
                              "procedures": [p for p in procs if p]})
    return cases


def _hit(parent, expected):
    node = parent["node_id"]
    return any(node == e or node.startswith(e + ".") or e.startswith(node + ".") for e in expected)


def evaluate(index, gold, modes, top_parents=5):
    table, firsts = {}, {}
    for mode in modes:
        t0, ranks = time.time(), []
        for q in gold:
            res = index.search(q["query"], mode=mode, top_parents=top_parents)
            hits = [i + 1 for i, p in enumerate(res) if _hit(p, q["expected"])]
            ranks.append(hits[0] if hits else None)
        firsts[mode] = ranks
        n = len(ranks)
        table[mode] = {"hit@1": sum(r == 1 for r in ranks) / n,
                       "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
                       "hit@5": sum(r is not None for r in ranks) / n,
                       "MRR": sum(1 / r for r in ranks if r) / n,
                       "sec/query": (time.time() - t0) / n}
    return table, firsts


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "query", "eval", "cases"])
    ap.add_argument("text", nargs="?", help="query text for the 'query' command")
    ap.add_argument("--chunks")
    ap.add_argument("--index", required=True)
    ap.add_argument("--models", required=True, help="folder holding minilm/ and reranker/")
    ap.add_argument("--gold")
    ap.add_argument("--admission")
    ap.add_argument("--mode", default="hybrid", help="bm25 | dense | hybrid | hybrid+rerank")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tabular", help="saved Tabular List JSON, to add a soft chapter boost in 'cases'")
    a = ap.parse_args()
    emb = Embedder(Path(a.models) / "minilm")
    if a.cmd == "build":
        idx = GuidelineIndex.build(a.chunks, emb, a.index)
        print(f"built index: {len(idx.children)} children, {len(idx.parents)} parents -> {a.index} | BM25 backend: {idx.bm25.backend}")
        return
    idx = GuidelineIndex.load(a.index, emb, Reranker(Path(a.models) / "reranker"))
    print(f"BM25 backend: {idx.bm25.backend} | dense: Chroma ({idx.col.count()} vectors)")
    if a.cmd == "query":
        for r in idx.search(a.text, mode=a.mode, top_parents=a.top):
            print(f"\n[{r['parent_id']}] score={r['score']:.3f} children={r['child_ids']}\n  {' > '.join(r['path'][-2:])}\n  {r['text'].split(chr(10), 1)[1][:200]}...")
    elif a.cmd == "eval":
        gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
        modes = ["bm25", "dense", "hybrid", "hybrid+rerank"]
        table, firsts = evaluate(idx, gold, modes, a.top)
        print(f"{len(gold)} queries, top {a.top} parents\n")
        print(f"{'mode':<16}" + "".join(f"{m:>11}" for m in next(iter(table.values()))))
        for mode, m in table.items():
            print(f"{mode:<16}" + "".join(f"{v:>11.2f}" for v in m.values()))
        print("\nrank of first correct parent (- = not in top):")
        print(f"{'query':<46}" + "".join(f"{m:>15}" for m in modes))
        for qi, q in enumerate(gold):
            print(f"{q['id'] + ' ' + q['query'][:40]:<46}" + "".join(f"{(firsts[m][qi] or '-'):>15}" for m in modes))
    elif a.cmd == "cases":
        tabular = None
        if a.tabular:
            from tabular_list import TabularList
            tabular = TabularList.load(a.tabular)
        for case in cases_from_stage8(a.admission):
            print(f"\n=== rank {case['rank']} {case['diagnosis']} -> {case['code']} | triggers: {case_triggers(case, tabular)}")
            for r in retrieve_for_case(idx, case, tabular, mode=a.mode):
                print(f"   [{r['parent_id']}] {r['path'][-1][:60]:<60} via {','.join(r['triggers'])}")


if __name__ == "__main__":
    main()
