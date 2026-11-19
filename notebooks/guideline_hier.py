"""Hierarchical (coarse-to-fine) retrieval over the ICD-10-CM Guidelines: an ALTERNATIVE to the flat child-chunk search in
guideline_rag.py. Nothing in guideline_rag.py, guidelines_chunker.py or the existing index is changed.

The flat search compares the query with every chunk of the whole document at once. This one walks the Guidelines'
own outline like a person would:

    Section I / II / III / IV   ->   A. / B. / C.   ->   4. Chapter 4 ...   ->   a. Diabetes ...   ->   1) Type of diabetes
    (choose the best few)           (best few of those' children)           ...                      (rules to read)

At every step only the children of the nodes kept so far are scored (MiniLM in Chroma + BM25, fused with RRF), the best
`beam` are kept and the search goes one level down. A node's "card" is its heading path, the headings below it and (for
nodes that carry rule text) the start of that text. Scores are blended down the path, so a rule in a strong branch
outranks an equally good match in a weak one.

It has the same search() interface as GuidelineIndex, so retrieve_for_case, evaluate and guideline_check work unchanged.

    python guideline_hier.py build --chunks ../../icd10cm_sources/guidelines_chunks_FY2026_v2.json \
        --index ../../icd10cm_sources/guideline_index_hier --models ../../icd10cm_sources/models
    python guideline_hier.py query --index ../../icd10cm_sources/guideline_index_hier --models ../../icd10cm_sources/models \
        --trace "Type 2 diabetes documented as in remission"
    python guideline_hier.py eval  --index ../../icd10cm_sources/guideline_index_hier --flat-index ../../icd10cm_sources/guideline_index_v2 \
        --models ../../icd10cm_sources/models --gold guideline_rag_gold.json
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from guideline_rag import Embedder, GuidelineIndex, _hit, evaluate, make_bm25, tokenize

COLLECTION = "guideline_tree"
ROOT = "ROOT"
CARD_WORDS = 120      # words of rule text in a node's card (MiniLM reads at most 256 tokens)
CARD_TITLES = 40      # headings below a node listed in its card
TOP_SECTIONS = {"I", "II", "III", "IV"}   # whole-Section introductions: broad text, not specific rules (see drop_sections)


def build_tree(parents):
    """Outline of the Guidelines: node id -> {title, parent, children, has_text, chapter}. Rule text hangs on the nodes
    that are chunk parents; the other nodes (e.g. I.C.4) only organise their children."""
    tree = {ROOT: {"title": "ICD-10-CM Official Guidelines", "parent": None, "children": [], "has_text": False, "chapter": None}}

    def add(nid, title, parent):
        if nid not in tree:
            tree[nid] = {"title": title, "parent": parent, "children": [], "has_text": False, "chapter": None}
            tree[parent]["children"].append(nid)

    for pid, p in parents.items():
        ids, titles = p["path_ids"], p["path_titles"]
        for i, nid in enumerate(ids):
            add(nid, titles[i], ids[i - 1] if i else ROOT)
        if pid != ids[-1]:                      # a second part of a long section, e.g. I.C.21.c.3#1
            add(pid, titles[-1] + " (part)", ids[-1])
        tree[pid]["has_text"] = True
        tree[pid]["chapter"] = p.get("chapter")
    for nid, n in tree.items():                 # chapter numbers for nodes under I.C.<n>
        m = re.match(r"I\.C\.(\d+)", nid)
        if n["chapter"] is None and m:
            n["chapter"] = int(m.group(1))
    return tree


def _body(text):
    return re.sub(r"^\[[^\]]+\][^\n]*\n?", "", text)


def make_cards(tree, parents):
    """Text that stands for each node when it is compared with a query."""
    def below(nid):
        out, queue = [], list(tree[nid]["children"])
        while queue and len(out) < CARD_TITLES:
            c = queue.pop(0)
            out.append(re.sub(r"^(\d+\)|\([a-z]\)|[A-Za-z0-9]+\.)\s*", "", tree[c]["title"]))
            queue.extend(tree[c]["children"])
        return out

    cards = {}
    for nid, n in tree.items():
        if nid == ROOT:
            continue
        chain, cur = [], nid
        while cur != ROOT and len(chain) < 3:
            chain.append(re.sub(r"^(\d+\)|\([a-z]\)|[A-Za-z0-9]+\.)\s*", "", tree[cur]["title"]))
            cur = tree[cur]["parent"]
        parts = [" > ".join(reversed(chain))]
        sub = below(nid)
        if sub:
            parts.append("Covers: " + "; ".join(sub))
        if n["has_text"]:
            parts.append(" ".join(_body(parents[nid]["text"]).split()[:CARD_WORDS]))
        cards[nid] = ". ".join(parts)
    return cards


class HierIndex:
    """Coarse-to-fine search over the Guidelines outline. Same search() interface as GuidelineIndex."""

    def __init__(self, tree, parents, cards, collection, embedder):
        self.tree, self.parents, self.cards, self.col, self.embedder = tree, parents, cards, collection, embedder
        self.ids = [n for n in tree if n != ROOT]
        self.pos = {n: i for i, n in enumerate(self.ids)}
        self.bm25 = make_bm25([tokenize(cards[n]) for n in self.ids])

    @classmethod
    def build(cls, chunks_path, embedder, out_dir):
        import chromadb
        data = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        parents = data["parents"]
        tree = build_tree(parents)
        cards = make_cards(tree, parents)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chunks_path, out / "chunks.json")
        ids = [n for n in tree if n != ROOT]
        emb = embedder.encode([cards[n] for n in ids])
        client = chromadb.PersistentClient(path=str(out / "chroma"))
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})
        col.add(ids=ids, embeddings=emb.tolist(), documents=[cards[n] for n in ids],
                metadatas=[{"parent": tree[n]["parent"], "has_text": bool(tree[n]["has_text"])} for n in ids])
        (out / "tree.json").write_text(json.dumps({"tree": tree, "cards": cards}, ensure_ascii=False), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({"fy": data.get("fy"), "n_nodes": len(ids), "n_text_nodes": len(parents),
                                                   "depth": max(len(p["path_ids"]) for p in parents.values())}))
        return cls(tree, parents, cards, col, embedder)

    @classmethod
    def load(cls, index_dir, embedder=None, reranker=None):
        import chromadb
        p = Path(index_dir)
        data = json.loads((p / "chunks.json").read_text(encoding="utf-8"))
        t = json.loads((p / "tree.json").read_text(encoding="utf-8"))
        col = chromadb.PersistentClient(path=str(p / "chroma")).get_collection(COLLECTION)
        return cls(t["tree"], data["parents"], t["cards"], col, embedder)

    def search(self, query, mode="hier", top_parents=5, chapter=None, chapter_boost=0.0, beam=5, decay=0.2,
               container_penalty=1.0, trace=None, drop_sections=False, **_):
        """Walk down the outline. beam = nodes kept per level; decay = weight of the path so far versus the new level;
        container_penalty < 1 would lower the score of a section's intro text when it has sub-rules of its own; tested on
        the gold queries it hurt badly (MRR 0.89 -> 0.5), because many correct answers ARE such sections, so it is off (1.0)."""
        qv = self.embedder.encode([query])[0].tolist()
        bm_all = self.bm25.scores(tokenize(query))
        frontier, found = [(ROOT, 1.0)], {}
        for depth in range(1, 10):
            cands = [(c, ps) for pid, ps in frontier for c in self.tree[pid]["children"]]
            if not cands:
                break
            cand_ids, path_prev = [c for c, _ in cands], dict(cands)
            res = self.col.query(query_embeddings=[qv], n_results=len(cand_ids),
                                 where={"parent": {"$in": [pid for pid, _ in frontier]}})
            dense = {cid: r for r, cid in enumerate(res["ids"][0])}
            bm = sorted((c for c in cand_ids if bm_all[self.pos[c]] > 0), key=lambda c: -bm_all[self.pos[c]])
            lexical = {c: r for r, c in enumerate(bm)}
            rrf = {c: (1 / (61 + dense[c]) if c in dense else 0.0) + (1 / (61 + lexical[c]) if c in lexical else 0.0)
                   for c in cand_ids}
            top = max(rrf.values()) or 1.0
            score = {}
            for c in cand_ids:
                s = rrf[c] / top
                if chapter is not None and chapter_boost and self.tree[c]["chapter"] == chapter:
                    s *= 1 + chapter_boost
                score[c] = decay * path_prev[c] + (1 - decay) * s
            for c in cand_ids:
                if self.tree[c]["has_text"] and not (drop_sections and c in TOP_SECTIONS):
                    shown = score[c] * (container_penalty if self.tree[c]["children"] else 1.0)
                    found[c] = max(found.get(c, 0.0), shown)
            inner = sorted((c for c in cand_ids if self.tree[c]["children"]), key=lambda c: -score[c])[:beam]
            if trace is not None:
                trace.append({"depth": depth, "kept": inner, "scored": sorted(
                    ((c, round(score[c], 3), bool(self.tree[c]["has_text"])) for c in cand_ids), key=lambda x: -x[1])[:6]})
            frontier = [(c, score[c]) for c in inner]
        out = []
        for pid, sc in sorted(found.items(), key=lambda kv: -kv[1])[:top_parents]:
            par = self.parents[pid]
            out.append({"parent_id": pid, "node_id": par["node_id"], "score": float(sc), "child_ids": [],
                        "path": par["path_titles"], "refs": par["refs"], "text": par["text"]})
        return out


class FusedIndex:
    """Reciprocal-rank fusion of the flat search and the hierarchical search (same search() interface).
    drop_sections=True also removes the whole-Section introductions (I, II, III, IV) that the flat half can return."""

    def __init__(self, flat, hier, drop_sections=False):
        self.flat, self.hier, self.drop_sections = flat, hier, drop_sections

    def search(self, query, mode="hybrid", top_parents=5, rrf_k=60, **kw):
        lists = [self.flat.search(query, mode="hybrid", top_parents=top_parents * 2, **kw),
                 self.hier.search(query, top_parents=top_parents * 2, **kw)]
        fused, keep = {}, {}
        for lst in lists:
            for rank, r in enumerate(lst):
                if self.drop_sections and r["parent_id"] in TOP_SECTIONS:
                    continue
                fused[r["parent_id"]] = fused.get(r["parent_id"], 0.0) + 1.0 / (rrf_k + rank + 1)
                keep.setdefault(r["parent_id"], r)
        order = sorted(fused, key=lambda p: -fused[p])[:top_parents]
        return [{**keep[p], "score": fused[p]} for p in order]


def print_trace(trace, tree):
    for t in trace:
        print(f"  level {t['depth']}: kept {t['kept']}")
        for nid, sc, text in t["scored"]:
            mark = "*" if nid in t["kept"] else " "
            print(f"     {mark} {sc:5.3f}  {nid:<14}{'[text] ' if text else '       '}{re.sub(r'^[^ ]+ ', '', tree[nid]['title'])[:70]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "query", "eval"])
    ap.add_argument("text", nargs="?")
    ap.add_argument("--chunks")
    ap.add_argument("--index", required=True, help="the hierarchical index folder")
    ap.add_argument("--flat-index", help="the existing flat index, for eval")
    ap.add_argument("--models", required=True)
    ap.add_argument("--gold")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--decay", type=float, default=0.2)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--trace", action="store_true", help="show, level by level, what was scored and which nodes were kept")
    a = ap.parse_args()
    embedder = Embedder(Path(a.models) / "minilm")

    if a.cmd == "build":
        idx = HierIndex.build(a.chunks, embedder, a.index)
        print(f"built hierarchical index: {len(idx.ids)} nodes ({len(idx.parents)} with rule text) -> {a.index}")
        return
    idx = HierIndex.load(a.index, embedder)
    if a.cmd == "query":
        trace = [] if a.trace else None
        res = idx.search(a.text, top_parents=a.top, beam=a.beam, decay=a.decay, trace=trace)
        if trace:
            print_trace(trace, idx.tree)
        for r in res:
            print(f"\n[{r['parent_id']}] score {r['score']:.3f}  {' > '.join(r['path'][-3:])}\n    {r['text'][:300]!r}")
        return
    flat = GuidelineIndex.load(a.flat_index, embedder, None)
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    gold = gold["queries"] if isinstance(gold, dict) and "queries" in gold else gold
    print(f"{len(gold)} gold queries (expected section known for each)\n")
    print(f"{'method':<28}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    rows = {"flat (current)": evaluate(flat, gold, ["hybrid"])[0]["hybrid"],
            f"hierarchical beam={a.beam}": evaluate(HierWrap(idx, a.beam, a.decay), gold, ["hier"])[0]["hier"],
            "flat + hierarchical (fused)": evaluate(FusedIndex(flat, HierWrap(idx, a.beam, a.decay)), gold, ["hybrid"])[0]["hybrid"]}
    for name, r in rows.items():
        print(f"{name:<28}{r['hit@1']:>8.2f}{r['hit@3']:>8.2f}{r['hit@5']:>8.2f}{r['MRR']:>8.2f}")
    print("\nper query: rank of the first correct section (- = not in top 5)")
    qs = [(q["query"][:62], q["expected"]) for q in gold]
    indexes = {"flat": flat, "hier": HierWrap(idx, a.beam, a.decay), "fused": FusedIndex(flat, HierWrap(idx, a.beam, a.decay))}
    print(f"{'query':<64}{'flat':>6}{'hier':>6}{'fused':>7}")
    for q in gold:
        ranks = []
        for name, ix in indexes.items():
            res = ix.search(q["query"], mode="hybrid", top_parents=5)
            hits = [i + 1 for i, p in enumerate(res) if _hit(p, q["expected"])]
            ranks.append(str(hits[0]) if hits else "-")
        print(f"{q['query'][:62]:<64}{ranks[0]:>6}{ranks[1]:>6}{ranks[2]:>7}")


class HierWrap:
    """Fixes beam/decay/penalty so evaluate() and FusedIndex can call search() with their usual arguments.
    drop_sections=True never returns the four whole-Section introductions (I, II, III, IV) as results; the search still walks
    through them, so their sub-rules are found as before."""

    def __init__(self, idx, beam=5, decay=0.2, penalty=1.0, drop_sections=False):
        self.idx, self.beam, self.decay, self.penalty, self.drop_sections = idx, beam, decay, penalty, drop_sections

    def search(self, query, **kw):
        return self.idx.search(query, beam=self.beam, decay=self.decay, container_penalty=self.penalty,
                               drop_sections=self.drop_sections, **kw)


if __name__ == "__main__":
    main()
