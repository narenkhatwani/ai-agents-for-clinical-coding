"""Chunk the ICD-10-CM Official Guidelines PDF along its own heading hierarchy.

Parents are rule sections (what the LLM reads); children are small pieces (what gets embedded and
searched), each prefixed with its heading path so it makes sense alone.

Heading levels, identified by font size / boldness / indent in the CMS PDF:
    Section I.  (bold 16)  >  A.  (bold 14, x~102)  >  1.  (bold 13, x~115-121)
    >  a.  (bold 14, x~170)  >  1)  (bold 13, x~199)  >  (a)  (regular 12, x~251)

    python guidelines_chunker.py --pdf fy2026_guidelines.pdf --out guidelines_chunks_FY2026.json
    python guidelines_chunker.py --pdf fy2026_guidelines.pdf --show I.B.16
"""
import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz

PARENT_MAX_TOKENS = 900
CHILD_MAX_TOKENS = 220
PARA_GAP = 20

_FOOTER = re.compile(r"^(ICD-10-CM Official Guidelines for Coding and Reporting|FY\s*\d{4}|Page \d+ of \d+)\s*$")
_SECTION_REF = re.compile(r"Section\s+([IV]+(?:\.[A-Za-z0-9]+)*)")
_NEW_PARA = re.compile(r"^(•|\(?[a-z0-9]{1,2}\)\s*\S)")


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.3)


def make_counter(tokenizer):
    """None -> chars/3.3 estimate; a path to MiniLM's vocab.txt -> exact offline counts; else a HF model name."""
    if not tokenizer:
        return est_tokens
    if str(tokenizer).endswith("vocab.txt"):
        from transformers import BertTokenizerFast
        tok = BertTokenizerFast(vocab_file=str(tokenizer), do_lower_case=True)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
    return lambda t: len(tok.encode(t, add_special_tokens=False))


@dataclass
class Node:
    level: int
    label: str
    title: str
    page: int
    parent: "Node" = None
    children: list = field(default_factory=list)
    paras: list = field(default_factory=list)
    id: str = ""

    def display(self):
        mark = {0: "Section {}.", 1: "{}.", 2: "{}.", 3: "{}.", 4: "{})", 5: "({})"}[self.level].format(
            self.label.replace("App", "Appendix ") if self.level == 0 else self.label)
        return f"{mark} {self.title}".strip()


def merged_lines(page):
    raw = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if spans:
                raw.append({"x": l["bbox"][0], "y": l["bbox"][1], "size": spans[0]["size"],
                            "bold": all("bold" in s["font"].lower() for s in spans),
                            "text": "".join(s["text"] for s in l["spans"]).strip()})
    raw.sort(key=lambda r: r["y"])
    out, group = [], []
    for r in raw:
        if group and r["y"] - group[0]["y"] >= 3:
            out.append(group)
            group = []
        group.append(r)
    if group:
        out.append(group)
    lines = []
    for g in out:
        g.sort(key=lambda r: r["x"])
        first = dict(g[0])
        first["text"] = " ".join(r["text"] for r in g)
        lines.append(first)
    return lines


def classify(ln):
    t, x, sz, bold = ln["text"], ln["x"], ln["size"], ln["bold"]
    if bold and sz >= 15.5:
        m = re.match(r"Section\s+([IV]+)\.\s*(.*)", t)
        if m:
            return 0, m.group(1), m.group(2)
        m = re.match(r"Appendix\s+([IVX]+)\s*(.*)", t)
        if m:
            return 0, "App" + m.group(1), m.group(2)
    if bold and sz >= 12.5:
        m = re.match(r"([A-Z])\.\s*(\S.*)", t)
        if m and x < 130 and sz >= 13.5:
            return 1, m.group(1), m.group(2)
        m = re.match(r"(\d+)\.\s*(\S.*)", t)
        if m and x < 140:
            return 2, m.group(1), m.group(2)
        m = re.match(r"([a-z])\.\s*(\S.*)", t)
        if m and x >= 150 and sz >= 13.5:
            return 3, m.group(1), m.group(2)
        m = re.match(r"(\d+)\)\s*(\S.*)", t)
        if m and x >= 180:
            return 4, m.group(1), m.group(2)
    if not bold and 11.5 <= sz <= 12.5 and x >= 240:
        m = re.match(r"\(([a-z])\)\s*([A-Z]\S.*)", t)
        if m:
            return 5, m.group(1), m.group(2)
    return None


def parse(pdf_path):
    doc = fitz.open(pdf_path)
    fy = re.search(r"FY\s*(\d{4})", doc[0].get_text())
    root = Node(-1, "", "ROOT", 0)
    stack, started = [root], False
    cur = None
    last_heading = None

    def flush():
        nonlocal cur
        if cur:
            cur["node"].paras.append((cur["page"], cur["text"]))
            cur = None

    for pno in range(len(doc)):
        for ln in merged_lines(doc[pno]):
            text = ln["text"]
            if _FOOTER.match(text):
                continue
            kind = classify(ln)
            if not started:
                if kind and kind[0] == 0 and kind[1] == "I":
                    started = True
                else:
                    continue
            # a wrapped heading continues on the next bold line; only for headings that are bold throughout. A heading with a
            # regular label and bold title, "(b) Type 2 diabetes mellitus in remission", is followed by bold BODY text in the
            # FY2026 PDF (narrative changes are printed in bold), which must not be absorbed into the title.
            if (not kind and last_heading and last_heading["bold"] and ln["bold"]
                    and abs(ln["size"] - last_heading["size"]) < 0.6
                    and last_heading["page"] == pno and ln["y"] - last_heading["y"] < 24):
                last_heading["node"].title = f'{last_heading["node"].title} {text}'.strip()
                last_heading["y"] = ln["y"]
                continue
            last_heading = None
            if kind:
                flush()
                level, label, title = kind
                while stack[-1].level >= level:
                    stack.pop()
                node = Node(level, label, title, pno + 1, parent=stack[-1])
                stack[-1].children.append(node)
                stack.append(node)
                last_heading = {"node": node, "size": ln["size"], "y": ln["y"], "page": pno, "bold": ln["bold"]}
                continue
            if stack[-1] is root:
                continue
            text = re.sub(r"^•\s*", "• ", text)
            joins = False
            if cur and not _NEW_PARA.match(text):
                if cur["page"] == pno and ln["y"] - cur["y"] <= PARA_GAP:
                    joins = True
                elif cur["page"] != pno and not re.search(r"[.:;)?!]$", cur["text"]):
                    joins = True
            if joins:
                cur["text"] += " " + text
                cur["y"], cur["page"] = ln["y"], pno
            else:
                flush()
                cur = {"node": stack[-1], "text": text, "page": pno, "y": ln["y"]}
    flush()
    seen = set()

    def assign(node, chain):
        for c in node.children:
            base = ".".join(chain + [c.label])
            c.id, k = base, 2
            while c.id in seen:
                c.id, k = f"{base}~{k}", k + 1
            seen.add(c.id)
            assign(c, chain + [c.label])

    assign(root, [])
    return root, (int(fy.group(1)) if fy else None)


def path_nodes(node):
    chain = []
    while node and node.level >= 0:
        chain.append(node)
        node = node.parent
    return list(reversed(chain))


def chapter_info(node):
    for n in reversed(path_nodes(node)):
        m = re.match(r"Chapter\s+(\d+):", n.title)
        if m:
            rng = re.search(r"\(([A-Z]\d{2}[^()]*)\)", n.title)
            return int(m.group(1)), (rng.group(1) if rng else None)
    return None, None


def short(node, limit=60):
    if node.level == 0:
        return node.label.replace("App", "Appendix ") if node.label.startswith("App") else f"Section {node.label}"
    d = node.display()
    return d if len(d) <= limit else d[:limit].rsplit(" ", 1)[0] + "…"


def prefix_for(node, budget_chars=260):
    chain = path_nodes(node)
    titles = [short(n, 90 if i == len(chain[-3:]) - 1 else 60) for i, n in enumerate(chain[-3:])]
    pre = f"{node.id} | " + " > ".join(titles)
    return pre if len(pre) <= budget_chars else f"{node.id} | {short(chain[-1], budget_chars - 20)}"


def flatten(node):
    out = [(node, p) for p in node.paras]
    for c in node.children:
        out.extend(flatten(c))
    return out


def subtree_tokens(node, count, memo):
    if node.id not in memo:
        memo[node.id] = sum(count(p[1]) for p in node.paras) + sum(subtree_tokens(c, count, memo) for c in node.children)
    return memo[node.id]


def pack(items, budget, count):
    groups, cur, tok = [], [], 0
    for it in items:
        t = count(it)
        if cur and tok + t > budget:
            groups.append(cur)
            cur, tok = [], 0
        cur.append(it)
        tok += t
    if cur:
        groups.append(cur)
    return groups


def split_long(paragraph, budget, count):
    sentences = re.split(r"(?<=[.;:?!])\s+(?=[A-Z“\"(•\d])", paragraph)
    pieces = []
    for s in sentences:
        if count(s) <= budget:
            pieces.append(s)
        else:
            words = s.split()
            step = max(1, int(len(words) * budget / max(count(s), 1)))
            pieces.extend(" ".join(words[i:i + step]) for i in range(0, len(words), step))
    return [" ".join(g) for g in pack(pieces, budget, count)]


def build(pdf_path, count=est_tokens):
    root, fy = parse(pdf_path)
    memo, parents, children = {}, {}, []

    def emit_parent(node, blocks, part):
        pid = node.id if part is None else f"{node.id}#{part}"
        pages = [p for n, (p, _) in blocks] or [node.page]
        chap, rng = chapter_info(node)
        lines, last_node = [], None
        for n, (_, text) in blocks:
            if n is not node and n is not last_node:
                lines.append(n.display())
            lines.append(text)
            last_node = n
        body = "\n".join(lines)
        parent = {"id": pid, "node_id": node.id, "part": part, "fy": fy,
                  "path_ids": [n.id for n in path_nodes(node)], "path_titles": [n.display() for n in path_nodes(node)],
                  "page_start": min(pages), "page_end": max(pages), "chapter": chap, "chapter_range": rng,
                  "refs": sorted({m.rstrip(".") for m in _SECTION_REF.findall(body)}),
                  "text": f"[{pid}] " + " > ".join(short(n, 90) for n in path_nodes(node)) + "\n" + body}
        parent["n_tokens_est"] = count(parent["text"])
        parents[pid] = parent
        k = 0
        i = 0
        while i < len(blocks):
            n = blocks[i][0]
            j = i
            while j < len(blocks) and blocks[j][0] is n:
                j += 1
            pre = prefix_for(n)
            budget = max(40, CHILD_MAX_TOKENS - count(pre) - 2)
            paras = []
            for _, (_, text) in blocks[i:j]:
                paras.extend([text] if count(text) <= budget else split_long(text, budget, count))
            for group in pack(paras, budget, count):
                cid = f"{pid}/c{k}"
                body_c = "\n".join(group)
                children.append({"id": cid, "parent_id": pid, "node_id": n.id, "fy": fy, "prefix": pre,
                                 "text": f"{pre}\n{body_c}", "chapter": chap, "chapter_range": rng,
                                 "page": blocks[i][1][0], "refs": sorted({m.rstrip(".") for m in _SECTION_REF.findall(body_c)})})
                k += 1
            i = j

    def walk(node):
        if node.level >= 0:
            total = subtree_tokens(node, count, memo)
            if total <= PARENT_MAX_TOKENS:
                blocks = flatten(node)
                if blocks:
                    emit_parent(node, blocks, None)
                return
            if node.paras:
                groups = pack([(node, p) for p in node.paras], PARENT_MAX_TOKENS, lambda b: count(b[1][1]))
                for part, g in enumerate(groups, 1):
                    emit_parent(node, g, part if len(groups) > 1 else None)
        for c in node.children:
            walk(c)

    walk(root)
    return {"fy": fy, "source": Path(pdf_path).name,
            "params": {"parent_max_tokens": PARENT_MAX_TOKENS, "child_max_tokens": CHILD_MAX_TOKENS},
            "parents": parents, "children": children}, root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out")
    ap.add_argument("--show", help="print the parent(s) and children whose id starts with this, e.g. I.B.16")
    ap.add_argument("--tokenizer", help="path to MiniLM vocab.txt (offline) or a HF model name, for exact token counts")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data, root = build(a.pdf, make_counter(a.tokenizer))
    kids = data["children"]
    print(f"FY{data['fy']}: {len(data['parents'])} parents, {len(kids)} children")
    if a.out:
        Path(a.out).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        print(f"saved -> {a.out}")
    if a.show:
        for pid, p in data["parents"].items():
            if pid == a.show or pid.startswith(a.show + "#"):
                print("\n=== PARENT", pid, f"({p['n_tokens_est']} tokens, p{p['page_start']}-{p['page_end']}) refs={p['refs']}")
                print(p["text"])
                for c in kids:
                    if c["parent_id"] == pid:
                        print(f"\n--- CHILD {c['id']}\n{c['text']}")


if __name__ == "__main__":
    main()
