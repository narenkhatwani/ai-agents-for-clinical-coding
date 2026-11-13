"""ICD-10-CM Tabular List -> keyed records with inherited notes, plus Excludes / code-first checks.

Source: the CDC/NCHS release zip ("icd10cm-table and index-YYYY.zip") or the bare
icd10cm-tabular-YYYY.xml inside it. The XML is a tree (chapter > section > diag > diag ...);
instructional notes sit on whichever node they apply to and are INHERITED by every descendant,
so a code's effective notes are the union along its ancestor path.

    python tabular_list.py --source "<zip or xml>" --out tabular_FY2026.json
    python tabular_list.py --source "<zip or xml>" --check E10.10,E11.9,S270XXA
"""
import argparse
import json
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

NOTE_TAGS = ("includes", "inclusionTerm", "excludes1", "excludes2",
             "codeFirst", "useAdditionalCode", "codeAlso", "sevenChrNote")

_CODE = r"[A-Z]\d[0-9A-Z](?:\.[0-9A-Z]{1,4})?"
_REF_RE = re.compile(rf"(?<![A-Za-z0-9.])({_CODE})(\.?-)?(?:({_CODE}))?")


def nodot(code: str) -> str:
    return re.sub(r"[.\-\s]", "", str(code or "")).upper()


def dotted(code: str) -> str:
    c = nodot(code)
    return c if len(c) <= 3 else f"{c[:3]}.{c[3:]}"


def _txt(el) -> str:
    return " ".join("".join(el.itertext()).split())


def _load_root(source):
    p = Path(source)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".xml") and "tabular" in n.lower()]
            if not names:
                raise FileNotFoundError(f"no tabular XML inside {p.name}")
            return ET.fromstring(z.read(names[0]))
    return ET.parse(p).getroot()


class TabularList:
    def __init__(self, nodes: dict, version: str):
        self.nodes = nodes
        self.version = version
        self.by_nodot = {n["nodot"]: n for n in nodes.values() if n["level"] == "diag"}
        self._prefixes = {c[:i] for c in self.by_nodot for i in range(3, len(c) + 1)}
        self._ref_cache = {}

    # ---- building / persistence -------------------------------------------------
    @classmethod
    def from_source(cls, source):
        root = _load_root(source)
        nodes = {}

        def add(el, level, key, code, title, parent):
            notes = {}
            for tag in NOTE_TAGS:
                for child in el.findall(tag):
                    items = [_txt(n) for n in child.findall("note")] or [_txt(child)]
                    notes.setdefault(tag, []).extend(i for i in items if i)
            ext = {e.get("char"): _txt(e) for d in el.findall("sevenChrDef") for e in d.findall("extension")}
            nodes[key] = {"key": key, "level": level, "code": code, "nodot": nodot(code) if level == "diag" else "",
                          "title": title, "parent": parent, "children": [], "notes": notes, "extensions": ext}
            if parent:
                nodes[parent]["children"].append(key)

        def walk_diag(el, parent):
            code = el.findtext("name", "").strip()
            add(el, "diag", code, code, el.findtext("desc", "").strip(), parent)
            for sub in el.findall("diag"):
                walk_diag(sub, code)

        for ch in root.findall("chapter"):
            ckey = f"ch{ch.findtext('name', '').strip()}"
            add(ch, "chapter", ckey, "", ch.findtext("desc", "").strip(), None)
            for sec in ch.findall("section"):
                skey = f"sec{sec.get('id')}"
                add(sec, "section", skey, "", sec.findtext("desc", "").strip(), ckey)
                for d in sec.findall("diag"):
                    walk_diag(d, skey)
        return cls(nodes, root.findtext("version", "").strip())

    def save(self, path):
        Path(path).write_text(json.dumps({"version": self.version, "nodes": self.nodes}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["nodes"], data["version"])

    # ---- lookup -----------------------------------------------------------------
    def ancestry(self, node):
        chain = []
        while node:
            chain.append(node)
            node = self.nodes.get(node["parent"]) if node["parent"] else None
        return list(reversed(chain))

    def extensions(self, node) -> dict:
        merged = {}
        for n in self.ancestry(node):
            merged.update(n["extensions"])
        return merged

    def resolve(self, code):
        """(node, seventh_char_or_None). Handles dotted/undotted codes and X-padded 7th characters."""
        c = nodot(code)
        node = self.by_nodot.get(c)
        if node:
            return node, None
        if len(c) >= 4:
            for i in range(len(c) - 1, 2, -1):
                base = self.by_nodot.get(c[:i])
                if base and set(c[i:-1]) <= {"X"} and c[-1] in self.extensions(base):
                    return base, c[-1]
        return None, None

    def effective(self, node, tag):
        out = []
        for anc in self.ancestry(node):
            for text in anc["notes"].get(tag, []):
                out.append({"text": text, "refs": self.refs(text), "source": anc["key"], "level": anc["level"]})
        return out

    # ---- code references inside notes -------------------------------------------
    def refs(self, text):
        if text in self._ref_cache:
            return self._ref_cache[text]
        found = []
        for m in _REF_RE.finditer(text):
            first, dash, second = nodot(m.group(1)), m.group(2), m.group(3)
            if dash and second:
                end = nodot(second)
                if first[:3] in self._prefixes:
                    found.append({"type": "range", "start": first, "end": end})
            elif first in self._prefixes:
                found.append({"type": "prefix", "value": first})
        self._ref_cache[text] = found
        return found

    @staticmethod
    def _matches(code_nodot, ref):
        if ref["type"] == "prefix":
            return code_nodot.startswith(ref["value"])
        s, e = ref["start"], ref["end"]
        return code_nodot[:len(s)] >= s and code_nodot[:len(e)] <= e

    # ---- checks -----------------------------------------------------------------
    def excludes_conflicts(self, codes, include_excludes2=False):
        """Excludes1 -> hard block for a set of codes on one encounter.

        Excludes2 means both codes MAY be reported together, so it is informational only and
        off by default (chapter-level Excludes2 notes match almost any unrelated pair)."""
        resolved = {c: self.resolve(c)[0] for c in codes}
        out, seen = [], set()
        tags = (("excludes1", "hard"),) + ((("excludes2", "soft"),) if include_excludes2 else ())
        for a, na in resolved.items():
            if na is None:
                continue
            for tag, strength in tags:
                for note in self.effective(na, tag):
                    for b, nb in resolved.items():
                        if b == a or nb is None:
                            continue
                        if any(self._matches(nodot(b), r) for r in note["refs"]):
                            key = (strength, frozenset((nodot(a), nodot(b))))
                            if key in seen:
                                continue
                            seen.add(key)
                            out.append({"strength": strength, "code": dotted(a), "conflicts_with": dotted(b),
                                        "note": note["text"], "note_source": note["source"]})
        return out

    def companion_check(self, codes):
        """'Code first' (required) and 'use additional / code also' (advisory) notes vs the other codes present.

        Not an error detector on its own: on 247 billed codes, 26 of 47 code-first notes were "unmet" and all
        were context-dependent (follow-up exams, pregnancy, "complicating:" headers) -- applicability needs judgement."""
        resolved = {c: self.resolve(c)[0] for c in codes}
        out = []
        for a, na in resolved.items():
            if na is None:
                continue
            for tag, strength in (("codeFirst", "required"), ("useAdditionalCode", "advisory"), ("codeAlso", "advisory")):
                notes = self.effective(na, tag)
                refs = [r for n in notes for r in n["refs"]]
                if not notes:
                    continue
                satisfied = [dotted(b) for b in resolved if b != a and any(self._matches(nodot(b), r) for r in refs)]
                out.append({"code": dotted(a), "note_type": tag, "strength": strength,
                            "status": "met" if satisfied else "not_met", "satisfied_by": satisfied,
                            "note_texts": [n["text"] for n in notes][:3], "has_code_refs": bool(refs),
                            "conditional": any(re.search(r"if applicable|such as|any associated", n["text"], re.I)
                                               for n in notes)})
        return out

    def more_specific(self, code):
        node, _ = self.resolve(code)
        if node is None:
            return []
        kids = [self.nodes[k] for k in node["children"]]
        if not kids and re.search(r"unspecified|\bNOS\b", node["title"], re.I) and node["parent"]:
            kids = [self.nodes[k] for k in self.nodes[node["parent"]]["children"] if k != node["key"]]
        return [{"code": k["code"], "title": k["title"]} for k in kids if k["level"] == "diag"]

    def record(self, code):
        """One flat, self-contained record for a code (own + inherited notes) -- the unit to embed or serve."""
        node, ext = self.resolve(code)
        if node is None:
            return None
        path = [n for n in self.ancestry(node)]
        return {"code": dotted(code), "base_code": node["code"], "seventh_char": ext,
                "seventh_meaning": self.extensions(node).get(ext) if ext else None,
                "title": node["title"], "fy": self.version,
                "path": [(n["code"] or n["key"]) for n in path],
                "is_leaf": not node["children"],
                "needs_7th": bool(self.extensions(node)),
                "notes": {t: [{"text": e["text"], "from": e["source"]} for e in self.effective(node, t)]
                          for t in NOTE_TAGS if self.effective(node, t)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out")
    ap.add_argument("--check", help="comma-separated codes to test for Excludes / code-first issues")
    a = ap.parse_args()
    tl = TabularList.from_source(a.source)
    n_diag = sum(1 for n in tl.nodes.values() if n["level"] == "diag")
    print(f"Tabular List FY{tl.version}: {len(tl.nodes):,} nodes, {n_diag:,} code entries")
    if a.out:
        tl.save(a.out)
        print(f"saved -> {a.out}")
    if a.check:
        codes = [c.strip() for c in a.check.split(",") if c.strip()]
        print(json.dumps({"unresolved": [c for c in codes if tl.resolve(c)[0] is None],
                          "excludes": tl.excludes_conflicts(codes),
                          "companions": tl.companion_check(codes)}, indent=2))


if __name__ == "__main__":
    main()
