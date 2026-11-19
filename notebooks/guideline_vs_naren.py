"""Naren's pipeline vs the Guidelines checker, per admission. No LLM calls; it only reads files that already exist.

BASELINE = the ICD code Naren's pipeline gives for the top-ranked diagnosis (rank 1, "rank1_predicted_icd" in his Stage 10
accuracy.json). The checker's result for the same row gives the CORRECTED code:

    verdict keep            -> corrected code = Naren's code (unchanged)
    verdict flag / replace  -> corrected code = the checker's proposed code, else the best code found for its suggested
                               condition (title search), else Naren's code (flagged but nothing better found)
    no checker record       -> unchanged (the checker only reviews codes that exist in Naren's icd_coding.json; it cannot
                               create a code where the pipeline produced none)

Scored against the billed PRIMARY ICD-10-CM code: exact; exact or a more specific child of the billed code (the billed list
uses older codes, e.g. billed I47.1 vs FY2026 I47.10); same 3-character family. "Named" is looser: the billed code appears
among the checker's top suggestions (what a coder reviewing the flag would see).

    python guideline_vs_naren.py --records ../patient_records --out ../../icd10cm_sources/naren_vs_checker \
        --arm control=../../icd10cm_sources/check_v2_control --arm fused=../../icd10cm_sources/check_v2_fused
"""
import argparse
import csv
import json
from pathlib import Path

FLAGGED = ("flag", "replace")


def nodot(code):
    return (code or "").replace(".", "").upper()


def load_checks(folder):
    f = Path(folder) / "checks.jsonl"
    if not f.exists():
        return {}
    return {r["key"]: r for r in (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}


def corrected_code(rec):
    """(corrected code, flagged, codes named) for one saved check."""
    llm, code = rec["llm"], nodot(rec["case"]["code"])
    verdict = str(llm.get("verdict", "")).lower()
    named = ([nodot(llm["proposed_code"])] if llm.get("proposed_code") else []) + \
            [nodot(c["code"]) for c in rec.get("codes_for_suggested_condition", [])[:3]]
    if verdict in FLAGGED and named:
        return named[0], True, named
    return code, verdict in FLAGGED, named


def score(code, primary):
    return {"exact": code == primary, "child": code.startswith(primary), "family": code[:3] == primary[:3]}


def mark(ok):
    return "yes" if ok else "no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=FOLDER",
                    help="a results folder (checks.jsonl) with the label to show for it, e.g. fused=../../icd10cm_sources/check_v2_fused; "
                         "repeat for each run to compare")
    ap.add_argument("--out", required=True, help="output file name without extension; writes .csv and .md")
    a = ap.parse_args()

    arms = {}
    for spec in a.arm:
        label, _, folder = spec.partition("=")
        if not folder:
            ap.error(f"--arm needs LABEL=FOLDER, got {spec!r}")
        arms[label] = load_checks(folder)
    rows = []
    for adm in sorted(Path(a.records).glob("patient_*/admissions/hadm_*")):
        acc, gt = adm / "accuracy" / "accuracy.json", adm / "ground_truth.json"
        if not (acc.exists() and gt.exists()):
            continue
        pid, hid = adm.parent.parent.name.replace("patient_", ""), adm.name.replace("hadm_", "")
        ev = json.loads(acc.read_text(encoding="utf-8"))["evaluation"]
        primary = nodot(json.loads(gt.read_text(encoding="utf-8"))["primary_icd_code"])
        naren = nodot(ev["icd_linked"].get("rank1_predicted_icd"))
        row = {"patient": pid, "hadm": hid, "gt_primary": primary, "naren_code": naren,
               **{f"naren_{k}": v for k, v in score(naren, primary).items()}}
        for arm, checks in arms.items():
            rec = next((r for r in checks.values() if r["patient_id"] == pid and r["hadm_id"] == hid
                        and r["case"]["rank"] == 1), None)
            if rec is None or nodot(rec["case"]["code"]) != naren:
                row[arm] = {"verdict": "not checked", "flagged": False, "code": naren, "named": naren == primary,
                            **score(naren, primary)}
                continue
            code, flagged, named = corrected_code(rec)
            row[arm] = {"verdict": str(rec["llm"].get("verdict", "")).lower(), "flagged": flagged, "code": code,
                        "named": primary in named or naren == primary, **score(code, primary)}
        rows.append(row)

    n = len(rows)
    head = " | ".join(f"{arm}: verdict | {arm}: result | {arm}: exact" for arm in arms)
    lines = [f"# Naren's pipeline vs the Guidelines checker ({n} admissions)", "",
             "Top-ranked diagnosis of each admission; its ICD code is Naren's prediction. Scored against the billed primary "
             "ICD-10-CM code.", "",
             f"| Patient | Billed primary | Naren code | Naren exact | {head} |", "|---|---|---|---|" + "---|---|---|" * len(arms)]
    for r in rows:
        cells = [r["patient"], r["gt_primary"], r["naren_code"] or "(none)", mark(r["naren_exact"])]
        for arm in arms:
            x = r[arm]
            cells += [x["verdict"], x["code"] or "(none)", mark(x["exact"])]
        lines.append("| " + " | ".join(cells) + " |")

    cols = "| | exact | exact or child of billed | same 3-char family | billed code named by checker |"
    summ = ["", "## Totals", "", cols, "|---|---|---|---|---|",
            f"| Naren alone | {sum(r['naren_exact'] for r in rows)}/{n} | {sum(r['naren_child'] for r in rows)}/{n} | "
            f"{sum(r['naren_family'] for r in rows)}/{n} | n/a |"]
    for arm in arms:
        summ.append(f"| + checker ({arm} arm) | " + " | ".join(f"{sum(r[arm][k] for r in rows)}/{n}"
                                                              for k in ("exact", "child", "family", "named")) + " |")
    summ += ["", "## What the checker changed", "", "| | fixed (wrong -> right) | broken (right -> wrong) | flagged, still wrong | not checked |",
             "|---|---|---|---|---|"]
    for arm in arms:
        fixed = sum(1 for r in rows if not r["naren_exact"] and r[arm]["exact"])
        broken = sum(1 for r in rows if r["naren_exact"] and not r[arm]["exact"])
        still = sum(1 for r in rows if r[arm]["flagged"] and not r[arm]["exact"])
        unchecked = sum(1 for r in rows if r[arm]["verdict"] == "not checked")
        summ.append(f"| {arm} arm | {fixed} | {broken} | {still} | {unchecked} |")
    lines += summ

    out = Path(a.out)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    flat = []
    for r in rows:
        d = {k: v for k, v in r.items() if k not in arms}
        for arm in arms:
            d.update({f"{arm}_{k}": v for k, v in r[arm].items()})
        flat.append(d)
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    print("\n".join(lines))
    print(f"\nwritten: {out.with_suffix('.md')} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
