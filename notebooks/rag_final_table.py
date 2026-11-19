"""Build the final-output table (Naren code, RAG verdict and final code, billed codes) from a finished run. No LLM calls.

    python rag_final_table.py ../../icd10cm_sources/rag_final_run ../patient_records

Reads <run>/rules/checks.jsonl, <run>/report/per_case.csv (from rag_compare_report.py) and each admission's ground_truth.json.
Writes <run>/report/final_output_with_billed.md and .csv.
"""
import json, csv, re, sys
from pathlib import Path
run = Path(sys.argv[1]); recs_dir = Path(sys.argv[2]); out = run/"report"
nd = lambda c: re.sub(r"[.\-\s]", "", str(c or "")).upper()
rows = [json.loads(l) for l in open(run/"rules"/"checks.jsonl", encoding="utf-8")]
status = {r["key"]: r["billing_status"] for r in csv.DictReader(open(out/"per_case.csv", encoding="utf-8"))}
gt = {}
for p in recs_dir.glob("patient_*/admissions/hadm_*/ground_truth.json"):
    d = json.load(open(p, encoding="utf-8")); gt[(str(d["patient_id"]), str(d["hadm_id"]))] = d
def final_codes(r):
    l = r["llm"]; v = l.get("verdict"); sel = (r.get("code_selection") or {}).get("selected") or []
    if v == "keep": return [r["case"]["code"]]
    return [l["proposed_code"]] if l.get("proposed_code") else sel
def rel(code, billed):
    c = nd(code); hits = [b for b in billed if nd(b) == c]
    if hits: return "yes"
    if any(nd(b).startswith(c) or c.startswith(nd(b)) for b in billed): return "related (more/less specific)"
    return "no"
NAMES = {"billed_primary": "billed (primary)", "billed_secondary": "billed (secondary)", "family_billed": "same family billed", "not_billed": "not billed"}
data = []; md = ["# Final output of the RAG with the billed codes added", "",
  "Billed codes come from each admission's ground_truth.json (evaluation labels, never shown to the model). 'Naren's code' is the code from his pipeline that the RAG checked. 'Naren's code billed?' is whether it appears in the billing record. 'RAG code in billed list' is whether the RAG's final code does (exact, or related = same code but billed more/less specific).", ""]
cur = None
for r in rows:
    k = (r["patient_id"], str(r["hadm_id"])); g = gt[k]; billed = g["icd10_codes"]
    fc = final_codes(r); fin = ", ".join(fc) if fc else "no code proposed"
    fb = "—" if not fc else (rel(fc[0], billed) if len(fc) == 1 else "; ".join(f"{c}: {rel(c, billed)}" for c in fc))
    st = NAMES.get(status.get(r["key"]), status.get(r["key"], ""))
    d = dict(patient=k[0], hadm=k[1], billed_primary=g["primary_icd_code"], billed_codes=" ".join(billed), rank=r["case"]["rank"],
             diagnosis=r["case"]["diagnosis"], naren_code=r["case"]["code"], naren_code_billed=st, rag_verdict=r["llm"]["verdict"],
             rag_final_code=fin, rag_code_in_billed_list=fb, provider_query="yes" if r["llm"].get("needs_provider_query") else "no")
    data.append(d)
    if k != cur:
        cur = k
        md += ["", f"## Patient {k[0]}, admission {k[1]}", "", f"**Billed primary:** {g['primary_icd_code']}  |  **All billed codes:** {', '.join(billed)}", "",
               "| # | Diagnosis | Naren's code | Naren's code billed? | RAG verdict | RAG final code | RAG code in billed list | Provider query |", "|---|---|---|---|---|---|---|---|"]
    fshort = fin.split(" (+")[0] if fin.startswith(r["case"]["code"]) else fin
    md.append(f"| {d['rank']} | {d['diagnosis']} | {d['naren_code']} | {st} | {d['rag_verdict']} | {fshort} | {fb} | {d['provider_query']} |")
(out/"final_output_with_billed.md").write_text("\n".join(md), encoding="utf-8")
with open(out/"final_output_with_billed.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
print(len(data), "rows")
