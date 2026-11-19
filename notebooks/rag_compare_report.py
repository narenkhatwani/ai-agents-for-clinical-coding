"""SCRIPT 2 of 2: compare the three methods on the 15 real project patients and write a report. No LLM calls, no cost.

  1. Naren's baseline   his pipeline's top-ranked ICD code per admission (read from his Stage 10 files)
  2. Control arm        same candidate codes + same LLM + same Tabular List facts, NO Guidelines text
  3. Rules arm          the control arm plus Guidelines rules from the RAG (fused hierarchical + flat retrieval by default)

It reads the run folder made by rag_hier_run.py (<run>/rules/, <run>/control/, and <run>/rules_raw/, <run>/control_raw/ if present)
and writes, next to it, <run>/report/report.md (the written comparison), per_admission.csv, per_case.csv and numbers.json.
The prose is generated from the numbers: every sentence that states a result is computed, not typed in.

    python rag_compare_report.py --records ../patient_records --run ../../icd10cm_sources/rag_hier_run \
        --tabular ../../icd10cm_sources/tabular_FY2026.json

To compare any two result folders instead (e.g. older runs): --rules <folder> --control <folder> --out <report folder>
[--retrieval flat|hier|fused] says how the rules arm retrieved, for the wording of the report.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import guideline_check_run as gcr
from guideline_vs_naren import corrected_code, nodot, score

FLAGGED = gcr.FLAGGED
RETRIEVAL_TEXT = {"flat": "the flat paragraph search", "hier": "the hierarchical (outline) search",
                  "fused": "the hierarchical and flat searches merged"}
FILLER = {"I", "II", "III", "IV"}   # whole-Section introductions: broad containers, not specific rules


def P(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def sign_p(b, c):
    """Two-sided exact sign test on b wins against c wins (ties excluded)."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def kappa(a_flags, b_flags):
    n = len(a_flags)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a_flags, b_flags)) / n
    pa, pb = sum(a_flags) / n, sum(b_flags) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def case_label(row):
    return f"{row['patient']} rank {row['rank']} {row['code']}"


def md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_admissions(records):
    out = []
    for pid, hid, adm in gcr.admissions(records):
        acc, gt = adm / "accuracy" / "accuracy.json", adm / "ground_truth.json"
        if not (acc.exists() and gt.exists()):
            continue
        ev = json.loads(acc.read_text(encoding="utf-8"))["evaluation"]
        g = json.loads(gt.read_text(encoding="utf-8"))
        icd = ev["icd_linked"]
        out.append({"patient": pid, "hadm": hid, "dir": adm, "primary": nodot(g["primary_icd_code"]),
                    "billed": [nodot(c) for c in g["icd10_codes"]], "naren": nodot(icd.get("rank1_predicted_icd")),
                    "naren_judged": bool(icd.get("is_rank1_icd_match")), "dx_match": bool(ev["diagnosis"].get("match"))})
    return out


def method_result(recs, adm):
    """The top-ranked row of one admission under one arm."""
    rec = next((r for r in recs.values() if r["patient_id"] == adm["patient"] and r["hadm_id"] == adm["hadm"]
                and r["case"]["rank"] == 1), None)
    if rec is None or nodot(rec["case"]["code"]) != adm["naren"]:
        code, flagged, verdict, named = adm["naren"], False, "not checked", []
    else:
        code, flagged, named = corrected_code(rec)
        verdict = str(rec["llm"].get("verdict", "")).lower()
    return {"verdict": verdict, "flagged": flagged, "code": code, **score(code, adm["primary"]),
            "named": adm["primary"] in named or adm["naren"] == adm["primary"]}


def arm_stats(rows, recs):
    """Row-level (every checked candidate code) statistics for one arm."""
    flagged = lambda r: r["verdict"] in FLAGGED
    billed = [r for r in rows if r["primary_status"] != "not_billed"]
    unbilled = [r for r in rows if r["primary_status"] == "not_billed"]
    vals = list(recs.values())
    toks = [r.get("prompt_tokens_est") or 0 for r in vals]
    return {"n": len(rows), "keep": sum(r["verdict"] == "keep" for r in rows), "flag": sum(r["verdict"] == "flag" for r in rows),
            "replace": sum(r["verdict"] == "replace" for r in rows), "flagged": sum(map(flagged, rows)),
            "billed_n": len(billed), "billed_flagged": sum(map(flagged, billed)),
            "unbilled_n": len(unbilled), "unbilled_flagged": sum(map(flagged, unbilled)),
            "by_status": {s: (sum(flagged(r) for r in rows if r["primary_status"] == s), sum(r["primary_status"] == s for r in rows))
                          for s in gcr.STATUS_LABEL},
            "provider_query": sum(bool(r["provider_query"]) for r in rows), "valid": sum(bool(r["valid"]) for r in rows),
            "retried": sum(int(r["attempts"]) > 1 for r in rows), "seconds": sum(float(r.get("seconds") or 0) for r in vals),
            "avg_tokens": sum(toks) / max(len(toks), 1), "calls": len(vals) + sum(int(r.get("attempts", 1)) > 1 for r in vals)}


def evidence_stats(recs):
    """What the rules arm showed and cited."""
    vals = list(recs.values())
    shown = [x for r in vals for x in (r.get("rules_shown") or [])]
    cited = [x for r in vals for x in ((r.get("llm") or {}).get("cited_rules") or [])]
    flagged = [r for r in vals if str((r.get("llm") or {}).get("verdict", "")).lower() in FLAGGED]
    return {"cases": len(vals), "cases_with_cite": sum(bool((r.get("llm") or {}).get("cited_rules")) for r in vals),
            "flagged": len(flagged), "flagged_with_cite": sum(bool((r.get("llm") or {}).get("cited_rules")) for r in flagged),
            "shown_total": len(shown), "shown_filler": sum(x in FILLER for x in shown), "shown_distinct": len(set(shown)),
            "cited_total": len(cited), "cited_filler": sum(x in FILLER for x in cited), "top_cited": Counter(cited).most_common(6),
            "ib18_cases": sum("I.B.18" in ((r.get("llm") or {}).get("cited_rules") or []) for r in vals)}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--run", help="run folder made by rag_hier_run.py")
    ap.add_argument("--rules", help="rules-arm results folder (instead of --run)")
    ap.add_argument("--control", help="control-arm results folder (instead of --run)")
    ap.add_argument("--out", help="report folder (default <run>/report)")
    ap.add_argument("--retrieval", choices=list(RETRIEVAL_TEXT), help="how the rules arm retrieved (default: from the run manifest, else hier)")
    ap.add_argument("--tabular", help="optional: Tabular List json, to count billed codes that FY2026 has made more specific")
    ap.add_argument("--control-repeat", help="a second, independent run of the control arm (rag_hier_run.py --arms control into a new folder, "
                                             "its control/ subfolder): adds a section showing how much a plain re-run differs by itself")
    a = ap.parse_args()

    run = Path(a.run) if a.run else None
    rules_dir = Path(a.rules) if a.rules else (run / "rules" if run else None)
    control_dir = Path(a.control) if a.control else (run / "control" if run else None)
    if not (rules_dir and control_dir):
        sys.exit("give --run, or both --rules and --control")
    out = Path(a.out) if a.out else (run / "report" if run else None)
    if out is None:
        sys.exit("--out is required without --run")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if run and (run / "manifest.json").exists() else {}
    retrieval = a.retrieval or manifest.get("index_type") or "hier"
    out.mkdir(parents=True, exist_ok=True)

    adms = load_admissions(a.records)
    adm_tuples = [(x["patient"], x["hadm"], x["dir"]) for x in adms]
    recs = {"control": gcr.load_done(control_dir), "rules": gcr.load_done(rules_dir)}
    if not recs["rules"] or not recs["control"]:
        sys.exit(f"no saved results in {rules_dir} or {control_dir}")
    rows = {arm: gcr.build_rows(adm_tuples, d) for arm, d in (("control", control_dir), ("rules", rules_dir))}
    n = len(adms)

    # ---- 1. final code per admission, three methods
    res = {x["patient"]: {"control": method_result(recs["control"], x), "rules": method_result(recs["rules"], x)} for x in adms}
    tot = {"naren": {k: sum(score(x["naren"], x["primary"])[k] for x in adms) for k in ("exact", "child", "family")}
           | {"named": sum(x["naren"] == x["primary"] for x in adms)}}
    for arm in ("control", "rules"):
        tot[arm] = {k: sum(res[x["patient"]][arm][k] for x in adms) for k in ("exact", "child", "family", "named")}
    change = {}
    for arm in ("control", "rules"):
        change[arm] = {
            "fixed": [x["patient"] for x in adms if score(x["naren"], x["primary"])["exact"] is False and res[x["patient"]][arm]["exact"]],
            "broken": [x["patient"] for x in adms if score(x["naren"], x["primary"])["exact"] and not res[x["patient"]][arm]["exact"]],
            "flagged_unfixed": [x["patient"] for x in adms if res[x["patient"]][arm]["flagged"] and not res[x["patient"]][arm]["exact"]
                                and not score(x["naren"], x["primary"])["exact"]],
            "unchanged": sum(res[x["patient"]][arm]["code"] == x["naren"] for x in adms),
            "not_checked": sum(res[x["patient"]][arm]["verdict"] == "not checked" for x in adms)}
    wins_r = sum(res[x["patient"]]["rules"]["exact"] and not res[x["patient"]]["control"]["exact"] for x in adms)
    wins_c = sum(res[x["patient"]]["control"]["exact"] and not res[x["patient"]]["rules"]["exact"] for x in adms)

    # ---- 2. row level
    st = {arm: arm_stats(rows[arm], recs[arm]) for arm in rows}
    ev = evidence_stats(recs["rules"])
    by_c, by_r = {r["key"]: r for r in rows["control"]}, {r["key"]: r for r in rows["rules"]}
    common = [k for k in by_r if k in by_c]
    fr = [by_r[k]["verdict"] in FLAGGED for k in common]
    fc = [by_c[k]["verdict"] in FLAGGED for k in common]
    both, only_r, only_c = (sum(x and y for x, y in zip(fr, fc)), sum(x and not y for x, y in zip(fr, fc)),
                            sum(y and not x for x, y in zip(fr, fc)))
    neither = len(common) - both - only_r - only_c
    disagreements = [k for k, x, y in zip(common, fr, fc) if x != y]
    tgt = {arm: [r for r in rows[arm] if r["true_target"]] for arm in rows}

    # ---- run-to-run noise: the control arm against a second run of itself
    noise = None
    if a.control_repeat:
        rep_recs = gcr.load_done(a.control_repeat)
        rep_rows = {r["key"]: r for r in gcr.build_rows(adm_tuples, a.control_repeat)}
        ck = [k for k in by_c if k in rep_rows]
        f1, f2 = [by_c[k]["verdict"] in FLAGGED for k in ck], [rep_rows[k]["verdict"] in FLAGGED for k in ck]
        rep_res = {x["patient"]: method_result(rep_recs, x) for x in adms}
        noise = {"cases": len(ck), "flag_orig": sum(f1), "flag_rep": sum(f2), "disagree": sum(x != y for x, y in zip(f1, f2)),
                 "kappa": kappa(f1, f2), "exact_orig": tot["control"]["exact"], "exact_rep": sum(rep_res[x["patient"]]["exact"] for x in adms),
                 "code_differs": sum(rep_res[x["patient"]]["code"] != res[x["patient"]]["control"]["code"] for x in adms),
                 "cases_list": [k for k, x, y in zip(ck, f1, f2) if x != y]}

    # ---- 3. effect of the code-selection step (needs the *_raw folders)
    selection = None
    if run and (run / "rules_raw" / "checks.jsonl").exists() and (run / "control_raw" / "checks.jsonl").exists():
        raw = {"rules": gcr.load_done(run / "rules_raw"), "control": gcr.load_done(run / "control_raw")}
        selection = {arm: {"raw": sum(method_result(raw[arm], x)["named"] for x in adms),
                           "final": sum(res[x["patient"]][arm]["named"] for x in adms),
                           "raw_exact": sum(method_result(raw[arm], x)["exact"] for x in adms),
                           "picked": sum(bool((r.get("code_selection") or {}).get("selected")) for r in recs[arm].values()),
                           "suggested": sum(bool(((r.get("llm") or {}).get("suggested_condition") or "").strip()) for r in recs[arm].values())}
                     for arm in raw}

    drift = None
    if a.tabular:
        from tabular_list import TabularList
        tab = TabularList.load(a.tabular)
        allb = [c for x in adms for c in x["billed"]]
        drift = (sum(1 for c in allb if (tab.record(c) or {}).get("is_leaf") is False), len(allb))

    # ---- write csv + json
    def cell(arm, x):
        r = res[x["patient"]][arm]
        return r["verdict"], r["code"], r["exact"]
    with (out / "per_admission.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient", "billed_primary", "naren_code", "naren_exact", "control_verdict", "control_result", "control_exact",
                    "rules_verdict", "rules_result", "rules_exact"])
        for x in adms:
            w.writerow([x["patient"], x["primary"], x["naren"], x["naren"] == x["primary"], *cell("control", x), *cell("rules", x)])
    with (out / "per_case.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "billing_status", "control_verdict", "rules_verdict", "rules_cited", "control_reason", "rules_reason"])
        for k in common:
            w.writerow([k, by_r[k]["primary_status"], by_c[k]["verdict"], by_r[k]["verdict"], by_r[k]["cited"],
                        (recs["control"][k]["llm"].get("reason") or "")[:300], (recs["rules"][k]["llm"].get("reason") or "")[:300]])
    numbers = {"admissions": n, "totals": tot, "changes": change, "rows": st, "evidence": ev, "agreement": dict(both=both, only_rules=only_r,
               only_control=only_c, neither=neither, kappa=kappa(fr, fc)), "selection": selection, "billed_code_drift": drift, "noise": noise}
    (out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")

    # ---- the report
    rt = RETRIEVAL_TEXT[retrieval] + ("; whole-Section introductions removed from the results" if manifest.get("drop_sections") else "")
    L = []
    L += [f"# Naren's baseline vs Control vs Rules (Guidelines RAG): comparison on the {n} real project patients", "",
          f"Generated {datetime.now():%Y-%m-%d %H:%M} by rag_compare_report.py from the saved results in `{rules_dir}` and `{control_dir}`. "
          f"No LLM was called to make this report; every number below is computed from those files.", ""]

    L += ["## 1. The three methods", "",
          md(["", "1. Naren's baseline", "2. Control arm", "3. Rules arm (the RAG)"], [
              ["What it is", "Naren's pipeline output (SNOMED-guided diagnosis, ICD code mapping, Stage 9 confirmation)",
               "A reviewer LLM re-checks each of Naren's candidate codes", "The same reviewer, now also shown Guidelines rules"],
              ["Input to the reviewer", "n/a", "Case evidence + Tabular List facts (title, 7th character, notes, Excludes1 conflicts)",
               f"As the control arm + up to 4 Guidelines rules retrieved with {rt}"],
              ["Model", "Naren's models", "Claude Opus 5, temperature 0.3 (Naren's settings)", "Claude Opus 5, temperature 0.3 (Naren's settings)"],
              ["Output", "One ICD-10-CM code per diagnosis", "keep / flag / replace, a suggested condition, a mapped code", "The same, plus the rules it relied on (cited by id)"],
              ["Extra cost", "none", f"{st['control']['calls']} LLM calls", f"{st['rules']['calls']} LLM calls, prompts about {st['rules']['avg_tokens'] / max(st['control']['avg_tokens'], 1):.1f}x larger"]]), "",
          "The control arm exists to isolate the Guidelines: it has the same model, cases and Tabular facts as the rules arm, so any "
          "difference between arms 2 and 3 is due to the retrieved Guidelines text, and any difference between arms 1 and 2 is due to the "
          "reviewer pass (second look plus Tabular facts) and not to the RAG.", ""]

    L += ["## 2. Headline", ""]
    L += [f"- **Final code for each admission's top diagnosis vs the billed primary code ({n} admissions).** Exact match: Naren alone "
          f"{tot['naren']['exact']}/{n}, control arm {tot['control']['exact']}/{n}, rules arm {tot['rules']['exact']}/{n}. Same 3-character family: "
          f"{tot['naren']['family']}, {tot['control']['family']}, {tot['rules']['family']} of {n}.",
          f"- **Fixes and breaks.** Control arm fixed {len(change['control']['fixed'])} and broke {len(change['control']['broken'])}; rules arm fixed "
          f"{len(change['rules']['fixed'])} and broke {len(change['rules']['broken'])}. Between the two arms, rules is better on {wins_r} admission(s) and "
          f"control on {wins_c} (sign test p = {sign_p(wins_r, wins_c):.2f}).",
          f"- **Flagging.** Over all {st['rules']['n']} checked candidate codes, the rules arm flagged {st['rules']['flagged']} and the control arm "
          f"{st['control']['flagged']}; they disagreed on {len(disagreements)} and agreed on {len(common) - len(disagreements)} (Cohen's kappa {kappa(fr, fc):.2f}).",
          f"- **Evidence.** The rules arm cited at least one Guidelines rule in {P(ev['cases_with_cite'], ev['cases'])} of its answers, and in "
          f"{P(ev['flagged_with_cite'], ev['flagged'])} of its flags. The control arm cites nothing.", ""]

    L += [f"## 3. Final code per admission", "",
          "The top-ranked diagnosis of each admission, scored against the billed primary ICD-10-CM code. For a flagged code the checker's "
          "result is its proposed code, else the best code found for its suggested condition, else the unchanged code. A `keep` leaves the code unchanged. "
          "*Exact or child* also accepts a more specific child of the billed code (the billed list uses older codes). *Named* means the billed "
          "code appears anywhere among the checker's suggestions.", "",
          md(["", "exact", "exact or child", "same 3-char family", "billed code named"], [
              ["Naren alone", P(tot['naren']['exact'], n), P(tot['naren']['child'], n), P(tot['naren']['family'], n), "n/a"],
              ["Control arm", P(tot['control']['exact'], n), P(tot['control']['child'], n), P(tot['control']['family'], n), P(tot['control']['named'], n)],
              ["Rules arm", P(tot['rules']['exact'], n), P(tot['rules']['child'], n), P(tot['rules']['family'], n), P(tot['rules']['named'], n)]]), "",
          md(["", "fixed (wrong to right)", "broken (right to wrong)", "flagged, still wrong", "unchanged", "not checked"], [
              [arm.capitalize() + " arm", ", ".join(change[arm]["fixed"]) or "none", ", ".join(change[arm]["broken"]) or "none",
               len(change[arm]["flagged_unfixed"]), change[arm]["unchanged"], change[arm]["not_checked"]] for arm in ("control", "rules")]), ""]
    judged = sum(x["naren_judged"] for x in adms)
    L += [f"Naren's own Stage 10 evaluation counts {judged}/{n} top-ranked codes as matching, because it also accepts judgements of clinical "
          f"equivalence; the strict string comparison used here is {tot['naren']['exact']}/{n} exact.", ""]
    if drift:
        L += [f"**Code-version drift.** {drift[0]} of the {drift[1]} billed codes in these admissions ({100 * drift[0] / drift[1]:.0f}%) are parent "
              f"categories in FY2026 that now need a more specific code, so a correct, more specific FY2026 code can fail the exact comparison. "
              f"That is why the *exact or child* column is shown.", ""]
    L += ["Per admission:", "",
          md(["patient", "billed primary", "Naren", "control: verdict / result", "rules: verdict / result"], [
              [x["patient"], x["primary"], x["naren"] + (" ✓" if x["naren"] == x["primary"] else ""),
               f"{res[x['patient']]['control']['verdict']} / {res[x['patient']]['control']['code']}" + (" ✓" if res[x['patient']]['control']['exact'] else ""),
               f"{res[x['patient']]['rules']['verdict']} / {res[x['patient']]['rules']['code']}" + (" ✓" if res[x['patient']]['rules']['exact'] else "")]
              for x in adms]), ""]

    L += ["## 4. Where each arm flags a candidate code", "",
          f"All {st['rules']['n']} candidate codes (up to 6 differential rows per admission, including rows Naren mapped only to supporting codes). "
          "A code is checked against the billing record only as a proxy: a flag on a code that was billed as primary or secondary is a possible false alarm, "
          "while a flag on a code that was not billed is a possible true detection, but a code that was not billed is not necessarily wrong.", "",
          md(["", "Control arm", "Rules arm"], [
              ["keep / flag / replace", f"{st['control']['keep']} / {st['control']['flag']} / {st['control']['replace']}",
               f"{st['rules']['keep']} / {st['rules']['flag']} / {st['rules']['replace']}"],
              ["flagged in total", P(st['control']['flagged'], st['control']['n']), P(st['rules']['flagged'], st['rules']['n'])],
              *[[f"flagged: {gcr.STATUS_LABEL[s]}", P(*st['control']['by_status'][s]), P(*st['rules']['by_status'][s])] for s in gcr.STATUS_LABEL],
              ["possible false alarms (flag on a billed code)", P(st['control']['billed_flagged'], st['control']['billed_n']), P(st['rules']['billed_flagged'], st['rules']['billed_n'])],
              ["asks for a provider query", st['control']['provider_query'], st['rules']['provider_query']]]), "",
          f"Agreement: both flagged {both}, both kept {neither}, only the rules arm flagged {only_r}, only the control arm flagged {only_c}.", ""]
    tl = []
    for arm in ("control", "rules"):
        for r in tgt[arm]:
            tl.append([arm, r["patient"], r["code"], r["gt_primary"], r["verdict"], "yes" if r["hit_primary_exact"] else "no"])
    if tl:
        L += ["**Target cases** (the diagnosis was right, Naren's linked ICD wrong, and the candidate was not billed):", "",
              md(["arm", "patient", "Naren's code", "billed primary", "verdict", "billed code among suggestions"], tl), ""]

    L += ["## 5. What the Guidelines text changed", ""]
    if disagreements:
        L += [f"The arms gave different flag decisions on {len(disagreements)} of {len(common)} cases. Reasons are the models' own words, shortened.", "",
              md(["case", "billing status", "control", "rules", "rules cited", "rules arm's reason"], [
                  [case_label(by_r[k]), by_r[k]["primary_status"], by_c[k]["verdict"], by_r[k]["verdict"],
                   by_r[k]["cited"].replace("|", ", ") or "-",
                   (recs["rules"][k]["llm"].get("reason") or "").replace("|", "/")[:170]] for k in disagreements]), ""]
    else:
        L += ["The two arms made identical flag decisions on every case.", ""]
    stricter = "fewer" if only_c > only_r else "more" if only_r > only_c else "the same number of"
    L += [f"Net effect: the rules arm flagged {stricter} codes than the control arm ({only_r} flagged only with Guidelines text, {only_c} only without).", ""]

    if noise:
        L += ["### 5a. How much a plain re-run differs by itself", "",
              "The control arm was run twice with identical settings (same model, prompts and cases; the model's temperature of 0.3 makes answers vary a little). "
              "Comparing the two runs shows how large a difference between two arms has to be before it means anything.", "",
              md(["", "control, run 1", "control, run 2"], [
                  ["flagged", P(noise["flag_orig"], noise["cases"]), P(noise["flag_rep"], noise["cases"])],
                  ["top codes that match the billed primary exactly", f"{noise['exact_orig']}/{n}", f"{noise['exact_rep']}/{n}"]]), "",
              f"The two runs gave different flag decisions on {noise['disagree']} of {noise['cases']} cases (kappa {noise['kappa']:.2f}) and a different final top code in "
              f"{noise['code_differs']} of {n} admissions. For comparison, the control and rules arms differ on {len(disagreements)} of {len(common)} cases. "
              + ("So the gap between the arms is no larger than the gap between two runs of the same arm: it cannot be attributed to the Guidelines text."
                 if len(disagreements) <= noise["disagree"] else
                 f"The arms differ on {len(disagreements) - noise['disagree']} more cases than a plain re-run does, which is the part that could be due to the Guidelines text "
                 "(still a small sample)."), ""]

    L += ["## 6. Evidence and auditability (rules arm)", "",
          md(["", "value"], [
              ["answers citing at least one rule", P(ev['cases_with_cite'], ev['cases'])],
              ["flags citing at least one rule", P(ev['flagged_with_cite'], ev['flagged'])],
              ["rules shown to the model per case (average)", f"{ev['shown_total'] / max(ev['cases'], 1):.1f}"],
              ["distinct rules shown over all cases", ev['shown_distinct']],
              ["shown rules that are whole-Section introductions (I, II, III, IV)", P(ev['shown_filler'], ev['shown_total'])],
              ["citations that are whole-Section introductions", P(ev['cited_filler'], ev['cited_total'])],
              ["answers citing I.B.18 (unspecified / sign / symptom codes)", P(ev['ib18_cases'], ev['cases'])],
              ["most cited", ", ".join(f"{r} ({c})" for r, c in ev['top_cited'])]]), ""]

    L += ["## 7. The code-lookup step", "",
          "When a reviewer suggests another condition, that text has to become an ICD-10-CM code. Title search alone ranks words, so a vague "
          "suggestion can map to an unrelated code; the code-selection step lets the LLM pick from a pool of candidates (it cannot invent codes).", ""]
    if selection:
        L += [md(["", "billed code named, title search only", "billed code named, after code selection", "suggestions resolved by the LLM"], [
            [arm.capitalize() + " arm", P(selection[arm]["raw"], n), P(selection[arm]["final"], n), P(selection[arm]["picked"], selection[arm]["suggested"])]
            for arm in ("control", "rules")]), ""]
    else:
        L += ["(The `*_raw` folders were not found, so the effect of the selection step is not shown.)", ""]

    L += ["## 8. Cost and time", "",
          md(["", "Naren's baseline", "Control arm", "Rules arm"], [
              ["LLM calls in this evaluation", 0, st['control']['calls'], st['rules']['calls']],
              ["average prompt tokens per call (estimate)", "-", f"{st['control']['avg_tokens']:.0f}", f"{st['rules']['avg_tokens']:.0f}"],
              ["total LLM time", "-", f"{st['control']['seconds'] / 60:.1f} min", f"{st['rules']['seconds'] / 60:.1f} min"],
              ["answers that needed a retry (failed validation)", "-", st['control']['retried'], st['rules']['retried']],
              ["answers valid", "-", P(st['control']['valid'], st['control']['n']), P(st['rules']['valid'], st['rules']['n'])]]), "",
          "(Naren's baseline costs nothing extra here because it is read from his saved files; his own pipeline cost is not included.)", ""]

    # ---- detailed comparison and interpretation
    dr, dc = tot['rules']['exact'] - tot['control']['exact'], tot['control']['exact'] - tot['naren']['exact']
    L += ["## 9. The three methods compared in detail", "",
          md(["", "Naren's baseline", "Control arm", "Rules arm"], [
              ["Top-code exact match", P(tot['naren']['exact'], n), P(tot['control']['exact'], n), P(tot['rules']['exact'], n)],
              ["Top-code same family", P(tot['naren']['family'], n), P(tot['control']['family'], n), P(tot['rules']['family'], n)],
              ["Codes fixed / broken", "-", f"{len(change['control']['fixed'])} / {len(change['control']['broken'])}", f"{len(change['rules']['fixed'])} / {len(change['rules']['broken'])}"],
              ["Flags on billed codes (possible false alarms)", "-", P(st['control']['billed_flagged'], st['control']['billed_n']), P(st['rules']['billed_flagged'], st['rules']['billed_n'])],
              ["Flags on not-billed codes", "-", P(st['control']['unbilled_flagged'], st['control']['unbilled_n']), P(st['rules']['unbilled_flagged'], st['rules']['unbilled_n'])],
              ["Gives a reason a coder can audit", "no", "free-text reason only", "reason plus rule ids"],
              ["Uses Guidelines text", "no", "no", "yes"],
              ["Cost", "none extra", f"{st['control']['calls']} calls", f"{st['rules']['calls']} calls, larger prompts"]]), "",
          "**Naren's baseline.** It is the starting point and the only method that can propose a diagnosis in the first place. Its codes are "
          "specific to the diagnosis text but are mapped from SNOMED, so they can be nonspecific (N93.9 for post-hysterectomy bleeding) or miss "
          "combination and sequencing rules.", "",
          f"**Control arm.** Adding a second LLM pass with exact Tabular List facts changed the top code in "
          f"{n - change['control']['unchanged']} of {n} admissions and moved exact matches from {tot['naren']['exact']} to {tot['control']['exact']} "
          f"({'+' if dc >= 0 else ''}{dc}). This is the improvement that needs no retrieval at all. (For a flagged code the 'result' is the first "
          "suggested code, which is sometimes a companion code to add rather than a true replacement, so 'changed' overstates how often the "
          "reviewer disagreed with the diagnosis itself.)", "",
          f"**Rules arm.** Adding retrieved Guidelines text ({rt}) moved exact matches from {tot['control']['exact']} to {tot['rules']['exact']} "
          f"({'+' if dr >= 0 else ''}{dr}) relative to the control arm. It changed flag decisions on {len(disagreements)} of {len(common)} cases and made the "
          f"answers citable: {P(ev['cases_with_cite'], ev['cases'])} cite a rule.", ""]

    L += ["## 10. What the findings mean", ""]
    pts = []
    if tot['rules']['exact'] > tot['control']['exact']:
        pts.append(f"The rules arm reached more exact matches than the control arm ({tot['rules']['exact']} vs {tot['control']['exact']}), but the paired "
                   f"comparison is {wins_r} wins to {wins_c} (sign test p = {sign_p(wins_r, wins_c):.2f}); with {n} admissions and one run this is not "
                   "evidence of a real accuracy gain.")
    elif tot['rules']['exact'] < tot['control']['exact']:
        pts.append(f"The Guidelines text did not improve exact matches: the rules arm has {tot['rules']['exact']} against the control arm's "
                   f"{tot['control']['exact']} (paired {wins_r} wins to {wins_c}, sign test p = {sign_p(wins_r, wins_c):.2f}). The difference is within "
                   "noise for this sample size, but it gives no support to an accuracy claim for the RAG.")
    else:
        pts.append(f"Exact matches are equal for the control and rules arms ({tot['rules']['exact']}/{n} each), so on these patients the retrieved "
                   "Guidelines text did not change the final accuracy.")
    if len(change['rules']['broken']) == 0 and len(change['control']['broken']) == 0:
        pts.append("Neither arm broke a code that Naren had right, so the reviewer pass is safe on these patients.")
    else:
        pts.append(f"Broken codes: control {change['control']['broken'] or 'none'}, rules {change['rules']['broken'] or 'none'}; each is a case where "
                   "the reviewer replaced a correct code.")
    if only_c > only_r:
        pts.append(f"The Guidelines text made the model more conservative: it flagged {only_c - only_r} fewer codes net. Rule I.B.18 (unspecified and "
                   f"symptom codes are acceptable when nothing more specific is documented) was cited in {P(ev['ib18_cases'], ev['cases'])} of the rules answers, "
                   "which fits that pattern. Whether those extra control flags were right needs a human coder's judgement, because 'not billed' is not 'wrong'.")
    elif only_r > only_c:
        pts.append(f"The Guidelines text made the model flag {only_r - only_c} more codes net than the control arm.")
    if ev['flagged'] and ev['flagged_with_cite'] == ev['flagged']:
        pts.append("Every flag in the rules arm names a Guidelines rule, which the control arm cannot do. This auditability is the clearest "
                   "benefit of the RAG in these results.")
    if ev['shown_filler']:
        pts.append(f"{P(ev['shown_filler'], ev['shown_total'])} of the rules shown to the model are whole-Section introductions, not specific rules; "
                   "the retriever spends some of its four slots on broad text, and citations of those (" + str(ev['cited_filler']) + ") add little audit value.")
    if selection:
        r_, c_ = selection['rules'], selection['control']
        pts.append(f"The code-selection step raised how often the billed code appears among the suggestions from {r_['raw']} to {r_['final']} (rules) and "
                   f"{c_['raw']} to {c_['final']} (control) of {n}; the weak point of the earlier pipeline was turning a correct suggestion into a code.")
    if noise:
        pts.append(f"Run-to-run noise: two runs of the control arm alone differ on {noise['disagree']} of {noise['cases']} flag decisions and {noise['code_differs']} of {n} "
                   f"final codes, against {len(disagreements)} flag decisions between the control and rules arms. Differences between arms of that size are "
                   "noise, and only a larger or consistent gap would be evidence for the Guidelines text.")
    pts.append("Many of Naren's remaining misses are different diagnoses or a different principal-diagnosis choice (for example a symptom code against a "
               "billed neoplasm code), which a code-level reviewer cannot repair: it verifies candidates, it does not choose the principal diagnosis, and "
               "it never sees laboratory values.")
    L += [f"{i}. {p}" for i, p in enumerate(pts, 1)] + [""]

    L += ["### Limits of this comparison", "",
          f"- **Small sample.** {n} admissions and {st['rules']['n']} candidate codes; one run per arm at temperature 0.3, so repeated runs would differ slightly. "
          "A difference of one or two admissions is noise.",
          "- **The billing record is a proxy.** Billed codes come from MIMIC-IV administrative coding, possibly under an older code version, and include "
          "choices the notes do not fully justify.",
          "- **Post-hoc fixes.** The prompt wording, the validator and the code-lookup step were improved after earlier results were seen; the "
          "controlled comparison above uses the same final versions for both arms.",
          "- **Opus 5 knows most coding rules.** A strong model gains little from textbook rules. The most promising place to look for an accuracy "
          "benefit is rules newer than the model's training data (FY2026 changes), or a weaker model.", ""]

    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out / 'report.md'} (+ per_admission.csv, per_case.csv, numbers.json)")


if __name__ == "__main__":
    main()
