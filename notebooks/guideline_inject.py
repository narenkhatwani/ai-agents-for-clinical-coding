"""Error-injection benchmark for the Guidelines checker (rules arm vs control arm).

Each scenario is a short SYNTHETIC case (written for this test, not a real patient) with one correct ICD-10-CM code
and one plausible wrong code. Both versions go through the checker on identical evidence:
    <scenario>|error  - the wrong code is the candidate; a good checker flags or replaces it
    <scenario>|clean  - the correct code is the candidate; a good checker keeps it
So we measure sensitivity (errors caught), false alarms (clean cases flagged) and whether the right code is found,
and the paired design lets us compare the two arms case by case.

Scenario groups say what is needed to catch the error:
    rule      a Coding Guideline decides it (the arm where retrieval should matter)
    evidence  reading the note is enough (no rule lookup needed)
    tabular   exact code-book facts (7th character, Excludes1) are enough

    python guideline_inject.py --list                                           # scenarios + code check, no LLM, no cost
    python guideline_inject.py --dry-run ...                                    # prompt sizes and cost estimate
    python guideline_inject.py --yes --arm rules   --out ../../icd10cm_sources/inject_rules ...
    python guideline_inject.py --yes --arm control --out ../../icd10cm_sources/inject_control ...
    python guideline_inject.py --compare-with ../../icd10cm_sources/inject_control --out ../../icd10cm_sources/inject_rules ...
    (shared options: --index ../../icd10cm_sources/guideline_index --models ../../icd10cm_sources/models
                     --tabular ../../icd10cm_sources/tabular_FY2026.json)
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import requests

import guideline_check as gc
from guideline_check_run import FLAGGED, NETWORK_RETRIES, OUT_TOKENS_PER_CALL, RETRY_FACTOR, append, load_done
from guideline_rag import Embedder, GuidelineIndex
from tabular_list import TabularList, nodot

SCENARIOS = [
    # ---- rule: a Coding Guideline decides it -------------------------------------------------------------------
    dict(id="complication_hemorrhage", group="rule", rule="I.B.16", wrong="K66.1", right="K91.840",
         diagnosis="Bleeding after gallbladder removal",
         evidence=["bloody output from the surgical drain on post-op day 1 after laparoscopic cholecystectomy",
                   "re-exploration found bleeding from the gallbladder fossa", "hemoglobin fell from 12.1 to 8.4 g/dL"],
         procedures=["laparoscopic cholecystectomy"]),
    dict(id="complication_wound_infection", group="rule", rule="I.B.16", wrong="L03.311", right="T81.41XA",
         diagnosis="Wound infection after hernia repair",
         evidence=["purulent drainage and erythema at the incision 6 days after open inguinal hernia repair",
                   "wound culture grew MSSA", "surgeon documents a superficial incisional surgical site infection"],
         procedures=["open inguinal hernia repair"]),
    dict(id="htn_ckd", group="rule", rule="I.C.9.a.2", wrong="I10", right="I12.9",
         diagnosis="Hypertension",
         evidence=["long-standing hypertension treated with lisinopril", "chronic kidney disease stage 3, eGFR 42",
                   "no other cause of CKD documented"]),
    dict(id="htn_heart_failure", group="rule", rule="I.C.9.a.1", wrong="I10", right="I11.0",
         diagnosis="Hypertension",
         evidence=["history of hypertension for 15 years", "chronic systolic heart failure, ejection fraction 30%",
                   "no other cause of heart failure documented"]),
    dict(id="diabetes_ckd", group="rule", rule="I.C.4.a", wrong="E11.9", right="E11.22",
         diagnosis="Type 2 diabetes mellitus",
         evidence=["type 2 diabetes on metformin, A1c 8.1", "CKD stage 3 with albuminuria, documented as diabetic nephropathy"]),
    dict(id="sepsis_sequencing", group="rule", rule="I.C.1.d.4", wrong="N39.0", right="A41.51",
         diagnosis="Sepsis due to E. coli urinary tract infection",
         evidence=["fever 39.2 C, heart rate 118, lactate 3.4", "urine and blood cultures grew E. coli",
                   "provider documents sepsis due to E. coli UTI"]),
    dict(id="history_of_dvt", group="rule", rule="I.C.21.c.4", wrong="I82.402", right="Z86.718",
         diagnosis="History of deep vein thrombosis",
         evidence=["left leg DVT 4 years ago, anticoagulation completed 3 years ago",
                   "no current symptoms and negative duplex this admission", "not on anticoagulants"]),
    dict(id="history_of_cancer", group="rule", rule="I.C.2.d", wrong="C50.911", right="Z85.3",
         diagnosis="History of breast cancer",
         evidence=["right breast cancer treated with lumpectomy and radiation in 2016",
                   "no evidence of disease on surveillance mammogram", "no current treatment"]),
    dict(id="old_mi", group="rule", rule="I.C.9.e", wrong="I21.4", right="I25.2",
         diagnosis="Prior myocardial infarction",
         evidence=["NSTEMI 4 months ago treated with a stent", "currently chest-pain free, troponin normal",
                   "admitted for elective surgery"]),
    dict(id="stroke_sequela", group="rule", rule="I.B.10", wrong="I63.9", right="I69.354",
         diagnosis="Left-sided weakness from past stroke",
         evidence=["right-handed patient", "ischemic stroke 3 years ago", "residual left arm and leg weakness",
                   "no new stroke this admission"]),
    dict(id="adverse_vs_poisoning", group="rule", rule="I.C.19.e.5", wrong="T45.511A", right="T45.515A",
         diagnosis="Hematuria due to warfarin",
         evidence=["INR 4.8", "warfarin taken exactly as prescribed", "gross hematuria with no urologic cause found"]),
    dict(id="symptom_integral", group="rule", rule="I.B.5", wrong="R07.9", right="I21.19",
         diagnosis="Chest pain",
         evidence=["chest pain with ST elevation in the inferior leads", "troponin markedly elevated",
                   "diagnosed inferior STEMI and taken to the cath lab"]),
    dict(id="anemia_in_ckd", group="rule", rule="I.A.13", wrong="D64.9", right="D63.1",
         diagnosis="Anemia in chronic kidney disease",
         evidence=["hemoglobin 8.9 g/dL with CKD stage 4", "provider documents anemia due to CKD", "started erythropoietin"]),
    dict(id="pathologic_fracture", group="rule", rule="I.C.13", wrong="S32.010A", right="M80.08XA",
         diagnosis="Vertebral compression fracture from osteoporosis",
         evidence=["L1 compression fracture after bending over to lift a light object", "DEXA T-score -3.1",
                   "provider documents osteoporotic pathological fracture"]),
    # ---- evidence: reading the note is enough ------------------------------------------------------------------
    dict(id="acute_on_chronic", group="evidence", rule="I.B.8", wrong="I50.22", right="I50.23",
         diagnosis="Acute on chronic systolic heart failure",
         evidence=["known HFrEF, ejection fraction 25%", "3 days of orthopnea and 10 lb weight gain", "treated with IV furosemide"]),
    dict(id="pressure_ulcer_stage", group="evidence", rule="I.C.12", wrong="L89.150", right="L89.153",
         diagnosis="Stage 3 sacral pressure ulcer",
         evidence=["full-thickness skin loss over the sacrum with visible subcutaneous fat", "no exposed muscle or bone",
                   "wound nurse documents stage 3"]),
    dict(id="laterality", group="evidence", rule="I.B.13", wrong="M17.11", right="M17.12",
         diagnosis="Osteoarthritis of the left knee",
         evidence=["left knee pain and crepitus", "x-ray: left knee joint-space narrowing and osteophytes"]),
    dict(id="pneumonia_organism", group="evidence", rule="", wrong="J18.9", right="J13",
         diagnosis="Pneumonia due to Streptococcus pneumoniae",
         evidence=["right lower lobe consolidation", "sputum and blood cultures grew Streptococcus pneumoniae"]),
    # ---- tabular: exact code-book facts are enough -------------------------------------------------------------
    dict(id="seventh_character", group="tabular", rule="", wrong="S72.001", right="S72.001A",
         diagnosis="Closed fracture of the neck of the right femur",
         evidence=["fall at home", "x-ray shows right femoral neck fracture", "admitted for surgical repair"]),
    dict(id="excludes1_pair", group="tabular", rule="", wrong="E11.9", right="E11.9", wrong_extra=["E10.9"],
         diagnosis="Type 2 diabetes mellitus", evidence=["type 2 diabetes on metformin, A1c 7.4"]),
    # ---- fy2026: passages the FY2026 Guidelines print in bold (narrative changes since FY2025). Written AFTER the first
    #      injection run and after the checker fixes, so these are a fresh test, not something the fixes were tuned on. --
    dict(id="t2dm_remission", group="fy2026", rule="I.C.4.a.1", wrong="E11.9", right="E11.A",
         diagnosis="Type 2 diabetes mellitus in remission",
         evidence=["type 2 diabetes diagnosed 6 years ago", "after weight loss A1c has been 5.6 for 12 months off all diabetes medication",
                   "endocrinologist documents type 2 diabetes in remission"]),
    dict(id="t2dm_resolved_not_remission", group="fy2026", rule="I.C.4.a.1", wrong="E11.A", right=None, expect_query=True,
         diagnosis="Type 2 diabetes mellitus, resolved",
         evidence=["type 2 diabetes diagnosed 8 years ago", "after bariatric surgery the provider writes that the diabetes has resolved",
                   "A1c 5.4, off all diabetes medication"]),
    dict(id="hiv_on_art", group="fy2026", rule="I.C.1.a.2", wrong="B20", right="Z21",
         diagnosis="HIV positive on antiretroviral therapy",
         evidence=["documented HIV positive status", "currently managed on antiretroviral medication, viral load undetectable",
                   "no HIV-related illness or AIDS documented"]),
    dict(id="hiv_prior_illness", group="fy2026", rule="I.C.1.a.2", wrong="Z21", right="B20",
         diagnosis="HIV infection with a prior HIV-related illness",
         evidence=["Pneumocystis pneumonia diagnosed as an HIV-related illness 3 years ago", "currently asymptomatic on antiretroviral therapy",
                   "admitted for elective knee surgery"]),
    dict(id="hiv_negative_followup", group="fy2026", rule="I.C.1.a.2", wrong="Z11.4", right="Z71.7",
         diagnosis="Follow-up visit to discuss a negative HIV test result",
         evidence=["patient returns to discuss the HIV test done last week", "HIV test result is negative", "counseling provided"]),
    dict(id="hiv_test_with_symptoms", group="fy2026", rule="I.C.1.a.2", wrong="Z11.4", right="R50.9",
         diagnosis="HIV testing in a patient with symptoms",
         evidence=["fever, night sweats and weight loss for 6 weeks", "HIV test ordered because of these symptoms",
                   "no HIV diagnosis established yet"]),
    dict(id="prophylactic_mastectomy", group="fy2026", rule="I.C.21.c.13", wrong="Z15.01", right="Z40.01",
         diagnosis="Admission for prophylactic bilateral mastectomy",
         evidence=["BRCA1 mutation carrier with a strong family history of breast cancer", "no breast cancer diagnosed",
                   "admitted for risk-reducing bilateral mastectomy"], procedures=["bilateral prophylactic mastectomy"]),
    dict(id="htn_cardiomegaly", group="fy2026", rule="I.C.9.a.1", wrong="I10", right="I11.9",
         diagnosis="Hypertension",
         evidence=["long-standing hypertension", "chest x-ray shows cardiomegaly",
                   "echocardiogram: left ventricular hypertrophy, preserved ejection fraction, no heart failure"]),
]


def build_items(tabular):
    items = []
    for s in SCENARIOS:
        for label in ("error", "clean"):
            if label == "clean" and s["right"] is None:  # no single correct code to use as the clean twin
                continue
            code = s["wrong"] if label == "error" else s["right"]
            rec = tabular.record(code)
            if rec is None:
                raise SystemExit(f"{s['id']}: code {code} not in the Tabular List")
            case = {"rank": 1, "diagnosis": s["diagnosis"], "code": code, "code_title": rec["title"],
                    "evidence": s["evidence"], "procedures": s.get("procedures", [])}
            codes = [code] + (s.get("wrong_extra", []) if label == "error" else [])
            items.append({"key": f"{s['id']}|{label}", "scenario": s, "label": label, "case": case, "codes": codes})
    return items


def list_scenarios(tabular):
    print(f"{len(SCENARIOS)} synthetic scenarios -> {len(build_items(tabular))} items (error item + clean twin each, where a single correct code exists)\n")
    print(f"{'id':<28}{'group':<10}{'wrong':<10}{'right':<10}rule")
    for s in SCENARIOS:
        build = {c: tabular.record(c) for c in (s["wrong"], s["right"]) if c}
        assert all(build.values()), s["id"]
        print(f"{s['id']:<28}{s['group']:<10}{s['wrong']:<10}{(s['right'] or '-'):<10}{s['rule']}")
        print(f"    wrong: {build[s['wrong']]['title'][:80]}" + (f"\n    right: {build[s['right']]['title'][:80]}" if s["right"] else "\n    right: (none: expected outcome is a provider query)"))
    from collections import Counter
    print("\ngroups:", dict(Counter(s["group"] for s in SCENARIOS)))


def dry_run(items, index, tabular, price_in, price_out, control):
    system = gc.SYSTEM_PROMPT_CONTROL if control else gc.SYSTEM_PROMPT
    toks = [gc.est_tokens(system + gc.prepare(i["case"], index, tabular, i["codes"], 12000, control=control)[3]) for i in items]
    calls, total = len(toks), sum(toks)
    cost = RETRY_FACTOR * (total * price_in + calls * OUT_TOKENS_PER_CALL * price_out) / 1e6
    print(f"arm: {'CONTROL (no Guidelines text)' if control else 'rules'}")
    print(f"{calls} calls | ~{total:,} input tokens (avg {total // calls}) | ~{calls * OUT_TOKENS_PER_CALL:,} output tokens (assumed {OUT_TOKENS_PER_CALL}/call)")
    print(f"estimated cost incl. {RETRY_FACTOR}x retries at ${price_in}/M in, ${price_out}/M out: ~${cost:.2f}")
    print("(prices are inputs, not looked up; Opus also spends hidden reasoning tokens, so check the OpenRouter Activity page)")


def retrieval_check(items, index, tabular):
    """Free (no LLM): for each error item, is the rule that governs the case among the rules the checker would be shown?"""
    print("expected rule shown to the model? (error items that name a rule; rules arm retrieval only)\n")
    hits = total = 0
    for it in items:
        s = it["scenario"]
        if it["label"] != "error" or not s["rule"]:
            continue
        shown = [r["parent_id"] for r in gc.prepare(it["case"], index, tabular, it["codes"], 12000)[4]]
        ok = rule_hit(s["rule"], shown)
        hits, total = hits + ok, total + 1
        print(f"  {s['id']:<28}{s['group']:<10}{s['rule']:<12}{'HIT ' if ok else 'miss'} shown={shown}")
    print(f"\n{hits}/{total} expected rules shown")


def run(items, index, tabular, title_search, config, call_llm_json, out_dir, not_available_error, control):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done, n_new = load_done(out), 0
    for it in items:
        if it["key"] in done:
            continue
        rec = None
        for attempt in range(1, NETWORK_RETRIES + 1):
            try:
                rec = gc.check_case(it["case"], index, tabular, title_search, it["codes"], config, call_llm_json, 12000, control=control)
                break
            except (not_available_error, requests.exceptions.RequestException) as exc:
                if attempt == NETWORK_RETRIES:
                    print(f"  Network still down after {NETWORK_RETRIES} tries at {it['key']}; progress is saved, re-run the same command.")
                    raise
                wait = min(30 * attempt, 180)
                print(f"  Network error at {it['key']} ({type(exc).__name__}); waiting {wait}s (retry {attempt}/{NETWORK_RETRIES - 1})...")
                time.sleep(wait)
            except Exception as exc:
                print(f"  ERROR {it['key']}: {type(exc).__name__}: {str(exc)[:120]}")
                append(out / "errors.jsonl", {"key": it["key"], "error": f"{type(exc).__name__}: {exc}"})
                break
        if rec is None:
            continue
        rec.update(key=it["key"], label=it["label"], scenario_id=it["scenario"]["id"])
        append(out / "checks.jsonl", rec)
        n_new += 1
        llm = rec["llm"]
        print(f"  {it['key']:<36}{it['case']['code']:<10}-> {str(llm.get('verdict')):<8} cited={llm.get('cited_rules')} valid={rec['valid']} ({rec['seconds']}s)")
    print(f"\nnew results this run: {n_new} | total saved: {len(load_done(out))} -> {out / 'checks.jsonl'}")


def rule_hit(expected, shown):
    return bool(expected) and any(s == expected or s.startswith(expected + ".") or expected.startswith(s + ".") for s in shown)


def code_match(found, right):
    """Exact match, or the found code lacks only the 7th character: the title search returns T81.41 for T81.41XA."""
    return found == right or (right.startswith(found) and 0 < len(right) - len(found) <= 3)


def score(out_dir):
    """One row per item: verdict, whether the right code was found, and whether the expected rule was shown/cited."""
    rows = []
    by_id = {s["id"]: s for s in SCENARIOS}
    for key, r in load_done(out_dir).items():
        s = by_id.get(r.get("scenario_id"))
        if s is None:
            continue
        llm = r["llm"]
        verdict = str(llm.get("verdict", "")).lower()
        found = ([nodot(llm["proposed_code"])] if llm.get("proposed_code") else []) + \
                [nodot(c["code"]) for c in r.get("codes_for_suggested_condition", [])[:3]]
        right = nodot(s["right"]) if s["right"] else None
        rows.append({"key": key, "scenario": s["id"], "group": s["group"], "label": r["label"], "code": r["case"]["code"],
                     "verdict": verdict, "flagged": verdict in FLAGGED, "valid": r["valid"],
                     "found_exact": bool(right) and r["label"] == "error" and any(code_match(f, right) for f in found),
                     "found_family": bool(right) and r["label"] == "error" and any(f[:3] == right[:3] for f in found),
                     "has_right": bool(right), "query_asked": bool(llm.get("needs_provider_query")),
                     "expect_query": bool(s.get("expect_query")),
                     "rule": s["rule"], "rule_shown": rule_hit(s["rule"], r.get("rules_shown", [])),
                     "rule_cited": rule_hit(s["rule"], llm.get("cited_rules") or []),
                     "cited": "|".join(llm.get("cited_rules") or []), "suggested": llm.get("suggested_condition", ""),
                     "reason": llm.get("reason", "")})
    return rows


def pct(n, d):
    return f"{n}/{d} ({100 * n // d}%)" if d else "n/a"


def summarize(out_dir, arm_name):
    rows = score(out_dir)
    if not rows:
        print("no results yet in", out_dir)
        return rows
    print(f"\n=== {arm_name} | {len(rows)} items | valid answers: {pct(sum(r['valid'] for r in rows), len(rows))}")
    print(f"{'group':<10}{'errors caught':>20}{'right code found':>20}{'clean flagged (false alarm)':>30}")
    for g in ("rule", "evidence", "tabular", "fy2026", "ALL"):
        sub = [r for r in rows if g == "ALL" or r["group"] == g]
        err, cln = [r for r in sub if r["label"] == "error"], [r for r in sub if r["label"] == "clean"]
        have = [r for r in err if r["has_right"]]
        print(f"{g:<10}{pct(sum(r['flagged'] for r in err), len(err)):>20}{pct(sum(r['found_exact'] for r in have), len(have)):>20}"
              f"{pct(sum(r['flagged'] for r in cln), len(cln)):>30}")
    err = [r for r in rows if r["label"] == "error" and r["rule"]]
    if err and any(r["rule_shown"] or r["rule_cited"] for r in rows):
        print(f"\nexpected rule retrieved: {pct(sum(r['rule_shown'] for r in err), len(err))} | "
              f"cited by the model: {pct(sum(r['rule_cited'] for r in err), len(err))}  (error items that name a rule)")
    with open(Path(out_dir) / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["key"]))
    print(f"per-item table -> {Path(out_dir) / 'summary.csv'}")
    return rows


def sign_test(only_a, only_b):
    """Two-sided exact sign test on the cases where the arms differ."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(rules_dir, control_dir):
    a = {r["key"]: r for r in score(rules_dir)}
    b = {r["key"]: r for r in score(control_dir)}
    keys = sorted(set(a) & set(b))
    print(f"\n=== RULES arm vs CONTROL arm on {len(keys)} common items (same LLM, same Tabular facts; control gets no Guidelines text)")
    for label, name, good in (("error", "ERRORS caught (flag or replace)", True), ("clean", "CLEAN cases wrongly flagged (false alarms)", False)):
        ks = [k for k in keys if a[k]["label"] == label]
        only_r = [k for k in ks if a[k]["flagged"] and not b[k]["flagged"]]
        only_c = [k for k in ks if b[k]["flagged"] and not a[k]["flagged"]]
        both = sum(a[k]["flagged"] and b[k]["flagged"] for k in ks)
        print(f"\n{name}: rules {pct(sum(a[k]['flagged'] for k in ks), len(ks))} | control {pct(sum(b[k]['flagged'] for k in ks), len(ks))}"
              f" | both {both} | only rules {len(only_r)} | only control {len(only_c)} | sign test p={sign_test(len(only_r), len(only_c)):.3f}")
        for k in only_r + only_c:
            print(f"    {k:<36} {'only rules' if k in only_r else 'only control'}")
    ek = [k for k in keys if a[k]["label"] == "error" and a[k]["has_right"]]
    print(f"\nright code found (error items): rules {pct(sum(a[k]['found_exact'] for k in ek), len(ek))} | control {pct(sum(b[k]['found_exact'] for k in ek), len(ek))}")
    print("\nper scenario (E = error item verdict, C = clean item verdict; want E=flag/replace, C=keep):")
    print(f"{'scenario':<28}{'group':<10}{'rules E / C':<20}{'control E / C':<20}{'rule shown':<12}rule cited")
    for s in SCENARIOS:
        e, c = f"{s['id']}|error", f"{s['id']}|clean"
        if s["right"] is None and e in a and e in b:
            print(f"{s['id']:<28}{s['group']:<10}{a[e]['verdict'] + ' (error only)':<20}{b[e]['verdict'] + ' (error only)':<20}"
                  f"{('yes' if a[e]['rule_shown'] else 'no') if s['rule'] else '-':<12}{('yes' if a[e]['rule_cited'] else 'no') if s['rule'] else '-'}"
                  f"   provider query asked: rules={a[e]['query_asked']} control={b[e]['query_asked']}")
        elif e in a and c in a and e in b and c in b:
            print(f"{s['id']:<28}{s['group']:<10}{a[e]['verdict'] + ' / ' + a[c]['verdict']:<20}{b[e]['verdict'] + ' / ' + b[c]['verdict']:<20}"
                  f"{('yes' if a[e]['rule_shown'] else 'no') if s['rule'] else '-':<12}{('yes' if a[e]['rule_cited'] else 'no') if s['rule'] else '-'}")


def load_index(a, embedder):
    if a.index_type == "flat":
        return GuidelineIndex.load(a.index, embedder, None)
    if not a.hier_index:
        sys.exit("--hier-index is required for --index-type hier or fused")
    import guideline_hier as gh
    hier = gh.HierWrap(gh.HierIndex.load(a.hier_index, embedder))
    return hier if a.index_type == "hier" else gh.FusedIndex(GuidelineIndex.load(a.index, embedder, None), hier)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index")
    ap.add_argument("--models")
    ap.add_argument("--index-type", choices=["flat", "hier", "fused"], default="flat",
                    help="flat = the child-chunk search (default); hier = coarse-to-fine search of guideline_hier.py "
                         "(--hier-index); fused = both, rank-fused")
    ap.add_argument("--hier-index", help="folder built by guideline_hier.py build (for --index-type hier or fused)")
    ap.add_argument("--tabular", required=True)
    ap.add_argument("--out")
    ap.add_argument("--arm", choices=["rules", "control"], default="rules")
    ap.add_argument("--compare-with", help="control-arm results folder to compare against --out (no LLM calls)")
    ap.add_argument("--list", action="store_true", help="show the scenarios and check every code against the Tabular List")
    ap.add_argument("--retrieval-check", action="store_true", help="free: is the governing rule among the rules shown? (no LLM)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm a real (paid) run")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--price-in", type=float, default=5.0)
    ap.add_argument("--price-out", type=float, default=25.0)
    a = ap.parse_args()

    tabular = TabularList.load(a.tabular)
    if a.list:
        list_scenarios(tabular)
        return
    if not a.out and not (a.dry_run or a.retrieval_check):
        sys.exit("--out is required")
    if a.compare_with:
        summarize(a.out, "RULES arm")
        summarize(a.compare_with, "CONTROL arm")
        compare(a.out, a.compare_with)
        return
    if a.summarize_only:
        summarize(a.out, f"{a.arm.upper()} arm")
        return
    if not (a.index and a.models):
        sys.exit("--index and --models are required for a dry run or a real run")
    items = build_items(tabular)
    control = a.arm == "control"
    embedder = Embedder(Path(a.models) / "minilm")
    index = load_index(a, embedder)
    if a.retrieval_check:
        retrieval_check(items, index, tabular)
        return
    if a.dry_run:
        dry_run(items, index, tabular, a.price_in, a.price_out, control)
        return

    gc.ensure_ca_bundle(Path(a.index).parent / "ca_bundle.pem")
    from dataclasses import replace
    from pipeline import ICD_CONFIRM_TEMPERATURE, LLMNotAvailableError, call_llm_json, get_llm_config
    config = replace(get_llm_config(), temperature=float(ICD_CONFIRM_TEMPERATURE))
    print(f"LLM (Naren's settings): {config.api_label}/{config.model} temperature={config.temperature} zdr={config.openrouter_zdr}")
    if not config.resolved_api_key():
        sys.exit(f"No API key. Set the {config.api_key_env} environment variable.")
    dry_run(items, index, tabular, a.price_in, a.price_out, control)
    if not a.yes:
        sys.exit("\nNot started. Re-run with --yes to spend the credits above (results already saved are skipped).")
    run(items, index, tabular, gc.TitleSearch(tabular, embedder, Path(a.index) / "chroma"), config, call_llm_json,
        a.out, LLMNotAvailableError, control)
    summarize(a.out, f"{a.arm.upper()} arm")


if __name__ == "__main__":
    main()
