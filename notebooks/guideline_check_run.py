"""Run the guideline check over every admission and score it against Naren's ground truth.

    python guideline_check_run.py --dry-run ...                      # no LLM, no cost: case counts, prompt sizes, cost estimate
    python guideline_check_run.py --yes ...                          # real run with Naren's LLM (needs OPENROUTER_API_KEY), resumable
    python guideline_check_run.py --yes --arm control --out <other>  # CONTROL arm: same cases, same LLM, NO Guidelines text
    python guideline_check_run.py --summarize-only ...               # rebuild the scored summary from saved results
    python guideline_check_run.py --compare-with <control dir> ...   # rules arm (--out) vs control arm, side by side (no LLM)

    python guideline_check_run.py --yes --index-type fused --hier-index <outline index> --include-supporting ...
                                                                     # outline+flat retrieval; also checks supporting-only rows

Common arguments: --records ../patient_records --index ../../icd10cm_sources/guideline_index
    --models ../../icd10cm_sources/models --tabular ../../icd10cm_sources/tabular_FY2026.json
    --out ../../icd10cm_sources/check_run

Nothing in Naren's repo is modified; results go to --out (checks.jsonl, errors.jsonl, summary.csv).
Billing status of each candidate code: billed_primary | billed_secondary | family_billed | not_billed.
A code billed as a secondary diagnosis but chosen as the main candidate is flagged "not_primary", not counted as an error.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

import guideline_check as gc
from guideline_rag import Embedder, GuidelineIndex, cases_from_stage8
from tabular_list import TabularList, nodot

NETWORK_RETRIES = 6  # waits 30s, 60s, ... up to 180s between tries (~10 min total) before giving up

RETRY_FACTOR = 1.3
OUT_TOKENS_PER_CALL = 500
FLAGGED = ("flag", "replace")


def admissions(records_dir, only=None):
    found = []
    for p in sorted(Path(records_dir).glob("patient_*/admissions/hadm_*")):
        if (p / "icd_coding.json").exists():
            pid, hid = p.parent.parent.name.replace("patient_", ""), p.name.replace("hadm_", "")
            if not only or pid in only:
                found.append((pid, hid, p))
    return found


def load_truth(adm):
    g = json.loads((adm / "ground_truth.json").read_text(encoding="utf-8"))
    billed = {nodot(c) for c in g["icd10_codes"]}
    primary, target_rank = nodot(g["primary_icd_code"]), None
    acc = adm / "accuracy" / "accuracy.json"
    if acc.exists():
        ev = json.loads(acc.read_text(encoding="utf-8"))["evaluation"]
        dx, icd = ev["diagnosis"], ev["icd_linked"]
        if dx.get("match") and not icd.get("icd_match") and dx.get("matched_diffdx_rank"):
            target_rank = int(dx["matched_diffdx_rank"])
    return billed, primary, target_rank


def primary_status(code, billed, primary):
    c = nodot(code)
    if c == primary:
        return "billed_primary"
    if c in billed:
        return "billed_secondary"
    if any(b[:3] == c[:3] for b in billed):
        return "family_billed"
    return "not_billed"


def load_index(a, embedder):
    """flat = child-chunk search (default); hier = outline search; fused = both merged (same options as guideline_inject.py)."""
    if a.index_type == "flat":
        return GuidelineIndex.load(a.index, embedder, None)
    if not a.hier_index:
        sys.exit("--hier-index is required for --index-type hier or fused")
    import guideline_hier as gh
    drop = getattr(a, "drop_sections", False)
    hier = gh.HierWrap(gh.HierIndex.load(a.hier_index, embedder), drop_sections=drop)
    return hier if a.index_type == "hier" else gh.FusedIndex(GuidelineIndex.load(a.index, embedder, None), hier, drop_sections=drop)


def dry_run(adms, index, tabular, max_cases, price_in, price_out, control=False, supporting=False):
    system = gc.SYSTEM_PROMPT_CONTROL if control else gc.SYSTEM_PROMPT
    calls = in_tokens = 0
    print(f"arm: {'CONTROL (no Guidelines text)' if control else 'rules'}")
    print(f"{'patient':<10}{'cases':>6}{'avg prompt tokens':>20}")
    for pid, hid, adm in adms:
        all_cases = cases_from_stage8(adm, supporting)
        codes = [c["code"] for c in all_cases]
        toks = [gc.est_tokens(system + gc.prepare(c, index, tabular, codes, 12000, control=control)[3])
                for c in all_cases[:max_cases]]
        calls += len(toks)
        in_tokens += sum(toks)
        print(f"{pid:<10}{len(toks):>6}{(sum(toks) // max(len(toks), 1)):>20}")
    cost = RETRY_FACTOR * (in_tokens * price_in + calls * OUT_TOKENS_PER_CALL * price_out) / 1e6
    print(f"\n{calls} calls | ~{in_tokens:,} input tokens | ~{calls * OUT_TOKENS_PER_CALL:,} output tokens (assumed {OUT_TOKENS_PER_CALL}/call)")
    print(f"estimated cost incl. {RETRY_FACTOR}x retries at ${price_in}/M in, ${price_out}/M out: ~${cost:.2f}")
    print("(prices are inputs, not looked up; Opus also spends hidden reasoning tokens, so check the OpenRouter Activity page)")
    return calls, cost


def load_done(out_dir):
    f = Path(out_dir) / "checks.jsonl"
    if not f.exists():
        return {}
    return {r["key"]: r for r in (json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip())}


def append(path, obj):
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run(adms, index, tabular, title_search, config, call_llm_json, out_dir, max_cases, not_available_error, control=False,
        supporting=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    n_new = 0
    for pid, hid, adm in adms:
        all_cases = cases_from_stage8(adm, supporting)
        codes = [c["code"] for c in all_cases]
        for case in all_cases[:max_cases]:
            key = f"{pid}|{hid}|{case['rank']}|{case['code']}"
            if key in done:
                continue
            rec = None
            for attempt in range(1, NETWORK_RETRIES + 1):
                try:
                    rec = gc.check_case(case, index, tabular, title_search, codes, config, call_llm_json, 12000, control=control)
                    break
                except (not_available_error, requests.exceptions.RequestException) as exc:
                    if attempt == NETWORK_RETRIES:
                        print(f"  Network still down after {NETWORK_RETRIES} tries at {key}; progress is saved, re-run the same command.")
                        raise
                    wait = min(30 * attempt, 180)
                    print(f"  Network error at {key} ({type(exc).__name__}); waiting {wait}s (retry {attempt}/{NETWORK_RETRIES - 1})...")
                    time.sleep(wait)
                except Exception as exc:
                    print(f"  ERROR {key}: {type(exc).__name__}: {str(exc)[:120]}")
                    append(out / "errors.jsonl", {"key": key, "error": f"{type(exc).__name__}: {exc}"})
                    break
            if rec is None:
                continue
            rec.update(key=key, patient_id=pid, hadm_id=hid)
            append(out / "checks.jsonl", rec)
            n_new += 1
            llm = rec["llm"]
            print(f"  {pid} rank {case['rank']} {case['code']:<8} -> {llm.get('verdict')} cited={llm.get('cited_rules')} valid={rec['valid']} ({rec['seconds']}s)")
    print(f"\nnew results this run: {n_new} | total saved: {len(load_done(out))} -> {out / 'checks.jsonl'}")


def build_rows(adms, out_dir):
    truth = {pid + "|" + hid: load_truth(adm) for pid, hid, adm in adms}
    rows = []
    for r in load_done(out_dir).values():
        if f"{r['patient_id']}|{r['hadm_id']}" not in truth:
            continue
        billed, primary, target_rank = truth[f"{r['patient_id']}|{r['hadm_id']}"]
        llm = r["llm"]
        status = primary_status(r["case"]["code"], billed, primary)
        suggestions = [nodot(c["code"]) for c in r.get("codes_for_suggested_condition", [])[:3]]
        found = ([nodot(llm["proposed_code"])] if llm.get("proposed_code") else []) + suggestions
        is_target = target_rank == r["case"]["rank"]
        rows.append({"key": r["key"], "patient": r["patient_id"], "hadm": r["hadm_id"], "rank": r["case"]["rank"],
                     "diagnosis": r["case"]["diagnosis"], "code": r["case"]["code"], "primary_status": status,
                     "not_primary": status in ("billed_secondary", "family_billed"),
                     "target_case": is_target, "true_target": is_target and status == "not_billed",
                     "verdict": str(llm.get("verdict", "")).lower(), "cited": "|".join(llm.get("cited_rules") or []),
                     "provider_query": llm.get("needs_provider_query"),
                     "suggested_condition": llm.get("suggested_condition", ""), "candidate_codes": "|".join(found),
                     "hit_primary_exact": primary in found, "hit_primary_family": any(c[:3] == primary[:3] for c in found),
                     "valid": r["valid"], "attempts": r["attempts"], "seconds": r["seconds"], "gt_primary": primary,
                     "arm": r.get("arm", "rules")})
    return rows


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


STATUS_LABEL = {"billed_primary": "billed as the PRIMARY code", "billed_secondary": "billed as a SECONDARY code (not primary)",
                "family_billed": "same 3-char family billed (different code)", "not_billed": "not billed"}


def summarize(adms, out_dir):
    rows = build_rows(adms, out_dir)
    if not rows:
        print("no saved results yet")
        return
    with (Path(out_dir) / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    flagged = lambda x: x["verdict"] in FLAGGED
    print(f"\n=== {len(rows)} cases | arm: {rows[0]['arm']} | valid answers: {pct(sum(r['valid'] for r in rows), len(rows))}")
    print("verdicts:", {v: sum(r['verdict'] == v for r in rows) for v in ('keep', 'flag', 'replace')})
    print("\nflag rate by how the candidate code appears in the billed codes:")
    for st, label in STATUS_LABEL.items():
        grp = [r for r in rows if r["primary_status"] == st]
        print(f"  {label:<46} flagged {pct(sum(map(flagged, grp)), len(grp))}")
    print("  (a flag on a code billed as primary or secondary is a possible false alarm; 'not primary' is NOT an error)")
    targets = [r for r in rows if r["true_target"]]
    print(f"\n--- target cases: diagnosis matched, linked ICD wrong, candidate not billed ({len(targets)} cases) ---")
    for r in targets:
        print(f"{r['patient']} rank {r['rank']} {r['code']:<8} GT primary {r['gt_primary']:<8} verdict={r['verdict']:<8} cited={r['cited']}")
        print(f"      suggested: {r['suggested_condition'][:70]!r} -> {r['candidate_codes'] or 'none'} | GT hit exact={r['hit_primary_exact']} family={r['hit_primary_family']}")
    if targets:
        print(f"target cases flagged: {pct(sum(map(flagged, targets)), len(targets))} | GT primary among candidates: "
              f"exact {pct(sum(r['hit_primary_exact'] for r in targets), len(targets))}, family {pct(sum(r['hit_primary_family'] for r in targets), len(targets))}")
    others = [r for r in rows if r["target_case"] and not r["true_target"]]
    if others:
        print("\n--- NOT a coding error: candidate is a legitimate billed code but not the billed PRIMARY (principal-diagnosis choice) ---")
        for r in others:
            print(f"{r['patient']} rank {r['rank']} {r['code']:<8} status={r['primary_status']} | billed primary is {r['gt_primary']} | verdict={r['verdict']}")
    print(f"\nper-case table -> {Path(out_dir) / 'summary.csv'}")


def compare(adms, dir_rules, dir_control):
    a = {r["key"]: r for r in build_rows(adms, dir_rules)}
    b = {r["key"]: r for r in build_rows(adms, dir_control)}
    keys = [k for k in a if k in b]
    if not keys:
        print("no common cases between the two result folders yet")
        return
    flagged = lambda x: x["verdict"] in FLAGGED
    print(f"\n=== RULES arm vs CONTROL arm on {len(keys)} common cases (same LLM, same Tabular facts; control gets no Guidelines text)")
    both = sum(flagged(a[k]) and flagged(b[k]) for k in keys)
    only_rules = [k for k in keys if flagged(a[k]) and not flagged(b[k])]
    only_ctrl = [k for k in keys if flagged(b[k]) and not flagged(a[k])]
    print(f"flagged in both: {both} | only with Guidelines text: {len(only_rules)} | only without: {len(only_ctrl)} | "
          f"neither: {len(keys) - both - len(only_rules) - len(only_ctrl)}")
    print(f"\n{'billing status of candidate code':<46}{'rules flagged':>16}{'control flagged':>18}")
    for st, label in STATUS_LABEL.items():
        ks = [k for k in keys if a[k]["primary_status"] == st]
        print(f"{label:<46}{pct(sum(flagged(a[k]) for k in ks), len(ks)):>16}{pct(sum(flagged(b[k]) for k in ks), len(ks)):>18}")
    tk = [k for k in keys if a[k]["true_target"]]
    print(f"\ntrue target cases ({len(tk)}): verdict and whether the billed primary appears among suggested codes")
    for k in tk:
        print(f"  {a[k]['patient']} {a[k]['code']:<8} GT {a[k]['gt_primary']:<7} rules: {a[k]['verdict']:<7} hit={a[k]['hit_primary_exact']!s:<5} | control: {b[k]['verdict']:<7} hit={b[k]['hit_primary_exact']}")
    diff = [k for k in keys if flagged(a[k]) != flagged(b[k])]
    if diff:
        print("\ncases where the arms disagree:")
        for k in diff:
            print(f"  {a[k]['patient']} {a[k]['code']:<8} ({a[k]['primary_status']}): rules={a[k]['verdict']} cited={a[k]['cited']} | control={b[k]['verdict']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--index", required=True, help="flat Guidelines index (also used for the title search)")
    ap.add_argument("--index-type", choices=["flat", "hier", "fused"], default="flat",
                    help="how Guidelines rules are retrieved: flat (default), hier (outline search), fused (both merged)")
    ap.add_argument("--hier-index", help="outline index folder; required for --index-type hier or fused")
    ap.add_argument("--include-supporting", action="store_true",
                    help="also check rows Naren mapped only to supporting codes (e.g. septic shock -> R65.21)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--tabular", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["rules", "control"], default="rules",
                    help="control = same cases and facts but no Guidelines text (use a separate --out folder)")
    ap.add_argument("--compare-with", help="control-arm results folder to compare against --out (no LLM calls)")
    ap.add_argument("--max-cases", type=int, default=6, help="differential rows checked per admission")
    ap.add_argument("--only", help="comma-separated patient ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm a real (paid) run")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--price-in", type=float, default=5.0, help="$ per million input tokens (check OpenRouter)")
    ap.add_argument("--price-out", type=float, default=25.0, help="$ per million output tokens (check OpenRouter)")
    a = ap.parse_args()

    adms = admissions(a.records, set(a.only.split(",")) if a.only else None)
    print(f"{len(adms)} admissions found")
    if a.compare_with:
        compare(adms, a.out, a.compare_with)
        return
    if a.summarize_only:
        summarize(adms, a.out)
        return
    control = a.arm == "control"
    tabular = TabularList.load(a.tabular)
    embedder = Embedder(Path(a.models) / "minilm")
    index = load_index(a, embedder)
    if a.dry_run:
        dry_run(adms, index, tabular, a.max_cases, a.price_in, a.price_out, control, a.include_supporting)
        return

    gc.ensure_ca_bundle(Path(a.index).parent / "ca_bundle.pem")
    from dataclasses import replace
    from pipeline import ICD_CONFIRM_TEMPERATURE, LLMNotAvailableError, call_llm_json, get_llm_config
    config = replace(get_llm_config(), temperature=float(ICD_CONFIRM_TEMPERATURE))
    print(f"LLM (Naren's settings): {config.api_label}/{config.model} temperature={config.temperature} zdr={config.openrouter_zdr}")
    if not config.resolved_api_key():
        sys.exit(f"No API key. Set the {config.api_key_env} environment variable.")
    dry_run(adms, index, tabular, a.max_cases, a.price_in, a.price_out, control, a.include_supporting)
    if not a.yes:
        sys.exit("\nNot started. Re-run with --yes to spend the credits above (results already saved are skipped).")
    run(adms, index, tabular, gc.TitleSearch(tabular, embedder, Path(a.index) / "chroma"), config, call_llm_json,
        a.out, a.max_cases, LLMNotAvailableError, control, a.include_supporting)
    summarize(adms, a.out)


if __name__ == "__main__":
    main()
