"""Re-run ONLY the code-lookup step (suggested condition -> ICD-10-CM codes) on results that are already saved. The verdicts,
cited rules and suggested conditions are copied unchanged; "codes_for_suggested_condition" is recomputed and the records are
written to a NEW folder (the source folder is never touched).

Two modes:

  default   re-run the title search with the current guideline_check.map_condition. No LLM, no cost.

  --select  title search builds a wider pool of candidate codes (including terms the LLM put in parentheses), then ONE small
            LLM call per suggestion picks the matching code(s) FROM THAT POOL ONLY (same model and settings as the checker).
            Why: word search cannot tell that "underlying systemic infection (sepsis)" means A41.9 and not "systemic
            bartonellosis"; reading the words in context can. Costs a few cents per folder; --dry-run shows the count first.

    python guideline_remap.py --src ../../icd10cm_sources/check_v2_fused --dst ../../icd10cm_sources/check_v2_fused_sel \
        --index ../../icd10cm_sources/guideline_index_v2 --models ../../icd10cm_sources/models \
        --tabular ../../icd10cm_sources/tabular_FY2026.json --select --yes

Works on guideline_check_run.py folders and on guideline_inject.py folders (one record per line in checks.jsonl).
With --select the picked codes go first in "codes_for_suggested_condition" (the rest of the pool follows) and the choice is
saved in "code_selection" for audit. A pick that is not in the pool is discarded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import guideline_check as gc
from guideline_rag import Embedder
from tabular_list import TabularList

POOL = 10
SELECT_SYSTEM = (
    "You are an expert ICD-10-CM coder. You are given a coding suggestion (a condition that needs its own code) and a list of "
    "candidate ICD-10-CM codes found by title search. Choose the candidate code(s) that best match the suggested condition, "
    "using the case context. Use ONLY codes from the list. A suggestion may name several conditions: give the best code for "
    "each, most important first, at most 3. If no candidate fits, return an empty list. Respond with JSON only.")
SELECT_SCHEMA = '{"codes": ["code from the list", "..."], "why": "one short sentence"}'
TOKENS_PER_CALL_IN, TOKENS_PER_CALL_OUT = 700, 80


def pool_for(title_search, tabular, rec, suggestion):
    """Candidate codes for a suggestion: the normal mapping (wider), plus a search for each parenthetical term."""
    anchor = gc.body_system(tabular, rec["case"]["code"])
    pool = list(gc.map_condition(title_search, suggestion, anchor, k=POOL))
    seen = {c["code"] for c in pool}
    for term in gc.parenthetical_terms(suggestion)[:2]:
        for hit in title_search.search(term, k=4):
            if hit["code"] not in seen:
                seen.add(hit["code"])
                pool.append(hit)
    return pool[:POOL + 4]


def select_prompt(rec, suggestion, pool):
    case, llm = rec["case"], rec.get("llm") or {}
    lines = [f"CASE diagnosis: {case.get('diagnosis')} | candidate code under review: {case.get('code')} {case.get('code_title', '')}",
             "EVIDENCE: " + "; ".join((case.get("evidence") or [])[:3]),
             f"SUGGESTION (condition needing its own code): {suggestion}",
             f"WHY THE REVIEWER SUGGESTED IT: {str(llm.get('reason', ''))[:400]}",
             "CANDIDATES:"] + [f"- {c['code']}: {c['title']}" for c in pool]
    return "\n".join(lines) + f"\n\nReturn JSON of this shape: {SELECT_SCHEMA}"


def choose_codes(rec, suggestion, title_search, tabular, call_llm_json, config, tries=3):
    """The --select step for one saved check: -> (codes_for_suggested_condition, code_selection). The LLM picks FROM the pool
    only; if it cannot be reached or picks nothing valid, the pool keeps its title-search order."""
    pool = pool_for(title_search, tabular, rec, suggestion)
    by_code = {c["code"]: c for c in pool}
    res = {}
    for attempt in range(1, tries + 1):
        try:
            res = call_llm_json(SELECT_SYSTEM, select_prompt(rec, suggestion, pool), config) or {}
            break
        except Exception as exc:  # network errors and LLMNotAvailableError alike: wait and retry, then keep the pool order
            if attempt == tries:
                print(f"  LLM unreachable at {rec['key']} ({type(exc).__name__}); keeping the title-search order")
            else:
                time.sleep(30 * attempt)
    chosen = [c for c in (res.get("codes") or []) if c in by_code][:3]
    codes = [by_code[c] for c in chosen] + [c for c in pool if c["code"] not in chosen]
    return codes, {"selected": chosen, "why": str(res.get("why", ""))[:300], "pool": [c["code"] for c in pool]}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder with checks.jsonl to re-map (left untouched)")
    ap.add_argument("--dst", required=True, help="new folder for the re-mapped checks.jsonl")
    ap.add_argument("--index", required=True, help="flat index folder (its chroma/ holds the cached title embeddings)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--tabular", required=True)
    ap.add_argument("--select", action="store_true", help="let the LLM pick from the candidate pool (costs a few cents)")
    ap.add_argument("--dry-run", action="store_true", help="with --select: only count the LLM calls and estimate the cost")
    ap.add_argument("--yes", action="store_true", help="with --select: confirm the (small) paid run")
    ap.add_argument("--price-in", type=float, default=5.0, help="$ per million input tokens (check OpenRouter)")
    ap.add_argument("--price-out", type=float, default=25.0, help="$ per million output tokens (check OpenRouter)")
    ap.add_argument("--quiet", action="store_true", help="do not list every changed suggestion")
    a = ap.parse_args()

    src, dst = Path(a.src) / "checks.jsonl", Path(a.dst)
    records = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    todo = [r for r in records if str((r.get("llm") or {}).get("suggested_condition") or "").strip()]
    if a.select:
        cost = (len(todo) * TOKENS_PER_CALL_IN * a.price_in + len(todo) * TOKENS_PER_CALL_OUT * a.price_out) / 1e6
        print(f"--select: {len(todo)} suggestions -> {len(todo)} small LLM calls, estimated cost ~${cost:.2f} "
              f"(assumed {TOKENS_PER_CALL_IN} in / {TOKENS_PER_CALL_OUT} out tokens per call; check the OpenRouter Activity page)")
        if a.dry_run:
            return
        if not a.yes:
            sys.exit("Not started. Re-run with --yes to spend it.")
    if (dst / "checks.jsonl").exists():
        sys.exit(f"{dst / 'checks.jsonl'} already exists; choose a new --dst (nothing is overwritten)")

    tabular = TabularList.load(a.tabular)
    title_search = gc.TitleSearch(tabular, Embedder(Path(a.models) / "minilm"), Path(a.index) / "chroma")
    call_llm_json = config = None
    if a.select:
        gc.ensure_ca_bundle(Path(a.index).parent / "ca_bundle.pem")
        from dataclasses import replace
        from pipeline import ICD_CONFIRM_TEMPERATURE, call_llm_json, get_llm_config
        config = replace(get_llm_config(), temperature=float(ICD_CONFIRM_TEMPERATURE))
        if not config.resolved_api_key():
            sys.exit(f"No API key. Set the {config.api_key_env} environment variable.")
        print(f"LLM (Naren's settings): {config.api_label}/{config.model} temperature={config.temperature}")

    dst.mkdir(parents=True, exist_ok=True)
    n = changed = picked = 0
    with (dst / "checks.jsonl").open("w", encoding="utf-8") as out:
        for rec in records:
            suggestion = str((rec.get("llm") or {}).get("suggested_condition") or "").strip()
            old = [c["code"] for c in rec.get("codes_for_suggested_condition", [])[:3]]
            if not suggestion:
                new = []
            elif not a.select:
                new = gc.map_condition(title_search, suggestion, gc.body_system(tabular, rec["case"]["code"]))
            else:
                new, rec["code_selection"] = choose_codes(rec, suggestion, title_search, tabular, call_llm_json, config)
                picked += bool(rec["code_selection"]["selected"])
            rec["codes_for_suggested_condition"] = new
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if [c["code"] for c in new[:3]] != old:
                changed += 1
                if not a.quiet:
                    print(f"{rec['key']:<42} {suggestion[:60]!r}\n      before {old}\n      after  {[(c['code'], c['title'][:34]) for c in new[:3]]}")
    print(f"\n{n} records written to {dst / 'checks.jsonl'} | suggestions whose top-3 codes changed: {changed}"
          + (f" | LLM picked from the pool: {picked}/{len(todo)}" if a.select else ""))


if __name__ == "__main__":
    main()
