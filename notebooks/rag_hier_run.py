"""SCRIPT 1 of 2: the Guidelines RAG with HIERARCHICAL retrieval, run on the 15 real project patients only.

Three methods are compared in the end (script 2, rag_compare_report.py, builds the report):

  1. NAREN'S BASELINE   his pipeline's ICD code for each diagnosis. Nothing is run here: it is read from his saved files.
  2. CONTROL ARM        the same candidate codes, the same LLM (Naren's settings), the same Tabular List facts, NO Guidelines text.
  3. RULES ARM          as the control arm, plus Guidelines rules retrieved by the FUSED search (default): the hierarchical search of
                        guideline_hier.py (down the Guidelines outline Section -> letter -> chapter -> topic -> rule) merged with the
                        flat paragraph search, so a rule the outline walk prunes can still arrive through the flat side.
                        --index-type hier uses the hierarchical search alone.

No made-up errors are injected: every case is a real candidate code from the 15 patients (all differential rows up to --max-cases per
admission, including rows Naren mapped only to supporting codes such as septic shock -> R65.21).

Steps (each saves as it goes; re-running the same command resumes and never pays twice for a finished case):
  a. rules arm  -> <out>/rules_raw/        control arm -> <out>/control_raw/   (or --reuse-control: see below)
  b. code-selection step: where an arm suggests another condition, a small LLM call picks the matching ICD-10-CM code from a
     title-search candidate pool (it cannot invent codes) -> <out>/rules/ and <out>/control/   (skip with --no-select)
  c. <out>/manifest.json and <out>/naren_baseline.csv

    python rag_hier_run.py --dry-run   <common arguments>      # free: case counts, prompt sizes, cost estimate
    python rag_hier_run.py --yes       <common arguments>      # the real run with Naren's LLM (needs OPENROUTER_API_KEY)

Common arguments: --records ../patient_records --index ../../icd10cm_sources/guideline_index_v2
    --hier-index ../../icd10cm_sources/guideline_index_hier --models ../../icd10cm_sources/models
    --tabular ../../icd10cm_sources/tabular_FY2026.json --out ../../icd10cm_sources/rag_fused_run

--reuse-control <folder>: the control arm never sees Guidelines text, so its result does not depend on the retriever. Pointing this at
an earlier control run (e.g. check_v2_control, same prompts and cases) copies it instead of paying for 80 calls again; any case it
lacks is still run.
--copy-control-from <earlier run folder>: the same, but copies the finished control_raw/ and control/ of a previous run of this script,
so even the code-selection step is not paid for again.
--arms rules | control | both: run one arm only. `--arms control` into a NEW folder repeats the control arm, so that
rag_compare_report.py --control-repeat can show how much a plain re-run of the same arm differs by itself (run-to-run noise).
Whole-Section introductions (I, II, III, IV) are dropped from the retrieved rules by default (--keep-sections restores them).

Nothing in Naren's repo or in earlier result folders is modified.
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import guideline_check as gc
import guideline_check_run as gcr
import guideline_remap as grm
from guideline_rag import Embedder
from guideline_vs_naren import nodot, score
from tabular_list import TabularList


def naren_baseline(adms, out_dir):
    """Naren's rank-1 ICD code per admission vs the billed primary code. Read from his saved Stage 10 files; no LLM."""
    rows = []
    for pid, hid, adm in adms:
        acc, gt = adm / "accuracy" / "accuracy.json", adm / "ground_truth.json"
        if not (acc.exists() and gt.exists()):
            continue
        ev = json.loads(acc.read_text(encoding="utf-8"))["evaluation"]
        primary = nodot(json.loads(gt.read_text(encoding="utf-8"))["primary_icd_code"])
        code = nodot(ev["icd_linked"].get("rank1_predicted_icd"))
        rows.append({"patient": pid, "hadm": hid, "billed_primary": primary, "naren_rank1_code": code,
                     **{f"naren_{k}": v for k, v in score(code, primary).items()}})
    if rows:
        with (Path(out_dir) / "naren_baseline.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def finalize(raw_dir, final_dir, select, title_search, tabular, call_llm_json, config):
    """Copy the raw results to the final folder; with select, also let the LLM pick codes for each suggested condition."""
    final = Path(final_dir)
    final.mkdir(parents=True, exist_ok=True)
    done = gcr.load_done(final)
    n = picked = 0
    for rec in gcr.load_done(raw_dir).values():
        if rec["key"] in done:
            continue
        suggestion = str((rec.get("llm") or {}).get("suggested_condition") or "").strip()
        if select and suggestion:
            rec["codes_for_suggested_condition"], rec["code_selection"] = grm.choose_codes(
                rec, suggestion, title_search, tabular, call_llm_json, config)
            picked += bool(rec["code_selection"]["selected"])
        gcr.append(final / "checks.jsonl", rec)
        n += 1
    print(f"  {final.name}: {n} new records written ({picked} suggestions resolved by the code-selection step) -> {final / 'checks.jsonl'}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--index", required=True, help="flat Guidelines index (also used for the title search)")
    ap.add_argument("--hier-index", required=True, help="outline (hierarchical) index folder")
    ap.add_argument("--index-type", choices=["hier", "fused"], default="fused",
                    help="fused = hierarchical (outline) search and flat paragraph search merged (default); hier = hierarchical only")
    ap.add_argument("--models", required=True)
    ap.add_argument("--tabular", required=True)
    ap.add_argument("--out", required=True, help="run folder; rules_raw/, control_raw/, rules/, control/ are created inside")
    ap.add_argument("--max-cases", type=int, default=6, help="differential rows checked per admission")
    ap.add_argument("--only", help="comma-separated patient ids")
    ap.add_argument("--skip-supporting", action="store_true", help="do not check rows Naren mapped only to supporting codes")
    ap.add_argument("--reuse-control", help="existing control-arm folder (checks.jsonl) to copy instead of re-running the control arm; "
                                            "the code-selection step is still applied to it")
    ap.add_argument("--copy-control-from", help="an earlier run folder of this script: copies its finished control_raw/ and control/ "
                                                "(selection included), so the control arm costs nothing")
    ap.add_argument("--arms", choices=["both", "rules", "control"], default="both",
                    help="which arm to run: rules, control, or both (default). A control-only run in a new folder measures run-to-run noise")
    ap.add_argument("--keep-sections", action="store_true",
                    help="keep the whole-Section introductions (I, II, III, IV) among the retrieved rules; by default they are dropped")
    ap.add_argument("--no-select", action="store_true", help="skip the code-selection step (title search order only)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="confirm the real (paid) run")
    ap.add_argument("--price-in", type=float, default=5.0, help="$ per million input tokens (check OpenRouter)")
    ap.add_argument("--price-out", type=float, default=25.0, help="$ per million output tokens (check OpenRouter)")
    a = ap.parse_args()

    adms = gcr.admissions(a.records, set(a.only.split(",")) if a.only else None)
    print(f"{len(adms)} admissions found (the real project patients; no injected errors)")
    out = Path(a.out)
    supporting = not a.skip_supporting
    tabular = TabularList.load(a.tabular)
    embedder = Embedder(Path(a.models) / "minilm")
    a.drop_sections = not a.keep_sections
    index = gcr.load_index(a, embedder)
    print(f"Guidelines retrieval for the rules arm: {a.index_type.upper()} ({'hierarchical only' if a.index_type == 'hier' else 'hierarchical + flat merged'})"
          f"; whole-Section introductions {'kept' if a.keep_sections else 'DROPPED from the results'}")
    do_rules, do_control = a.arms in ("both", "rules"), a.arms in ("both", "control")
    control_free = bool(a.copy_control_from)

    total = 0.0
    for control in (False, True):
        if (control and not do_control) or (not control and not do_rules):
            continue
        if control and (a.reuse_control or control_free):
            print(f"\ncontrol arm: reusing {a.copy_control_from or a.reuse_control} (only missing cases would be run)")
            continue
        _, cost = gcr.dry_run(adms, index, tabular, a.max_cases, a.price_in, a.price_out, control, supporting)
        total += cost
    n_sel = (do_rules + (do_control and not control_free)) * 0.3
    print(f"\nTOTAL estimated cost: ~${total:.2f}" + ("" if a.no_select else f" + about ${n_sel:.1f} for the code-selection step") +
          " (estimates; check the OpenRouter Activity page)")
    if a.dry_run or not a.yes:
        sys.exit("\nNot started. Re-run with --yes to spend the credits above (finished cases are saved and skipped).")

    gc.ensure_ca_bundle(Path(a.index).parent / "ca_bundle.pem")
    from dataclasses import replace
    from pipeline import ICD_CONFIRM_TEMPERATURE, LLMNotAvailableError, call_llm_json, get_llm_config
    config = replace(get_llm_config(), temperature=float(ICD_CONFIRM_TEMPERATURE))
    if not config.resolved_api_key():
        sys.exit(f"No API key. Set the {config.api_key_env} environment variable.")
    print(f"LLM (Naren's settings): {config.api_label}/{config.model} temperature={config.temperature} zdr={config.openrouter_zdr}")
    out.mkdir(parents=True, exist_ok=True)
    base = naren_baseline(adms, out)
    print(f"Naren's baseline (read from his files, no LLM): top-ranked code equals the billed primary exactly in "
          f"{sum(r['naren_exact'] for r in base)}/{len(base)} admissions, same 3-char family in {sum(r['naren_family'] for r in base)}/{len(base)}")
    title_search = gc.TitleSearch(tabular, embedder, Path(a.index) / "chroma")

    if control_free:
        for sub in ("control_raw", "control"):
            src_dir = Path(a.copy_control_from) / sub
            if not (src_dir / "checks.jsonl").exists():
                sys.exit(f"--copy-control-from: {src_dir / 'checks.jsonl'} not found")
            if not (out / sub / "checks.jsonl").exists():
                (out / sub).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_dir / "checks.jsonl", out / sub / "checks.jsonl")
        print(f"Control arm copied (finished, selection included) from {a.copy_control_from}: {len(gcr.load_done(out / 'control'))} records")

    if do_rules:
        print(f"\n=== RULES arm ({'fused hierarchical + flat' if a.index_type == 'fused' else 'hierarchical'} Guidelines retrieval) ===")
        gcr.run(adms, index, tabular, title_search, config, call_llm_json, out / "rules_raw", a.max_cases, LLMNotAvailableError, False, supporting)
    ctrl_raw = out / "control_raw"
    if do_control:
        print("\n=== CONTROL arm (no Guidelines text) ===")
        if a.reuse_control:
            src = Path(a.reuse_control) / "checks.jsonl"
            if not src.exists():
                sys.exit(f"--reuse-control: {src} not found")
            if not (ctrl_raw / "checks.jsonl").exists():
                ctrl_raw.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, ctrl_raw / "checks.jsonl")
                print(f"  copied {len(gcr.load_done(ctrl_raw))} saved control results from {src}")
        gcr.run(adms, index, tabular, title_search, config, call_llm_json, ctrl_raw, a.max_cases, LLMNotAvailableError, True, supporting)

    print("\n=== code-selection step ===" if not a.no_select else "\n=== copying results (code-selection step skipped) ===")
    if do_rules:
        finalize(out / "rules_raw", out / "rules", not a.no_select, title_search, tabular, call_llm_json, config)
    if do_control:
        finalize(out / "control_raw", out / "control", not a.no_select, title_search, tabular, call_llm_json, config)

    arms_done = [arm for arm, on in (("rules", do_rules), ("control", do_control or control_free)) if on]
    for arm in arms_done:
        print(f"\n######## {arm.upper()} arm summary ########")
        gcr.summarize(adms, out / arm)
    (out / "manifest.json").write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"), "index_type": a.index_type, "flat_index": a.index,
        "hier_index": a.hier_index, "max_cases": a.max_cases, "include_supporting": supporting, "code_selection": not a.no_select,
        "drop_sections": a.drop_sections, "arms_run": a.arms, "reuse_control": a.reuse_control, "copy_control_from": a.copy_control_from,
        "llm": f"{config.api_label}/{config.model}", "temperature": config.temperature, "n_admissions": len(adms),
        "records": {arm: len(gcr.load_done(out / arm)) for arm in arms_done}}, indent=1), encoding="utf-8")
    nxt = (f"python rag_compare_report.py --records {a.records} --run {a.out} --tabular {a.tabular}" if len(arms_done) == 2
           else "(a single-arm run: use it as --control-repeat in rag_compare_report.py)")
    print(f"\nDone. Run folder: {out}\nNext: {nxt}")


if __name__ == "__main__":
    main()
