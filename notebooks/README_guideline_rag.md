# Guidelines-grounded code checker (RAG + Tabular List)

An add-on to the clinical coding pipeline. It takes the ICD-10-CM codes that the Stage 8/9 pipeline proposes for each
diagnosis and **verifies each one** against two official sources:

| Source | Used for | How it is used |
|---|---|---|
| **ICD-10-CM Tabular List** (FY2026 XML) | Exact facts about a code: title, Excludes1/2, code-first / use-additional notes, inherited notes, 7th characters | Exact lookup, not retrieval. A code's notes are inherited from its parents. |
| **ICD-10-CM Official Guidelines** (FY2026 PDF) | The coding rules (complications, unspecified codes, sequencing, ...) | Chunked along the PDF's own heading outline, then retrieved per case (RAG). |

The LLM is **Naren's configured LLM** (`pipeline.get_llm_config` / `call_llm_json`: Claude Opus 5 via OpenRouter, ZDR on).
Nothing in `pipeline.py`, the stage notebooks, or any of Naren's saved outputs is modified. The scripts only read them.

For every candidate code the checker returns a structured verdict: `keep`, `flag` or `replace`, with the cited rule ids.
If it suggests a different condition it names it in plain English. The code is then found by title search and an LLM pick
from a candidate pool. The LLM can never invent a code: it may only choose from the allowed list it is given.

> The checker never reads the clinical notes. Its input is Naren's diagnosis row (diagnosis, candidate code and title,
> up to 4 supporting-evidence phrases, up to 5 procedures) plus the Tabular facts and, in the rules arm, retrieved rules.

## What is compared

| Method | What it is |
|---|---|
| **Naren's baseline** | His pipeline's rank-1 ICD code per diagnosis. Read from his saved files, no LLM call. |
| **Control arm** | Same cases, same LLM, same Tabular facts, **no Guidelines text**. Isolates the effect of the Guidelines. |
| **Rules arm** | Control arm plus retrieved Guidelines rules (fused flat + hierarchical retrieval). |

Real data only: the 15 admissions in `patient_records/`. No injected errors in the main evaluation.

## Files

**Build the knowledge base**

| Script | Purpose |
|---|---|
| `tabular_list.py` | Parses the CDC Tabular List XML/zip into keyed records with inherited notes; Excludes / code-first checks. |
| `guidelines_chunker.py` | Splits the Guidelines PDF along its heading hierarchy (parents = rule sections, children = embedded pieces). |
| `guideline_rag.py` | Flat hybrid retrieval (MiniLM vectors + BM25, reciprocal rank fusion, optional cross-encoder rerank). Also builds cases from Stage 8 output. |
| `guideline_hier.py` | Hierarchical retrieval: beam search down the Guidelines outline (Section > letter > chapter > topic > rule). Same interface as the flat index. |
| `guideline_rag_gold.json` | Small gold set of query -> governing-rule pairs used by the `eval` commands. |

**The checker**

| Script | Purpose |
|---|---|
| `guideline_check.py` | Prompt, verdict schema, validator, retry, title-search mapping. Can check one admission (`--admission`). |
| `guideline_check_run.py` | Runs the checker over all admissions (rules or control arm), resumable, scored against Naren's data. |
| `guideline_remap.py` | Re-runs only the code-lookup step on saved results, optionally with the LLM pick (`--select`). |

**Real-patient evaluation (the main experiment)**

| Script | Purpose |
|---|---|
| `rag_hier_run.py` | **Script 1.** Runs rules arm + control arm on the 15 patients and writes Naren's baseline next to them. |
| `rag_compare_report.py` | **Script 2.** Reads a run folder and writes the three-way comparison report. No LLM calls, no cost. |
| `guideline_vs_naren.py` | Per-admission table of Naren's codes vs. the checker's. No LLM calls. |

**Extra tests**

| Script | Purpose |
|---|---|
| `guideline_inject.py` | Synthetic error-injection benchmark (28 scenarios, 55 items: an error case plus, mostly, a clean twin) for rules vs. control. |
| `llm_ping.py` | Tiny OpenRouter diagnostic (a fraction of a cent) to trace 400/404 errors to one setting. |

## Setup

Python packages: `pymupdf`, `numpy`, `torch`, `transformers`, `chromadb`, `rank_bm25`, `requests`, `certifi`.

Inputs (kept **outside the repo**; the paths below assume a sibling folder `../../icd10cm_sources/`):

- `icd10cm-table-and-index-2026.zip` from the CDC/NCHS ICD-10-CM release page (contains the Tabular XML).
- The FY2026 *ICD-10-CM Official Guidelines for Coding and Reporting* PDF from CMS/CDC.
- `models/minilm/` (a MiniLM-L6 sentence embedder) and `models/reranker/` (`cross-encoder/ms-marco-MiniLM-L-12-v2`), downloaded locally.
- An OpenRouter key, supplied only through the environment:

```powershell
$env:OPENROUTER_API_KEY = "<your key>"   # never written to a file; notebooks/settings.json keeps it null
```

Run all commands from `notebooks/`. The index and result folders are generated data (they contain MIMIC-derived text) and
must **not** be committed.

## Run order

All paths below are relative to `notebooks/`. `S=../../icd10cm_sources`.

**1. Build the knowledge base (once, no LLM, free)**

```bash
python tabular_list.py --source "$S/icd10cm-table-and-index-2026.zip" --out $S/tabular_FY2026.json
python guidelines_chunker.py --pdf $S/ICD-10-CM-October-2025-Guidelines.pdf --out $S/guidelines_chunks_FY2026_v2.json
python guideline_rag.py build --chunks $S/guidelines_chunks_FY2026_v2.json --models $S/models --index $S/guideline_index_v2
python guideline_hier.py build --chunks $S/guidelines_chunks_FY2026_v2.json --index $S/guideline_index_hier --models $S/models
```

Optional retrieval check against the gold set: `python guideline_rag.py eval --index $S/guideline_index_v2 --models $S/models --gold guideline_rag_gold.json`

**2. Dry run (free): case counts, prompt sizes, cost estimate**

```bash
python rag_hier_run.py --dry-run --records ../patient_records --index $S/guideline_index_v2 --hier-index $S/guideline_index_hier \
  --models $S/models --tabular $S/tabular_FY2026.json --out $S/rag_final_run
```

**3. Real run (paid, resumable; re-running the same command continues where it stopped)**

```bash
python rag_hier_run.py --yes --records ../patient_records --index $S/guideline_index_v2 --hier-index $S/guideline_index_hier \
  --models $S/models --tabular $S/tabular_FY2026.json --out $S/rag_final_run
```

Defaults: fused retrieval, whole-Section introductions dropped, up to 6 differential rows per admission, rows Naren mapped
only to supporting codes included, LLM code-selection step on. Other flags: `--arms rules|control|both`, `--only <patient ids>`,
`--keep-sections`, `--no-select`, `--reuse-control`, `--copy-control-from`.

**4. (Optional) measure run-to-run noise** by repeating the control arm into a new folder:

```bash
python rag_hier_run.py --yes --arms control --records ../patient_records --index $S/guideline_index_v2 --hier-index $S/guideline_index_hier \
  --models $S/models --tabular $S/tabular_FY2026.json --out $S/rag_final_noise
```

**5. Report (free)**

```bash
python rag_compare_report.py --records ../patient_records --run $S/rag_final_run --tabular $S/tabular_FY2026.json \
  --control-repeat $S/rag_final_noise/control
```

Writes `<run>/report/report.md`, `per_admission.csv`, `per_case.csv` and `numbers.json`.

**6. (Optional) per-admission Naren-vs-checker table**

```bash
python guideline_vs_naren.py --records ../patient_records --out $S/naren_vs_checker \
  --arm control=$S/rag_final_run/control --arm rules=$S/rag_final_run/rules
```

**Injection benchmark (optional)**

```bash
python guideline_inject.py --list --tabular $S/tabular_FY2026.json            # scenarios + code check, free
python guideline_inject.py --yes --arm rules   --index $S/guideline_index_v2 --index-type fused --hier-index $S/guideline_index_hier \
  --models $S/models --tabular $S/tabular_FY2026.json --out $S/inject_rules_fused
python guideline_inject.py --yes --arm control --index $S/guideline_index_v2 \
  --models $S/models --tabular $S/tabular_FY2026.json --out $S/inject_control
python guideline_inject.py --compare-with $S/inject_control --out $S/inject_rules_fused --tabular $S/tabular_FY2026.json
```

## Results (15 admissions, Claude Opus 5, temperature 0.3, single runs)

| | Naren | Control | Rules |
|---|---|---|---|
| Top code equals the billed primary code (of 15) | 1 | 3 | 2 |
| Same 3-character family as billed primary (of 15) | 5 | 7 | 7 |
| Candidate codes flagged (of 80) | - | 38 | 37 |

- **No accuracy gain from the Guidelines text.** The rules arm is not better than the control arm. Opus already knows these
  textbook rules. Hierarchical/fused retrieval raised how often the governing rule is shown to the model (flat 15/25,
  hierarchical 20/25, fused 21/25 on held-out injection scenarios) but did not change verdicts.
- **What the Guidelines add is auditability.** Every rules-arm flag cites a specific rule.
- **The rules arm is less stable than the control arm.** Its flag count moved between 27 and 37 across runs with small input
  changes. The control arm stayed at 36-38 and differs from its own repeat on 2 of 80 decisions. Noise for the rules arm has not been measured.
- **"Not in the billing record" is not "wrong".** A flag on a code that was not billed is a prompt for coder review, not an error.
  About 19 of 247 billed codes are parent categories in FY2026 (version drift, e.g. billed I47.1 vs. FY2026 I47.10).
- **Fixes made after seeing results are post-hoc** and are not validated on independent data.

## Limitations

- 15 admissions, one run per arm; no confidence intervals are meaningful at this size.
- Only the first principal code per differential row is checked, and at most 6 rows per admission. About 100 other package codes are never reviewed.
- The checker sees Naren's extracted evidence phrases, not the note text, so it cannot find a code that needs labs or free text it was not given.
- Billing records are the only reference standard and are incomplete.

## Final output files

- `docs/RAG_summary.md`: what the RAG is, its steps, and why each component was chosen.
- `results/final_output_with_billed.md` / `.csv`: for each of the 80 checked codes, Naren's code, the RAG's verdict and final code, and the billed codes for the admission (evaluation labels, never shown to the model). Built by `notebooks/rag_final_table.py`.
