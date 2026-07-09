# Clinical Coding Pipeline

Multi-stage Jupyter workflow for AI-assisted clinical diagnosis coding on MIMIC-IV.

**All pipeline code lives in `notebooks/`** — four stage notebooks plus one shared module (`pipeline.py`) and optional settings (`00_settings.ipynb`).

## Quick start

```bash
pip install pandas requests jupyter
jupyter notebook notebooks/00_settings.ipynb   # edit paths / LLM provider
jupyter notebook notebooks/stage_01_cohort_selection.ipynb
# … then stages 2 → 3 → 4 in order
```

For OpenRouter:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## Should I re-run all stages?

**Yes — re-run all four stages** after this update:

| Artifact | Action |
|----------|--------|
| `data/cohort/cohort.pkl` | **Re-run Stage 1** — adds structured vitals, labs, radiology |
| `data/stage_02/ie_checkpoint.json` | **Delete** (old format / wrong admission count) |
| Stages 2–4 | Re-run — agents now use note + structured MIMIC data |

Stage 1 takes longer now (~2–5 min) because it loads labs/vitals/reports for each admission before collapsing to the latest stay.

## Data sources (per patient)

| Source | MIMIC table | Used for |
|--------|-------------|----------|
| Discharge note (latest) | `mimic-iv-note/discharge.csv.gz` | Primary narrative text |
| Prior admissions | same + ICD-10 | History context (excerpts) |
| Vitals (ICU) | `icu/chartevents.csv.gz` | HR, BP, SpO2, temp, RR |
| Vitals (ward) | `hosp/omr.csv.gz` | BP, pulse, BMI during stay |
| Labs | `hosp/labevents.csv.gz` | Abnormal-prioritized lab panel |
| Radiology reports | `mimic-iv-note/radiology.csv.gz` | Imaging report excerpts |
| Ground truth | `hosp/diagnoses_icd.csv.gz` | ICD-10 labels (evaluation) |

The LLM is instructed to **prefer structured vitals/labs over note text** when they conflict.

## Pipeline stages

### Stage 0 — Settings (`00_settings.ipynb`)

- Edit `TEST_MODE`, MIMIC path, `LLM_PROVIDER`, rate-limit delay
- Writes `notebooks/settings.json` (loaded by `pipeline.py`)

### Stage 1 — Cohort selection (`stage_01_cohort_selection.ipynb`)

**What it does**

1. Samples patients with ≥2 admissions (≥1 in test mode), each with ICD-10 labels and a discharge note ≥500 chars
2. Loads **structured vitals, labs, and radiology** for every admission
3. Collapses to **one row per patient**: latest admission = full note + structured data; prior stays = history

**Outputs**

- `data/cohort/cohort.pkl` — one row per patient
- `data/cohort/cohort_index.json` — human-readable index

**Key columns:** `clinical_note`, `clinical_context_text`, `structured_vitals`, `structured_labs`, `structured_reports`, `admission_history`

### Stage 2 — Information extraction (`stage_02_information_extraction.ipynb`)

**What it does**

- Runs Qwen 2.5 7B IE on the **latest note + structured MIMIC context + prior admission history**
- Checkpoints after each patient (`ie_checkpoint.json`) for resume on rate limits

**Outputs**

- `data/stage_02_information_extraction/information_extractions.json`
- `data/stage_02_information_extraction/ie_checkpoint.json`

### Stage 3 — Symptom tree (`stage_03_symptom_tree.ipynb`)

**What it does**

- Builds a hierarchical symptom tree from note + IE + structured data + history
- One tree per patient (latest admission)

**Outputs**

- `data/stage_03_symptom_tree/symptom_tree_results.json`

### Stage 4 — Export (`stage_04_export_patient_records.ipynb`)

**What it does**

- Writes per-patient folders with notes, structured data, IE, symptom trees, ground truth

**Outputs** (`patient_records/` or `patient_records_test/`)

```
patient_<id>/
  admission_history.json / .txt
  symptom_tree.json / .txt
  admissions/hadm_<latest>/
    clinical_note.txt
    clinical_context.txt          # vitals + labs + radiology text
    structured_vitals.json
    structured_labs.json
    radiology_reports.json
    information_extraction.json
    symptom_tree.json
    ground_truth.txt
```

## Test mode

Set in `00_settings.ipynb`:

```python
"TEST_MODE": True   # 1 patient → data/test/ and patient_records_test/
```

## LLM backends

| Backend | `LLM_PROVIDER` | Model |
|---------|------------------|-------|
| OpenRouter | `"openrouter"` | `qwen/qwen-2.5-7b-instruct` |
| Ollama (local) | `"ollama"` | `qwen2.5:7b` |

### MIMIC data + OpenRouter (Zero Data Retention)

Credentialed MIMIC data must not be retained by third-party LLM services. When using OpenRouter:

1. Set **`OPENROUTER_ZDR: true`** in `00_settings.ipynb` (default) — the pipeline sends `"provider": { "zdr": true }` on every API call, routing only to [ZDR endpoints](https://openrouter.ai/docs/guides/features/zdr).
2. In [OpenRouter privacy settings](https://openrouter.ai/settings/privacy), do **not** enable prompt logging or “use inputs/outputs” discounts.
3. **Ollama local** remains PhysioNet’s recommended option if you want zero third-party exposure.

Restart the Jupyter kernel after changing provider or ZDR settings.

## Project layout

| Path | Purpose |
|------|---------|
| `notebooks/00_settings.ipynb` | User-editable settings |
| `notebooks/settings.json` | Saved settings (gitignored optional) |
| `notebooks/pipeline.py` | **Single shared module** — cohort, LLM, agents, I/O, export |
| `notebooks/stage_01_*.ipynb` … `stage_04_*.ipynb` | Run in order |
| `data/` | Intermediate artifacts |
| `patient_records/` | Final export |
| `MISC/` | Previous Streamlit prototype |

## Planned future stages

5. Ontology routing (Infectious / Cardiovascular / Respiratory)  
6. Evidence scoring  
7. Prune low-likelihood branches  
8. Retrieve guidelines / ICD codes  
9. Final diagnosis + confidence + reasoning trace
