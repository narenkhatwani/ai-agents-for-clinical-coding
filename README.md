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
| `data/cohort/cohort.pkl` | **Re-run Stage 1** — redacted latest notes + rich prior history |
| `data/stage_02/ie_checkpoint.json` | **Delete** and re-run stages 2–4 |

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
3. Collapses to **one row per patient**:
   - **Latest admission** → discharge note with **discharge package redacted** (diagnosis, instructions, meds, disposition, condition, followup, transitional issues) + Hospital Course `# Problem:` titles scrubbed; full note in `clinical_note_full`; ICD-10 in ground truth only
   - **Prior admissions** → detailed history (`clinical_detail`: HPI, hospital course, prior discharge diagnoses, ICD-10)

**Outputs**

- `data/cohort/cohort.pkl` — one row per patient
- `data/cohort/cohort_index.json` — human-readable index

**Key columns:** `clinical_note` (redacted, for LLM), `clinical_note_full`, `clinical_context_text`, `admission_history`, `ground_truth_icd10`

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
  admission_history.json / .txt   (detailed prior stays)
  symptom_tree.json / .txt
  admissions/hadm_<latest>/
    clinical_note.txt              (redacted discharge package — sent to LLM)
    clinical_note_full.txt         (original)
    redacted_discharge_sections.txt
    ground_truth.json / .txt       (ICD-10 — evaluation only)
    clinical_context.txt
    structured_vitals.json / structured_labs.json / radiology_reports.json
    information_extraction.json / symptom_tree.json
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
| `notebooks/settings.json` | Saved settings |
| `notebooks/pipeline.py` | Cohort, LLM, agents, I/O, export |
| `notebooks/snomed_ct.py` | Offline SNOMED CT mapping + ancestor context |
| `notebooks/stage_01_*.ipynb` … `stage_07_*.ipynb` | Run in order |
| `data/SnomedCT_*` | Offline RF2 SNOMED CT package (not committed) |
| `data/snomed_index/` | Cached SNOMED index pickle |
| `data/stage_05_snomed_mapping/` | Entity → SNOMED mappings |
| `data/stage_06_snomed_ancestors/` | Ancestors + attribute context (filtered) |
| `patient_records/` | Per-patient export |
| `MISC/` | Previous Streamlit prototype |

### Stage 5 — SNOMED CT mapping (`stage_05_snomed_mapping.ipynb`)

- Reads IE entities from `patient_records/`
- Maps each term to offline SNOMED CT (exact + fuzzy lexical match)
- Writes `data/stage_05_snomed_mapping/snomed_mappings.json` and per-admission `snomed_mapping.json`

### Stage 6 — Ancestors & relations (`stage_06_snomed_ancestors.ipynb`)

For each mapped concept:

1. **2 Is-a ancestors** (parent + grandparent)
2. **Outbound** attribute destinations only (no inverses — avoids drug→poisoning noise):
   - `cause_of` (Due to), `has_causative_agent`, `has_finding_site`, `has_associated_morphology`, `has_pathological_process`
   - plus `after`, `associated_with`, `occurrence`, `clinical_course`
   - plus Is-a parent of each attribute target
3. **Score** with local **MiniLM** (`all-MiniLM-L6-v2`) vs the **patient IE entity term**
4. **Retain** if similarity ≥ **0.70**; mark **high_confidence** if ≥ **0.80**
5. Full unfiltered lists stay in `ancestors_depth2_all` / `attribute_relations_all`
6. Per-admission: `snomed_ancestors.json` (full) + `snomed_retained.json` / `.txt` (retained only)
7. Relationship **weights deferred** (`weights: null`)

Requires: `pip install sentence-transformers` (model downloads once, then runs offline).

After changing attribute typeIds, rebuild the SNOMED index with `force_rebuild=True` once.

### Stage 7 — Scored differential diagnosis (`stage_07_differential_diagnosis.ipynb`)

For each latest admission under `patient_records/`:

1. Loads **symptom tree** + **retained SNOMED** (`snomed_retained.json`)
2. Loads **all prior ICD-10 codes** from `admission_history.json` (PMH/comorbidity only — not current GT)
3. Optionally includes structured clinical context + IE (current stay)
4. Structured prompt: **ROLE / CONTEXT / TASK / CONSTRAINTS**, temperature **0.4**
5. LLM produces a **ranked multi-diagnosis differential** with scores **0–100**
6. Checkpoint resume via `diff_dx_checkpoint.json`
7. Writes:
   - `data/stage_07_differential_diagnosis/differential_diagnoses.json`
   - per admission: `differential_diagnosis.json` + `differential_diagnosis.txt`

Re-run: delete `diff_dx_checkpoint.json` if you want all patients regenerated under this prompt.

## Planned future stages

8. ICD / guideline retrieval over scored differentials  
9. Final coding package + reasoning trace  
10. Evaluation vs ground-truth ICD-10