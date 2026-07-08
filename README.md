# Clinical Coding Pipeline

Multi-stage Jupyter notebook workflow for clinical diagnosis coding.

## Pipeline Stages

Run notebooks **in order** from `notebooks/`:

| Stage | Notebook | Description | Output |
|-------|----------|-------------|--------|
| **1** | [`stage_01_cohort_selection.ipynb`](notebooks/stage_01_cohort_selection.ipynb) | Choose 15 MIMIC patients (≥2 admissions, ICD-10 labels, discharge notes) | `data/cohort/` |
| **2** | [`stage_02_information_extraction.ipynb`](notebooks/stage_02_information_extraction.ipynb) | Ollama NLP information extraction | `data/stage_02_information_extraction/` |
| **3** | [`stage_03_symptom_tree.ipynb`](notebooks/stage_03_symptom_tree.ipynb) | Ollama LLM hierarchical symptom trees | `data/stage_03_symptom_tree/` |
| **4** | [`stage_04_export_patient_records.ipynb`](notebooks/stage_04_export_patient_records.ipynb) | Export per-patient folders | `patient_records/` |

[`clinical_coding_pipeline.ipynb`](clinical_coding_pipeline.ipynb) is an index pointing to these stage notebooks.

```
Stage 1: Cohort Selection
      ↓  data/cohort/cohort.pkl
Stage 2: Information Extraction (Ollama NLP)
      ↓  data/stage_02_information_extraction/
Stage 3: Symptom Tree (Ollama LLM)
      ↓  data/stage_03_symptom_tree/
Stage 4: Export → patient_records/
```

Data source: **MIMIC-IV v3.1** + **MIMIC-IV-Note v2.2** from local PhysioNet files. Only admissions with **ICD-10 ground-truth diagnoses** are included.

Default MIMIC path (edit in `pipeline_config.py` if needed):

`/Users/narenkhatwani/Desktop/physionet.org/files`

## Test mode (1 patient)

Set in `pipeline_config.py` before running Stage 1:

```python
TEST_MODE = True   # 1 patient, min 1 admission — fast end-to-end smoke test
TEST_MODE = False  # 15 patients, min 2 admissions — full cohort
```

Test mode writes to **separate dirs** so it does not overwrite a full run:

| | Test mode | Full mode |
|--|-----------|-----------|
| Artifacts | `data/test/` | `data/` |
| Export | `patient_records_test/` | `patient_records/` |

Run all 4 stage notebooks in order — each stage reads the test cohort automatically.

## Run

```bash
pip install pandas requests jupyter
jupyter notebook notebooks/stage_01_cohort_selection.ipynb
```

**Ollama must be running** when `LLM_PROVIDER = "ollama"`. Use a **text** model (not vision):

```bash
ollama pull qwen2.5:7b
```

### LLM provider (`pipeline_config.py`)

```python
# Local Ollama
LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "qwen2.5:7b"

# OpenRouter (recommended for cloud — many models, one API key)
LLM_PROVIDER = "openrouter"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
export OPENROUTER_API_KEY="sk-or-..."

# Generic OpenAI-compatible API
LLM_PROVIDER = "api"
API_BASE_URL = "https://api.openai.com/v1"
API_MODEL = "gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
```

OpenRouter model IDs use the `provider/model` format (see [openrouter.ai/models](https://openrouter.ai/models)). Optional attribution headers in `pipeline_config.py`:

```python
OPENROUTER_HTTP_REFERER = "https://github.com/yourname/yourrepo"
OPENROUTER_APP_TITLE = "ai-agents-for-clinical-coding"
```

Other API examples:

| Provider | `API_BASE_URL` | Key env var |
|----------|----------------|-------------|
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |

Restart the Jupyter kernel after changing `LLM_PROVIDER` or API settings.

If a request times out, Stage 2 saves a checkpoint to `data/.../stage_02_information_extraction/ie_checkpoint.json` — re-run the extraction cell to resume.

| Setting | Default | Location |
|---------|---------|----------|
| Provider | `ollama` / `openrouter` / `api` | `LLM_PROVIDER` |
| OpenRouter model | `google/gemini-2.0-flash-001` | `OPENROUTER_MODEL` |
| OpenRouter key | env `OPENROUTER_API_KEY` | or `OPENROUTER_API_KEY` in config |
| Timeout | 600s | `LLM_TIMEOUT_SECONDS` |
| IE note length | 4000 chars | `IE_MAX_NOTE_CHARS` |

## Project Layout

| Path | Description |
|------|-------------|
| `notebooks/stage_01_*.ipynb` … `stage_04_*.ipynb` | Stage notebooks (run in order) |
| `pipeline_config.py` | Shared paths, **TEST_MODE**, and constants |
| `cohort_selection.py` | Stage 1 — load/save MIMIC cohort |
| `llm_client.py` | Ollama + API LLM client |
| `ollama_agents.py` | Clinical IE and symptom tree agents |
| `pipeline_io.py` | Load/save artifacts between stages |
| `export_patient_data.py` | Stage 4 — per-patient folder export |
| `data/` | Pipeline artifacts (created by notebooks) |
| `patient_records/` | Final export (created by Stage 4) |
| `MISC/` | Previous Streamlit app and related code |

## Planned Future Stages

5. Ontology Routing Agent (Infectious / Cardiovascular / Respiratory)
6. Evidence Scoring Agents
7. Prune Low-Likelihood Branches
8. Retrieve Guidelines / Literature / ICD Codes
9. Final Diagnosis + Confidence + Reasoning Trace
