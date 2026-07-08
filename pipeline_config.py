"""Shared paths and constants for the multi-stage clinical coding pipeline."""

from pathlib import Path

# Repo root (parent of this file)
REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Test mode — set True to run all stages on 1 patient only (fast smoke test).
# Uses separate artifact dirs so full runs are not overwritten:
#   data/test/...              vs  data/...
#   patient_records_test/      vs  patient_records/
# ---------------------------------------------------------------------------
TEST_MODE = False

# Local MIMIC-IV paths (PhysioNet download) — edit if needed
PHYSIONET_ROOT = Path("/Users/narenkhatwani/Desktop/physionet.org/files")
MIMIC_BASE = PHYSIONET_ROOT / "mimiciv/3.1"
DISCHARGE_NOTES_PATH = PHYSIONET_ROOT / "mimic-iv-note/2.2/note/discharge.csv.gz"

# Cohort selection (overridden when TEST_MODE is True)
N_PATIENTS = 1 if TEST_MODE else 15
MIN_ADMISSIONS_PER_PATIENT = 1 if TEST_MODE else 2
MIN_NOTE_CHARS = 500
RANDOM_SEED = 42
MAX_NOTE_CHARS = 8000

# LLM backend for stages 2–3: "ollama" | "api" | "openrouter"
LLM_PROVIDER = "ollama"

# Ollama (when LLM_PROVIDER = "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Generic API (when LLM_PROVIDER = "api") — OpenAI, Groq, Together, Azure, etc.
API_BASE_URL = "https://api.openai.com/v1"
API_MODEL = "gpt-4o-mini"
API_KEY_ENV = "OPENAI_API_KEY"
API_KEY = None  # optional inline key (prefer environment variable)

# OpenRouter (when LLM_PROVIDER = "openrouter")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_KEY = None  # or: export OPENROUTER_API_KEY=sk-or-...
OPENROUTER_HTTP_REFERER = ""  # optional — your site URL for OpenRouter rankings
OPENROUTER_APP_TITLE = "ai-agents-for-clinical-coding"

# Shared LLM settings
LLM_TIMEOUT_SECONDS = 600
LLM_MAX_RETRIES = 2
IE_MAX_NOTE_CHARS = 4000  # shorter input for faster IE; full note kept in cohort for stage 3
SYMPTOM_TREE_MAX_NOTE_CHARS = 8000

# Backward-compatible aliases
OLLAMA_TIMEOUT_SECONDS = LLM_TIMEOUT_SECONDS
OLLAMA_MAX_RETRIES = LLM_MAX_RETRIES

# Pipeline artifacts (written by stage notebooks)
DATA_DIR = REPO_ROOT / "data" / "test" if TEST_MODE else REPO_ROOT / "data"
COHORT_DIR = DATA_DIR / "cohort"
STAGE_02_DIR = DATA_DIR / "stage_02_information_extraction"
STAGE_03_DIR = DATA_DIR / "stage_03_symptom_tree"
EXPORT_DIR = REPO_ROOT / "patient_records_test" if TEST_MODE else REPO_ROOT / "patient_records"

COHORT_PICKLE = COHORT_DIR / "cohort.pkl"
COHORT_INDEX_JSON = COHORT_DIR / "cohort_index.json"
IE_RESULTS_JSON = STAGE_02_DIR / "information_extractions.json"
IE_CHECKPOINT_JSON = STAGE_02_DIR / "ie_checkpoint.json"
SYMPTOM_TREE_RESULTS_JSON = STAGE_03_DIR / "symptom_tree_results.json"


def get_llm_config(for_symptom_tree: bool = False):
    """Build LLMConfig from pipeline settings (Ollama, API, or OpenRouter)."""
    from llm_client import LLMConfig

    max_note = SYMPTOM_TREE_MAX_NOTE_CHARS if for_symptom_tree else IE_MAX_NOTE_CHARS
    provider = LLM_PROVIDER.lower().strip()
    if provider not in ("ollama", "api", "openrouter"):
        raise ValueError(
            f"LLM_PROVIDER must be 'ollama', 'api', or 'openrouter', got: {LLM_PROVIDER!r}"
        )

    if provider == "openrouter":
        extra_headers: dict = {}
        if OPENROUTER_HTTP_REFERER:
            extra_headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_APP_TITLE:
            extra_headers["X-Title"] = OPENROUTER_APP_TITLE
        return LLMConfig(
            provider="api",
            model=OPENROUTER_MODEL,
            api_base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            api_key_env=OPENROUTER_API_KEY_ENV,
            api_extra_headers=extra_headers,
            api_label="openrouter",
            max_note_chars=max_note,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

    if provider == "api":
        return LLMConfig(
            provider="api",
            model=API_MODEL,
            api_base_url=API_BASE_URL,
            api_key=API_KEY,
            api_key_env=API_KEY_ENV,
            api_label="api",
            max_note_chars=max_note,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

    return LLMConfig(
        provider="ollama",
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        max_note_chars=max_note,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )


def llm_provider_label() -> str:
    if LLM_PROVIDER == "openrouter":
        return f"OpenRouter ({OPENROUTER_MODEL})"
    if LLM_PROVIDER == "api":
        return f"API ({API_MODEL})"
    return f"Ollama ({OLLAMA_MODEL})"


def pipeline_mode_label() -> str:
    return f"TEST ({N_PATIENTS} patient)" if TEST_MODE else f"FULL ({N_PATIENTS} patients)"


def print_pipeline_banner() -> None:
    """Print current mode and paths (call at the top of each stage notebook)."""
    print("=" * 60)
    print(f"Pipeline mode : {pipeline_mode_label()}")
    print(f"LLM provider  : {llm_provider_label()}")
    print(f"Admissions/patient (min): {MIN_ADMISSIONS_PER_PATIENT}")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Export dir    : {EXPORT_DIR}")
    if TEST_MODE:
        print("TEST_MODE=True — set TEST_MODE=False in pipeline_config.py for full run")
    print("=" * 60)
