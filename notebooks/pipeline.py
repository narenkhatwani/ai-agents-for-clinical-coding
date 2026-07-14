"""
Clinical coding pipeline — single module used by stage notebooks.

Edit settings in notebooks/00_settings.ipynb (writes settings.json) or change defaults below.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import requests


# Repo root (parent of this file)
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Load KEY=VALUE pairs from .env into os.environ (does not override existing env vars)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

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

# ---------------------------------------------------------------------------
# LLM backend — Qwen 2.5 7B (same model, two ways to run)
#
#   OpenRouter (cloud, fast to start):  LLM_PROVIDER = "openrouter"
#                                      export OPENROUTER_API_KEY=sk-or-...
#
#   Ollama (local, compare speed):      LLM_PROVIDER = "ollama"
#                                      ollama pull qwen2.5:7b && ollama serve
#
# Restart the Jupyter kernel after switching LLM_PROVIDER.
# ---------------------------------------------------------------------------
LLM_PROVIDER = "openrouter"  # "openrouter" | "ollama" | "api"

# Paired Qwen 2.5 7B model IDs (equivalent instruct-tuned weights)
QWEN_7B_OPENROUTER = "qwen/qwen-2.5-7b-instruct"
QWEN_7B_OLLAMA = "qwen2.5:7b"

# Ollama (when LLM_PROVIDER = "ollama")
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = QWEN_7B_OLLAMA

# Generic API (when LLM_PROVIDER = "api") — OpenAI, Groq, Together, Azure, etc.
API_BASE_URL = "https://api.openai.com/v1"
API_MODEL = "gpt-4o-mini"
API_KEY_ENV = "OPENAI_API_KEY"
API_KEY = None  # optional inline key (prefer environment variable)

# OpenRouter (when LLM_PROVIDER = "openrouter")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = QWEN_7B_OPENROUTER
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_API_KEY = None  # or: export OPENROUTER_API_KEY=sk-or-...
OPENROUTER_HTTP_REFERER = ""  # optional — your site URL for OpenRouter rankings
OPENROUTER_APP_TITLE = "ai-agents-for-clinical-coding"
# Zero Data Retention — route only to OpenRouter endpoints with ZDR (required for MIMIC credentialed data)
OPENROUTER_ZDR = True

# Shared LLM settings
LLM_TIMEOUT_SECONDS = 600
LLM_MAX_RETRIES = 2
LLM_RATE_LIMIT_RETRIES = 6  # OpenRouter 429 — wait and retry
LLM_REQUEST_DELAY_SECONDS = 3.0  # pause between admissions (avoid rate limits)
IE_MAX_NOTE_CHARS = 4000  # shorter input for faster IE; full note kept in cohort for stage 3
SYMPTOM_TREE_MAX_NOTE_CHARS = 8000
HISTORY_NOTE_EXCERPT_CHARS = 800  # prior admission note snippet in history context

# Stage 7 — prune candidates with evidence score below this cutoff (see stage 6 scoring guide:
# 0-20 minimal, 21-40 weak, 41-60 moderate, 61-80 strong, 81-100 very strong)
PRUNE_SCORE_THRESHOLD = 40

# Backward-compatible aliases
OLLAMA_TIMEOUT_SECONDS = LLM_TIMEOUT_SECONDS
OLLAMA_MAX_RETRIES = LLM_MAX_RETRIES

# Pipeline artifacts (written by stage notebooks)
DATA_DIR = REPO_ROOT / "data" / "test" if TEST_MODE else REPO_ROOT / "data"
COHORT_DIR = DATA_DIR / "cohort"
STAGE_01B_DIR = DATA_DIR / "stage_01b_redact_notes"
STAGE_02_DIR = DATA_DIR / "stage_02_information_extraction"
STAGE_03_DIR = DATA_DIR / "stage_03_symptom_tree"
STAGE_05_DIR = DATA_DIR / "stage_05_ontology_routing"
STAGE_06_DIR = DATA_DIR / "stage_06_evidence_scoring"
STAGE_07_DIR = DATA_DIR / "stage_07_pruned_branches"
STAGE_08_DIR = DATA_DIR / "stage_08_guideline_icd_retrieval"
EXPORT_DIR = REPO_ROOT / "patient_records_test" if TEST_MODE else REPO_ROOT / "patient_records"


_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


def _apply_settings_json() -> None:
    if not _SETTINGS_PATH.exists():
        return
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    g = globals()
    for key, val in data.items():
        if key in g:
            g[key] = val
    if data.get("PHYSIONET_ROOT"):
        pr = Path(data["PHYSIONET_ROOT"])
        g["PHYSIONET_ROOT"] = pr
        g["MIMIC_BASE"] = pr / "mimiciv/3.1"
        g["DISCHARGE_NOTES_PATH"] = pr / "mimic-iv-note/2.2/note/discharge.csv.gz"
        g["RADIOLOGY_NOTES_PATH"] = pr / "mimic-iv-note/2.2/note/radiology.csv.gz"
    if "TEST_MODE" in data:
        tm = bool(data["TEST_MODE"])
        g["TEST_MODE"] = tm
        g["N_PATIENTS"] = 1 if tm else 15
        g["MIN_ADMISSIONS_PER_PATIENT"] = 1 if tm else 2
        g["DATA_DIR"] = g["REPO_ROOT"] / "data" / "test" if tm else g["REPO_ROOT"] / "data"
        g["COHORT_DIR"] = g["DATA_DIR"] / "cohort"
        g["STAGE_01B_DIR"] = g["DATA_DIR"] / "stage_01b_redact_notes"
        g["STAGE_02_DIR"] = g["DATA_DIR"] / "stage_02_information_extraction"
        g["STAGE_03_DIR"] = g["DATA_DIR"] / "stage_03_symptom_tree"
        g["STAGE_05_DIR"] = g["DATA_DIR"] / "stage_05_ontology_routing"
        g["STAGE_06_DIR"] = g["DATA_DIR"] / "stage_06_evidence_scoring"
        g["STAGE_07_DIR"] = g["DATA_DIR"] / "stage_07_pruned_branches"
        g["STAGE_08_DIR"] = g["DATA_DIR"] / "stage_08_guideline_icd_retrieval"
        g["EXPORT_DIR"] = g["REPO_ROOT"] / "patient_records_test" if tm else g["REPO_ROOT"] / "patient_records"


def _refresh_artifact_paths() -> None:
    global COHORT_PICKLE, COHORT_INDEX_JSON, IE_RESULTS_JSON, IE_CHECKPOINT_JSON, SYMPTOM_TREE_RESULTS_JSON, REDACTION_CHECKPOINT_JSON
    COHORT_PICKLE = COHORT_DIR / "cohort.pkl"
    COHORT_INDEX_JSON = COHORT_DIR / "cohort_index.json"
    REDACTION_CHECKPOINT_JSON = STAGE_01B_DIR / "redaction_checkpoint.json"
    IE_RESULTS_JSON = STAGE_02_DIR / "information_extractions.json"
    IE_CHECKPOINT_JSON = STAGE_02_DIR / "ie_checkpoint.json"
    SYMPTOM_TREE_RESULTS_JSON = STAGE_03_DIR / "symptom_tree_results.json"



_apply_settings_json()
_refresh_artifact_paths()


def get_llm_config(for_symptom_tree: bool = False):
    """Build LLMConfig from pipeline settings (Ollama, API, or OpenRouter)."""
    # llm in this module LLMConfig

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
            openrouter_zdr=OPENROUTER_ZDR,
            max_note_chars=max_note,
            timeout_seconds=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
            rate_limit_retries=LLM_RATE_LIMIT_RETRIES,
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
            rate_limit_retries=LLM_RATE_LIMIT_RETRIES,
        )

    return LLMConfig(
        provider="ollama",
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        max_note_chars=max_note,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        rate_limit_retries=0,
    )


def llm_provider_label() -> str:
    if LLM_PROVIDER == "openrouter":
        zdr = "ZDR on" if OPENROUTER_ZDR else "ZDR off"
        return f"OpenRouter ({OPENROUTER_MODEL}, {zdr})"
    if LLM_PROVIDER == "api":
        return f"API ({API_MODEL})"
    return f"Ollama ({OLLAMA_MODEL})"


def qwen_pair_hint() -> str:
    """Reminder of the equivalent model on the other backend."""
    if LLM_PROVIDER == "openrouter":
        return f"Local equivalent: ollama pull {QWEN_7B_OLLAMA}"
    if LLM_PROVIDER == "ollama":
        return f"Cloud equivalent: {QWEN_7B_OPENROUTER} on OpenRouter"
    return ""


def pipeline_mode_label() -> str:
    return f"TEST ({N_PATIENTS} patient)" if TEST_MODE else f"FULL ({N_PATIENTS} patients)"


def print_pipeline_banner() -> None:
    """Print current mode and paths (call at the top of each stage notebook)."""
    print("=" * 60)
    print(f"Pipeline mode : {pipeline_mode_label()}")
    print(f"LLM provider  : {llm_provider_label()}")
    hint = qwen_pair_hint()
    if hint:
        print(f"Qwen pair     : {hint}")
    print(f"Admissions/patient (min): {MIN_ADMISSIONS_PER_PATIENT}")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Export dir    : {EXPORT_DIR}")
    if TEST_MODE:
        print("TEST_MODE=True — set TEST_MODE=False in 00_settings.ipynb for full run")
    if LLM_PROVIDER == "openrouter" and OPENROUTER_ZDR:
        print("OpenRouter ZDR : enabled (provider.zdr=true on every request)")
    elif LLM_PROVIDER == "openrouter":
        print("OpenRouter ZDR : DISABLED — not suitable for MIMIC credentialed data")
    print("=" * 60)


# =============================================================================
# Structured MIMIC data — vitals, labs, radiology reports
# =============================================================================

VITAL_ITEMIDS = {
    220045: "Heart Rate",
    220210: "Respiratory Rate",
    220277: "SpO2",
    220179: "SBP (non-invasive)",
    220180: "DBP (non-invasive)",
    223762: "Temperature (C)",
    220052: "MAP (arterial)",
}

ICU_CHARTEVENTS_PATH = MIMIC_BASE / "icu/chartevents.csv.gz"
LABEVENTS_PATH = MIMIC_BASE / "hosp/labevents.csv.gz"
D_LABITEMS_PATH = MIMIC_BASE / "hosp/d_labitems.csv.gz"
OMR_PATH = MIMIC_BASE / "hosp/omr.csv.gz"
RADIOLOGY_NOTES_PATH = PHYSIONET_ROOT / "mimic-iv-note/2.2/note/radiology.csv.gz"

MAX_LABS_PER_ADMISSION = 40
MAX_RAD_REPORTS_PER_ADMISSION = 5
RADIOLOGY_REPORT_EXCERPT_CHARS = 1500
OMR_VITAL_NAMES = {
    "Blood Pressure", "Pulse", "Temperature", "Respiratory Rate",
    "O2 saturation", "SpO2", "Weight (Lbs)", "BMI (kg/m2)",
}


def _summarize_vital_series(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    nums = pd.to_numeric(df["valuenum"], errors="coerce").dropna()
    if nums.empty:
        return {
            "name": label,
            "value": str(df["value"].iloc[-1]),
            "unit": str(df["valueuom"].iloc[-1]) if "valueuom" in df.columns else "",
            "source": df["source"].iloc[0] if "source" in df.columns else "mimic",
        }
    return {
        "name": label,
        "min": float(nums.min()),
        "max": float(nums.max()),
        "last": float(nums.iloc[-1]),
        "unit": str(df["valueuom"].dropna().iloc[-1]) if df["valueuom"].notna().any() else "",
        "n_readings": int(len(nums)),
        "source": df["source"].iloc[0] if "source" in df.columns else "mimic",
    }


def load_chartevents_vitals(hadm_ids: set, subject_ids: set) -> Dict[int, List[Dict[str, Any]]]:
    if not ICU_CHARTEVENTS_PATH.exists():
        return {int(h): [] for h in hadm_ids}
    hadm_ids = {int(h) for h in hadm_ids}
    subject_ids = {int(s) for s in subject_ids}
    rows: List[pd.DataFrame] = []
    usecols = {"subject_id", "hadm_id", "charttime", "itemid", "valuenum", "value", "valueuom"}
    for chunk in pd.read_csv(
        ICU_CHARTEVENTS_PATH, usecols=lambda c: c in usecols, chunksize=250_000
    ):
        m = (
            chunk["subject_id"].isin(subject_ids)
            & chunk["hadm_id"].isin(hadm_ids)
            & chunk["itemid"].isin(VITAL_ITEMIDS)
        )
        if m.any():
            part = chunk[m].copy()
            part["source"] = "icu_chartevents"
            rows.append(part)
    if not rows:
        return {int(h): [] for h in hadm_ids}
    df = pd.concat(rows, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["label"] = df["itemid"].map(VITAL_ITEMIDS)
    out: Dict[int, List[Dict[str, Any]]] = {int(h): [] for h in hadm_ids}
    for hadm_id, grp in df.groupby("hadm_id"):
        summaries = []
        for label, sub in grp.sort_values("charttime").groupby("label"):
            summaries.append(_summarize_vital_series(sub, str(label)))
        out[int(hadm_id)] = summaries
    return out


def load_omr_vitals(admissions: pd.DataFrame) -> Dict[int, List[Dict[str, Any]]]:
    if not OMR_PATH.exists():
        return {int(h): [] for h in admissions["hadm_id"]}
    subjects = set(admissions["subject_id"].astype(int))
    omr = pd.read_csv(
        OMR_PATH, usecols=["subject_id", "chartdate", "result_name", "result_value"]
    )
    omr = omr[omr["subject_id"].isin(subjects) & omr["result_name"].isin(OMR_VITAL_NAMES)].copy()
    omr["chartdate"] = pd.to_datetime(omr["chartdate"], errors="coerce")
    out: Dict[int, List[Dict[str, Any]]] = {int(h): [] for h in admissions["hadm_id"]}
    adm = admissions.copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    for _, row in adm.iterrows():
        hadm_id = int(row["hadm_id"])
        subj = int(row["subject_id"])
        start, end = row["admittime"], row["dischtime"]
        sub = omr[omr["subject_id"] == subj]
        if pd.notna(start):
            sub = sub[sub["chartdate"] >= start - pd.Timedelta(days=2)]
        if pd.notna(end):
            sub = sub[sub["chartdate"] <= end + pd.Timedelta(days=1)]
        if sub.empty:
            continue
        items = []
        for name, g in sub.sort_values("chartdate").groupby("result_name"):
            last = g.iloc[-1]
            items.append({
                "name": str(name),
                "value": str(last["result_value"]),
                "unit": "",
                "charttime": str(last["chartdate"].date()) if pd.notna(last["chartdate"]) else "",
                "source": "omr",
            })
        out[hadm_id] = items
    return out


def load_labs_for_hadm_ids(hadm_ids: set) -> Dict[int, List[Dict[str, Any]]]:
    hadm_ids = {int(h) for h in hadm_ids}
    if not LABEVENTS_PATH.exists():
        return {int(h): [] for h in hadm_ids}
    labels = pd.read_csv(D_LABITEMS_PATH, usecols=["itemid", "label"])
    labels = labels.drop_duplicates("itemid")
    label_map = dict(zip(labels["itemid"], labels["label"]))
    rows: List[pd.DataFrame] = []
    usecols = {
        "hadm_id", "charttime", "itemid", "valuenum", "value", "valueuom",
        "flag", "ref_range_lower", "ref_range_upper",
    }
    for chunk in pd.read_csv(
        LABEVENTS_PATH, usecols=lambda c: c in usecols, chunksize=500_000
    ):
        m = chunk["hadm_id"].isin(hadm_ids)
        if m.any():
            rows.append(chunk[m])
    out: Dict[int, List[Dict[str, Any]]] = {int(h): [] for h in hadm_ids}
    if not rows:
        return out
    df = pd.concat(rows, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["name"] = df["itemid"].map(label_map).fillna("Unknown lab")
    for hadm_id, grp in df.groupby("hadm_id"):
        grp = grp.sort_values(["flag", "charttime"], ascending=[True, False])
        abnormal = grp[grp["flag"].notna() & (grp["flag"] != "")]
        normal = grp[~(grp["flag"].notna() & (grp["flag"] != ""))]
        ordered = pd.concat([abnormal, normal]).drop_duplicates("name", keep="first")
        ordered = ordered.head(MAX_LABS_PER_ADMISSION)
        labs = []
        for _, r in ordered.iterrows():
            val = r["valuenum"] if pd.notna(r["valuenum"]) else r["value"]
            ref = ""
            if pd.notna(r.get("ref_range_lower")) and pd.notna(r.get("ref_range_upper")):
                ref = f"{r['ref_range_lower']}-{r['ref_range_upper']}"
            labs.append({
                "name": str(r["name"]),
                "value": str(val),
                "unit": str(r["valueuom"]) if pd.notna(r["valueuom"]) else "",
                "flag": str(r["flag"]) if pd.notna(r["flag"]) else "",
                "charttime": str(r["charttime"]) if pd.notna(r["charttime"]) else "",
                "ref_range": ref,
            })
        out[int(hadm_id)] = labs
    return out


def load_radiology_reports(hadm_ids: set) -> Dict[int, List[Dict[str, Any]]]:
    hadm_ids = {int(h) for h in hadm_ids}
    if not RADIOLOGY_NOTES_PATH.exists():
        return {int(h): [] for h in hadm_ids}
    rows: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        RADIOLOGY_NOTES_PATH, usecols=["hadm_id", "charttime", "text"], chunksize=100_000
    ):
        m = chunk["hadm_id"].isin(hadm_ids) & chunk["text"].notna()
        if m.any():
            rows.append(chunk[m])
    out: Dict[int, List[Dict[str, Any]]] = {int(h): [] for h in hadm_ids}
    if not rows:
        return out
    df = pd.concat(rows, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    for hadm_id, grp in df.groupby("hadm_id"):
        grp = grp.sort_values("charttime", ascending=False).head(MAX_RAD_REPORTS_PER_ADMISSION)
        reports = []
        for _, r in grp.iterrows():
            text = str(r["text"])
            excerpt = text[:RADIOLOGY_REPORT_EXCERPT_CHARS]
            if len(text) > RADIOLOGY_REPORT_EXCERPT_CHARS:
                excerpt += "\n[... report truncated ...]"
            reports.append({
                "type": "radiology",
                "charttime": str(r["charttime"]) if pd.notna(r["charttime"]) else "",
                "text_excerpt": excerpt,
            })
        out[int(hadm_id)] = reports
    return out


def format_structured_clinical_context(
    vitals: List[Dict[str, Any]],
    labs: List[Dict[str, Any]],
    reports: List[Dict[str, Any]],
) -> str:
    lines = [
        "STRUCTURED MIMIC DATA (objective vitals, labs, radiology — prefer over note text when they conflict):",
        "",
        "VITALS:",
    ]
    if not vitals:
        lines.append("  (none recorded in MIMIC for this admission)")
    else:
        for v in vitals:
            if "min" in v:
                lines.append(
                    f"  • {v['name']}: last={v.get('last')} min={v.get('min')} max={v.get('max')} "
                    f"{v.get('unit', '')} ({v.get('n_readings', 0)} readings, {v.get('source', '')})"
                )
            else:
                lines.append(f"  • {v['name']}: {v.get('value')} {v.get('unit', '')} ({v.get('source', '')})")
    lines.append("")
    lines.append("LABS (abnormal prioritized):")
    if not labs:
        lines.append("  (none recorded)")
    else:
        for lab in labs[:MAX_LABS_PER_ADMISSION]:
            flag = f" [{lab['flag']}]" if lab.get("flag") else ""
            lines.append(f"  • {lab['name']}: {lab.get('value')} {lab.get('unit', '')}{flag}")
    lines.append("")
    lines.append("RADIOLOGY REPORTS:")
    if not reports:
        lines.append("  (none recorded)")
    else:
        for i, rep in enumerate(reports, 1):
            lines.append(f"  --- Report {i} ({rep.get('charttime', '')}) ---")
            for ln in str(rep.get("text_excerpt", "")).splitlines()[:20]:
                lines.append(f"    {ln}")
    lines.append("")
    return "\n".join(lines)


def enrich_cohort_structured_data(cohort: pd.DataFrame) -> pd.DataFrame:
    """Attach vitals, labs, and radiology reports per admission row."""
    hadm_ids = set(cohort["hadm_id"].astype(int))
    subject_ids = set(cohort["subject_id"].astype(int))
    print(f"Loading structured MIMIC data for {len(hadm_ids)} admissions...")
    icu_vitals = load_chartevents_vitals(hadm_ids, subject_ids)
    omr_vitals = load_omr_vitals(
        cohort[["hadm_id", "subject_id", "admittime", "dischtime"]].drop_duplicates("hadm_id")
    )
    labs = load_labs_for_hadm_ids(hadm_ids)
    reports = load_radiology_reports(hadm_ids)

    def merge_vitals(hadm_id: int) -> List[Dict[str, Any]]:
        return icu_vitals.get(int(hadm_id), []) + omr_vitals.get(int(hadm_id), [])

    cohort = cohort.copy()
    cohort["structured_vitals"] = cohort["hadm_id"].astype(int).map(merge_vitals)
    cohort["structured_labs"] = cohort["hadm_id"].astype(int).map(lambda h: labs.get(int(h), []))
    cohort["structured_reports"] = cohort["hadm_id"].astype(int).map(lambda h: reports.get(int(h), []))
    cohort["clinical_context_text"] = cohort.apply(
        lambda r: format_structured_clinical_context(
            r["structured_vitals"], r["structured_labs"], r["structured_reports"]
        ),
        axis=1,
    )
    n_v = (cohort["structured_vitals"].apply(len) > 0).sum()
    n_l = (cohort["structured_labs"].apply(len) > 0).sum()
    n_r = (cohort["structured_reports"].apply(len) > 0).sum()
    print(f"  Admissions with vitals: {n_v}/{len(cohort)} | labs: {n_l} | radiology: {n_r}")
    return cohort



import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import requests

Provider = Literal["ollama", "api"]


@dataclass
class LLMConfig:
    provider: Provider = "ollama"
    model: str = "qwen2.5:7b"
    # Ollama
    base_url: str = "http://localhost:11434"
    # API (OpenAI-compatible)
    api_base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    api_extra_headers: Dict[str, str] = field(default_factory=dict)
    api_label: str = "api"  # used in logs and extraction_method tags
    openrouter_zdr: bool = False  # OpenRouter: route only to zero-retention endpoints
    # Shared
    max_note_chars: int = 4000
    timeout_seconds: int = 600
    max_retries: int = 2
    rate_limit_retries: int = 6  # extra retries for HTTP 429/503 (OpenRouter)
    num_predict: int = 4096  # Ollama
    max_tokens: int = 4096  # API
    temperature: float = 0.1
    json_mode: bool = True  # request JSON object from API when supported

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env) or os.environ.get("LLM_API_KEY")

    def method_prefix(self) -> str:
        if self.provider == "ollama":
            return "ollama"
        return self.api_label or "api"


# Backward-compatible alias used by notebooks / older code
OllamaConfig = LLMConfig


class LLMNotAvailableError(RuntimeError):
    pass


class OllamaNotAvailableError(LLMNotAvailableError):
    pass


def _strip_json_wrappers(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(
        r"<(?:think|redacted_thinking)>.*?</(?:think|redacted_thinking)>",
        "",
        cleaned,
        flags=re.I | re.DOTALL,
    )
    return cleaned.strip()


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_json_wrappers(text)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def truncate_note(note: str, max_chars: int) -> str:
    note = note or ""
    if len(note) <= max_chars:
        return note
    return note[:max_chars] + "\n\n[... note truncated for LLM input ...]"


def check_llm(config: LLMConfig, strict: bool = True) -> Tuple[bool, str]:
    if config.provider == "ollama":
        return _check_ollama(config, strict=strict)
    return _check_api(config)


def require_llm(config: LLMConfig) -> str:
    ok, message = check_llm(config)
    if not ok:
        raise LLMNotAvailableError(message)
    return message


def check_ollama(config: LLMConfig, strict: bool = True) -> Tuple[bool, str]:
    return _check_ollama(config, strict=strict)


def _check_ollama(config: LLMConfig, strict: bool = True) -> Tuple[bool, str]:
    try:
        response = requests.get(f"{config.base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        if config.model in models:
            return True, config.model
        if not strict and models:
            return True, models[0]
        if not models:
            return False, "No models installed. Run: ollama pull qwen2.5:7b"
        return False, (
            f"Model '{config.model}' not found. Installed: {', '.join(models[:5])}. "
            f"Run: ollama pull {config.model}"
        )
    except requests.exceptions.ConnectionError:
        return False, "Ollama is not running. Start with: ollama serve"
    except Exception as exc:
        return False, str(exc)


def _check_api(config: LLMConfig) -> Tuple[bool, str]:
    key = config.resolved_api_key()
    if not key:
        return False, (
            f"API key not set. Export {config.api_key_env} or LLM_API_KEY, "
            "or set API_KEY in pipeline_config.py"
        )
    if not config.model:
        return False, "API_MODEL is not set in pipeline_config.py"
    return True, config.model


def warn_if_slow_model(model: str, provider: Provider) -> None:
    if provider != "ollama":
        return
    lower = model.lower()
    if "vl" in lower or "vision" in lower:
        print(
            f"WARNING: '{model}' is a vision model and is slow for text-only NLP. "
            "Prefer a text model, e.g. ollama pull qwen2.5:7b"
        )


def call_llm_chat(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: Optional[str] = None,
    json_mode: Optional[bool] = None,
) -> str:
    use_model = model or config.model
    use_json = config.json_mode if json_mode is None else json_mode

    if config.provider == "ollama":
        return _call_ollama_chat(system_prompt, user_prompt, config, use_model)
    return _call_api_chat(system_prompt, user_prompt, config, use_model, json_mode=use_json)


def _call_ollama_chat(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: str,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
        },
    }
    url = f"{config.base_url.rstrip('/')}/api/chat"
    label = "Ollama"

    last_error: Optional[Exception] = None
    for attempt in range(config.max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=config.timeout_seconds)
        except requests.exceptions.ConnectionError as exc:
            raise LLMNotAvailableError(
                "Cannot connect to Ollama. Is 'ollama serve' running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            last_error = _timeout_error(config, label, attempt)
            if attempt < config.max_retries:
                _sleep_retry(attempt, reason="Timeout")
                continue
            raise last_error from exc

        if response.status_code == 404:
            raise LLMNotAvailableError(f"Model '{model}' not found. Run: ollama pull {model}")
        response.raise_for_status()
        return (response.json().get("message") or {}).get("content", "")

    raise last_error or TimeoutError(f"{label} request failed")


def _call_api_chat(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: str,
    json_mode: bool,
) -> str:
    api_key = config.resolved_api_key()
    if not api_key:
        raise LLMNotAvailableError(
            f"API key not set. Export {config.api_key_env} or LLM_API_KEY."
        )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if config.api_label == "openrouter" and config.openrouter_zdr:
        payload["provider"] = {"zdr": True}

    url = f"{config.api_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **config.api_extra_headers,
    }
    label = config.api_label or "API LLM"
    max_attempts = max(config.max_retries, config.rate_limit_retries) + 1

    last_error: Optional[Exception] = None
    rate_limit_attempts = 0
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=config.timeout_seconds
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMNotAvailableError(
                f"Cannot connect to API at {config.api_base_url}: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            last_error = _timeout_error(config, label, attempt)
            if attempt < config.max_retries:
                _sleep_retry(attempt, reason="Timeout")
                continue
            raise last_error from exc

        if response.status_code == 401:
            raise LLMNotAvailableError("API authentication failed (401). Check your API key.")
        if response.status_code == 404:
            hint = ""
            if config.api_label == "openrouter" and config.openrouter_zdr:
                hint = (
                    " No ZDR endpoint may be available for this model — "
                    "check https://openrouter.ai/api/v1/endpoints/zdr or disable OPENROUTER_ZDR."
                )
            raise LLMNotAvailableError(
                f"Model '{model}' not found at {config.api_base_url}.{hint}"
            )
        if response.status_code in (429, 503):
            rate_limit_attempts += 1
            if rate_limit_attempts <= config.rate_limit_retries:
                wait = _retry_after_seconds(response, rate_limit_attempts - 1)
                print(
                    f"  Rate limited ({response.status_code}) — "
                    f"waiting {wait:.0f}s (retry {rate_limit_attempts}/{config.rate_limit_retries})..."
                )
                time.sleep(wait)
                continue
            raise LLMNotAvailableError(_format_api_error(response))
        if not response.ok:
            raise LLMNotAvailableError(_format_api_error(response))

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Empty API response: {json.dumps(data)[:300]}")
        return (choices[0].get("message") or {}).get("content", "")

    raise last_error or TimeoutError(f"{label} request failed")


def _timeout_error(config: LLMConfig, label: str, attempt: int) -> TimeoutError:
    return TimeoutError(
        f"{label} request timed out after {config.timeout_seconds}s "
        f"(attempt {attempt + 1}/{config.max_retries + 1})"
    )


def _sleep_retry(attempt: int, reason: str = "Timeout") -> None:
    wait = 5 * (attempt + 1)
    print(f"  {reason} — retrying in {wait}s...")
    time.sleep(wait)


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 5.0)
        except ValueError:
            pass
    # 15s, 30s, 60s, 90s, 120s, 150s
    return min(15.0 * (2**attempt), 180.0)


def _format_api_error(response: requests.Response) -> str:
    status = response.status_code
    body = response.text[:800]
    if status == 429 and "openrouter" in (response.url or ""):
        hint = (
            "OpenRouter rate limit. Wait and re-run (checkpoint resumes), "
            "add credits at openrouter.ai/settings/credits, "
            "or link a provider key at openrouter.ai/settings/integrations (BYOK)."
        )
        return f"API error 429 (rate limited). {hint}\nDetails: {body}"
    return f"API error {status}: {body}"


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    raw = call_llm_chat(system_prompt, user_prompt, config, model=model, json_mode=True)
    parsed = parse_json_object(raw)
    if parsed is None:
        raise ValueError(
            f"Could not parse JSON from {config.provider} LLM. Preview: {raw[:300]}"
        )
    return parsed


import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd



def assert_mimic_paths(
    mimic_base: Path = MIMIC_BASE,
    discharge_notes_path: Path = DISCHARGE_NOTES_PATH,
) -> None:
    missing = [p for p in (mimic_base, discharge_notes_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "MIMIC path not found:\n" + "\n".join(f"  - {p}" for p in missing)
        )


def load_icd10_ground_truth(hadm_ids: set) -> pd.DataFrame:
    """All ICD-10 diagnoses for the given admissions, with titles."""
    dx = pd.read_csv(
        MIMIC_BASE / "hosp/diagnoses_icd.csv.gz",
        usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    dx = dx[(dx["icd_version"] == 10) & (dx["hadm_id"].isin(hadm_ids))].copy()

    icd_dict = pd.read_csv(
        MIMIC_BASE / "hosp/d_icd_diagnoses.csv.gz",
        usecols=["icd_code", "icd_version", "long_title"],
    )
    icd_dict = icd_dict[icd_dict["icd_version"] == 10]
    dx = dx.merge(icd_dict, on=["icd_code", "icd_version"], how="left")
    dx = dx.sort_values(["hadm_id", "seq_num"])

    primary = dx[dx["seq_num"] == 1].copy()
    primary = primary.rename(
        columns={"icd_code": "primary_icd_code", "long_title": "primary_dx_title"}
    )

    all_dx = (
        dx.groupby("hadm_id")
        .agg(
            ground_truth_icd10=("icd_code", list),
            ground_truth_dx_titles=("long_title", list),
            n_diagnoses=("icd_code", "count"),
        )
        .reset_index()
    )

    return primary[["hadm_id", "primary_icd_code", "primary_dx_title"]].merge(
        all_dx, on="hadm_id", how="inner"
    )


def _admission_summary(row: pd.Series, include_note_excerpt: bool = True) -> Dict[str, Any]:
    """Structured summary of one admission for history context."""
    summary: Dict[str, Any] = {
        "hadm_id": int(row["hadm_id"]),
        "admission_id": str(row["admission_id"]),
        "admittime": str(row.get("admittime", "")),
        "dischtime": str(row.get("dischtime", "")),
        "admission_type": row.get("admission_type"),
        "primary_icd_code": row.get("primary_icd_code"),
        "primary_dx_title": row.get("primary_dx_title"),
        "ground_truth_icd10": row.get("ground_truth_icd10", []),
        "ground_truth_dx_titles": row.get("ground_truth_dx_titles", []),
        "n_diagnoses": int(row["n_diagnoses"]) if pd.notna(row.get("n_diagnoses")) else None,
    }
    if include_note_excerpt:
        note = row.get("clinical_note") or row.get("text") or ""
        if note:
            excerpt = str(note)[:HISTORY_NOTE_EXCERPT_CHARS]
            if len(str(note)) > HISTORY_NOTE_EXCERPT_CHARS:
                excerpt += "\n[... excerpt truncated ...]"
            summary["note_excerpt"] = excerpt
    vitals = row.get("structured_vitals") or []
    labs = row.get("structured_labs") or []
    reports = row.get("structured_reports") or []
    if vitals:
        summary["vitals_summary"] = vitals[:8]
    if labs:
        summary["labs_summary"] = labs[:12]
    if reports:
        summary["reports_summary"] = [
            {
                "type": r.get("type"),
                "charttime": r.get("charttime"),
                "excerpt": (r.get("text_excerpt") or "")[:400],
            }
            for r in reports[:2]
        ]
    return summary


def collapse_to_latest_admission(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    One row per patient: latest admission note + prior admissions as history.

    Latest = most recent admittime. Prior admissions become `admission_history`.
    """
    cohort = cohort.sort_values(["subject_id", "admittime"]).copy()
    latest_idx = cohort.groupby("subject_id")["admittime"].idxmax()
    latest = cohort.loc[latest_idx].copy()

    history_map: Dict[str, List[Dict[str, Any]]] = {}
    for patient_id, grp in cohort.groupby("patient_id"):
        grp = grp.sort_values("admittime")
        if len(grp) <= 1:
            history_map[str(patient_id)] = []
            continue
        prior = grp.iloc[:-1]
        history_map[str(patient_id)] = [
            _admission_summary(row) for _, row in prior.iterrows()
        ]

    latest["admission_history"] = latest["patient_id"].astype(str).map(history_map)
    latest["n_prior_admissions"] = latest["admission_history"].apply(len)
    latest["n_total_admissions"] = latest["n_prior_admissions"] + 1
    latest["is_latest_admission"] = True
    latest = latest.reset_index(drop=True)
    return latest


def format_admission_history_text(history: List[Dict[str, Any]]) -> str:
    """Human-readable prior admission history for LLM prompts."""
    if not history:
        return "No prior admissions in cohort."

    lines = ["PRIOR ADMISSION HISTORY (oldest → most recent before current stay):", ""]
    for i, adm in enumerate(history, start=1):
        lines.append(f"--- Prior admission {i} | hadm_id={adm.get('hadm_id')} ---")
        lines.append(f"  Admit: {adm.get('admittime')} | Discharge: {adm.get('dischtime')}")
        lines.append(f"  Type: {adm.get('admission_type')}")
        lines.append(
            f"  Primary ICD-10: {adm.get('primary_icd_code')} — {adm.get('primary_dx_title')}"
        )
        codes = adm.get("ground_truth_icd10") or []
        if codes:
            lines.append(f"  All ICD-10 ({len(codes)}): {', '.join(codes[:8])}")
        excerpt = adm.get("note_excerpt")
        if excerpt:
            lines.append("  Note excerpt:")
            for line in str(excerpt).splitlines()[:12]:
                lines.append(f"    {line}")
        vitals = adm.get("vitals_summary") or []
        if vitals:
            lines.append(f"  Vitals ({len(vitals)}):")
            for v in vitals[:4]:
                if "last" in v:
                    lines.append(f"    • {v.get('name')}: last={v.get('last')} ({v.get('source', '')})")
                else:
                    lines.append(f"    • {v.get('name')}: {v.get('value')}")
        labs = adm.get("labs_summary") or []
        if labs:
            lines.append(f"  Key labs ({len(labs)}):")
            for lab in labs[:5]:
                flag = f" [{lab.get('flag')}]" if lab.get("flag") else ""
                lines.append(f"    • {lab.get('name')}: {lab.get('value')}{flag}")
        lines.append("")
    return "\n".join(lines)


def load_mimic_cohort(
    n_patients: int = N_PATIENTS,
    min_admissions: int = MIN_ADMISSIONS_PER_PATIENT,
    min_note_chars: int = MIN_NOTE_CHARS,
    seed: int = RANDOM_SEED,
    max_note_chars: int = MAX_NOTE_CHARS,
    latest_note_only: bool = True,
) -> pd.DataFrame:
    """
    Sample n MIMIC patients with multi-admission discharge notes and ICD-10 labels.

    When latest_note_only=True (default), returns one row per patient:
      - clinical_note from the **latest** admission
      - prior admissions in `admission_history` (metadata + note excerpts)
    """
    assert_mimic_paths()
    rng = random.Random(seed)

    patients = pd.read_csv(
        MIMIC_BASE / "hosp/patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age"],
    )
    admissions = pd.read_csv(
        MIMIC_BASE / "hosp/admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
    )
    for col in ("admittime", "dischtime"):
        admissions[col] = pd.to_datetime(admissions[col], errors="coerce")

    print("Loading discharge notes (this may take ~20s)...")
    notes = pd.read_csv(
        DISCHARGE_NOTES_PATH,
        usecols=["note_id", "subject_id", "hadm_id", "charttime", "text"],
    )
    notes = notes[notes["text"].notna()].copy()
    notes["text_len"] = notes["text"].str.len()
    notes = notes[notes["text_len"] >= min_note_chars]
    notes = notes.sort_values("text_len", ascending=False).drop_duplicates(
        ["subject_id", "hadm_id"], keep="first"
    )

    gt = load_icd10_ground_truth(set(notes["hadm_id"]))
    notes = notes.merge(gt, on="hadm_id", how="inner")

    adm_counts = notes.groupby("subject_id")["hadm_id"].nunique()
    eligible = adm_counts[adm_counts >= min_admissions].index.tolist()
    if len(eligible) < n_patients:
        raise ValueError(
            f"Only {len(eligible)} eligible patients with ICD-10 + notes found, "
            f"need {n_patients}."
        )

    picked_subjects = rng.sample(eligible, n_patients)
    cohort = notes[notes["subject_id"].isin(picked_subjects)].copy()
    cohort = cohort.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    cohort = cohort.merge(patients, on="subject_id", how="left")

    cohort["patient_id"] = cohort["subject_id"].astype(str)
    cohort["admission_id"] = cohort["hadm_id"].astype(str)
    cohort["clinical_note"] = cohort["text"].str.slice(0, max_note_chars)
    cohort["note_type"] = "discharge"
    cohort = cohort.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    cohort = enrich_cohort_structured_data(cohort)

    if latest_note_only:
        cohort = collapse_to_latest_admission(cohort)
        print(
            f"Patient-centric cohort: {cohort['patient_id'].nunique()} patients, "
            f"latest note only (avg {cohort['n_prior_admissions'].mean():.1f} prior admissions as history)"
        )
    return cohort


def save_cohort(
    cohort_df: pd.DataFrame,
    pickle_path: Union[str, Path] = COHORT_PICKLE,
    index_path: Union[str, Path] = COHORT_INDEX_JSON,
    seed: int = RANDOM_SEED,
) -> Path:
    """Persist cohort for downstream stage notebooks."""
    pickle_path = Path(pickle_path)
    index_path = Path(index_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)

    cohort_df.to_pickle(pickle_path)

    if "n_total_admissions" not in cohort_df.columns:
        counts = cohort_df.groupby("patient_id")["hadm_id"].transform("count")
        cohort_df = cohort_df.copy()
        cohort_df["n_total_admissions"] = counts
        cohort_df["n_prior_admissions"] = counts - 1

    patient_summary = (
        cohort_df.groupby("patient_id")
        .agg(
            subject_id=("subject_id", "first"),
            n_admissions=("n_total_admissions", "first"),
            n_prior_admissions=("n_prior_admissions", "first"),
            latest_hadm_id=("hadm_id", "first"),
            primary_codes=("primary_icd_code", "first"),
        )
        .reset_index()
    )

    index: Dict[str, Any] = {
        "stage": 1,
        "description": "MIMIC cohort — latest admission note + prior admission history",
        "cohort_mode": "latest_note_with_history",
        "test_mode": TEST_MODE,
        "generated_at": datetime.now().isoformat(),
        "random_seed": seed,
        "n_patients": int(cohort_df["patient_id"].nunique()),
        "n_index_admissions": int(len(cohort_df)),
        "avg_prior_admissions": float(cohort_df.get("n_prior_admissions", pd.Series([0])).mean()),
        "subject_ids": sorted(cohort_df["subject_id"].unique().tolist()),
        "patients": patient_summary.to_dict(orient="records"),
        "cohort_pickle": str(pickle_path),
    }
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return pickle_path


def load_cohort(
    pickle_path: Union[str, Path] = COHORT_PICKLE,
    index_path: Optional[Union[str, Path]] = COHORT_INDEX_JSON,
) -> pd.DataFrame:
    """Load cohort saved by Stage 1."""
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Cohort not found at {pickle_path}. Run notebooks/stage_01_cohort_selection.ipynb first."
        )
    cohort = pd.read_pickle(pickle_path)
    # Migrate legacy multi-row-per-patient pickles
    if "admission_history" not in cohort.columns and cohort["patient_id"].duplicated().any():
        cohort = collapse_to_latest_admission(cohort)
    if "clinical_context_text" not in cohort.columns:
        print(
            "WARNING: cohort.pkl has no structured vitals/labs/reports — "
            "re-run stage_01_cohort_selection.ipynb"
        )
    return cohort


def load_cohort_index(index_path: Union[str, Path] = COHORT_INDEX_JSON) -> Dict[str, Any]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Cohort index not found at {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


# =============================================================================
# Stage 1.5 — Redact diagnosis information from the latest clinical note
#
# Prevents the diagnosis stated in the discharge note itself (e.g. a
# "Discharge Diagnosis:" section, or "consistent with pneumonia" in the
# narrative) from leaking into what Stage 2 (IE) and Stage 3 (symptom tree)
# see, which would let the LLM "cheat" instead of reasoning from symptoms.
#
# Only the LATEST admission's clinical_note is redacted — prior-admission
# history already exposes its diagnosis explicitly via structured fields
# (primary_icd_code / primary_dx_title in admission_history), so redacting
# those note excerpts too would not hide anything additional.
#
# Two passes:
#   1. Deterministic section stripping (no LLM) — removes whole sections
#      whose header names this admission's diagnosis outright.
#   2. LLM pass over what's left — finds inline narrative diagnosis mentions
#      regex can't reliably catch (e.g. in Brief Hospital Course). The LLM
#      only returns verbatim spans to redact; code does the actual string
#      replacement, so it can't rewrite or drop unrelated text.
# =============================================================================

_DIAGNOSIS_SECTION_HEADERS = {
    "discharge diagnosis",
    "discharge diagnoses",
    "final diagnosis",
    "final diagnoses",
    "primary diagnosis",
    "secondary diagnosis",
    "secondary diagnoses",
    "admission diagnosis",
    "admission diagnoses",
}

_SECTION_HEADER_RE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9 /\-]{2,60}):[ \t]?", re.MULTILINE)

_DIAGNOSIS_REDACTION_TOKEN = "[DIAGNOSIS REDACTED]"


def redact_diagnosis_sections(note: str) -> Tuple[str, List[str]]:
    """Strip content of known diagnosis-revealing section headers. Deterministic, no LLM.

    Returns (redacted_note, redacted_section_names).
    """
    if not note:
        return note, []

    matches = list(_SECTION_HEADER_RE.finditer(note))
    if not matches:
        return note, []

    redacted_sections: List[str] = []
    pieces: List[str] = []
    cursor = 0
    for i, m in enumerate(matches):
        header_text = m.group(1).strip()
        section_start = m.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(note)

        pieces.append(note[cursor:section_start])
        if header_text.lower() in _DIAGNOSIS_SECTION_HEADERS:
            pieces.append(f"\n{_DIAGNOSIS_REDACTION_TOKEN}\n")
            redacted_sections.append(header_text)
        else:
            pieces.append(note[section_start:section_end])
        cursor = section_end

    pieces.append(note[cursor:])
    return "".join(pieces), redacted_sections


def _build_diagnosis_redaction_system_prompt() -> str:
    return f"""You are a clinical Diagnosis Redaction Agent.
You are given a clinical note (already stripped of explicit Discharge/Final/Primary/Admission
Diagnosis sections) that will be used to test a DIFFERENT system's ability to infer the diagnosis
from symptoms, vitals, labs, and clinical course alone. Find any remaining sentences or phrases
that STATE OR STRONGLY IMPLY a named diagnosis conclusion for THIS admission (e.g. "consistent
with pneumonia", "diagnosed with heart failure", "found to have a pulmonary embolism",
"impression: sepsis").

Do NOT flag: symptoms, vital signs, lab values, physical exam findings, imaging findings described
objectively (e.g. "bilateral infiltrates on CXR" is fine; "CXR consistent with pneumonia" is not),
medications, procedures, or past medical history unrelated to naming THIS admission's diagnosis.

Return ONLY valid JSON (no markdown):
{{
  "diagnosis_mentions": [
    {{"text": "exact verbatim substring from the note that states/implies the diagnosis", "diagnosis_hint": "the diagnosis name being revealed"}}
  ]
}}
Copy "text" EXACTLY as it appears in the note (same casing, punctuation, whitespace) — it will be
used for an exact string replacement, so paraphrasing will fail to match. If nothing else reveals
the diagnosis, return an empty list."""


def redact_diagnosis_mentions_llm(
    note: str,
    config: Optional[LLMConfig] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """LLM finds remaining inline diagnosis mentions; code does the exact-match replacement."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)

    result = call_llm_json(
        _build_diagnosis_redaction_system_prompt(),
        f"Clinical note:\n\n{note}",
        config,
        model=model,
    )

    redacted_note = note
    applied: List[Dict[str, str]] = []
    for mention in result.get("diagnosis_mentions") or []:
        text = mention.get("text", "")
        if text and text in redacted_note:
            redacted_note = redacted_note.replace(text, _DIAGNOSIS_REDACTION_TOKEN)
            applied.append(mention)
    return redacted_note, applied


def _redact_note_by_section(note: str, config: LLMConfig) -> Tuple[str, List[Dict[str, str]]]:
    """Run the LLM mention-finder per section rather than once on the whole note.

    A 7B model's recall drops sharply on long multi-section notes — verified empirically:
    an obvious inline mention ("found to have community-acquired pneumonia") was caught
    3/3 times when the LLM saw just that section, but 0/3 times when it saw the full note.
    Splitting by section keeps each LLM call's context short, which fixes recall, at the
    cost of one LLM call per section instead of one per note.
    """
    if not note:
        return note, []

    matches = list(_SECTION_HEADER_RE.finditer(note))
    all_mentions: List[Dict[str, str]] = []

    def _redact_chunk(text: str, header: Optional[str]) -> str:
        stripped = text.strip()
        if not stripped or stripped == _DIAGNOSIS_REDACTION_TOKEN:
            return text
        redacted_text, mentions = redact_diagnosis_mentions_llm(text, config=config)
        for mention in mentions:
            mention["section"] = header or "(preamble)"
        all_mentions.extend(mentions)
        return redacted_text

    if not matches:
        return _redact_chunk(note, None), all_mentions

    pieces: List[str] = []
    if matches[0].start() > 0:
        pieces.append(_redact_chunk(note[: matches[0].start()], None))

    for i, m in enumerate(matches):
        header_text = m.group(1).strip()
        section_start = m.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        pieces.append(note[m.start():section_start])  # header line itself, unchanged
        pieces.append(_redact_chunk(note[section_start:section_end], header_text))

    return "".join(pieces), all_mentions


_KEYWORD_STOPWORDS = {
    "of", "the", "and", "or", "with", "without", "unspecified", "other", "due",
    "to", "in", "a", "an", "type", "acute", "chronic", "disorder", "disease",
    "not", "elsewhere", "classified", "specified",
}
_KEYWORD_STEM_LEN = 6


def _extract_diagnosis_keywords(dx_titles: List[str]) -> List[str]:
    """Significant word stems from the admission's own ground-truth ICD-10 title(s).

    Also generates the adjective form for the Greek -sis/-tic suffix pair (sepsis→septic,
    cirrhosis→cirrhotic, necrosis→necrotic, thrombosis→thrombotic, psychosis→psychotic,
    ...) since prefix-stemming alone misses it — verified in practice: "septic shock"
    mentioned three times in a note whose primary diagnosis was "sepsis" was caught only
    once, because "sepsis" and "septic" diverge before the 6-character stem length.
    """
    stems: set = set()
    for title in dx_titles or []:
        for w in re.findall(r"[a-zA-Z]+", str(title).lower()):
            if len(w) <= 4 or w in _KEYWORD_STOPWORDS:
                continue
            stems.add(w[:_KEYWORD_STEM_LEN] if len(w) > _KEYWORD_STEM_LEN else w)
            if w.endswith("sis") and len(w) > 5:
                stems.add(w[:-3] + "tic")
    return sorted(stems, key=len, reverse=True)


def redact_known_diagnosis_terms(note: str, dx_titles: List[str]) -> Tuple[str, List[str]]:
    """Deterministic backstop pass: redact any remaining mention of a word stem drawn from
    the admission's OWN ground-truth diagnosis title(s). Ground truth is used here only to
    REMOVE it from the note, never to inform any prediction — this guards against the LLM
    passes missing a paraphrased or secondary mention (verified to happen in practice: e.g.
    "consistent with a right lower lobe pneumonic process" was missed by the LLM pass even
    when "found to have community-acquired pneumonia" in the same sentence was caught).
    Crude prefix-stemming means it isn't exhaustive either (e.g. "sepsis" won't catch
    "septic") — treat this as an additional layer, not a guarantee.
    """
    if not note or not dx_titles:
        return note, []
    keywords = _extract_diagnosis_keywords(dx_titles)
    if not keywords:
        return note, []
    redacted = note
    hit_terms: List[str] = []
    for stem in keywords:
        pattern = re.compile(rf"\b{re.escape(stem)}\w*", re.IGNORECASE)
        if pattern.search(redacted):
            hit_terms.append(stem)
            redacted = pattern.sub(_DIAGNOSIS_REDACTION_TOKEN, redacted)
    return redacted, hit_terms


def redact_diagnosis_agent(
    clinical_note: str,
    admission_id: str,
    patient_id: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    primary_dx_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Three-pass diagnosis redaction:
    1. Deterministic section stripping (Discharge/Final/Primary/Admission Diagnosis headers).
    2. Per-section LLM pass for inline narrative mentions the sections don't cover.
    3. Deterministic keyword backstop using ONLY the admission's PRIMARY diagnosis title, if
       supplied — catches mentions passes 1-2 miss. Deliberately NOT the full
       ground_truth_dx_titles list: that list is every coded diagnosis for the admission
       (comorbidities, complications, past conditions — often 20+ for a complex patient) and
       using all of them as redaction keywords was verified to gut legitimate Past Medical
       History / HPI content, not just the target diagnosis. Ground truth is used only to
       redact, never passed to any downstream prediction stage.
    """
    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)

    section_redacted, redacted_sections = redact_diagnosis_sections(clinical_note or "")
    llm_redacted_note, llm_mentions = _redact_note_by_section(section_redacted, config)
    dx_titles = [primary_dx_title] if primary_dx_title else []
    final_note, keyword_hits = redact_known_diagnosis_terms(llm_redacted_note, dx_titles)

    return {
        "type": "diagnosis_redaction",
        "original_note": clinical_note,
        "redacted_note": final_note,
        "redacted_sections": redacted_sections,
        "llm_redacted_mentions": llm_mentions,
        "keyword_backstop_hits": keyword_hits,
        "n_sections_redacted": len(redacted_sections),
        "n_llm_mentions_redacted": len(llm_mentions),
        "n_keyword_backstop_hits": len(keyword_hits),
        "original_char_count": len(clinical_note or ""),
        "redacted_char_count": len(final_note or ""),
        "_method": f"sections+{config.method_prefix()}_llm+keyword_backstop:{model}",
        "_agent": "diagnosis_redaction",
        "patient_id": patient_id,
        "admission_id": admission_id,
        "generated_at": datetime.now().isoformat(),
    }


def load_redaction_checkpoint(
    path: Union[str, Path] = REDACTION_CHECKPOINT_JSON,
) -> Dict[str, Dict[str, Any]]:
    """Return completed redactions keyed by patient_id (for resume after a network error)."""
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["patient_id"]): r for r in payload.get("results", [])}


def save_redaction_checkpoint(
    records: List[Dict[str, Any]],
    path: Union[str, Path] = REDACTION_CHECKPOINT_JSON,
) -> Path:
    """Save progress after each patient — this stage makes several LLM calls per note
    (one per section), so a mid-run network error should not lose completed work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "1.5",
        "checkpoint": True,
        "updated_at": datetime.now().isoformat(),
        "n_completed": len(records),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_redaction_results(
    results_df: pd.DataFrame,
    dir_path: Union[str, Path] = STAGE_01B_DIR,
) -> Path:
    """Write one JSON file per patient, plus an index summarizing all of them."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for stale in dir_path.glob("redacted_note_*.json"):
        stale.unlink()

    index_records = []
    for record in results_df.to_dict(orient="records"):
        patient_id = str(record["patient_id"])
        file_name = f"redacted_note_{patient_id}.json"
        (dir_path / file_name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        index_records.append({
            "patient_id": patient_id,
            "admission_id": record.get("admission_id"),
            "hadm_id": record.get("hadm_id"),
            "file": file_name,
            "n_sections_redacted": record.get("n_sections_redacted"),
            "n_llm_mentions_redacted": record.get("n_llm_mentions_redacted"),
        })

    index_payload = {
        "stage": "1.5",
        "description": "Diagnosis redaction — latest clinical note only (one file per patient)",
        "generated_at": datetime.now().isoformat(),
        "n_patients": len(index_records),
        "patients": index_records,
    }
    (dir_path / "redacted_notes_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dir_path


def load_redaction_results(
    dir_path: Union[str, Path] = STAGE_01B_DIR,
) -> pd.DataFrame:
    dir_path = Path(dir_path)
    index_path = dir_path / "redacted_notes_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Redacted notes index not found at {index_path}. "
            "Run notebooks/stage_01b_redact_notes.ipynb first."
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = [
        json.loads((dir_path / entry["file"]).read_text(encoding="utf-8"))
        for entry in index_payload.get("patients", [])
    ]
    return pd.DataFrame(records)


import json
from datetime import datetime
from typing import Any, Dict, List, Optional


IE_SYSTEM_PROMPT = """You are a clinical Information Extraction Agent (NLP).
Read the CURRENT admission clinical note AND structured MIMIC data (vitals, labs, radiology). Prior admissions are HISTORY only.
Prefer structured vitals/labs over note text when they conflict. Tag prior-admission findings as status "history".

Return ONLY valid JSON (no markdown) with this schema:
{
  "symptoms": [{"term": "", "status": "present|absent|history", "evidence": "verbatim phrase from note"}],
  "vitals": [{"name": "", "value": "", "unit": ""}],
  "labs": [{"name": "", "value": "", "unit": "", "flag": "high|low|normal|unknown"}],
  "diagnoses_mentioned": [{"term": "", "certainty": "confirmed|suspected|rule_out"}],
  "medications": [{"name": "", "dose": "", "route": "", "status": "started|continued|stopped"}],
  "procedures": [{"name": "", "result": ""}],
  "negations": [""],
  "temporal": [{"finding": "", "onset": ""}]
}
Use standard clinical terminology. Copy evidence verbatim from the note."""

SYMPTOM_TREE_SYSTEM_PROMPT = """You are a clinical Symptom Tree Agent.
Given a clinical note, structured MIMIC vitals/labs/reports, and information extraction, build a hierarchical symptom tree
for ontology routing (Infectious, Cardiovascular, Respiratory, etc.).

Return ONLY valid JSON (no markdown):
{
  "root": "ClinicalPresentation",
  "reasoning": "1-2 sentence summary of dominant clinical picture",
  "branches": [
    {
      "category": "constitutional|respiratory|cardiovascular|infectious|neurologic|gi|renal|other",
      "ontology_hint": "Infectious Diseases|Cardiovascular|Respiratory|Other",
      "symptoms": [
        {
          "term": "standardized symptom name",
          "status": "present|absent|history",
          "severity": "mild|moderate|severe|unknown",
          "evidence": "verbatim phrase",
          "related_findings": ["supporting lab/vital/diagnosis"],
          "children": [
            {"term": "more specific sub-symptom", "status": "present", "evidence": ""}
          ]
        }
      ]
    }
  ],
  "key_symptoms": ["fever", "cough"],
  "red_flags": ["hypotension", "AMS"]
}
Group related symptoms under the correct branch. Use the extraction JSON as a guide but reason over the full note."""


def require_ollama(config: LLMConfig) -> str:
    """Backward-compatible alias."""
    return require_llm(config)


def call_ollama_json(
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Backward-compatible alias."""
    return call_llm_json(system_prompt, user_prompt, config, model=model)


def information_extraction_agent(
    clinical_note: str,
    config: Optional[LLMConfig] = None,
    admission_history: Optional[List[Dict[str, Any]]] = None,
    history_text: Optional[str] = None,
    clinical_context_text: Optional[str] = None,
) -> Dict[str, Any]:
    """NLP information extraction via configured LLM."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)
    note = truncate_note(clinical_note, config.max_note_chars)

    history_block = ""
    if history_text:
        history_block = f"\n\n{history_text}\n"
    elif admission_history:
        history_block = f"\n\n{format_admission_history_text(admission_history)}\n"

    context_block = f"\n\n{clinical_context_text}\n" if clinical_context_text else ""
    user_prompt = (
        f"CURRENT ADMISSION — clinical note:\n\n{note}"
        f"{context_block}"
        f"{history_block}"
    )

    extracted = call_llm_json(IE_SYSTEM_PROMPT, user_prompt, config, model=model)
    extracted["_method"] = f"{config.method_prefix()}_nlp:{model}"
    extracted["_agent"] = "information_extraction"
    if admission_history is not None:
        extracted["_n_prior_admissions"] = len(admission_history)
    return extracted


def symptom_tree_agent(
    clinical_note: str,
    extracted: Dict[str, Any],
    admission_id: str,
    patient_id: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    admission_history: Optional[List[Dict[str, Any]]] = None,
    history_text: Optional[str] = None,
    clinical_context_text: Optional[str] = None,
) -> Dict[str, Any]:
    """LLM-built hierarchical symptom tree from note + information extraction."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)
    note = truncate_note(clinical_note, config.max_note_chars)

    extraction_for_prompt = {
        k: v for k, v in extracted.items() if not str(k).startswith("_")
    }

    history_block = ""
    if history_text:
        history_block = f"\n\nPrior admission history:\n{history_text}"
    elif admission_history:
        history_block = f"\n\n{format_admission_history_text(admission_history)}"

    context_block = f"\n\n{clinical_context_text}\n" if clinical_context_text else ""
    user_prompt = (
        f"Admission ID: {admission_id}\n"
        f"Patient ID: {patient_id or 'unknown'}\n\n"
        f"CURRENT admission clinical note:\n{note}\n"
        f"{context_block}"
        f"{history_block}\n\n"
        f"Information extraction JSON (current admission):\n"
        f"{json.dumps(extraction_for_prompt, indent=2)}"
    )

    tree = call_llm_json(SYMPTOM_TREE_SYSTEM_PROMPT, user_prompt, config, model=model)
    tree["type"] = "symptom_tree"
    tree["_method"] = f"{config.method_prefix()}_llm:{model}"
    tree["_agent"] = "symptom_tree"
    tree["patient_id"] = patient_id
    tree["admission_id"] = admission_id
    tree["generated_at"] = datetime.now().isoformat()
    return tree


def aggregate_patient_symptom_tree_llm(
    admission_trees: List[Dict[str, Any]],
    patient_id: str,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """LLM merges admission-level symptom trees into a patient-level view."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)

    system = """You are a clinical Symptom Tree Agent.
Merge multiple admission-level symptom trees for ONE patient into a single patient-level tree.
Track which symptoms recurred across admissions. Return ONLY valid JSON:
{
  "root": "PatientClinicalHistory",
  "reasoning": "summary across admissions",
  "n_admissions": 0,
  "branches": [...same branch schema as admission trees...],
  "recurrent_symptoms": [{"term": "", "admissions": ["hadm_id1", "hadm_id2"]}],
  "key_symptoms": []
}"""

    user_prompt = json.dumps(
        {
            "patient_id": patient_id,
            "admission_trees": [
                {k: v for k, v in t.items() if not str(k).startswith("_")}
                for t in admission_trees
            ],
        },
        indent=2,
    )
    tree = call_llm_json(system, user_prompt, config, model=model)
    tree["type"] = "symptom_tree_aggregate"
    tree["_method"] = f"{config.method_prefix()}_llm:{model}"
    tree["_agent"] = "symptom_tree_aggregate"
    tree["patient_id"] = patient_id
    tree["generated_at"] = datetime.now().isoformat()
    tree["n_admissions"] = len(admission_trees)
    return tree


# =============================================================================
# Ontology Routing Agent — route symptom tree branches to candidate diseases
# =============================================================================

ONTOLOGY_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "infectious": {
        "ontology_name": "Infectious Diseases",
        "diseases": [
            "Pneumonia",
            "Sepsis",
            "Urinary Tract Infection",
            "Cellulitis / Skin and Soft Tissue Infection",
            "Bacteremia",
            "Clostridioides difficile Infection",
            "Tuberculosis",
            "Viral Infection (e.g. COVID-19, Influenza)",
        ],
    },
    "cardiovascular": {
        "ontology_name": "Cardiovascular",
        "diseases": [
            "Heart Failure",
            "Myocardial Infarction / Acute Coronary Syndrome",
            "Atrial Fibrillation / Arrhythmia",
            "Hypertensive Emergency",
            "Cardiogenic Shock",
            "Pericarditis / Pericardial Effusion",
        ],
    },
    "respiratory": {
        "ontology_name": "Respiratory",
        "diseases": [
            "COPD Exacerbation",
            "Asthma Exacerbation",
            "Pulmonary Embolism",
            "Acute Respiratory Failure / ARDS",
            "Pleural Effusion",
            "Pneumothorax",
        ],
    },
    "neurologic": {
        "ontology_name": "Neurologic",
        "diseases": [
            "Ischemic / Hemorrhagic Stroke",
            "Altered Mental Status / Encephalopathy",
            "Seizure Disorder",
            "Delirium",
        ],
    },
    "gi": {
        "ontology_name": "Gastrointestinal",
        "diseases": [
            "Gastrointestinal Bleed",
            "Acute Pancreatitis",
            "Bowel Obstruction",
            "Cirrhosis / Hepatic Decompensation",
            "Cholangitis / Cholecystitis",
        ],
    },
    "renal": {
        "ontology_name": "Renal",
        "diseases": [
            "Acute Kidney Injury",
            "Chronic Kidney Disease Exacerbation",
            "Electrolyte Disturbance",
            "Volume Overload",
        ],
    },
    "constitutional": {
        "ontology_name": "Constitutional / General",
        "diseases": [
            "Fever of Unknown Origin",
            "Dehydration",
            "Malignancy-related Symptoms",
            "Failure to Thrive",
        ],
    },
    "other": {
        "ontology_name": "Other",
        "diseases": [],
    },
}


def format_ontology_taxonomy_text() -> str:
    lines = [
        "ONTOLOGY TAXONOMY (route candidates only within these categories/diseases; "
        "return zero candidates for a branch if nothing fits):",
        "",
    ]
    for category, info in ONTOLOGY_TAXONOMY.items():
        lines.append(f"- {category} → {info['ontology_name']}")
        for disease in info["diseases"]:
            lines.append(f"    • {disease}")
    return "\n".join(lines)


def _build_ontology_routing_system_prompt() -> str:
    return f"""You are a clinical Ontology Routing Agent.
Given a patient's symptom tree (branches grouped by clinical category), route each branch to candidate diseases
drawn ONLY from the fixed ontology taxonomy below. Do not invent diseases outside this taxonomy; if nothing in a
branch's category fits well, return an empty candidates list for that branch.

{format_ontology_taxonomy_text()}

Return ONLY valid JSON (no markdown):
{{
  "reasoning": "1-2 sentence summary of overall routing rationale",
  "routed_branches": [
    {{
      "category": "infectious|cardiovascular|respiratory|neurologic|gi|renal|constitutional|other",
      "ontology_name": "exact ontology_name from the taxonomy for this category",
      "candidates": [
        {{
          "disease": "exact disease name from the taxonomy list",
          "confidence": "low|medium|high",
          "supporting_symptoms": ["symptom terms from this branch that support this disease"],
          "rationale": "brief reason referencing symptoms/severity/red flags"
        }}
      ]
    }}
  ]
}}
Base confidence on symptom severity, evidence strength, and red flags in the tree. A branch may route to zero,
one, or multiple candidate diseases."""


def ontology_routing_agent(
    symptom_tree: Dict[str, Any],
    admission_id: str,
    patient_id: Optional[str] = None,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """LLM routes symptom tree branches to candidate diseases in the fixed ontology taxonomy."""
    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)

    tree_for_prompt = {k: v for k, v in symptom_tree.items() if not str(k).startswith("_")}
    user_prompt = (
        f"Admission ID: {admission_id}\n"
        f"Patient ID: {patient_id or 'unknown'}\n\n"
        f"Symptom tree JSON:\n{json.dumps(tree_for_prompt, indent=2)}"
    )

    routing = call_llm_json(_build_ontology_routing_system_prompt(), user_prompt, config, model=model)
    routing["type"] = "ontology_routing"

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    flattened = [
        {
            "disease": candidate.get("disease"),
            "category": branch.get("category"),
            "ontology_name": branch.get("ontology_name"),
            "confidence": candidate.get("confidence"),
            "supporting_symptoms": candidate.get("supporting_symptoms", []),
            "rationale": candidate.get("rationale", ""),
        }
        for branch in routing.get("routed_branches", [])
        for candidate in branch.get("candidates", [])
    ]
    flattened.sort(key=lambda c: confidence_rank.get(str(c.get("confidence")).lower(), 3))
    routing["top_candidates"] = flattened

    routing["_method"] = f"{config.method_prefix()}_llm:{model}"
    routing["_agent"] = "ontology_routing"
    routing["patient_id"] = patient_id
    routing["admission_id"] = admission_id
    routing["generated_at"] = datetime.now().isoformat()
    return routing


# =============================================================================
# Evidence Scoring Agent — score ontology-routed candidates against evidence
# =============================================================================

def _build_evidence_scoring_system_prompt() -> str:
    return """You are a clinical Evidence Scoring Agent.
Given a set of candidate diagnoses (produced by an Ontology Routing Agent) plus the full symptom tree and
information extraction for one admission, score how strongly the DOCUMENTED evidence supports each candidate.

Return ONLY valid JSON (no markdown):
{
  "reasoning": "1-2 sentence summary of the overall evidence picture",
  "scored_candidates": [
    {
      "disease": "exact disease name, copied from the candidates provided",
      "category": "category, copied from the candidate provided",
      "score": 0-100,
      "supporting_evidence": ["specific symptoms/vitals/labs/findings that support this diagnosis"],
      "contradicting_evidence": ["documented findings that argue against this diagnosis, if any"],
      "missing_evidence": ["key confirmatory evidence that is not documented, if relevant"],
      "rationale": "brief justification for the score"
    }
  ]
}
Score guide: 0-20 minimal support, 21-40 weak, 41-60 moderate, 61-80 strong, 81-100 very strong/near-definitive.
Weigh objective vitals/labs more heavily than subjective complaints alone. Consider severity and red flags noted
in the symptom tree. Score every candidate you were given — do not add new candidates or drop any."""


def _candidate_score(candidate: Dict[str, Any]) -> float:
    try:
        return float(candidate.get("score", 0))
    except (TypeError, ValueError):
        return 0.0


def evidence_scoring_agent(
    ontology_routing: Dict[str, Any],
    symptom_tree: Dict[str, Any],
    extracted: Dict[str, Any],
    admission_id: str,
    patient_id: Optional[str] = None,
    config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """LLM scores each ontology-routed candidate diagnosis against the documented evidence."""
    config = config or LLMConfig()
    model = config.model

    candidates_for_prompt = [
        {k: v for k, v in c.items() if not str(k).startswith("_")}
        for c in (ontology_routing.get("top_candidates") or [])
    ]

    if not candidates_for_prompt:
        return {
            "type": "evidence_scoring",
            "reasoning": "No candidates were routed for this admission — nothing to score.",
            "scored_candidates": [],
            "_method": "none:no_candidates",
            "_agent": "evidence_scoring",
            "patient_id": patient_id,
            "admission_id": admission_id,
            "generated_at": datetime.now().isoformat(),
        }

    require_llm(config)
    warn_if_slow_model(model, config.provider)

    tree_for_prompt = {k: v for k, v in symptom_tree.items() if not str(k).startswith("_")}
    extracted_for_prompt = {k: v for k, v in extracted.items() if not str(k).startswith("_")}

    user_prompt = (
        f"Admission ID: {admission_id}\n"
        f"Patient ID: {patient_id or 'unknown'}\n\n"
        f"Candidate diagnoses to score (from ontology routing):\n"
        f"{json.dumps(candidates_for_prompt, indent=2)}\n\n"
        f"Symptom tree JSON:\n{json.dumps(tree_for_prompt, indent=2)}\n\n"
        f"Information extraction JSON:\n{json.dumps(extracted_for_prompt, indent=2)}"
    )

    scoring = call_llm_json(_build_evidence_scoring_system_prompt(), user_prompt, config, model=model)
    scoring["type"] = "evidence_scoring"
    scoring["scored_candidates"] = sorted(
        scoring.get("scored_candidates") or [], key=_candidate_score, reverse=True
    )

    scoring["_method"] = f"{config.method_prefix()}_llm:{model}"
    scoring["_agent"] = "evidence_scoring"
    scoring["patient_id"] = patient_id
    scoring["admission_id"] = admission_id
    scoring["generated_at"] = datetime.now().isoformat()
    return scoring


# =============================================================================
# Branch Pruning — deterministic cutoff on Stage 6 evidence scores
# =============================================================================

def prune_low_likelihood_branches(
    evidence_scoring: Dict[str, Any],
    admission_id: str,
    patient_id: Optional[str] = None,
    score_threshold: float = PRUNE_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    """Drop candidates whose Stage 6 evidence score is below score_threshold. No LLM call."""
    scored_candidates = evidence_scoring.get("scored_candidates") or []
    kept = [c for c in scored_candidates if _candidate_score(c) >= score_threshold]
    pruned = [c for c in scored_candidates if _candidate_score(c) < score_threshold]

    return {
        "type": "branch_pruning",
        "score_threshold": score_threshold,
        "n_input_candidates": len(scored_candidates),
        "n_kept": len(kept),
        "n_pruned": len(pruned),
        "kept_candidates": kept,
        "pruned_candidates": pruned,
        "_method": f"threshold:{score_threshold}",
        "_agent": "branch_pruning",
        "patient_id": patient_id,
        "admission_id": admission_id,
        "generated_at": datetime.now().isoformat(),
    }


# =============================================================================
# Guideline & ICD-10 Retrieval Agent
#
# ICD-10 codes are grounded: matched against MIMIC's real ICD-10-CM dictionary
# (same d_icd_diagnoses.csv.gz used for ground truth in Stage 1), never invented.
# Guideline citations are a curated static reference table, not LLM free recall —
# curated citations are best-effort and should be spot-checked by a clinician
# before any real use; some diseases have no single dedicated society guideline
# and are noted as such rather than assigned a citation that overstates precision.
# =============================================================================

GUIDELINE_REFERENCE_TAXONOMY: Dict[str, Dict[str, str]] = {
    # Infectious
    "Pneumonia": {
        "organization": "ATS/IDSA", "year": "2019",
        "citation": "Metlay JP et al. Diagnosis and Treatment of Adults with Community-Acquired "
                     "Pneumonia. Am J Respir Crit Care Med. 2019.",
    },
    "Sepsis": {
        "organization": "SCCM/ESICM", "year": "2021",
        "citation": "Evans L et al. Surviving Sepsis Campaign: International Guidelines for "
                     "Management of Sepsis and Septic Shock 2021. Crit Care Med. 2021.",
    },
    "Urinary Tract Infection": {
        "organization": "IDSA/ESCMID", "year": "2011",
        "citation": "Gupta K et al. International Clinical Practice Guidelines for the Treatment "
                     "of Acute Uncomplicated Cystitis and Pyelonephritis in Women. Clin Infect Dis. 2011.",
    },
    "Cellulitis / Skin and Soft Tissue Infection": {
        "organization": "IDSA", "year": "2014",
        "citation": "Stevens DL et al. Practice Guidelines for the Diagnosis and Management of "
                     "Skin and Soft Tissue Infections. Clin Infect Dis. 2014.",
    },
    "Bacteremia": {
        "organization": "SCCM/ESICM (when septic)", "year": "2021",
        "citation": "No single dedicated bacteremia guideline; managed per Surviving Sepsis "
                     "Campaign 2021 when associated with sepsis, or source-specific IDSA "
                     "guidelines (e.g. catheter-related, endocarditis) depending on source.",
    },
    "Clostridioides difficile Infection": {
        "organization": "IDSA/SHEA", "year": "2021",
        "citation": "Johnson S et al. Clinical Practice Guideline by the IDSA and SHEA: 2021 "
                     "Focused Update Guidelines on Management of Clostridioides difficile "
                     "Infection in Adults. Clin Infect Dis. 2021.",
    },
    "Tuberculosis": {
        "organization": "ATS/CDC/IDSA", "year": "2016",
        "citation": "Nahid P et al. Official ATS/CDC/IDSA Clinical Practice Guidelines: "
                     "Treatment of Drug-Susceptible Tuberculosis. Clin Infect Dis. 2016.",
    },
    "Viral Infection (e.g. COVID-19, Influenza)": {
        "organization": "NIH / IDSA", "year": "ongoing / 2019",
        "citation": "NIH COVID-19 Treatment Guidelines Panel (living guideline); Uyeki TM et al. "
                     "IDSA Clinical Practice Guidelines: Seasonal Influenza. Clin Infect Dis. 2019.",
    },
    # Cardiovascular
    "Heart Failure": {
        "organization": "AHA/ACC/HFSA", "year": "2022",
        "citation": "Heidenreich PA et al. 2022 AHA/ACC/HFSA Guideline for the Management of "
                     "Heart Failure. Circulation. 2022.",
    },
    "Myocardial Infarction / Acute Coronary Syndrome": {
        "organization": "ACC/AHA", "year": "2025",
        "citation": "2025 ACC/AHA/ACEP/NAEMSP/SCAI Guideline for the Management of Patients With "
                     "Acute Coronary Syndromes. J Am Coll Cardiol. 2025 (supersedes the separate "
                     "2013 STEMI and 2014 NSTE-ACS guidelines).",
    },
    "Atrial Fibrillation / Arrhythmia": {
        "organization": "ACC/AHA/ACCP/HRS", "year": "2023",
        "citation": "Joglar JA et al. 2023 ACC/AHA/ACCP/HRS Guideline for the Diagnosis and "
                     "Management of Atrial Fibrillation. Circulation. 2023.",
    },
    "Hypertensive Emergency": {
        "organization": "ACC/AHA", "year": "2017",
        "citation": "Whelton PK et al. 2017 ACC/AHA Guideline for the Prevention, Detection, "
                     "Evaluation, and Management of High Blood Pressure in Adults. Hypertension. 2018.",
    },
    "Cardiogenic Shock": {
        "organization": "SCAI", "year": "2022",
        "citation": "Naidu SS et al. SCAI SHOCK Stage Classification Expert Consensus Update. "
                     "J Am Coll Cardiol. 2022.",
    },
    "Pericarditis / Pericardial Effusion": {
        "organization": "ESC", "year": "2015",
        "citation": "Adler Y et al. 2015 ESC Guidelines for the Diagnosis and Management of "
                     "Pericardial Diseases. Eur Heart J. 2015.",
    },
    # Respiratory
    "COPD Exacerbation": {
        "organization": "GOLD", "year": "2024",
        "citation": "Global Initiative for Chronic Obstructive Lung Disease (GOLD). Global "
                     "Strategy for the Diagnosis, Management, and Prevention of COPD: 2024 Report.",
    },
    "Asthma Exacerbation": {
        "organization": "GINA", "year": "2024",
        "citation": "Global Initiative for Asthma (GINA). Global Strategy for Asthma Management "
                     "and Prevention: 2024 Update.",
    },
    "Pulmonary Embolism": {
        "organization": "ESC", "year": "2019",
        "citation": "Konstantinides SV et al. 2019 ESC Guidelines for the Diagnosis and "
                     "Management of Acute Pulmonary Embolism. Eur Heart J. 2019.",
    },
    "Acute Respiratory Failure / ARDS": {
        "organization": "ATS/ESICM/SCCM", "year": "2017",
        "citation": "Fan E et al. An Official ATS/ESICM/SCCM Clinical Practice Guideline: "
                     "Mechanical Ventilation in Adult Patients with ARDS. Am J Respir Crit Care Med. 2017.",
    },
    "Pleural Effusion": {
        "organization": "BTS", "year": "2023",
        "citation": "Roberts ME et al. British Thoracic Society Guideline for Pleural Disease. Thorax. 2023.",
    },
    "Pneumothorax": {
        "organization": "BTS", "year": "2023",
        "citation": "British Thoracic Society Guideline for Pleural Disease (covers pneumothorax "
                     "management). Thorax. 2023.",
    },
    # Neurologic
    "Ischemic / Hemorrhagic Stroke": {
        "organization": "AHA/ASA", "year": "2019 / 2022",
        "citation": "Powers WJ et al. 2019 AHA/ASA Guideline for the Early Management of Acute "
                     "Ischemic Stroke. Greenberg SM et al. 2022 AHA/ASA Guideline for the "
                     "Management of Spontaneous Intracerebral Hemorrhage. Stroke.",
    },
    "Altered Mental Status / Encephalopathy": {
        "organization": "SCCM", "year": "2018",
        "citation": "Devlin JW et al. Clinical Practice Guidelines for the Prevention and "
                     "Management of Pain, Agitation/Sedation, Delirium, Immobility, and Sleep "
                     "Disruption in Adult ICU Patients (PADIS). Crit Care Med. 2018.",
    },
    "Seizure Disorder": {
        "organization": "AES", "year": "2016",
        "citation": "Glauser T et al. Evidence-Based Guideline: Treatment of Convulsive Status "
                     "Epilepticus in Children and Adults. American Epilepsy Society. Epilepsy Curr. 2016.",
    },
    "Delirium": {
        "organization": "SCCM", "year": "2018",
        "citation": "Devlin JW et al. PADIS Guidelines (Pain, Agitation/Sedation, Delirium, "
                     "Immobility, Sleep). Crit Care Med. 2018.",
    },
    # GI
    "Gastrointestinal Bleed": {
        "organization": "ACG", "year": "2021",
        "citation": "Laine L et al. ACG Clinical Guideline: Upper Gastrointestinal and Ulcer "
                     "Bleeding. Am J Gastroenterol. 2021. (Lower GI bleeding: Sengupta N et al. "
                     "ACG Clinical Guideline, 2023.)",
    },
    "Acute Pancreatitis": {
        "organization": "ACG", "year": "2024",
        "citation": "ACG Clinical Guideline: Management of Acute Pancreatitis (Tenner S et al., "
                     "updated 2024).",
    },
    "Bowel Obstruction": {
        "organization": "WSES", "year": "2018",
        "citation": "Ten Broek RPG et al. Bologna Guidelines for Diagnosis and Management of "
                     "Adhesive Small Bowel Obstruction: 2017 Update. World Society of Emergency "
                     "Surgery. World J Emerg Surg. 2018.",
    },
    "Cirrhosis / Hepatic Decompensation": {
        "organization": "AASLD", "year": "2021",
        "citation": "Biggins SW et al. AASLD Practice Guidance: Diagnosis, Evaluation, and "
                     "Management of Ascites, Spontaneous Bacterial Peritonitis and Hepatorenal "
                     "Syndrome. Hepatology. 2021.",
    },
    "Cholangitis / Cholecystitis": {
        "organization": "Tokyo Guidelines", "year": "2018",
        "citation": "Kiriyama S et al. Tokyo Guidelines 2018 (TG18): Diagnostic Criteria and "
                     "Severity Grading of Acute Cholangitis and Cholecystitis. J Hepatobiliary "
                     "Pancreat Sci. 2018.",
    },
    # Renal
    "Acute Kidney Injury": {
        "organization": "KDIGO", "year": "2012",
        "citation": "KDIGO Clinical Practice Guideline for Acute Kidney Injury. Kidney Int Suppl. 2012.",
    },
    "Chronic Kidney Disease Exacerbation": {
        "organization": "KDIGO", "year": "2024",
        "citation": "KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of "
                     "Chronic Kidney Disease. Kidney Int. 2024.",
    },
    "Electrolyte Disturbance": {
        "organization": "ESE/ESICM/ERA-EDTA", "year": "2014",
        "citation": "Spasovski G et al. Clinical Practice Guideline on Diagnosis and Treatment of "
                     "Hyponatraemia. Eur J Endocrinol. 2014. (Other electrolyte disorders lack a "
                     "single unifying guideline; managed per disorder-specific society statements.)",
    },
    "Volume Overload": {
        "organization": "AHA/ACC/HFSA", "year": "2022",
        "citation": "Managed per congestion/decongestion recommendations in the 2022 AHA/ACC/HFSA "
                     "Heart Failure Guideline (no single dedicated volume-overload guideline).",
    },
    # Constitutional
    "Fever of Unknown Origin": {
        "organization": "General / consensus", "year": "2007",
        "citation": "Bleeker-Rovers CP et al. A Prospective Multicenter Study on Fever of Unknown "
                     "Origin: Diagnostic Procedures and Outcome. Medicine (Baltimore). 2007. "
                     "(No single society guideline; systematic diagnostic protocols are used.)",
    },
    "Dehydration": {
        "organization": "General supportive care", "year": "n/a",
        "citation": "No single major society guideline for adult dehydration; managed per general "
                     "fluid resuscitation principles (e.g. Surviving Sepsis Campaign fluid "
                     "resuscitation recommendations if hypovolemic/septic).",
    },
    "Malignancy-related Symptoms": {
        "organization": "NCCN", "year": "ongoing",
        "citation": "National Comprehensive Cancer Network (NCCN) Guidelines for Supportive Care "
                     "(e.g. Cancer-Related Fatigue, Palliative Care) — cancer type and symptom specific.",
    },
    "Failure to Thrive": {
        "organization": "General / geriatric consensus", "year": "n/a",
        "citation": "No single major society guideline; evaluated per general geriatric "
                     "failure-to-thrive workup principles.",
    },
}


_ICD10_DICT_CACHE: Optional[pd.DataFrame] = None

_ICD10_SEARCH_STOPWORDS = {
    "of", "the", "and", "or", "with", "without", "unspecified", "other", "due",
    "to", "in", "a", "an", "type", "acute", "chronic", "disorder", "disease",
}


def load_icd10_dictionary() -> pd.DataFrame:
    """All ICD-10-CM diagnosis codes + titles from MIMIC's d_icd_diagnoses.csv.gz (cached)."""
    global _ICD10_DICT_CACHE
    if _ICD10_DICT_CACHE is not None:
        return _ICD10_DICT_CACHE
    path = MIMIC_BASE / "hosp/d_icd_diagnoses.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"ICD-10 dictionary not found at {path}. Check MIMIC_BASE / PHYSIONET_ROOT in "
            "notebooks/00_settings.ipynb — Stage 8 needs local MIMIC-IV access."
        )
    icd = pd.read_csv(path, usecols=["icd_code", "icd_version", "long_title"])
    icd = icd[icd["icd_version"] == 10].drop_duplicates("icd_code").reset_index(drop=True)
    icd["_title_lower"] = icd["long_title"].str.lower()
    _ICD10_DICT_CACHE = icd
    return icd


def search_icd10_candidates(
    disease_name: str,
    icd_dict: pd.DataFrame,
    top_n: int = 8,
) -> List[Dict[str, Any]]:
    """Keyword-overlap search over real ICD-10-CM titles. Returns up to top_n candidates."""
    query_terms = [
        t for t in re.findall(r"[a-z]+", disease_name.lower())
        if t not in _ICD10_SEARCH_STOPWORDS and len(t) > 2
    ]
    if not query_terms:
        return []
    score = pd.Series(0, index=icd_dict.index)
    for term in query_terms:
        score = score + icd_dict["_title_lower"].str.contains(term, regex=False)
    matched = icd_dict.assign(_match_score=score)
    matched = matched[matched["_match_score"] > 0].sort_values("_match_score", ascending=False).head(top_n)
    return [
        {"icd_code": r["icd_code"], "long_title": r["long_title"], "match_score": int(r["_match_score"])}
        for _, r in matched.iterrows()
    ]


def _build_guideline_icd_retrieval_system_prompt() -> str:
    return """You are a clinical Guideline & ICD-10 Retrieval Agent.
For each candidate diagnosis (already evidence-scored and pruned), you are given:
- a shortlist of REAL candidate ICD-10-CM codes (from MIMIC's ICD-10 dictionary) — pick from this
  list only, never invent a code or title
- a curated clinical guideline citation for this diagnosis, if one exists
- the supporting/contradicting clinical evidence for this diagnosis at this admission

Return ONLY valid JSON (no markdown):
{
  "reasoning": "1-2 sentence summary",
  "retrieved": [
    {
      "disease": "exact disease name, copied from the candidate given",
      "icd10_codes": [
        {"code": "exact code copied from the provided shortlist", "title": "exact title copied from the shortlist", "why": "brief reason this code fits the evidence"}
      ],
      "guideline_relevance_note": "1-2 sentences connecting the provided guideline citation to this patient's specific evidence, or 'No curated guideline available for this diagnosis.' if none was provided"
    }
  ]
}
Pick 1-3 ICD-10 codes per candidate from ONLY that candidate's shortlist — if none of the
shortlisted codes fit well, return an empty icd10_codes list rather than guessing. Never invent
codes, titles, or guideline citations beyond what was given to you."""


def guideline_icd_retrieval_agent(
    kept_candidates: List[Dict[str, Any]],
    admission_id: str,
    patient_id: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    icd_dict: Optional[pd.DataFrame] = None,
    icd_shortlist_size: int = 8,
) -> Dict[str, Any]:
    """LLM retrieves ICD-10 codes (grounded in MIMIC's dictionary) and curated guideline notes."""
    if not kept_candidates:
        return {
            "type": "guideline_icd_retrieval",
            "reasoning": "No candidates survived pruning for this admission — nothing to retrieve.",
            "retrieved": [],
            "_method": "none:no_candidates",
            "_agent": "guideline_icd_retrieval",
            "patient_id": patient_id,
            "admission_id": admission_id,
            "generated_at": datetime.now().isoformat(),
        }

    config = config or LLMConfig()
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)

    icd_dict = icd_dict if icd_dict is not None else load_icd10_dictionary()

    candidates_for_prompt = []
    shortlist_by_disease: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in kept_candidates:
        disease = candidate.get("disease", "")
        shortlist = search_icd10_candidates(disease, icd_dict, top_n=icd_shortlist_size)
        shortlist_by_disease[disease] = shortlist
        candidates_for_prompt.append({
            "disease": disease,
            "category": candidate.get("category"),
            "evidence_score": candidate.get("score"),
            "supporting_evidence": candidate.get("supporting_evidence", []),
            "contradicting_evidence": candidate.get("contradicting_evidence", []),
            "icd10_shortlist": shortlist,
            "curated_guideline": GUIDELINE_REFERENCE_TAXONOMY.get(disease),
        })

    user_prompt = (
        f"Admission ID: {admission_id}\n"
        f"Patient ID: {patient_id or 'unknown'}\n\n"
        f"Candidates to retrieve for:\n{json.dumps(candidates_for_prompt, indent=2)}"
    )

    retrieval = call_llm_json(
        _build_guideline_icd_retrieval_system_prompt(), user_prompt, config, model=model
    )
    retrieval["type"] = "guideline_icd_retrieval"

    validated = []
    for item in retrieval.get("retrieved") or []:
        disease = item.get("disease", "")
        valid_codes = {c["icd_code"] for c in shortlist_by_disease.get(disease, [])}
        raw_codes = item.get("icd10_codes") or []
        codes = [c for c in raw_codes if c.get("code") in valid_codes]
        dropped = len(raw_codes) - len(codes)
        item["icd10_codes"] = codes
        if dropped:
            item["_n_hallucinated_codes_dropped"] = dropped
        validated.append(item)
    retrieval["retrieved"] = validated

    retrieval["_method"] = f"{config.method_prefix()}_llm:{model}"
    retrieval["_agent"] = "guideline_icd_retrieval"
    retrieval["patient_id"] = patient_id
    retrieval["admission_id"] = admission_id
    retrieval["generated_at"] = datetime.now().isoformat()
    return retrieval


import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd



def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_ie_results(
    results_df: pd.DataFrame,
    path: Union[str, Path] = IE_RESULTS_JSON,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    _ensure_parent(path)
    records = results_df.to_dict(orient="records")
    payload = {
        "stage": 2,
        "description": "Ollama information extraction",
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "results": records,
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_ie_results(path: Union[str, Path] = IE_RESULTS_JSON) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"IE results not found at {path}. Run notebooks/stage_02_information_extraction.ipynb first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["results"])


def load_ie_checkpoint(path: Union[str, Path] = IE_CHECKPOINT_JSON) -> Dict[int, Dict[str, Any]]:
    """Return completed extractions keyed by hadm_id (for resume)."""
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["hadm_id"]): r for r in payload.get("results", [])}


def save_ie_checkpoint(records: List[Dict[str, Any]], path: Union[str, Path] = IE_CHECKPOINT_JSON) -> Path:
    """Save progress after each admission so a timeout does not lose work."""
    path = Path(path)
    _ensure_parent(path)
    payload = {
        "stage": 2,
        "checkpoint": True,
        "updated_at": datetime.now().isoformat(),
        "n_completed": len(records),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_symptom_tree_results(
    results_df: pd.DataFrame,
    patient_symptom_trees: Dict[str, Dict[str, Any]],
    path: Union[str, Path] = SYMPTOM_TREE_RESULTS_JSON,
) -> Path:
    path = Path(path)
    _ensure_parent(path)
    records = results_df.to_dict(orient="records")
    payload = {
        "stage": 3,
        "description": "Ollama symptom tree (admission + patient aggregate)",
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "n_patients": len(patient_symptom_trees),
        "results": records,
        "patient_symptom_trees": patient_symptom_trees,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_symptom_tree_results(
    path: Union[str, Path] = SYMPTOM_TREE_RESULTS_JSON,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Symptom tree results not found at {path}. Run notebooks/stage_03_symptom_tree.ipynb first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["results"]), payload.get("patient_symptom_trees", {})


def save_ontology_routing_results(
    results_df: pd.DataFrame,
    dir_path: Union[str, Path] = STAGE_05_DIR,
) -> Path:
    """Write one JSON file per patient, plus an index summarizing all of them."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for stale in dir_path.glob("ontology_routing_*.json"):
        stale.unlink()

    index_records = []
    for record in results_df.to_dict(orient="records"):
        patient_id = str(record["patient_id"])
        file_name = f"ontology_routing_{patient_id}.json"
        (dir_path / file_name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        index_records.append({
            "patient_id": patient_id,
            "admission_id": record.get("admission_id"),
            "hadm_id": record.get("hadm_id"),
            "file": file_name,
            "n_candidates": record.get("n_candidates"),
            "top_candidate": record.get("top_candidate"),
        })

    index_payload = {
        "stage": 5,
        "description": "Ontology routing — symptom tree branches routed to candidate diseases (one file per patient)",
        "generated_at": datetime.now().isoformat(),
        "n_patients": len(index_records),
        "patients": index_records,
    }
    (dir_path / "ontology_routing_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dir_path


def load_ontology_routing_results(
    dir_path: Union[str, Path] = STAGE_05_DIR,
) -> pd.DataFrame:
    dir_path = Path(dir_path)
    index_path = dir_path / "ontology_routing_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Ontology routing index not found at {index_path}. "
            "Run notebooks/stage_05_ontology_routing.ipynb first."
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = [
        json.loads((dir_path / entry["file"]).read_text(encoding="utf-8"))
        for entry in index_payload.get("patients", [])
    ]
    return pd.DataFrame(records)


def save_evidence_scoring_results(
    results_df: pd.DataFrame,
    dir_path: Union[str, Path] = STAGE_06_DIR,
) -> Path:
    """Write one JSON file per patient, plus an index summarizing all of them."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for stale in dir_path.glob("evidence_scoring_*.json"):
        stale.unlink()

    index_records = []
    for record in results_df.to_dict(orient="records"):
        patient_id = str(record["patient_id"])
        file_name = f"evidence_scoring_{patient_id}.json"
        (dir_path / file_name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        index_records.append({
            "patient_id": patient_id,
            "admission_id": record.get("admission_id"),
            "hadm_id": record.get("hadm_id"),
            "file": file_name,
            "n_scored": record.get("n_scored"),
            "top_diagnosis": record.get("top_diagnosis"),
            "top_score": record.get("top_score"),
        })

    index_payload = {
        "stage": 6,
        "description": "Evidence scoring — candidate diagnoses scored against documented evidence (one file per patient)",
        "generated_at": datetime.now().isoformat(),
        "n_patients": len(index_records),
        "patients": index_records,
    }
    (dir_path / "evidence_scoring_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dir_path


def load_evidence_scoring_results(
    dir_path: Union[str, Path] = STAGE_06_DIR,
) -> pd.DataFrame:
    dir_path = Path(dir_path)
    index_path = dir_path / "evidence_scoring_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Evidence scoring index not found at {index_path}. "
            "Run notebooks/stage_06_evidence_scoring.ipynb first."
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = [
        json.loads((dir_path / entry["file"]).read_text(encoding="utf-8"))
        for entry in index_payload.get("patients", [])
    ]
    return pd.DataFrame(records)


def save_pruning_results(
    results_df: pd.DataFrame,
    dir_path: Union[str, Path] = STAGE_07_DIR,
) -> Path:
    """Write one JSON file per patient, plus an index summarizing all of them."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for stale in dir_path.glob("branch_pruning_*.json"):
        stale.unlink()

    index_records = []
    for record in results_df.to_dict(orient="records"):
        patient_id = str(record["patient_id"])
        file_name = f"branch_pruning_{patient_id}.json"
        (dir_path / file_name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        index_records.append({
            "patient_id": patient_id,
            "admission_id": record.get("admission_id"),
            "hadm_id": record.get("hadm_id"),
            "file": file_name,
            "n_kept": record.get("n_kept"),
            "n_pruned": record.get("n_pruned"),
        })

    index_payload = {
        "stage": 7,
        "description": "Branch pruning — deterministic cutoff on Stage 6 evidence scores (one file per patient)",
        "generated_at": datetime.now().isoformat(),
        "score_threshold": PRUNE_SCORE_THRESHOLD,
        "n_patients": len(index_records),
        "patients": index_records,
    }
    (dir_path / "branch_pruning_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dir_path


def load_pruning_results(
    dir_path: Union[str, Path] = STAGE_07_DIR,
) -> pd.DataFrame:
    dir_path = Path(dir_path)
    index_path = dir_path / "branch_pruning_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Branch pruning index not found at {index_path}. "
            "Run notebooks/stage_07_prune_branches.ipynb first."
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = [
        json.loads((dir_path / entry["file"]).read_text(encoding="utf-8"))
        for entry in index_payload.get("patients", [])
    ]
    return pd.DataFrame(records)


def save_retrieval_results(
    results_df: pd.DataFrame,
    dir_path: Union[str, Path] = STAGE_08_DIR,
) -> Path:
    """Write one JSON file per patient, plus an index summarizing all of them."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    for stale in dir_path.glob("guideline_icd_retrieval_*.json"):
        stale.unlink()

    index_records = []
    for record in results_df.to_dict(orient="records"):
        patient_id = str(record["patient_id"])
        file_name = f"guideline_icd_retrieval_{patient_id}.json"
        (dir_path / file_name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        index_records.append({
            "patient_id": patient_id,
            "admission_id": record.get("admission_id"),
            "hadm_id": record.get("hadm_id"),
            "file": file_name,
            "n_retrieved": record.get("n_retrieved"),
        })

    index_payload = {
        "stage": 8,
        "description": "Guideline & ICD-10 retrieval — grounded ICD-10 codes + curated guideline "
                        "relevance notes for kept candidates (one file per patient)",
        "generated_at": datetime.now().isoformat(),
        "n_patients": len(index_records),
        "patients": index_records,
    }
    (dir_path / "guideline_icd_retrieval_index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dir_path


def load_retrieval_results(
    dir_path: Union[str, Path] = STAGE_08_DIR,
) -> pd.DataFrame:
    dir_path = Path(dir_path)
    index_path = dir_path / "guideline_icd_retrieval_index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Guideline/ICD retrieval index not found at {index_path}. "
            "Run notebooks/stage_08_guideline_icd_retrieval.ipynb first."
        )
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = [
        json.loads((dir_path / entry["file"]).read_text(encoding="utf-8"))
        for entry in index_payload.get("patients", [])
    ]
    return pd.DataFrame(records)


import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _format_section(title: str, char: str = "=") -> str:
    line = char * len(title)
    return f"{title}\n{line}\n"


def format_ground_truth_txt(row: pd.Series) -> str:
    lines = [
        _format_section("GROUND TRUTH DIAGNOSES"),
        f"Admission ID     : {row.get('admission_id', row.get('hadm_id'))}",
        f"Primary ICD-10   : {row.get('primary_icd_code', row.get('icd_code', 'N/A'))}",
        f"Primary Title    : {row.get('primary_dx_title', row.get('long_title', 'N/A'))}",
        f"Total diagnoses  : {row.get('n_diagnoses', 'N/A')}",
        "",
        "All ICD-10 codes (ordered):",
        "-" * 40,
    ]
    codes = row.get("ground_truth_icd10") or []
    titles = row.get("ground_truth_dx_titles") or []
    for i, code in enumerate(codes):
        title = titles[i] if i < len(titles) else ""
        lines.append(f"  {i+1:2d}. {code} — {title}")
    return "\n".join(lines) + "\n"


def format_extraction_txt(extracted: Dict[str, Any], meta: Dict[str, Any]) -> str:
    lines = [
        _format_section("INFORMATION EXTRACTION"),
        f"Patient ID       : {meta.get('patient_id')}",
        f"Admission ID     : {meta.get('admission_id')}",
        f"Method           : {meta.get('extraction_method', 'unknown')}",
        "",
    ]

    def block(name: str, items: List[Any], formatter) -> None:
        lines.append(_format_section(name, "-"))
        if not items:
            lines.append("  (none)\n")
            return
        for item in items:
            lines.append(formatter(item))
        lines.append("")

    block("SYMPTOMS", extracted.get("symptoms", []),
          lambda s: f"  • [{s.get('status')}] {s.get('term')} — \"{s.get('evidence', '')}\"")
    block("VITALS", extracted.get("vitals", []),
          lambda v: f"  • {v.get('name')}: {v.get('value')} {v.get('unit', '')}".strip())
    block("LABS", extracted.get("labs", []),
          lambda l: f"  • {l.get('name')}: {l.get('value')} {l.get('unit', '')} [{l.get('flag', '')}]".strip())
    block("DIAGNOSES MENTIONED", extracted.get("diagnoses_mentioned", []),
          lambda d: f"  • [{d.get('certainty')}] {d.get('term')}")
    block("MEDICATIONS", extracted.get("medications", []),
          lambda m: f"  • {m.get('name')} ({m.get('status')})")
    block("PROCEDURES", extracted.get("procedures", []),
          lambda p: f"  • {p.get('name')}: {p.get('result', '')}")
    block("NEGATIONS", extracted.get("negations", []),
          lambda n: f"  • {n}")
    block("TEMPORAL", extracted.get("temporal", []),
          lambda t: f"  • {t.get('finding')}: {t.get('onset')}")

    return "\n".join(lines)


def _format_symptom_node_lines(node: Dict[str, Any], indent: int = 2) -> List[str]:
    lines: List[str] = []
    prefix = " " * indent
    term = node.get("term", "")
    status = node.get("status", "")
    severity = node.get("severity", "")
    evidence = node.get("evidence", "")
    related = node.get("related_findings") or []

    status_label = status
    if severity and severity != "unknown":
        status_label = f"{status}, {severity}"
    lines.append(f"{prefix}• [{status_label}] {term}")
    if evidence:
        lines.append(f"{prefix}    evidence: \"{evidence}\"")
    if related:
        lines.append(f"{prefix}    related: {', '.join(str(r) for r in related)}")
    for child in node.get("children") or []:
        lines.extend(_format_symptom_node_lines(child, indent + 4))
    return lines


def format_symptom_tree_txt(tree: Dict[str, Any]) -> str:
    lines = [
        _format_section("SYMPTOM TREE"),
        f"Type             : {tree.get('type')}",
        f"Method           : {tree.get('_method', 'unknown')}",
        f"Root             : {tree.get('root', 'N/A')}",
        f"Patient ID       : {tree.get('patient_id', 'N/A')}",
        f"Admission ID     : {tree.get('admission_id', 'N/A (aggregate)')}",
    ]
    if tree.get("n_admissions"):
        lines.append(f"Admissions       : {tree.get('n_admissions')}")
    if tree.get("reasoning"):
        lines.extend(["", "Reasoning:", f"  {tree['reasoning']}", ""])

    key_symptoms = tree.get("key_symptoms") or []
    if key_symptoms:
        lines.append(f"Key symptoms     : {', '.join(key_symptoms)}")
    red_flags = tree.get("red_flags") or []
    if red_flags:
        lines.append(f"Red flags        : {', '.join(red_flags)}")
    recurrent = tree.get("recurrent_symptoms") or []
    if recurrent:
        lines.append("")
        lines.append(_format_section("RECURRENT SYMPTOMS", "-"))
        for item in recurrent:
            adms = ", ".join(str(a) for a in item.get("admissions", []))
            lines.append(f"  • {item.get('term')} [admissions: {adms}]")

    branches = tree.get("branches") or []
    if branches:
        lines.append("")
        for branch in branches:
            cat = branch.get("category", "other")
            hint = branch.get("ontology_hint", "")
            lines.append(_format_section(f"{cat.upper()} → {hint}", "-"))
            for symptom in branch.get("symptoms") or []:
                lines.extend(_format_symptom_node_lines(symptom, indent=2))
            lines.append("")

    return "\n".join(lines)


def format_patient_summary_txt(patient_id: str, cohort_rows: pd.DataFrame, n_admissions: int) -> str:
    first = cohort_rows.iloc[0]
    lines = [
        _format_section("PATIENT SUMMARY"),
        f"Patient ID       : {patient_id}",
        f"Subject ID       : {first.get('subject_id', patient_id)}",
        f"Gender           : {first.get('gender', 'N/A')}",
        f"Age (anchor)     : {first.get('anchor_age', 'N/A')}",
        f"Admissions       : {n_admissions}",
        "",
        "Admission history:",
        "-" * 40,
    ]
    for _, row in cohort_rows.iterrows():
        lines.append(
            f"  • hadm {row['hadm_id']} | {row.get('admittime', '')} | "
            f"{row.get('admission_type', '')} | primary: {row.get('primary_icd_code', row.get('icd_code', 'N/A'))}"
        )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_cohort_to_folders(
    cohort_df: pd.DataFrame,
    results_df: pd.DataFrame,
    output_dir: Path | str = "patient_records",
    patient_symptom_trees: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Export all patients to organized folders with txt + json artifacts.
    Returns the output root path.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    cohort_by_patient = {pid: grp for pid, grp in cohort_df.groupby("patient_id")}
    results_by_patient = {pid: grp for pid, grp in results_df.groupby("patient_id")}

    index: List[Dict[str, Any]] = []

    for patient_id, patient_cohort in cohort_by_patient.items():
        patient_dir = root / f"patient_{patient_id}"
        admissions_dir = patient_dir / "admissions"
        admissions_dir.mkdir(parents=True, exist_ok=True)

        patient_results = results_by_patient.get(patient_id)
        if patient_results is None:
            continue

        # Patient-centric: one latest admission row per patient
        cohort_row = patient_cohort.iloc[0]
        result_row = patient_results.iloc[0]
        hadm_id = str(cohort_row["hadm_id"])
        adm_dir = admissions_dir / f"hadm_{hadm_id}"
        adm_dir.mkdir(parents=True, exist_ok=True)

        extracted = result_row["extracted"]
        tree = result_row.get("symptom_tree")
        if tree is None or (isinstance(tree, float) and pd.isna(tree)):
            raise ValueError(
                f"Missing symptom_tree for patient_id={patient_id}. "
                "Run stages 2–3 before export."
            )

        admission_history = cohort_row.get("admission_history") or []
        if isinstance(admission_history, float) and pd.isna(admission_history):
            admission_history = []

        # Prior admission history (metadata only)
        if admission_history:
            _write_json(patient_dir / "admission_history.json", admission_history)
            _write_text(
                patient_dir / "admission_history.txt",
                format_admission_history_text(admission_history),
            )

        # Clinical note (latest admission)
        _write_text(
            adm_dir / "clinical_note.txt",
            cohort_row.get("clinical_note", cohort_row.get("text", "")),
        )

        ctx = cohort_row.get("clinical_context_text") or ""
        if ctx:
            _write_text(adm_dir / "clinical_context.txt", ctx)
        vitals = cohort_row.get("structured_vitals") or []
        labs = cohort_row.get("structured_labs") or []
        reports = cohort_row.get("structured_reports") or []
        if vitals:
            _write_json(adm_dir / "structured_vitals.json", vitals)
        if labs:
            _write_json(adm_dir / "structured_labs.json", labs)
        if reports:
            _write_json(adm_dir / "radiology_reports.json", reports)

        metadata = {
            "patient_id": patient_id,
            "subject_id": int(cohort_row["subject_id"]),
            "hadm_id": int(cohort_row["hadm_id"]),
            "admission_id": str(cohort_row["admission_id"]),
            "admittime": str(cohort_row.get("admittime", "")),
            "dischtime": str(cohort_row.get("dischtime", "")),
            "admission_type": cohort_row.get("admission_type"),
            "gender": cohort_row.get("gender"),
            "anchor_age": int(cohort_row["anchor_age"]) if pd.notna(cohort_row.get("anchor_age")) else None,
            "note_type": cohort_row.get("note_type", "discharge"),
            "text_len": int(cohort_row.get("text_len", 0)),
            "n_structured_vitals": len(vitals),
            "n_structured_labs": len(labs),
            "n_radiology_reports": len(reports),
            "is_latest_admission": True,
            "n_prior_admissions": int(cohort_row.get("n_prior_admissions", 0)),
            "n_total_admissions": int(cohort_row.get("n_total_admissions", 1)),
            "ground_truth": {
                "primary_icd_code": cohort_row.get("primary_icd_code", cohort_row.get("icd_code")),
                "primary_dx_title": cohort_row.get("primary_dx_title", cohort_row.get("long_title")),
                "icd10_codes": cohort_row.get("ground_truth_icd10", []),
                "dx_titles": cohort_row.get("ground_truth_dx_titles", []),
                "n_diagnoses": int(cohort_row.get("n_diagnoses", 0)) if pd.notna(cohort_row.get("n_diagnoses")) else None,
            },
        }
        _write_json(adm_dir / "metadata.json", metadata)
        _write_text(adm_dir / "ground_truth.txt", format_ground_truth_txt(cohort_row))

        ie_meta = {
            "patient_id": patient_id,
            "admission_id": hadm_id,
            "extraction_method": result_row.get("extraction_method"),
            "n_prior_admissions": metadata["n_prior_admissions"],
        }
        _write_json(adm_dir / "information_extraction.json", extracted)
        _write_text(adm_dir / "information_extraction.txt", format_extraction_txt(extracted, ie_meta))

        _write_json(adm_dir / "symptom_tree.json", tree)
        _write_text(adm_dir / "symptom_tree.txt", format_symptom_tree_txt(tree))

        branch_symptoms = sum(
            len(b.get("symptoms") or []) for b in (tree.get("branches") or [])
        )
        admission_index = [{
            "hadm_id": hadm_id,
            "admittime": str(cohort_row.get("admittime", "")),
            "primary_icd_code": metadata["ground_truth"]["primary_icd_code"],
            "primary_dx_title": metadata["ground_truth"]["primary_dx_title"],
            "symptom_count": branch_symptoms,
            "symptom_tree_method": tree.get("_method"),
            "is_latest": True,
        }]

        n_adm = int(cohort_row.get("n_total_admissions", 1))
        patient_summary = {
            "patient_id": patient_id,
            "subject_id": int(cohort_row["subject_id"]),
            "gender": cohort_row.get("gender"),
            "anchor_age": int(cohort_row["anchor_age"]) if pd.notna(cohort_row.get("anchor_age")) else None,
            "n_admissions": n_adm,
            "n_prior_admissions": metadata["n_prior_admissions"],
            "latest_hadm_id": hadm_id,
            "admissions": admission_index,
            "generated_at": datetime.now().isoformat(),
        }
        _write_json(patient_dir / "patient_summary.json", patient_summary)
        _write_text(
            patient_dir / "patient_summary.txt",
            format_patient_summary_txt(patient_id, patient_cohort, n_adm),
        )

        patient_tree = (patient_symptom_trees or {}).get(patient_id) or tree
        _write_json(patient_dir / "symptom_tree.json", patient_tree)
        _write_text(patient_dir / "symptom_tree.txt", format_symptom_tree_txt(patient_tree))

        index.append({
            "patient_id": patient_id,
            "folder": str(patient_dir.relative_to(root)),
            "n_admissions": n_adm,
            "n_prior_admissions": metadata["n_prior_admissions"],
            "latest_hadm_id": hadm_id,
            "admissions": admission_index,
        })

    cohort_index = {
        "generated_at": datetime.now().isoformat(),
        "cohort_mode": "latest_note_with_history",
        "n_patients": len(index),
        "n_latest_admissions": int(len(results_df)),
        "patients": index,
    }
    _write_json(root / "cohort_index.json", cohort_index)

    readme = f"""Clinical Coding Pipeline — Patient Records Export
Generated: {cohort_index['generated_at']}

{len(index)} patients | latest admission note + prior history
Agents: LLM information extraction + symptom tree

Folder layout per patient:
  patient_<subject_id>/
    patient_summary.txt / .json
    admission_history.txt / .json   (prior admissions)
    symptom_tree.txt / .json
    admissions/
      hadm_<latest_id>/
        clinical_note.txt
        clinical_context.txt
        structured_vitals.json
        structured_labs.json
        radiology_reports.json
        metadata.json
        ground_truth.txt
        information_extraction.txt / .json
        symptom_tree.txt / .json
"""
    _write_text(root / "README.txt", readme)

    return root
