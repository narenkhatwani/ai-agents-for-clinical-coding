"""
Clinical coding pipeline — single module used by stage notebooks.

Edit settings in notebooks/00_settings.ipynb (writes settings.json) or change defaults below.
"""

from __future__ import annotations

import csv
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
HISTORY_NOTE_EXCERPT_CHARS = 800  # legacy short excerpt
HISTORY_CLINICAL_DETAIL_CHARS = 3500  # prior admission rich history for LLM context

# Backward-compatible aliases
OLLAMA_TIMEOUT_SECONDS = LLM_TIMEOUT_SECONDS
OLLAMA_MAX_RETRIES = LLM_MAX_RETRIES

# Pipeline artifacts (written by stage notebooks)
DATA_DIR = REPO_ROOT / "data" / "test" if TEST_MODE else REPO_ROOT / "data"
COHORT_DIR = DATA_DIR / "cohort"
STAGE_02_DIR = DATA_DIR / "stage_02_information_extraction"
STAGE_03_DIR = DATA_DIR / "stage_03_symptom_tree"
STAGE_07_DIR = DATA_DIR / "stage_07_differential_diagnosis"
STAGE_08_DIR = DATA_DIR / "stage_08_icd_coding"
STAGE_09_DIR = DATA_DIR / "stage_09_icd_confirmation"
STAGE_10_DIR = DATA_DIR / "stage_10_evaluation"
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
        g["STAGE_02_DIR"] = g["DATA_DIR"] / "stage_02_information_extraction"
        g["STAGE_03_DIR"] = g["DATA_DIR"] / "stage_03_symptom_tree"
        g["STAGE_07_DIR"] = g["DATA_DIR"] / "stage_07_differential_diagnosis"
        g["STAGE_08_DIR"] = g["DATA_DIR"] / "stage_08_icd_coding"
        g["STAGE_09_DIR"] = g["DATA_DIR"] / "stage_09_icd_confirmation"
        g["STAGE_10_DIR"] = g["DATA_DIR"] / "stage_10_evaluation"
        g["EXPORT_DIR"] = g["REPO_ROOT"] / "patient_records_test" if tm else g["REPO_ROOT"] / "patient_records"


def _refresh_artifact_paths() -> None:
    global COHORT_PICKLE, COHORT_INDEX_JSON, IE_RESULTS_JSON, IE_CHECKPOINT_JSON
    global SYMPTOM_TREE_RESULTS_JSON, DIFF_DX_RESULTS_JSON, DIFF_DX_CHECKPOINT_JSON
    global ICD_CODING_RESULTS_JSON
    global ICD_CONFIRM_RESULTS_JSON, ICD_CONFIRM_CHECKPOINT_JSON
    global EVAL_SUMMARY_JSON, EVAL_COHORT_TXT, EVAL_PRIMARY_CSV, EVAL_DIFFDX_CSV, EVAL_ICD_CSV
    COHORT_PICKLE = COHORT_DIR / "cohort.pkl"
    COHORT_INDEX_JSON = COHORT_DIR / "cohort_index.json"
    IE_RESULTS_JSON = STAGE_02_DIR / "information_extractions.json"
    IE_CHECKPOINT_JSON = STAGE_02_DIR / "ie_checkpoint.json"
    SYMPTOM_TREE_RESULTS_JSON = STAGE_03_DIR / "symptom_tree_results.json"
    DIFF_DX_RESULTS_JSON = STAGE_07_DIR / "differential_diagnoses.json"
    DIFF_DX_CHECKPOINT_JSON = STAGE_07_DIR / "diff_dx_checkpoint.json"
    ICD_CODING_RESULTS_JSON = STAGE_08_DIR / "icd_coding_results.json"
    ICD_CONFIRM_RESULTS_JSON = STAGE_09_DIR / "icd_confirmation_results.json"
    ICD_CONFIRM_CHECKPOINT_JSON = STAGE_09_DIR / "icd_confirmation_checkpoint.json"
    EVAL_SUMMARY_JSON = STAGE_10_DIR / "accuracy_summary.json"
    EVAL_COHORT_TXT = STAGE_10_DIR / "cohort_metrics.txt"
    EVAL_DIFFDX_CSV = STAGE_10_DIR / "primary_diffdx_match.csv"
    EVAL_ICD_CSV = STAGE_10_DIR / "primary_icd_match.csv"
    EVAL_PRIMARY_CSV = EVAL_DIFFDX_CSV



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


def _repair_truncated_json(text: str) -> Optional[str]:
    """Best-effort close of truncated JSON objects/arrays (common when max_tokens hits)."""
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]
    # Drop trailing incomplete string after last quote imbalance
    in_string = False
    escape = False
    stack: List[str] = []
    last_good = 0
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            last_good = i
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
                last_good = i
            else:
                break
        elif ch in ",:" and not stack:
            break
        else:
            last_good = i

    if in_string:
        # Close the open string and trim after last complete value if possible
        s = s + '"'
    # Remove trailing comma / incomplete key
    s = re.sub(r",\s*$", "", s.rstrip())
    s = re.sub(r",\s*\"[^\"]*$", "", s)  # dangling key
    s = re.sub(r":\s*$", ": null", s)
    while True:
        # Recompute stack after cleanup
        in_string = False
        escape = False
        stack = []
        for ch in s:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]" and stack and ch == stack[-1]:
                stack.pop()
        if in_string:
            s += '"'
            continue
        if not stack:
            break
        s += "".join(reversed(stack))
        break
    return s


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_json_wrappers(text)
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

    repaired = _repair_truncated_json(cleaned)
    if repaired:
        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                data["_json_repaired"] = True
                return data
        except json.JSONDecodeError:
            pass
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
    if config.api_label == "openrouter":
        provider_prefs: Dict[str, Any] = {}
        if config.openrouter_zdr:
            provider_prefs["zdr"] = True
        # Prefer Phala (higher completion cap) over Together (~2048) for ZDR Qwen 7B
        provider_prefs["order"] = ["Phala", "Together"]
        provider_prefs["allow_fallbacks"] = True
        payload["provider"] = provider_prefs

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
        choice = choices[0]
        content = (choice.get("message") or {}).get("content", "")
        finish = choice.get("finish_reason") or choice.get("native_finish_reason")
        if finish in ("length", "max_tokens"):
            print(
                f"  Warning: LLM output truncated (finish_reason={finish}). "
                "Will try to repair JSON or retry."
            )
        return content

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
    compact_hint = (
        "\n\nIMPORTANT: Return COMPLETE valid JSON only. "
        "Keep evidence phrases SHORT (≤15 words). Prefer ≤12 items per list."
    )
    prompts = [user_prompt, user_prompt + compact_hint]
    last_raw = ""
    for attempt, prompt in enumerate(prompts[: max(config.max_retries, 1) + 1]):
        raw = call_llm_chat(system_prompt, prompt, config, model=model, json_mode=True)
        last_raw = raw or ""
        parsed = parse_json_object(last_raw)
        if parsed is not None:
            if parsed.pop("_json_repaired", False):
                print("  Note: repaired truncated JSON from LLM output")
            return parsed
        if attempt < max(config.max_retries, 1):
            print(f"  JSON parse failed — retrying with compact prompt ({attempt + 1})...")
            time.sleep(2)
    raise ValueError(
        f"Could not parse JSON from {config.provider} LLM. Preview: {last_raw[:300]}"
    )


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


# =============================================================================
# Clinical note sections — redaction (latest stay) & rich prior history
# =============================================================================

_NOTE_SECTION_HEADERS = [
    ("chief_complaint", r"Chief Complaint\s*:"),
    ("hpi", r"History of Present Illness\s*:"),
    ("past_medical_history", r"Past Medical History\s*:"),
    ("hospital_course", r"(?:Brief )?Hospital Course\s*:"),
    ("discharge_diagnosis", r"Discharge Diagnos(?:is|es)\s*:"),
    ("discharge_condition", r"Discharge Condition\s*:"),
    ("discharge_instructions", r"Discharge Instructions\s*:"),
    ("followup", r"Follow(?:\-|\s)?up Instructions\s*:"),
    ("physical_exam", r"Physical Exam(?:ination)?\s*:"),
    ("pertinent_results", r"Pertinent Results\s*:"),
]


def _header_spans(note: str) -> List[Tuple[str, int, int]]:
    """Return (section_key, start, end) for known section headers in note."""
    spans: List[Tuple[str, int, int]] = []
    for key, pattern in _NOTE_SECTION_HEADERS:
        for m in re.finditer(pattern, note, flags=re.IGNORECASE):
            spans.append((key, m.start(), m.end()))
    spans.sort(key=lambda x: x[1])
    return spans


def extract_note_sections(note: str) -> Dict[str, str]:
    """Parse common MIMIC discharge note sections."""
    note = note or ""
    if not note.strip():
        return {}
    spans = _header_spans(note)
    if not spans:
        return {}
    sections: Dict[str, str] = {}
    for i, (key, _start, end) in enumerate(spans):
        next_start = spans[i + 1][1] if i + 1 < len(spans) else len(note)
        body = note[end:next_start].strip()
        if body and key not in sections:
            sections[key] = body
    return sections


def _truncate_block(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


# Headers that start the terminal discharge package (and similar label leaks).
# When any of these appears (typically after Hospital Course), we cut from the
# earliest match through end-of-note for the coding input.
_DISCHARGE_PACKAGE_HEADERS = [
    r"Discharge Medications\s*:",
    r"Discharge Disposition\s*:",
    r"Discharge Diagnos(?:is|es)\s*:",
    r"Final Diagnos(?:is|es)\s*:",
    r"Primary Diagnos(?:is|es)\s*:",
    r"Secondary Diagnos(?:is|es)\s*:",
    r"Principal Diagnos(?:is|es)\s*:",
    r"Discharge Condition\s*:",
    r"Discharge Instructions\s*:",
    r"Follow(?:\-|\s)?up Instructions\s*:",
    r"Transitional Issues\s*:",
    r"Facility\s*:",
    r"Pending Results\s*:",
]

# Anywhere-in-note admitting labels (not only at end)
_ADMITTING_DX_HEADERS = [
    r"Admission Diagnos(?:is|es)\s*:",
    r"Admitting Diagnos(?:is|es)\s*:",
]

_DISCHARGE_PACKAGE_RE = re.compile(
    r"(?im)^\s*(?:" + "|".join(_DISCHARGE_PACKAGE_HEADERS) + r")"
)
_ADMITTING_DX_RE = re.compile(
    r"(?is)(\n\s*(?:Admission Diagnos(?:is|es)|Admitting Diagnos(?:is|es))\s*:"
    r".*?)(?=\n\s*[A-Z][A-Za-z0-9 \/\-]{2,60}\s*:|\Z)"
)
# Hospital Course problem-list titles: "# ASCITES." or "# SVT:"
_HC_PROBLEM_HEADER_RE = re.compile(
    r"(?m)^(\s*#\s+)([^:\n.]{1,80})([.:]\s*)"
)


def _scrub_hospital_course_problem_headers(note: str) -> Tuple[str, List[str]]:
    """Replace '# DIAGNOSIS TITLE:' headers with '# Problem:' — keep narrative body."""
    removed: List[str] = []

    def _repl(m: re.Match) -> str:
        title = m.group(2).strip()
        if title.lower() in ("problem", "problems", "issue", "issues"):
            return m.group(0)
        removed.append(f"# {title}{m.group(3).rstrip()}")
        return f"{m.group(1)}Problem{m.group(3)}"

    return _HC_PROBLEM_HEADER_RE.sub(_repl, note), removed


def redact_latest_note_for_coding(note: str) -> Tuple[str, str]:
    """
    Redact label-leaking sections from the latest admission discharge note.

    Removes:
      - Terminal discharge package (meds, disposition, diagnosis, condition,
        instructions, followup, transitional issues, facility, pending results)
      - Admission / Admitting Diagnosis blocks anywhere in the note
      - Hospital Course '# PROBLEM:' style titles (body kept)

    Returns (coding_note, redacted_text) for evaluation-only storage.
    """
    note = note or ""
    if not note.strip():
        return note, ""

    redacted_parts: List[str] = []
    coding = note

    # 1) Admission / Admitting Diagnosis anywhere
    def _admit_repl(match: re.Match) -> str:
        block = match.group(1).strip()
        if len(block) > 10:
            redacted_parts.append(block)
        return "\n[Admission Diagnosis REDACTED — see ground_truth.json]\n"

    coding = _ADMITTING_DX_RE.sub(_admit_repl, coding)

    # 2) Truncate from earliest terminal discharge-package header to EOF
    package_match = _DISCHARGE_PACKAGE_RE.search(coding)
    if package_match:
        cut = package_match.start()
        # Prefer cutting at a line boundary
        line_start = coding.rfind("\n", 0, cut)
        cut_at = line_start if line_start != -1 else cut
        removed = coding[cut_at:].strip()
        if removed:
            redacted_parts.append(removed)
        coding = (
            coding[:cut_at].rstrip()
            + "\n\n[DISCHARGE PACKAGE REDACTED — see ground_truth.json / "
            "redacted_discharge_sections.txt]\n"
        )

    # 3) Scrub Hospital Course problem-list titles
    coding, hc_titles = _scrub_hospital_course_problem_headers(coding)
    if hc_titles:
        redacted_parts.append(
            "Hospital Course problem titles scrubbed:\n  - " + "\n  - ".join(hc_titles)
        )

    coding = re.sub(r"\n{3,}", "\n\n", coding).strip()
    redacted_text = "\n\n---\n\n".join(redacted_parts)
    return coding, redacted_text


def build_prior_admission_clinical_detail(row: pd.Series) -> str:
    """
    Rich prior-stay context for the latest note's history block.

    Includes narrative (CC, HPI, hospital course) AND prior discharge diagnoses
    (allowed in history — only the latest stay labels are withheld).
    """
    note = str(row.get("text") or row.get("clinical_note") or "")
    sections = extract_note_sections(note)
    parts: List[str] = []

    cc = sections.get("chief_complaint")
    if cc:
        parts.append(f"Chief Complaint:\n{_truncate_block(cc, 400)}")

    hpi = sections.get("hpi")
    if hpi:
        parts.append(f"History of Present Illness:\n{_truncate_block(hpi, 1200)}")

    pmh = sections.get("past_medical_history")
    if pmh:
        parts.append(f"Past Medical History:\n{_truncate_block(pmh, 600)}")

    course = sections.get("hospital_course")
    if course:
        parts.append(f"Hospital Course:\n{_truncate_block(course, 1500)}")

    dx_note = sections.get("discharge_diagnosis")
    if dx_note:
        parts.append(f"Discharge Diagnosis (prior stay — from note):\n{_truncate_block(dx_note, 800)}")
    else:
        codes = row.get("ground_truth_icd10") or []
        titles = row.get("ground_truth_dx_titles") or []
        if codes:
            lines = ["Discharge Diagnosis (prior stay — from billing records):"]
            for i, code in enumerate(codes[:12]):
                title = titles[i] if i < len(titles) else ""
                lines.append(f"  • {code} — {title}")
            parts.append("\n".join(lines))

    detail = "\n\n".join(parts)
    if not detail and note:
        detail = _truncate_block(note, HISTORY_CLINICAL_DETAIL_CHARS)
    return _truncate_block(detail, HISTORY_CLINICAL_DETAIL_CHARS)


def apply_latest_note_redaction(cohort: pd.DataFrame) -> pd.DataFrame:
    """Set clinical_note (redacted) and clinical_note_full on latest-admission rows."""
    cohort = cohort.copy()
    full_notes: List[str] = []
    coding_notes: List[str] = []
    redacted_blocks: List[str] = []

    for _, row in cohort.iterrows():
        raw = str(row.get("text") or row.get("clinical_note") or "")
        full = raw[:MAX_NOTE_CHARS]
        coding, redacted = redact_latest_note_for_coding(raw)
        coding = coding[:MAX_NOTE_CHARS]
        full_notes.append(full)
        coding_notes.append(coding)
        redacted_blocks.append(redacted)

    cohort["clinical_note_full"] = full_notes
    cohort["clinical_note"] = coding_notes
    cohort["redacted_diagnosis_text"] = redacted_blocks
    n_redacted = sum(1 for r in redacted_blocks if r.strip())
    print(
        f"  Latest notes: {n_redacted}/{len(cohort)} had discharge-package / "
        "label sections redacted for coding"
    )
    return cohort


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
        summary["clinical_detail"] = build_prior_admission_clinical_detail(row)
        sections = extract_note_sections(str(note))
        if sections.get("discharge_diagnosis"):
            summary["discharge_diagnosis_text"] = sections["discharge_diagnosis"][:1200]
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
    latest = apply_latest_note_redaction(latest)
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
        titles = adm.get("ground_truth_dx_titles") or []
        if codes:
            lines.append(f"  Billing ICD-10 ({len(codes)}):")
            for j, code in enumerate(codes[:10]):
                title = titles[j] if j < len(titles) else ""
                lines.append(f"    • {code} — {title}")
        detail = adm.get("clinical_detail")
        if detail:
            lines.append("  Clinical history (prior stay — detailed):")
            for line in str(detail).splitlines():
                lines.append(f"    {line}")
        elif (excerpt := adm.get("note_excerpt")):
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
    if "clinical_note_full" not in cohort.columns:
        print("Migrating cohort: applying discharge-diagnosis redaction to latest notes...")
        cohort = apply_latest_note_redaction(cohort)
    return cohort


def load_cohort_index(index_path: Union[str, Path] = COHORT_INDEX_JSON) -> Dict[str, Any]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Cohort index not found at {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


import json
from datetime import datetime
from typing import Any, Dict, List, Optional


IE_SYSTEM_PROMPT = """You are a clinical Information Extraction Agent (NLP).
Read the CURRENT admission clinical note AND structured MIMIC data (vitals, labs, radiology). Prior admissions are HISTORY only.
Discharge package (diagnosis, instructions, meds, disposition, condition, followup,
transitional issues) and Hospital Course problem titles are REDACTED — do not infer
labels from placeholders. Prefer structured vitals/labs over note text when they conflict.
Tag prior-admission findings as status "history".

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
Use standard clinical terminology. Keep evidence SHORT (≤15 words). Return COMPLETE JSON only."""

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


# ---------------------------------------------------------------------------
# Stage 7 — Scored differential diagnosis
# ---------------------------------------------------------------------------
DIFF_DX_TEMPERATURE = 0.4

DIFF_DX_ROLE = """\
You are an experienced hospital clinician and clinical coding specialist. You formulate \
working differential diagnoses for the CURRENT (latest) admission to support clinical \
coding and documentation. You routinely integrate:
- hierarchical symptom trees for the current stay,
- retained SNOMED CT ontology routes (Is-a parents; outbound finding site, morphology, \
  due to / causative agent, pathological process, after, associated with, occurrence, \
  clinical course),
- structured vitals/labs for the current stay,
- PRIOR admission ICD-10 lists as known past medical history / comorbidity context.

You produce multiple scored differentials when the presentation is multi-factorial or uncertain."""

DIFF_DX_TASK = """\
Using only the CONTEXT below, produce a SCORED differential diagnosis for the CURRENT admission.

Multiple diagnoses are expected and welcome (primary, secondary, complications, \
and competing alternatives).

For each candidate diagnosis:
1. Assign a numeric likelihood score from 0–100 relative to the other candidates \
   (higher = more likely given the provided evidence; scores need not sum to 100).
2. Assign confidence: high | medium | low.
3. Assign category: primary | secondary | complication | risk_factor | differential.
4. Cite supporting evidence drawn from the symptom tree, retained SNOMED links, \
   and/or prior ICDs (label which source when relevant).
5. Note opposing evidence when the context weighs against the diagnosis.
6. Optionally align to a retained SNOMED term when it matches the diagnosis concept.

Also provide:
- summary: 2–4 sentence synthesis of the CURRENT presentation (with PMH only as context)
- most_likely: the top working diagnosis string for THIS admission
- rule_outs: considered but weakly supported or contradicted
- uncertain_areas: what missing data would change ranking
- Aim for 3–8 differentials when evidence allows (more is fine if justified)

Return COMPLETE valid JSON only (no markdown fences), matching the OUTPUT schema."""

DIFF_DX_CONSTRAINTS = """\
- Base every judgment solely on the CONTEXT provided. Do not invent symptoms, labs, \
  imaging, or history that are not present.
- PRIOR admission ICD-10 codes are known past diagnoses / comorbidities only. Use them \
  to inform risk, chronic conditions, and recurrence — do NOT copy them wholesale as the \
  only current-admission diagnoses unless the CURRENT context strongly supports them \
  for this stay.
- Do NOT use or guess CURRENT discharge diagnoses (they may be redacted). Current-stay \
  ground-truth ICD is never provided.
- Do not use private outside-case knowledge about this specific patient.
- Prefer diagnoses that could plausibly explain the CURRENT presentation for coding support.
- Keep evidence phrases short (≤20 words).
- Order the differential highest score → lowest score.
- If evidence is weak, still list candidates with lower scores and state that in reasoning.
- Evaluate candidates independently before ranking them."""

DIFF_DX_OUTPUT_SCHEMA = """\
{
  "summary": "2–4 sentence clinical synthesis of the current presentation",
  "most_likely": "primary working diagnosis string",
  "differential": [
    {
      "rank": 1,
      "diagnosis": "condition name",
      "score": 85,
      "confidence": "high|medium|low",
      "category": "primary|secondary|complication|risk_factor|differential",
      "snomed_aligned": "optional related SNOMED term from retained context, or empty string",
      "supporting_evidence": ["short bullet from tree / SNOMED / prior ICD"],
      "opposing_evidence": ["short bullet if any, else empty array"],
      "reasoning": "1–2 sentences why this scores as it does"
    }
  ],
  "rule_outs": [{"diagnosis": "", "why": ""}],
  "uncertain_areas": ["what would change ranking if known"],
  "n_candidates": 0
}"""

DIFF_DX_SYSTEM_PROMPT = (
    "You are a differential diagnosis agent for clinical coding support. "
    "Follow the ROLE, CONTEXT, TASK, and CONSTRAINTS in the user message exactly. "
    "Return complete valid JSON only."
)


def slim_symptom_tree_for_prompt(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Drop internal metadata so the prompt stays focused."""
    if not tree:
        return {}
    keep = {
        k: v
        for k, v in tree.items()
        if not str(k).startswith("_") and k not in ("generated_at",)
    }
    return keep


def slim_retained_for_prompt(retained_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact retained SNOMED context for the LLM."""
    entities = retained_payload.get("entities") or []
    out: List[Dict[str, Any]] = []
    for ent in entities:
        links = []
        for r in ent.get("retained") or []:
            links.append(
                {
                    "relation": r.get("relation"),
                    "term": r.get("term"),
                    "concept_id": r.get("concept_id"),
                    "similarity": r.get("cosine_similarity"),
                    "high_confidence": r.get("high_confidence"),
                }
            )
        out.append(
            {
                "entity": ent.get("term"),
                "kind": ent.get("kind"),
                "snomed": ent.get("snomed_preferred_term"),
                "snomed_id": ent.get("snomed_concept_id"),
                "retained_links": links,
            }
        )
    return out


def build_prior_icd_context(
    admission_history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Option B: structured prior-admission package for DiffDx.

    Includes ALL billed ICD-10 codes (+ titles) from every prior admission.
    Does not include current-stay ground truth.
    """
    if not admission_history:
        return []
    out: List[Dict[str, Any]] = []
    for adm in admission_history:
        if not isinstance(adm, dict):
            continue
        codes = list(adm.get("ground_truth_icd10") or [])
        titles = list(adm.get("ground_truth_dx_titles") or [])
        icd_list: List[Dict[str, str]] = []
        for j, code in enumerate(codes):
            title = titles[j] if j < len(titles) else ""
            icd_list.append(
                {
                    "code": str(code),
                    "title": str(title) if title is not None else "",
                }
            )
        # Ensure primary is listed even if missing from code list
        primary = adm.get("primary_icd_code")
        if primary and not any(x["code"] == str(primary) for x in icd_list):
            icd_list.insert(
                0,
                {
                    "code": str(primary),
                    "title": str(adm.get("primary_dx_title") or ""),
                },
            )
        out.append(
            {
                "hadm_id": adm.get("hadm_id") or adm.get("admission_id"),
                "admittime": adm.get("admittime"),
                "dischtime": adm.get("dischtime"),
                "admission_type": adm.get("admission_type"),
                "primary_icd_code": adm.get("primary_icd_code"),
                "primary_dx_title": adm.get("primary_dx_title"),
                "n_diagnoses": len(icd_list) or adm.get("n_diagnoses"),
                "icd10_diagnoses": icd_list,  # ALL prior ICDs
            }
        )
    return out


def format_prior_icd_context_text(prior_blocks: List[Dict[str, Any]]) -> str:
    """Human-readable prior ICD block for the prompt CONTEXT section."""
    if not prior_blocks:
        return (
            "No prior admissions in export. "
            "Do not invent past diagnoses."
        )
    lines = [
        "PRIOR ADMISSIONS — known past ICD-10 diagnoses only "
        "(NOT current discharge diagnoses; use as PMH / comorbidity / recurrence context).",
        "",
    ]
    for i, adm in enumerate(prior_blocks, start=1):
        lines.append(
            f"--- Prior admission {i} | hadm_id={adm.get('hadm_id')} ---"
        )
        lines.append(
            f"  Admit: {adm.get('admittime')} | Discharge: {adm.get('dischtime')}"
        )
        lines.append(f"  Type: {adm.get('admission_type')}")
        lines.append(
            f"  Primary: {adm.get('primary_icd_code')} — {adm.get('primary_dx_title')}"
        )
        icds = adm.get("icd10_diagnoses") or []
        lines.append(f"  All ICD-10 ({len(icds)}):")
        for row in icds:
            title = row.get("title") or ""
            if title:
                lines.append(f"    • {row.get('code')} — {title}")
            else:
                lines.append(f"    • {row.get('code')}")
        lines.append("")
    return "\n".join(lines)


def build_diff_dx_context_block(
    patient_id: str,
    hadm_id: str,
    tree_slim: Dict[str, Any],
    retained_slim: List[Dict[str, Any]],
    prior_icd_blocks: Optional[List[Dict[str, Any]]] = None,
    clinical_context_text: Optional[str] = None,
    ie_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Assemble the CONTEXT section for the structured DiffDx prompt."""
    prior_blocks = prior_icd_blocks or []
    parts: List[str] = [
        f"Patient ID: {patient_id}",
        f"Current admission ID (HADM): {hadm_id}",
        "",
        "Materials below come from the coding pipeline for this patient.",
        "CURRENT stay: symptom tree, retained SNOMED, optional context/IE.",
        "PRIOR stays: all billed ICD-10 codes only (no current-stay ground truth).",
        "",
        "--- PRIOR ADMISSION ICD-10 CONTEXT ---",
        format_prior_icd_context_text(prior_blocks),
        "",
        "--- CURRENT SYMPTOM TREE ---",
        json.dumps(tree_slim, indent=2, ensure_ascii=False),
        "",
        "--- RETAINED SNOMED CT ONTOLOGY CONTEXT (current entities) ---",
        "(MiniLM-filtered Is-a + outbound attributes for anatomic/process alignment.)",
        json.dumps(retained_slim, indent=2, ensure_ascii=False),
    ]

    if clinical_context_text:
        ctx = clinical_context_text.strip()
        if len(ctx) > 2500:
            ctx = ctx[:2500] + "\n...[truncated]"
        parts.extend(
            [
                "",
                "--- STRUCTURED CLINICAL CONTEXT (current stay vitals/labs excerpts) ---",
                ctx,
            ]
        )

    if ie_summary:
        keys = (
            "symptoms",
            "diagnoses_mentioned",
            "procedures",
            "medications",
            "labs",
            "temporal",
        )
        slim_ie = {k: ie_summary.get(k) for k in keys if ie_summary.get(k)}
        if slim_ie:
            parts.extend(
                [
                    "",
                    "--- INFORMATION EXTRACTION (current stay; may overlap symptom tree) ---",
                    json.dumps(slim_ie, indent=2, ensure_ascii=False),
                ]
            )

    return "\n".join(parts)


def build_diff_dx_user_prompt(context: str) -> str:
    """Full structured user prompt: ROLE / CONTEXT / TASK / CONSTRAINTS / OUTPUT."""
    return (
        f"ROLE:\n{DIFF_DX_ROLE}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"TASK:\n{DIFF_DX_TASK}\n\n"
        f"CONSTRAINTS:\n{DIFF_DX_CONSTRAINTS}\n\n"
        f"OUTPUT SCHEMA (return only this JSON object):\n{DIFF_DX_OUTPUT_SCHEMA}"
    )


def differential_diagnosis_agent(
    symptom_tree: Dict[str, Any],
    retained_snomed: Dict[str, Any],
    patient_id: str,
    hadm_id: str,
    clinical_context_text: Optional[str] = None,
    ie_summary: Optional[Dict[str, Any]] = None,
    admission_history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[LLMConfig] = None,
    temperature: float = DIFF_DX_TEMPERATURE,
) -> Dict[str, Any]:
    """
    LLM scored differential diagnosis.

    Inputs:
      - current symptom tree + retained SNOMED
      - optional current clinical context / IE
      - prior admission_history with ALL ICD-10 codes (Option B)
    Never uses current-stay ground-truth ICD / discharge package.
    Prompt format: ROLE / CONTEXT / TASK / CONSTRAINTS; default temperature=0.4.
    """
    from dataclasses import replace

    base = config or LLMConfig()
    config = replace(base, temperature=float(temperature))
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)

    tree_slim = slim_symptom_tree_for_prompt(symptom_tree)
    retained_slim = slim_retained_for_prompt(retained_snomed or {})
    prior_blocks = build_prior_icd_context(admission_history)

    context = build_diff_dx_context_block(
        patient_id=str(patient_id),
        hadm_id=str(hadm_id),
        tree_slim=tree_slim,
        retained_slim=retained_slim,
        prior_icd_blocks=prior_blocks,
        clinical_context_text=clinical_context_text,
        ie_summary=ie_summary,
    )
    user_prompt = build_diff_dx_user_prompt(context)

    result = call_llm_json(DIFF_DX_SYSTEM_PROMPT, user_prompt, config, model=model)

    # Normalize differential ranking/scores
    differentials = result.get("differential") or result.get("differentials") or []
    if not isinstance(differentials, list):
        differentials = []
    cleaned: List[Dict[str, Any]] = []
    for i, item in enumerate(differentials):
        if not isinstance(item, dict):
            continue
        score = item.get("score", item.get("confidence_score", 0))
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        if 0.0 < score_f <= 1.0:
            score_f = score_f * 100.0
        score_f = max(0.0, min(100.0, score_f))
        cleaned.append(
            {
                "rank": int(item.get("rank") or i + 1),
                "diagnosis": str(item.get("diagnosis") or item.get("name") or "").strip(),
                "score": round(score_f, 1),
                "confidence": str(item.get("confidence") or "medium").lower(),
                "category": str(item.get("category") or "differential"),
                "snomed_aligned": item.get("snomed_aligned") or item.get("snomed") or "",
                "supporting_evidence": item.get("supporting_evidence") or [],
                "opposing_evidence": item.get("opposing_evidence") or [],
                "reasoning": item.get("reasoning") or "",
            }
        )
    cleaned = [c for c in cleaned if c["diagnosis"]]
    cleaned.sort(key=lambda x: (-x["score"], x["rank"]))
    for i, c in enumerate(cleaned, start=1):
        c["rank"] = i

    n_prior_icds = sum(len(p.get("icd10_diagnoses") or []) for p in prior_blocks)

    result["differential"] = cleaned
    result["n_candidates"] = len(cleaned)
    if cleaned and not result.get("most_likely"):
        result["most_likely"] = cleaned[0]["diagnosis"]
    result["type"] = "differential_diagnosis"
    result["_method"] = f"{config.method_prefix()}_llm:{model}"
    result["_agent"] = "differential_diagnosis"
    result["_temperature"] = config.temperature
    result["patient_id"] = str(patient_id)
    result["hadm_id"] = str(hadm_id)
    result["generated_at"] = datetime.now().isoformat()
    result["inputs"] = {
        "symptom_tree": True,
        "retained_snomed_entities": len(retained_slim),
        "retained_links": sum(len(e.get("retained_links") or []) for e in retained_slim),
        "clinical_context": bool(clinical_context_text),
        "information_extraction": bool(ie_summary),
        "prior_admissions": len(prior_blocks),
        "prior_icd_codes": n_prior_icds,
        "prior_icd_context": True,
        "prompt_format": "ROLE/CONTEXT/TASK/CONSTRAINTS",
        "temperature": config.temperature,
    }
    result["prior_icd_context"] = prior_blocks
    return result


def format_differential_diagnosis_txt(result: Dict[str, Any]) -> str:
    lines = [
        _format_section("DIFFERENTIAL DIAGNOSIS (scored)"),
        f"Patient ID       : {result.get('patient_id', 'N/A')}",
        f"HADM ID          : {result.get('hadm_id', 'N/A')}",
        f"Method           : {result.get('_method', 'unknown')}",
        f"Temperature      : {result.get('_temperature', result.get('inputs', {}).get('temperature', 'N/A'))}",
        f"Generated        : {result.get('generated_at', 'N/A')}",
        f"Candidates       : {result.get('n_candidates', len(result.get('differential') or []))}",
        f"Most likely      : {result.get('most_likely', 'N/A')}",
    ]
    inputs = result.get("inputs") or {}
    if inputs.get("prior_admissions") is not None:
        lines.append(
            f"Prior admissions : {inputs.get('prior_admissions')} "
            f"({inputs.get('prior_icd_codes', 0)} ICD codes as PMH context)"
        )
    if result.get("summary"):
        lines.extend(["", "Summary:", f"  {result['summary']}"])
    lines.append("")
    lines.append(_format_section("RANKED DIFFERENTIAL", "-"))
    for item in result.get("differential") or []:
        lines.append(
            f"  #{item.get('rank')}  [{item.get('score')}/100 | {item.get('confidence')}]  "
            f"{item.get('diagnosis')}  ({item.get('category')})"
        )
        if item.get("snomed_aligned"):
            lines.append(f"      SNOMED aligned : {item['snomed_aligned']}")
        for e in item.get("supporting_evidence") or []:
            lines.append(f"      + {e}")
        for e in item.get("opposing_evidence") or []:
            lines.append(f"      − {e}")
        if item.get("reasoning"):
            lines.append(f"      why: {item['reasoning']}")
        lines.append("")
    rule_outs = result.get("rule_outs") or []
    if rule_outs:
        lines.append(_format_section("RULE-OUTS", "-"))
        for r in rule_outs:
            if isinstance(r, dict):
                lines.append(f"  • {r.get('diagnosis')}: {r.get('why')}")
            else:
                lines.append(f"  • {r}")
        lines.append("")
    uncertain = result.get("uncertain_areas") or []
    if uncertain:
        lines.append(_format_section("UNCERTAIN / NEED MORE DATA", "-"))
        for u in uncertain:
            lines.append(f"  • {u}")
        lines.append("")
    return "\n".join(lines)



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


def list_admission_export_dirs(export_dir: Union[str, Path] = None) -> List[Dict[str, Any]]:
    """
    Discovery helper: patient_records/patient_*/admissions/hadm_* with required Stage 7 inputs.
    """
    export_dir = Path(export_dir or EXPORT_DIR)
    rows: List[Dict[str, Any]] = []
    for adm in sorted(export_dir.glob("patient_*/admissions/hadm_*")):
        if not adm.is_dir():
            continue
        pid = ""
        hid = ""
        for part in adm.parts:
            if part.startswith("patient_"):
                pid = part.replace("patient_", "", 1)
            if part.startswith("hadm_"):
                hid = part.replace("hadm_", "", 1)
        tree_path = adm / "symptom_tree.json"
        retained_path = adm / "snomed_retained.json"
        # fallbacks
        if not tree_path.exists():
            alt = export_dir / f"patient_{pid}" / "symptom_tree.json"
            if alt.exists():
                tree_path = alt
        rows.append(
            {
                "patient_id": pid,
                "hadm_id": hid,
                "admission_dir": adm,
                "symptom_tree_path": tree_path,
                "retained_path": retained_path,
                "has_symptom_tree": tree_path.exists(),
                "has_retained": retained_path.exists(),
            }
        )
    return rows


def save_diff_dx_results(
    records: List[Dict[str, Any]],
    path: Union[str, Path] = None,
) -> Path:
    path = Path(path or DIFF_DX_RESULTS_JSON)
    _ensure_parent(path)
    payload = {
        "stage": 7,
        "description": "Scored differential diagnosis from symptom tree + retained SNOMED context",
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_diff_dx_checkpoint(
    path: Union[str, Path] = None,
) -> Dict[str, Dict[str, Any]]:
    path = Path(path or DIFF_DX_CHECKPOINT_JSON)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    # key = "patient_id|hadm_id"
    out: Dict[str, Dict[str, Any]] = {}
    for row in data.get("results") or []:
        key = f"{row.get('patient_id')}|{row.get('hadm_id')}"
        out[key] = row
    return out


def save_diff_dx_checkpoint(
    records: List[Dict[str, Any]],
    path: Union[str, Path] = None,
) -> Path:
    path = Path(path or DIFF_DX_CHECKPOINT_JSON)
    _ensure_parent(path)
    payload = {
        "stage": 7,
        "checkpoint": True,
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def export_diff_dx_to_admission(
    result: Dict[str, Any],
    admission_dir: Path,
) -> None:
    """Write differential_diagnosis.json + .txt into one admission folder."""
    admission_dir = Path(admission_dir)
    admission_dir.mkdir(parents=True, exist_ok=True)
    _write_json(admission_dir / "differential_diagnosis.json", result)
    _write_text(
        admission_dir / "differential_diagnosis.txt",
        format_differential_diagnosis_txt(result),
    )


def run_stage07_cohort(
    export_dir: Union[str, Path] = None,
    config: Optional[LLMConfig] = None,
    temperature: float = DIFF_DX_TEMPERATURE,
    delay_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run Stage 7 DiffDx over admissions that have a symptom tree."""
    export_dir = Path(export_dir or EXPORT_DIR)
    delay = LLM_REQUEST_DELAY_SECONDS if delay_seconds is None else delay_seconds
    cfg = config or get_llm_config()
    STAGE_07_DIR.mkdir(parents=True, exist_ok=True)

    done = load_diff_dx_checkpoint()
    records = list(done.values())
    admissions = list_admission_export_dirs(export_dir)
    todo = [
        a
        for a in admissions
        if a.get("has_symptom_tree")
        and f"{a['patient_id']}|{a['hadm_id']}" not in done
    ]
    print(f"Stage 7: {len(done)} done, {len(todo)} remaining (temp={temperature})")
    for i, adm in enumerate(todo, start=1):
        pid, hid = adm["patient_id"], adm["hadm_id"]
        adm_dir = Path(adm["admission_dir"])
        print(f"[{i}/{len(todo)}] DiffDx patient={pid} hadm={hid}...")
        tree = json.loads(Path(adm["symptom_tree_path"]).read_text(encoding="utf-8"))
        retained = {}
        if adm.get("has_retained"):
            retained = json.loads(Path(adm["retained_path"]).read_text(encoding="utf-8"))
        hist_path = export_dir / f"patient_{pid}" / "admission_history.json"
        history = []
        if hist_path.exists():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        ctx_path = adm_dir / "clinical_context.txt"
        clinical_context = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else None
        ie = None
        ie_path = adm_dir / "information_extraction.json"
        if ie_path.exists():
            ie = json.loads(ie_path.read_text(encoding="utf-8"))
        try:
            result = differential_diagnosis_agent(
                symptom_tree=tree,
                retained_snomed=retained,
                patient_id=pid,
                hadm_id=hid,
                clinical_context_text=clinical_context,
                ie_summary=ie,
                admission_history=history,
                config=cfg,
                temperature=temperature,
            )
        except (ValueError, TimeoutError, LLMNotAvailableError) as exc:
            print(f"  ERROR: {exc}")
            result = {
                "patient_id": pid,
                "hadm_id": hid,
                "type": "differential_diagnosis",
                "error": str(exc),
                "differential": [],
                "n_candidates": 0,
                "generated_at": datetime.now().isoformat(),
            }
        export_diff_dx_to_admission(result, adm_dir)
        records.append(result)
        save_diff_dx_checkpoint(records)
        top = (result.get("differential") or [{}])[0]
        print(
            f"  most_likely={result.get('most_likely')!r} | "
            f"n={result.get('n_candidates')} | top_score={top.get('score')}"
        )
        if i < len(todo) and delay and delay > 0:
            time.sleep(delay)

    out_path = save_diff_dx_results(records)
    print(f"Saved Stage 7 → {out_path}")
    return {
        "stage": 7,
        "n_admissions": len(records),
        "path": str(out_path),
        "results": records,
    }


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

        # Clinical notes (latest admission): redacted for coding; full for audit
        coding_note = cohort_row.get("clinical_note", "")
        full_note = cohort_row.get("clinical_note_full") or cohort_row.get("text", coding_note)
        redacted_dx = cohort_row.get("redacted_diagnosis_text") or ""

        _write_text(adm_dir / "clinical_note.txt", coding_note)
        _write_text(adm_dir / "clinical_note_full.txt", full_note)
        if redacted_dx.strip():
            _write_text(adm_dir / "redacted_discharge_sections.txt", redacted_dx)
            # Backward-compatible alias
            _write_text(adm_dir / "redacted_diagnosis_sections.txt", redacted_dx)

        gt_payload = {
            "patient_id": patient_id,
            "hadm_id": int(cohort_row["hadm_id"]),
            "primary_icd_code": cohort_row.get("primary_icd_code"),
            "primary_dx_title": cohort_row.get("primary_dx_title"),
            "icd10_codes": cohort_row.get("ground_truth_icd10", []),
            "dx_titles": cohort_row.get("ground_truth_dx_titles", []),
            "n_diagnoses": int(cohort_row.get("n_diagnoses", 0)) if pd.notna(cohort_row.get("n_diagnoses")) else None,
            "redacted_from_note": bool(redacted_dx.strip()),
            "note": "Labels for evaluation only — not shown to LLM agents",
        }
        _write_json(adm_dir / "ground_truth.json", gt_payload)

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
            "note_redacted_for_coding": bool(str(cohort_row.get("redacted_diagnosis_text") or "").strip()),
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
        clinical_note.txt              (redacted discharge package — LLM / coding input)
        clinical_note_full.txt         (original discharge note)
        redacted_discharge_sections.txt (removed discharge package + HC titles)
        ground_truth.json / .txt       (ICD-10 labels — eval only)
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


# ---------------------------------------------------------------------------
# Stage 9 — LLM confirmation of Stage 8 ICD packages (map-backed adds)
# ---------------------------------------------------------------------------
ICD_CONFIRM_TEMPERATURE = 0.3
DX_SEMANTIC_MATCH_THRESHOLD = 0.70  # MiniLM vs GT primary title (Stage 10)
ICD_TITLE_SEMANTIC_THRESHOLD = 0.70  # MiniLM ICD/dx title vs GT title
ICD_TITLE_NEAR_THRESHOLD = 0.50  # related but not counted as a hit

ICD_CONFIRM_ROLE = """\
You are a hospital clinical coding specialist reviewing a draft ICD-10-CM package \
produced by SNOMED CT ExtendedMap. You confirm, drop, or replace mapped codes and \
name clinically indicated conditions that are missing from the draft. You never \
invent ICD-10 strings — you only select codes already listed, or name conditions \
in English so they can be mapped later."""

ICD_CONFIRM_TASK = """\
Using only the CONTEXT below, review the Stage 8 draft ICD-10-CM packages.

1. For each DiffDx diagnosis, KEEP, DROP, or REPLACE its principal code.
   - KEEP if the listed candidate fits the current stay.
   - DROP if the map is anatomically/clinically wrong (e.g. eyelid code for abdominal adhesions).
   - REPLACE by naming a better CONDITION (English), not an ICD code.
   If several principal candidates are listed, pick one of those codes only.
2. Confirm supporting codes that belong on a billing-style list for THIS stay \
   (comorbidities, complications). Drop symptom-only codes (isolated R-codes) \
   unless they are the best available description of an unmapped problem.
3. Name MISSING conditions that should be coded for this stay (active comorbidities, \
   status/aftercare, complications supported by the current context or clearly \
   still relevant PMH). Do NOT dump every prior-admission ICD as a current code \
   unless the current context supports it.
4. Name the PRIMARY condition for this admission (usually the top DiffDx, refined if needed).

Return COMPLETE valid JSON only, matching the OUTPUT schema."""

ICD_CONFIRM_CONSTRAINTS = """\
- Do NOT invent ICD-10-CM codes. selected_code / confirmed_codes / dropped_codes \
  MUST be copied from codes listed in CONTEXT (Stage 8 packages).
- For missing or replaced items, give a condition NAME (e.g. "essential hypertension"), \
  never a guessed code.
- Do NOT use or guess current-stay ground-truth / discharge diagnoses (they are redacted).
- Base every keep/drop/add on the CONTEXT only. Do not invent labs, imaging, or history.
- Prior-admission ICDs are PMH only — include as current codes only when still relevant.
- Prefer specific mapped conditions over unspecified catch-alls when evidence supports it.
- Keep reasons ≤25 words."""

ICD_CONFIRM_OUTPUT_SCHEMA = """\
{
  "review_summary": "2–4 sentences: what you confirmed, dropped, and added",
  "primary_condition": "primary diagnosis name for this stay",
  "principal_selections": [
    {
      "rank": 1,
      "diagnosis": "DiffDx diagnosis string",
      "action": "keep|drop|replace",
      "selected_code": "code from that diagnosis's candidates, or empty if drop/replace",
      "replace_with_condition": "English condition name if action=replace, else empty",
      "reason": "short reason"
    }
  ],
  "confirmed_codes": ["Stage-8 codes to keep as secondaries, if any"],
  "dropped_codes": [{"code": "Stage-8 code", "reason": "why dropped"}],
  "missing_conditions": [
    {
      "condition": "English condition name",
      "role": "comorbidity|secondary|complication|status",
      "evidence": "short evidence from context",
      "as_primary": false
    }
  ]
}"""

ICD_CONFIRM_SYSTEM_PROMPT = (
    "You are an ICD-10-CM coding review agent. "
    "Follow the ROLE, CONTEXT, TASK, and CONSTRAINTS in the user message exactly. "
    "Return complete valid JSON only. Never invent ICD codes."
)


def _undot_icd_code(code: str) -> str:
    c = (code or "").strip().upper().replace(".", "").replace(" ", "")
    return c.replace("?", "")


def _dot_icd_code(code: str) -> str:
    c = _undot_icd_code(code)
    if not c:
        return ""
    if "." in (code or ""):
        return (code or "").strip().upper()
    if len(c) > 3 and c[0].isalpha():
        return c[:3] + "." + c[3:]
    return c


def _icd_family(code: str) -> str:
    c = _undot_icd_code(code)
    return c[:3] if len(c) >= 3 else c


def _stage08_code_index(stage08_row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Undotted code → first Stage 8 package row (principal preferred)."""
    by_code: Dict[str, Dict[str, Any]] = {}
    for d in stage08_row.get("differential") or []:
        diagnosis = d.get("diagnosis") or ""
        for c in d.get("icd10_package") or []:
            key = _undot_icd_code(c.get("code") or "")
            if not key:
                continue
            row = {**c, "from_diagnosis": diagnosis, "from_rank": d.get("rank")}
            prev = by_code.get(key)
            if prev is None or (
                str(c.get("role") or "") == "principal"
                and str(prev.get("role") or "") != "principal"
            ):
                by_code[key] = row
    return by_code


def slim_stage08_for_prompt(stage08_row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact Stage 8 packages for the confirmation LLM (no map-advice walls)."""
    diffs = []
    for d in stage08_row.get("differential") or []:
        principals = []
        supporting = []
        for c in d.get("icd10_package") or []:
            item = {
                "code": c.get("code"),
                "title": c.get("title") or "",
                "always": bool(c.get("always")),
            }
            if c.get("role") == "supporting":
                item["from"] = c.get("from_term") or ""
                supporting.append(item)
            else:
                principals.append(item)
        diffs.append(
            {
                "rank": d.get("rank"),
                "diagnosis": d.get("diagnosis"),
                "score": d.get("score"),
                "confidence": d.get("confidence"),
                "category": d.get("category"),
                "principal_candidates": principals[:6],
                "supporting": supporting[:6],
            }
        )
    return {
        "most_likely": stage08_row.get("most_likely"),
        "summary": stage08_row.get("summary"),
        "differential": diffs,
    }


def slim_ie_for_confirm(ie_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not ie_summary:
        return {}
    keys = ("symptoms", "diagnoses_mentioned", "procedures", "medications")
    out = {}
    for k in keys:
        val = ie_summary.get(k)
        if val:
            out[k] = val[:12] if isinstance(val, list) else val
    return out


def build_icd_confirm_context(
    patient_id: str,
    hadm_id: str,
    stage08_slim: Dict[str, Any],
    prior_icd_blocks: Optional[List[Dict[str, Any]]] = None,
    clinical_context_text: Optional[str] = None,
    ie_summary: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = [
        f"Patient ID: {patient_id}",
        f"Current admission ID (HADM): {hadm_id}",
        "",
        "Draft ICD packages below were produced by SNOMED US ExtendedMap (Stage 8).",
        "You may only KEEP codes that appear in those packages. Missing items = condition names.",
        "Current-stay ground-truth ICD is NOT provided and must not be guessed.",
        "",
        "--- PRIOR ADMISSION ICD-10 CONTEXT (PMH only) ---",
        format_prior_icd_context_text(prior_icd_blocks or []),
        "",
        "--- STAGE 8 DRAFT ICD PACKAGES ---",
        json.dumps(stage08_slim, indent=2, ensure_ascii=False),
    ]
    if clinical_context_text:
        ctx = clinical_context_text.strip()
        if len(ctx) > 2200:
            ctx = ctx[:2200] + "\n...[truncated]"
        parts.extend(["", "--- STRUCTURED CLINICAL CONTEXT (current stay) ---", ctx])
    slim_ie = slim_ie_for_confirm(ie_summary)
    if slim_ie:
        parts.extend(
            [
                "",
                "--- INFORMATION EXTRACTION (current stay) ---",
                json.dumps(slim_ie, indent=2, ensure_ascii=False),
            ]
        )
    return "\n".join(parts)


def build_icd_confirm_user_prompt(context: str) -> str:
    return (
        f"ROLE:\n{ICD_CONFIRM_ROLE}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"TASK:\n{ICD_CONFIRM_TASK}\n\n"
        f"CONSTRAINTS:\n{ICD_CONFIRM_CONSTRAINTS}\n\n"
        f"OUTPUT SCHEMA (return only this JSON object):\n{ICD_CONFIRM_OUTPUT_SCHEMA}"
    )


def _lookup_map_for_condition(
    condition: str,
    snomed_index: Any,
    map_index: Any,
) -> Optional[Dict[str, Any]]:
    from snomed_ct import map_condition_name_to_icd

    mapped = map_condition_name_to_icd(condition, snomed_index, map_index)
    primary = mapped.get("icd_primary")
    if not primary:
        return None
    return {
        "code": primary.get("code"),
        "title": primary.get("title") or "",
        "always": bool(primary.get("always")),
        "source": "added_mapped",
        "snomed_resolution": mapped.get("snomed_resolution"),
        "icd_candidates": mapped.get("icd_candidates") or [],
        "condition": condition,
    }


def _passthrough_stage08(stage08_row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """If the LLM fails, keep Stage 8 principals so Stage 10 can still score."""
    final = []
    seen = set()
    for d in stage08_row.get("differential") or []:
        code = d.get("icd10_primary")
        key = _undot_icd_code(code or "")
        if not key or key in seen:
            continue
        seen.add(key)
        final.append(
            {
                "code": d.get("icd10_primary"),
                "title": d.get("icd10_primary_title") or "",
                "role": "primary" if len(final) == 0 else "secondary",
                "source": "stage8_passthrough",
                "condition": d.get("diagnosis"),
                "action": "keep",
                "reason": reason,
            }
        )
    primary = final[0] if final else {}
    return {
        "review_summary": reason,
        "primary_condition": stage08_row.get("most_likely"),
        "primary_icd_code": primary.get("code"),
        "primary_icd_title": primary.get("title"),
        "final_codes": final,
        "dropped_codes": [],
        "added_conditions": [],
        "unmapped_additions": [],
        "fallback": "stage8_passthrough",
    }


def resolve_stage09_review(
    llm_out: Dict[str, Any],
    stage08_row: Dict[str, Any],
    snomed_index: Any,
    map_index: Any,
) -> Dict[str, Any]:
    """Apply LLM keep/drop/replace + map missing condition names through ExtendedMap."""
    allowed = _stage08_code_index(stage08_row)
    dropped_keys = set()
    dropped_rows: List[Dict[str, Any]] = []
    for row in llm_out.get("dropped_codes") or []:
        if isinstance(row, str):
            code, reason = row, ""
        else:
            code, reason = row.get("code") or "", row.get("reason") or ""
        key = _undot_icd_code(code)
        if key in allowed:
            dropped_keys.add(key)
            dropped_rows.append(
                {
                    "code": allowed[key].get("code"),
                    "title": allowed[key].get("title") or "",
                    "reason": reason,
                }
            )

    keep_keys: List[str] = []
    replacements: List[Dict[str, Any]] = []

    for sel in llm_out.get("principal_selections") or []:
        if not isinstance(sel, dict):
            continue
        action = str(sel.get("action") or "keep").lower().strip()
        selected = _undot_icd_code(sel.get("selected_code") or sel.get("code") or "")
        diagnosis = str(sel.get("diagnosis") or "")
        reason = str(sel.get("reason") or "")
        if action == "drop":
            if selected:
                dropped_keys.add(selected)
                if selected in allowed:
                    dropped_rows.append(
                        {
                            "code": allowed[selected].get("code"),
                            "title": allowed[selected].get("title") or "",
                            "reason": reason or "principal dropped",
                        }
                    )
            continue
        if action == "replace":
            cond = str(sel.get("replace_with_condition") or "").strip()
            if cond:
                replacements.append(
                    {"condition": cond, "reason": reason, "from_diagnosis": diagnosis}
                )
            continue
        # keep
        if selected and selected in allowed and selected not in dropped_keys:
            if selected not in keep_keys:
                keep_keys.append(selected)

    for code in llm_out.get("confirmed_codes") or []:
        key = _undot_icd_code(str(code))
        if key in allowed and key not in dropped_keys and key not in keep_keys:
            keep_keys.append(key)

    # If the LLM kept nothing valid, fall back to Stage 8 rank-1 principal
    if not keep_keys:
        diffs = stage08_row.get("differential") or []
        if diffs and diffs[0].get("icd10_primary"):
            k = _undot_icd_code(diffs[0]["icd10_primary"])
            if k in allowed:
                keep_keys.append(k)

    added: List[Dict[str, Any]] = []
    unmapped_additions: List[Dict[str, Any]] = []
    pending_conditions: List[Dict[str, Any]] = list(replacements)
    for miss in llm_out.get("missing_conditions") or []:
        if not isinstance(miss, dict):
            continue
        cond = str(miss.get("condition") or "").strip()
        if not cond:
            continue
        pending_conditions.append(
            {
                "condition": cond,
                "reason": miss.get("evidence") or "",
                "role": miss.get("role") or "comorbidity",
                "as_primary": bool(miss.get("as_primary")),
            }
        )

    used_codes = set(keep_keys)
    for item in pending_conditions:
        mapped = _lookup_map_for_condition(item["condition"], snomed_index, map_index)
        if not mapped:
            unmapped_additions.append(
                {"condition": item["condition"], "reason": item.get("reason") or ""}
            )
            continue
        key = _undot_icd_code(mapped["code"])
        if key in used_codes:
            continue
        used_codes.add(key)
        added.append(
            {
                **mapped,
                "role": item.get("role") or "comorbidity",
                "action": "add",
                "reason": item.get("reason") or "",
                "as_primary": bool(item.get("as_primary")),
            }
        )

    primary_condition = str(
        llm_out.get("primary_condition") or stage08_row.get("most_likely") or ""
    ).strip()

    final: List[Dict[str, Any]] = []
    seen_final = set()

    def append_final(row: Dict[str, Any], role: str) -> None:
        key = _undot_icd_code(row.get("code") or "")
        if not key or key in seen_final:
            return
        seen_final.add(key)
        final.append(
            {
                "code": _dot_icd_code(row.get("code") or ""),
                "title": row.get("title") or "",
                "role": role,
                "source": row.get("source") or "stage8_confirmed",
                "condition": row.get("condition") or row.get("from_diagnosis") or "",
                "action": row.get("action") or "keep",
                "reason": row.get("reason") or "",
                "always": bool(row.get("always")),
            }
        )

    # Primary: mapped primary_condition if possible, else first kept Stage 8 code
    primary_mapped = _lookup_map_for_condition(primary_condition, snomed_index, map_index)
    if primary_mapped:
        primary_mapped["source"] = (
            "added_mapped"
            if _undot_icd_code(primary_mapped["code"]) not in allowed
            else "stage8_confirmed"
        )
        primary_mapped["condition"] = primary_condition
        primary_mapped["action"] = "keep"
        append_final(primary_mapped, "primary")
    elif keep_keys:
        src = {**allowed[keep_keys[0]], "condition": primary_condition, "action": "keep"}
        append_final(src, "primary")

    for key in keep_keys:
        src = {
            **allowed[key],
            "source": "stage8_confirmed",
            "action": "keep",
            "condition": allowed[key].get("from_diagnosis") or "",
        }
        append_final(src, "secondary")

    for row in added:
        role = "primary" if row.get("as_primary") and not final else "secondary"
        append_final(row, role)

    primary = next((c for c in final if c.get("role") == "primary"), final[0] if final else {})
    return {
        "review_summary": llm_out.get("review_summary") or "",
        "primary_condition": primary_condition,
        "primary_icd_code": primary.get("code"),
        "primary_icd_title": primary.get("title"),
        "final_codes": final,
        "dropped_codes": dropped_rows,
        "added_conditions": added,
        "unmapped_additions": unmapped_additions,
        "llm_raw": {
            "principal_selections": llm_out.get("principal_selections") or [],
            "n_confirmed": len(llm_out.get("confirmed_codes") or []),
            "n_missing_named": len(llm_out.get("missing_conditions") or []),
        },
        "fallback": None,
    }


def icd_confirmation_agent(
    stage08_row: Dict[str, Any],
    snomed_index: Any,
    map_index: Any,
    patient_id: str,
    hadm_id: str,
    clinical_context_text: Optional[str] = None,
    ie_summary: Optional[Dict[str, Any]] = None,
    admission_history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[LLMConfig] = None,
    temperature: float = ICD_CONFIRM_TEMPERATURE,
) -> Dict[str, Any]:
    """
    LLM reviews Stage 8 ICD packages; missing items are condition names mapped
    through SNOMED ExtendedMap. Never uses current-stay ground truth.
    """
    from dataclasses import replace

    base = config or LLMConfig()
    config = replace(base, temperature=float(temperature))
    model = config.model
    require_llm(config)
    warn_if_slow_model(model, config.provider)

    prior_blocks = build_prior_icd_context(admission_history)
    slim = slim_stage08_for_prompt(stage08_row)
    context = build_icd_confirm_context(
        patient_id=str(patient_id),
        hadm_id=str(hadm_id),
        stage08_slim=slim,
        prior_icd_blocks=prior_blocks,
        clinical_context_text=clinical_context_text,
        ie_summary=ie_summary,
    )
    user_prompt = build_icd_confirm_user_prompt(context)

    try:
        llm_out = call_llm_json(ICD_CONFIRM_SYSTEM_PROMPT, user_prompt, config, model=model)
        resolved = resolve_stage09_review(llm_out, stage08_row, snomed_index, map_index)
        error = None
    except (ValueError, TimeoutError, LLMNotAvailableError) as exc:
        resolved = _passthrough_stage08(stage08_row, f"LLM failed: {exc}")
        error = str(exc)
        llm_out = {}

    result = {
        **resolved,
        "patient_id": str(patient_id),
        "hadm_id": str(hadm_id),
        "most_likely": resolved.get("primary_condition") or stage08_row.get("most_likely"),
        "stage8_most_likely": stage08_row.get("most_likely"),
        "stage": 9,
        "type": "icd_confirmation",
        "_method": f"{config.method_prefix()}_llm:{model}",
        "_agent": "icd_confirmation",
        "_temperature": config.temperature,
        "generated_at": datetime.now().isoformat(),
        "error": error,
        "inputs": {
            "stage8_packages": True,
            "clinical_context": bool(clinical_context_text),
            "information_extraction": bool(ie_summary),
            "prior_admissions": len(prior_blocks),
            "prior_icd_codes": sum(len(p.get("icd10_diagnoses") or []) for p in prior_blocks),
            "prompt_format": "ROLE/CONTEXT/TASK/CONSTRAINTS",
            "temperature": config.temperature,
            "map_backed_additions": True,
        },
        "n_final_codes": len(resolved.get("final_codes") or []),
        "n_dropped": len(resolved.get("dropped_codes") or []),
        "n_added": len(resolved.get("added_conditions") or []),
    }
    return result


def format_icd_confirmed_txt(result: Dict[str, Any]) -> str:
    lines = [
        _format_section("ICD-10-CM CONFIRMATION (Stage 9)"),
        f"Patient ID       : {result.get('patient_id', 'N/A')}",
        f"HADM ID          : {result.get('hadm_id', 'N/A')}",
        f"Method           : {result.get('_method', 'unknown')}",
        f"Temperature      : {result.get('_temperature', 'N/A')}",
        f"Generated        : {result.get('generated_at', 'N/A')}",
        f"Primary condition: {result.get('primary_condition') or result.get('most_likely')}",
        f"Primary ICD      : {result.get('primary_icd_code')} — {result.get('primary_icd_title') or ''}",
        f"Final codes      : {result.get('n_final_codes', len(result.get('final_codes') or []))}",
        f"Dropped / added  : {result.get('n_dropped', 0)} / {result.get('n_added', 0)}",
    ]
    if result.get("fallback"):
        lines.append(f"Fallback         : {result.get('fallback')}")
    if result.get("error"):
        lines.append(f"Error            : {result.get('error')}")
    if result.get("review_summary"):
        lines.extend(["", "Review:", f"  {result['review_summary']}"])
    lines.append("")
    lines.append(_format_section("FINAL CODE LIST", "-"))
    for i, row in enumerate(result.get("final_codes") or [], start=1):
        lines.append(
            f"  {i:2d}. [{row.get('role')}] {row.get('code')} — {row.get('title') or ''}"
        )
        extra = []
        if row.get("condition"):
            extra.append(row["condition"])
        if row.get("source"):
            extra.append(row["source"])
        if row.get("action"):
            extra.append(row["action"])
        if extra:
            lines.append(f"      ({' | '.join(extra)})")
        if row.get("reason"):
            lines.append(f"      why: {row['reason']}")
    dropped = result.get("dropped_codes") or []
    if dropped:
        lines.append("")
        lines.append(_format_section("DROPPED STAGE-8 CODES", "-"))
        for row in dropped:
            lines.append(
                f"  • {row.get('code')} — {row.get('title') or ''}  ({row.get('reason') or ''})"
            )
    unmapped = result.get("unmapped_additions") or []
    if unmapped:
        lines.append("")
        lines.append(_format_section("NAMED BUT UNMAPPED CONDITIONS", "-"))
        for row in unmapped:
            lines.append(f"  • {row.get('condition')}  ({row.get('reason') or ''})")
    lines.append("")
    return "\n".join(lines)


def load_icd_confirm_checkpoint(
    path: Union[str, Path] = None,
) -> Dict[str, Dict[str, Any]]:
    path = Path(path or ICD_CONFIRM_CHECKPOINT_JSON)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for row in data.get("results") or []:
        out[f"{row.get('patient_id')}|{row.get('hadm_id')}"] = row
    return out


def save_icd_confirm_checkpoint(
    records: List[Dict[str, Any]],
    path: Union[str, Path] = None,
) -> Path:
    path = Path(path or ICD_CONFIRM_CHECKPOINT_JSON)
    _ensure_parent(path)
    payload = {
        "stage": 9,
        "checkpoint": True,
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_icd_confirm_results(
    records: List[Dict[str, Any]],
    path: Union[str, Path] = None,
) -> Path:
    path = Path(path or ICD_CONFIRM_RESULTS_JSON)
    _ensure_parent(path)
    payload = {
        "stage": 9,
        "description": (
            "LLM confirmation of Stage 8 ICD packages; missing conditions "
            "mapped via SNOMED ExtendedMap (no free-form ICD invention)"
        ),
        "generated_at": datetime.now().isoformat(),
        "n_admissions": len(records),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def export_icd_confirm_to_admission(result: Dict[str, Any], admission_dir: Path) -> None:
    admission_dir = Path(admission_dir)
    admission_dir.mkdir(parents=True, exist_ok=True)
    _write_json(admission_dir / "icd_coding_confirmed.json", result)
    _write_text(
        admission_dir / "icd_coding_confirmed.txt",
        format_icd_confirmed_txt(result),
    )


def run_stage09_cohort(
    export_dir: Union[str, Path] = None,
    snomed_index: Any = None,
    map_index: Any = None,
    config: Optional[LLMConfig] = None,
    temperature: float = ICD_CONFIRM_TEMPERATURE,
    delay_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run Stage 9 over all admissions that have Stage 8 `icd_coding.json`."""
    from snomed_ct import build_icd10cm_map_index, build_snomed_index, find_snomed_root

    export_dir = Path(export_dir or EXPORT_DIR)
    delay = LLM_REQUEST_DELAY_SECONDS if delay_seconds is None else delay_seconds
    cfg = config or get_llm_config()
    if snomed_index is None or map_index is None:
        root = find_snomed_root(REPO_ROOT / "data")
        if snomed_index is None:
            snomed_index = build_snomed_index(
                snomed_root=root,
                cache_path=REPO_ROOT / "data" / "snomed_index" / "snomed_index.pkl",
            )
        if map_index is None:
            map_index = build_icd10cm_map_index(
                snomed_root=root,
                cache_path=REPO_ROOT / "data" / "snomed_index" / "icd10cm_extended_map.pkl",
            )

    STAGE_09_DIR.mkdir(parents=True, exist_ok=True)
    done = load_icd_confirm_checkpoint()
    records = list(done.values())
    admissions = list_admission_export_dirs(export_dir)
    todo = []
    for adm in admissions:
        key = f"{adm['patient_id']}|{adm['hadm_id']}"
        icd_path = Path(adm["admission_dir"]) / "icd_coding.json"
        if key not in done and icd_path.exists():
            todo.append(adm)

    print(f"Stage 9: {len(done)} done, {len(todo)} remaining")
    for i, adm in enumerate(todo, start=1):
        pid, hid = adm["patient_id"], adm["hadm_id"]
        adm_dir = Path(adm["admission_dir"])
        print(f"[{i}/{len(todo)}] Confirm ICD patient={pid} hadm={hid}...")
        stage08 = json.loads((adm_dir / "icd_coding.json").read_text(encoding="utf-8"))
        hist_path = export_dir / f"patient_{pid}" / "admission_history.json"
        history = []
        if hist_path.exists():
            history = json.loads(hist_path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        ctx_path = adm_dir / "clinical_context.txt"
        clinical_context = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else None
        ie = None
        ie_path = adm_dir / "information_extraction.json"
        if ie_path.exists():
            ie = json.loads(ie_path.read_text(encoding="utf-8"))
        result = icd_confirmation_agent(
            stage08_row=stage08,
            snomed_index=snomed_index,
            map_index=map_index,
            patient_id=pid,
            hadm_id=hid,
            clinical_context_text=clinical_context,
            ie_summary=ie,
            admission_history=history,
            config=cfg,
            temperature=temperature,
        )
        export_icd_confirm_to_admission(result, adm_dir)
        records.append(result)
        save_icd_confirm_checkpoint(records)
        print(
            f"  primary={result.get('primary_condition')!r} | "
            f"icd={result.get('primary_icd_code')} | "
            f"n={result.get('n_final_codes')} added={result.get('n_added')} "
            f"dropped={result.get('n_dropped')}"
        )
        if i < len(todo) and delay and delay > 0:
            time.sleep(delay)

    out_path = save_icd_confirm_results(records)
    print(f"Saved Stage 9 → {out_path}")
    return {
        "stage": 9,
        "n_admissions": len(records),
        "path": str(out_path),
        "results": records,
    }


# ---------------------------------------------------------------------------
# Stage 10 — Accuracy vs current-stay ground truth
# ---------------------------------------------------------------------------
def _load_gt(admission_dir: Path) -> Dict[str, Any]:
    path = admission_dir / "ground_truth.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stage8_predicted_codes(stage08_row: Dict[str, Any]) -> Dict[str, Any]:
    all_codes: List[Dict[str, str]] = []
    principals: List[Dict[str, str]] = []
    seen_all = set()
    seen_p = set()
    diffs = stage08_row.get("differential") or []
    top = diffs[0] if diffs else {}
    primary_code = top.get("icd10_primary")
    primary_title = top.get("icd10_primary_title") or ""
    primary_dx = stage08_row.get("most_likely") or top.get("diagnosis") or ""
    for d in diffs:
        for c in d.get("icd10_package") or []:
            key = _undot_icd_code(c.get("code") or "")
            if not key or key in seen_all:
                continue
            seen_all.add(key)
            all_codes.append(
                {
                    "code": c.get("code"),
                    "title": c.get("title") or "",
                    "role": c.get("role"),
                    "diagnosis": d.get("diagnosis") or "",
                    "condition": d.get("diagnosis") or "",
                }
            )
            if c.get("role") == "principal" and key not in seen_p:
                seen_p.add(key)
                principals.append(
                    {
                        "code": c.get("code"),
                        "title": c.get("title") or "",
                        "diagnosis": d.get("diagnosis") or "",
                    }
                )
    diagnoses: List[str] = []
    diagnosis_items: List[Dict[str, Any]] = []
    seen_dx = set()
    for i, d in enumerate(diffs):
        name = " ".join(str(d.get("diagnosis") or "").split())
        key = name.lower()
        if name and key not in seen_dx:
            seen_dx.add(key)
            diagnoses.append(name)
            score = d.get("score")
            try:
                score = float(score) if score is not None and score != "" else None
            except (TypeError, ValueError):
                score = None
            diagnosis_items.append(
                {
                    "diagnosis": name,
                    "rank": int(d.get("rank") or (i + 1)),
                    "score": score,
                    "confidence": str(d.get("confidence") or "").strip(),
                }
            )
    if primary_dx and primary_dx.strip().lower() not in seen_dx:
        diagnoses.insert(0, primary_dx.strip())
        diagnosis_items.insert(
            0,
            {
                "diagnosis": primary_dx.strip(),
                "rank": 1,
                "score": None,
                "confidence": "",
            },
        )
    return {
        "primary_dx": primary_dx,
        "primary_code": primary_code,
        "primary_title": primary_title,
        "diagnoses": diagnoses,
        "diagnosis_items": diagnosis_items,
        "all_codes": all_codes,
        "principal_codes": principals,
    }


def _stage9_predicted_codes(stage09_row: Dict[str, Any]) -> Dict[str, Any]:
    codes = []
    seen = set()
    for c in stage09_row.get("final_codes") or []:
        key = _undot_icd_code(c.get("code") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        codes.append(
            {
                "code": c.get("code"),
                "title": c.get("title") or "",
                "role": c.get("role"),
                "diagnosis": c.get("condition") or "",
                "condition": c.get("condition") or "",
            }
        )
    primary_dx = (
        stage09_row.get("primary_condition")
        or stage09_row.get("most_likely")
        or ""
    )
    diagnoses: List[str] = []
    seen_dx = set()
    for name in [primary_dx] + [c.get("condition") or c.get("diagnosis") or "" for c in codes]:
        text = " ".join(str(name or "").split())
        key = text.lower()
        if text and key not in seen_dx:
            seen_dx.add(key)
            diagnoses.append(text)
    return {
        "primary_dx": primary_dx,
        "primary_code": stage09_row.get("primary_icd_code"),
        "primary_title": stage09_row.get("primary_icd_title") or "",
        "diagnoses": diagnoses,
        "all_codes": codes,
        "principal_codes": [c for c in codes if c.get("role") == "primary"] or codes[:1],
    }


def _text_sim(embedder: Any, a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return 0.0
    if a.lower() == b.lower():
        return 1.0

    def _one(x: str, y: str) -> float:
        try:
            if embedder is not None:
                return max(0.0, min(1.0, float(embedder.similarity(x, y))))
        except Exception:
            pass
        return max(0.0, min(1.0, cosine_similarity_text_local(x, y)))

    best = _one(a, b)
    ea, eb = _expand_clinical_text(a), _expand_clinical_text(b)
    if ea and eb and (ea != a.lower() or eb != b.lower()):
        best = max(best, _one(ea, eb))
    return best


def _match_texts(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for key in ("title", "diagnosis", "condition", "match_text"):
        val = str(row.get(key) or "").strip()
        low = val.lower()
        if val and low not in seen:
            seen.add(low)
            out.append(val)
    return out


def _best_text_sim(embedder: Any, lefts: List[str], rights: List[str]) -> float:
    best = 0.0
    for a in lefts or []:
        for b in rights or []:
            if a and b:
                best = max(best, _text_sim(embedder, a, b))
    return best


def _code_set_metrics(
    predicted: List[Dict[str, Any]],
    gt_codes: List[str],
    gt_titles: List[str],
    exclude_r_codes: bool = False,
    embedder: Any = None,
    semantic_threshold: float = ICD_TITLE_SEMANTIC_THRESHOLD,
    near_threshold: float = ICD_TITLE_NEAR_THRESHOLD,
) -> Dict[str, Any]:
    def keep(code: str) -> bool:
        fam = _icd_family(code)
        if exclude_r_codes and fam.startswith("R"):
            return False
        return True

    pred_keys = []
    pred_map = {}
    for row in predicted:
        code = row.get("code") or ""
        if not keep(code):
            continue
        key = _undot_icd_code(code)
        if key and key not in pred_map:
            pred_keys.append(key)
            pred_map[key] = row

    gt_map = {}
    gt_keys = []
    for i, code in enumerate(gt_codes or []):
        if not keep(code):
            continue
        key = _undot_icd_code(code)
        if not key:
            continue
        gt_keys.append(key)
        title = gt_titles[i] if i < len(gt_titles or []) else ""
        gt_map[key] = {"code": _dot_icd_code(code), "title": title}

    pred_set, gt_set = set(pred_keys), set(gt_keys)
    hits = sorted(pred_set & gt_set)
    misses = [k for k in gt_keys if k not in pred_set]
    extras = [k for k in pred_keys if k not in gt_set]
    n_pred, n_gt = len(pred_set), len(gt_set)
    precision = (len(hits) / n_pred) if n_pred else 0.0
    recall = (len(hits) / n_gt) if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    def label(key: str, source: str) -> Dict[str, str]:
        if source == "gt":
            row = gt_map.get(key) or {}
            return {"code": row.get("code") or _dot_icd_code(key), "title": row.get("title") or ""}
        row = pred_map.get(key) or {}
        return {
            "code": row.get("code") or _dot_icd_code(key),
            "title": row.get("title") or (gt_map.get(key) or {}).get("title") or "",
        }

    # Greedy 1-1 MiniLM pairing of leftover titles (same meaning, different codes)
    pair_scores = []
    for pk in extras:
        ptexts = _match_texts(pred_map.get(pk) or {})
        for gk in misses:
            gtitle = (gt_map.get(gk) or {}).get("title") or ""
            sim = _best_text_sim(embedder, ptexts, [gtitle])
            pair_scores.append((sim, pk, gk))
    pair_scores.sort(key=lambda x: -x[0])

    used_p, used_g = set(), set()
    semantic_pairs: List[Dict[str, Any]] = []
    for sim, pk, gk in pair_scores:
        if pk in used_p or gk in used_g:
            continue
        if sim < semantic_threshold - 1e-9:
            continue
        used_p.add(pk)
        used_g.add(gk)
        prow = pred_map.get(pk) or {}
        grow = gt_map.get(gk) or {}
        semantic_pairs.append(
            {
                "predicted_code": prow.get("code") or _dot_icd_code(pk),
                "predicted_title": prow.get("title") or "",
                "predicted_diagnosis": prow.get("diagnosis") or prow.get("condition") or "",
                "ground_truth_code": grow.get("code") or _dot_icd_code(gk),
                "ground_truth_title": grow.get("title") or "",
                "similarity": round(float(sim), 4),
                "match": "semantic",
            }
        )

    near_pairs: List[Dict[str, Any]] = []
    for sim, pk, gk in pair_scores:
        if pk in used_p or gk in used_g:
            continue
        if sim < near_threshold - 1e-9:
            continue
        used_p.add(pk)
        used_g.add(gk)
        prow = pred_map.get(pk) or {}
        grow = gt_map.get(gk) or {}
        near_pairs.append(
            {
                "predicted_code": prow.get("code") or _dot_icd_code(pk),
                "predicted_title": prow.get("title") or "",
                "predicted_diagnosis": prow.get("diagnosis") or prow.get("condition") or "",
                "ground_truth_code": grow.get("code") or _dot_icd_code(gk),
                "ground_truth_title": grow.get("title") or "",
                "similarity": round(float(sim), 4),
                "match": "related",
            }
        )

    n_sem = len(hits) + len(semantic_pairs)
    sem_precision = (n_sem / n_pred) if n_pred else 0.0
    sem_recall = (n_sem / n_gt) if n_gt else 0.0
    sem_f1 = (
        (2 * sem_precision * sem_recall / (sem_precision + sem_recall))
        if (sem_precision + sem_recall)
        else 0.0
    )
    semantic_misses = [label(k, "gt") for k in misses if k not in used_g]
    semantic_extras = [label(k, "pred") for k in extras if k not in used_p]

    return {
        "n_predicted": n_pred,
        "n_ground_truth": n_gt,
        "n_hits": len(hits),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hits": [label(k, "gt") for k in hits],
        "misses": [label(k, "gt") for k in misses],
        "extras": [label(k, "pred") for k in extras],
        "semantic_threshold": semantic_threshold,
        "n_semantic_hits": n_sem,
        "semantic_precision": round(sem_precision, 4),
        "semantic_recall": round(sem_recall, 4),
        "semantic_f1": round(sem_f1, 4),
        "semantic_pairs": semantic_pairs,
        "related_pairs": near_pairs,
        "semantic_misses": semantic_misses,
        "semantic_extras": semantic_extras,
    }


def _score_primary_icd(
    pred_code: Optional[str],
    gt_code: Optional[str],
    pred_title: str = "",
    gt_title: str = "",
    pred_dx: str = "",
    embedder: Any = None,
    threshold: float = ICD_TITLE_SEMANTIC_THRESHOLD,
) -> Dict[str, Any]:
    p, g = _undot_icd_code(pred_code or ""), _undot_icd_code(gt_code or "")
    exact = bool(p and g and p == g)
    family = bool(p and g and _icd_family(p) == _icd_family(g))
    # Score the mapped ICD title, not the DiffDx name (wrong code + right name is not an ICD hit).
    title_for_match = (pred_title or "").strip() or ((pred_dx or "").strip() if p else "")
    hit = _clinical_match(title_for_match, gt_title, embedder, threshold)
    return {
        "predicted": _dot_icd_code(pred_code or "") if pred_code else None,
        "ground_truth": _dot_icd_code(gt_code or "") if gt_code else None,
        "predicted_title": pred_title or "",
        "ground_truth_title": gt_title or "",
        "exact": exact,
        "family": family,
        "semantic_similarity": round(float(hit.get("similarity") or 0), 4),
        "semantic_match": bool(exact or (p and hit.get("match"))),
        "match_reason": (
            "exact_code" if exact else (hit.get("reason") if (p and hit.get("match")) else "no_match")
        ),
        "clinical_reason": (
            "exact_code" if exact else (hit.get("reason") if (p and hit.get("match")) else "no_match")
        ),
        "match_type": (
            "exact"
            if exact
            else (
                "similarity"
                if (p and hit.get("reason") == "semantic")
                else ("clinical_rule" if (p and hit.get("match")) else "no_match")
            )
        ),
        "threshold": threshold,
        "predicted_family": _icd_family(p) if p else None,
        "ground_truth_family": _icd_family(g) if g else None,
        "match_scope": "rank1_vs_primary",
    }


def _score_primary_icd_multi(
    pred_code: Optional[str],
    gt_primary_code: Optional[str],
    pred_title: str,
    gt_primary_title: str,
    pred_dx: str,
    all_pred_codes: Optional[List[Dict[str, Any]]],
    gt_codes: Optional[List[str]],
    gt_titles: Optional[List[str]],
    embedder: Any = None,
    threshold: float = ICD_TITLE_SEMANTIC_THRESHOLD,
) -> Dict[str, Any]:
    """Rank-1 vs GT primary, else any package code vs GT primary, else rank-1 vs any billed GT code."""
    base = _score_primary_icd(
        pred_code,
        gt_primary_code,
        pred_title=pred_title,
        gt_title=gt_primary_title,
        pred_dx=pred_dx,
        embedder=embedder,
        threshold=threshold,
    )
    if base.get("semantic_match"):
        return base

    def alt_ok(hit: Dict[str, Any]) -> bool:
        if hit.get("exact") or hit.get("family"):
            return True
        reason = str(hit.get("match_reason") or "")
        if hit.get("semantic_match") and reason in {
            "exact", "contained", "term_subset", "same_condition", "exact_code",
        }:
            return True
        return bool(hit.get("semantic_match")) and float(hit.get("semantic_similarity") or 0) >= 0.85

    best: Optional[Dict[str, Any]] = None
    for row in all_pred_codes or []:
        hit = _score_primary_icd(
            row.get("code"),
            gt_primary_code,
            pred_title=row.get("title") or "",
            gt_title=gt_primary_title,
            pred_dx=row.get("diagnosis") or row.get("condition") or "",
            embedder=embedder,
            threshold=threshold,
        )
        if not alt_ok(hit):
            continue
        hit["match_scope"] = "package_vs_primary"
        hit["match_reason"] = "package_vs_primary"
        hit["clinical_reason"] = hit.get("clinical_reason") or hit.get("match_reason")
        if best is None or float(hit.get("semantic_similarity") or 0) > float(
            best.get("semantic_similarity") or 0
        ):
            best = hit
    if best:
        return best

    for i, code in enumerate(gt_codes or []):
        title = gt_titles[i] if i < len(gt_titles or []) else ""
        hit = _score_primary_icd(
            pred_code,
            code,
            pred_title=pred_title,
            gt_title=title,
            pred_dx=pred_dx,
            embedder=embedder,
            threshold=threshold,
        )
        if alt_ok(hit):
            hit["match_scope"] = "rank1_vs_gt_list"
            hit["match_reason"] = "rank1_vs_gt_list"
            hit["clinical_reason"] = hit.get("clinical_reason") or hit.get("match_reason")
            hit["ground_truth_primary"] = _dot_icd_code(gt_primary_code or "") if gt_primary_code else None
            hit["ground_truth_primary_title"] = gt_primary_title or ""
            return hit
    return base


# Clinical abbreviations / synonyms so "Refractory VT" matches "Ventricular tachycardia"
_CLINICAL_ABBREV = (
    ("hfpef", "heart failure with preserved ejection fraction"),
    ("hfref", "heart failure with reduced ejection fraction"),
    ("nstemi", "non st elevation myocardial infarction"),
    ("stemi", "st elevation myocardial infarction"),
    ("adhf", "acute decompensated heart failure"),
    ("svt", "supraventricular tachycardia"),
    ("dka", "diabetic ketoacidosis"),
    ("aki", "acute kidney injury"),
    ("ckd", "chronic kidney disease"),
    ("chf", "congestive heart failure"),
    ("copd", "chronic obstructive pulmonary disease"),
    ("esrd", "end stage renal disease"),
    ("uti", "urinary tract infection"),
    ("pna", "pneumonia"),
    ("rul", "right upper lobe"),
    ("rml", "right middle lobe"),
    ("rll", "right lower lobe"),
    ("lul", "left upper lobe"),
    ("lll", "left lower lobe"),
    ("cva", "cerebrovascular accident"),
    ("dvt", "deep vein thrombosis"),
    ("cad", "coronary artery disease"),
    ("htn", "hypertension"),
    ("vt", "ventricular tachycardia"),
    ("vf", "ventricular fibrillation"),
    ("hf", "heart failure"),
    ("mi", "myocardial infarction"),
    ("pe", "pulmonary embolism"),
)
_DX_FILLER_RE = re.compile(
    r"\b(unspecified|organism|initial encounter|subsequent encounter|"
    r"other specified|except renal pelvis|except|type)\b",
    re.I,
)
_DX_STOP = {
    "a", "an", "the", "of", "with", "due", "to", "and", "or", "in", "on", "for",
    "from", "by", "as", "at", "into", "over", "after", "before", "without",
    "unspecified", "organism", "initial", "encounter", "other", "specified",
    "except", "type", "left", "right", "upper", "middle", "lower", "lobe",
}


def _expand_clinical_text(text: str) -> str:
    s = (text or "").lower().replace("-", " ").replace("/", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    for abbr, full in _CLINICAL_ABBREV:
        s = re.sub(rf"\b{re.escape(abbr)}\b", full, s)
    s = s.replace("diabetic ketoacidosis", "diabetes ketoacidosis")
    s = s.replace("diabetic", "diabetes")
    s = s.replace("nephrolithiasis", "renal calculus")
    s = s.replace("kidney stones", "renal calculus")
    s = s.replace("kidney stone", "renal calculus")
    s = s.replace("renal cell carcinoma", "malignant neoplasm kidney")
    s = s.replace("kidney cancer", "malignant neoplasm kidney")
    s = s.replace("renal cancer", "malignant neoplasm kidney")
    s = s.replace("calculous", "calculus")
    s = s.replace("haemorrhage", "hemorrhage")
    s = s.replace("post operative", "postprocedural")
    s = s.replace("postoperative", "postprocedural")
    s = s.replace("post procedural", "postprocedural")
    s = _DX_FILLER_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _dx_tokens(text: str) -> set:
    return {t for t in _expand_clinical_text(text).split() if t not in _DX_STOP and len(t) > 1}


def _clinical_match(predicted: str, ground_truth: str, embedder: Any, threshold: float) -> Dict[str, Any]:
    """
    YES if same diagnosis despite abbreviations, extra ICD wording, or combination titles.
    Does not count unrelated diseases that merely share a generic word (e.g. anemia).
    """
    pred = (predicted or "").strip()
    gt = (ground_truth or "").strip()
    if not pred or not gt:
        return {"match": False, "reason": "missing", "similarity": 0.0}
    if pred.lower() == gt.lower():
        return {"match": True, "reason": "exact", "similarity": 1.0}

    sim = _text_sim(embedder, pred, gt)
    pe, ge = _expand_clinical_text(pred), _expand_clinical_text(gt)
    sim_e = _text_sim(embedder, pe, ge) if pe and ge else 0.0
    best = max(sim, sim_e)

    if len(pe) >= 10 and pe in ge:
        return {"match": True, "reason": "contained", "similarity": best}
    if len(ge) >= 10 and ge in pe:
        return {"match": True, "reason": "contained", "similarity": best}

    tp, tg = _dx_tokens(pred), _dx_tokens(gt)
    if tp and tg:
        shorter, longer = (tp, tg) if len(tp) <= len(tg) else (tg, tp)
        strong = any(len(t) >= 8 for t in shorter)
        if shorter <= longer and (len(shorter) >= 2 or strong):
            return {"match": True, "reason": "term_subset", "similarity": best}

    def both(phrases: List[str]) -> bool:
        return any(p in pe for p in phrases) and any(p in ge for p in phrases)

    if both(["sepsis", "septic"]):
        return {"match": True, "reason": "same_condition", "similarity": best}
    if both(["heart failure", "congestive heart"]):
        return {"match": True, "reason": "same_condition", "similarity": best}
    if both(["ventricular tachycardia"]):
        return {"match": True, "reason": "same_condition", "similarity": best}
    if both(["cholangitis"]):
        return {"match": True, "reason": "same_condition", "similarity": best}
    if both(["calculus", "renal calculus"]) and (
        any(x in pe for x in ("renal", "kidney", "ureter", "hydronephrosis", "nephro"))
        and any(x in ge for x in ("renal", "kidney", "ureter", "hydronephrosis", "nephro"))
    ):
        return {"match": True, "reason": "same_condition", "similarity": best}
    if both(["hemorrhage"]) and (
        any(x in pe for x in ("postprocedural", "hysterectomy", "procedur"))
        and any(x in ge for x in ("postprocedural", "hysterectomy", "procedur"))
    ):
        return {"match": True, "reason": "same_condition", "similarity": best}
    if both(["ketoacidosis"]):
        return {"match": True, "reason": "same_condition", "similarity": best}

    if best >= threshold - 1e-9:
        return {"match": True, "reason": "semantic", "similarity": best}
    return {"match": False, "reason": "no_match", "similarity": best}


def _best_clinical_against(
    predicted_list: List[str],
    gt_list: List[str],
    embedder: Any,
    threshold: float,
) -> Dict[str, Any]:
    """Best clinical match of any predicted string vs any ground-truth string."""
    best: Optional[Dict[str, Any]] = None

    def sort_key(row: Dict[str, Any]) -> Tuple:
        return (
            int(bool(row.get("match"))),
            float(row.get("similarity") or 0),
            -int(row.get("pred_rank") or 99),
            -int(row.get("gt_index") or 99),
        )

    for i, pred in enumerate(predicted_list or []):
        pred_s = " ".join(str(pred or "").split())
        if not pred_s:
            continue
        for j, gt in enumerate(gt_list or []):
            gt_s = " ".join(str(gt or "").split())
            if not gt_s:
                continue
            hit = _clinical_match(pred_s, gt_s, embedder, threshold)
            rec = {
                **hit,
                "predicted": pred_s,
                "ground_truth": gt_s,
                "pred_rank": i + 1,
                "gt_index": j,
            }
            if best is None or sort_key(rec) > sort_key(best):
                best = rec
    return best or {
        "match": False,
        "reason": "missing",
        "similarity": 0.0,
        "predicted": "",
        "ground_truth": "",
        "pred_rank": None,
        "gt_index": None,
    }


def _alt_clinical_ok(hit: Dict[str, Any]) -> bool:
    """Alternate-list hits need a structured clinical match; MiniLM-only 0.70 is too loose."""
    if not hit.get("match"):
        return False
    reason = str(hit.get("reason") or "")
    if reason in {"exact", "contained", "term_subset", "same_condition"}:
        return True
    return float(hit.get("similarity") or 0) >= 0.85


def _score_diagnosis(
    predicted_dx: str,
    gt_title: str,
    embedder: Any,
    threshold: float = DX_SEMANTIC_MATCH_THRESHOLD,
    other_predicted: Optional[List[str]] = None,
    other_gt: Optional[List[str]] = None,
) -> Dict[str, Any]:
    pred = (predicted_dx or "").strip()
    gt = (gt_title or "").strip()
    hit = _clinical_match(pred, gt, embedder, threshold)
    scope = "rank1_vs_primary"
    matched_pred, matched_gt = pred, gt
    pred_rank, gt_index = 1, 0

    if not hit.get("match"):
        alt = _best_clinical_against(other_predicted or [], [gt], embedder, threshold)
        if _alt_clinical_ok(alt):
            hit = alt
            scope = "diffdx_vs_primary"
            matched_pred = alt.get("predicted") or pred
            matched_gt = gt
            pred_rank = int(alt.get("pred_rank") or 0)
            gt_index = 0

    if not hit.get("match"):
        alt = _best_clinical_against([pred], other_gt or [], embedder, threshold)
        if _alt_clinical_ok(alt):
            hit = alt
            scope = "rank1_vs_gt_list"
            matched_pred = pred
            matched_gt = alt.get("ground_truth") or gt
            pred_rank = 1
            gt_index = int(alt.get("gt_index") or 0)

    clinical_reason = str(hit.get("reason") or "no_match")
    reason = clinical_reason
    if hit.get("match") and scope != "rank1_vs_primary":
        reason = scope
    if clinical_reason == "exact":
        match_type = "exact"
    elif clinical_reason == "semantic":
        match_type = "similarity"
    elif hit.get("match"):
        match_type = "clinical_rule"
    else:
        match_type = "no_match"
    return {
        "predicted": pred,
        "ground_truth": gt,
        "matched_predicted": matched_pred,
        "matched_ground_truth": matched_gt,
        "match_scope": scope,
        "match_pred_rank": pred_rank,
        "match_gt_index": gt_index,
        "exact": clinical_reason == "exact",
        "semantic_similarity": round(float(hit.get("similarity") or 0), 4),
        "semantic_match": bool(hit.get("match")),
        "match_reason": reason,
        "clinical_reason": clinical_reason,
        "match_type": match_type,
        "threshold": threshold,
    }


def cosine_similarity_text_local(a: str, b: str) -> float:
    """Tiny char-ngram cosine so Stage 10 works even if snomed_ct import fails."""
    def ngrams(text: str, n: int = 3):
        s = f" {(text or '').lower()} "
        if len(s) < n:
            return {}
        c: Dict[str, int] = {}
        for i in range(len(s) - n + 1):
            g = s[i : i + n]
            c[g] = c.get(g, 0) + 1
        return c

    ca, cb = ngrams(a), ngrams(b)
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    dot = sum(ca.get(k, 0) * cb.get(k, 0) for k in keys)
    na = sum(v * v for v in ca.values()) ** 0.5
    nb = sum(v * v for v in cb.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def evaluate_admission_accuracy(
    admission_dir: Path,
    embedder: Any = None,
) -> Dict[str, Any]:
    admission_dir = Path(admission_dir)
    gt = _load_gt(admission_dir)
    s8_path = admission_dir / "icd_coding.json"
    s9_path = admission_dir / "icd_coding_confirmed.json"
    stage08 = json.loads(s8_path.read_text(encoding="utf-8")) if s8_path.exists() else {}
    stage09 = json.loads(s9_path.read_text(encoding="utf-8")) if s9_path.exists() else {}

    pid = str(gt.get("patient_id") or stage08.get("patient_id") or stage09.get("patient_id") or "")
    hid = str(gt.get("hadm_id") or stage08.get("hadm_id") or stage09.get("hadm_id") or "")
    gt_codes = list(gt.get("icd10_codes") or [])
    gt_titles = list(gt.get("dx_titles") or [])
    gt_primary_code = gt.get("primary_icd_code")
    gt_primary_title = gt.get("primary_dx_title") or ""

    pred8 = _stage8_predicted_codes(stage08) if stage08 else {}
    pred9 = _stage9_predicted_codes(stage09) if stage09 else {}
    items8 = list(pred8.get("diagnosis_items") or [])

    def _attach_diffdx_meta(block: Dict[str, Any], diagnoses: List[str], items: List[Dict[str, Any]]) -> None:
        block["diagnoses"] = list(diagnoses or [])
        block["n_diffdx"] = len(block["diagnoses"])
        block["diagnosis_items"] = items
        rank1 = items[0] if items else {}
        block["rank1_score"] = rank1.get("score")
        block["rank1_confidence"] = rank1.get("confidence") or ""
        matched_name = (block.get("matched_predicted") or block.get("predicted") or "").strip().lower()
        matched_rank = block.get("match_pred_rank") if block.get("semantic_match") else 1
        found: Dict[str, Any] = {}
        if matched_rank:
            for it in items:
                if int(it.get("rank") or 0) == int(matched_rank):
                    found = it
                    break
        if not found and matched_name:
            for it in items:
                if str(it.get("diagnosis") or "").strip().lower() == matched_name:
                    found = it
                    break
        if not found:
            found = rank1
        block["matched_score"] = found.get("score")
        block["matched_confidence"] = found.get("confidence") or ""

    dx8 = _score_diagnosis(
        pred8.get("primary_dx") or "",
        gt_primary_title,
        embedder,
        other_predicted=pred8.get("diagnoses") or [],
        other_gt=gt_titles,
    )
    _attach_diffdx_meta(dx8, pred8.get("diagnoses") or [], items8)
    dx9 = (
        _score_diagnosis(
            pred9.get("primary_dx") or "",
            gt_primary_title,
            embedder,
            other_predicted=pred9.get("diagnoses") or [],
            other_gt=gt_titles,
        )
        if pred9
        else None
    )
    if dx9 is not None:
        # Stage 9 names; scores still come from Stage 7/8 DiffDx items when the name matches.
        _attach_diffdx_meta(dx9, pred9.get("diagnoses") or [], items8)

    icd8_primary = _score_primary_icd_multi(
        pred8.get("primary_code"),
        gt_primary_code,
        pred8.get("primary_title") or "",
        gt_primary_title,
        pred8.get("primary_dx") or "",
        pred8.get("all_codes") or [],
        gt_codes,
        gt_titles,
        embedder=embedder,
    )
    icd9_primary = (
        _score_primary_icd_multi(
            pred9.get("primary_code"),
            gt_primary_code,
            pred9.get("primary_title") or "",
            gt_primary_title,
            pred9.get("primary_dx") or "",
            pred9.get("all_codes") or [],
            gt_codes,
            gt_titles,
            embedder=embedder,
        )
        if pred9
        else None
    )

    set8 = _code_set_metrics(
        pred8.get("all_codes") or [], gt_codes, gt_titles, embedder=embedder
    )
    set8_nr = _code_set_metrics(
        pred8.get("all_codes") or [],
        gt_codes,
        gt_titles,
        exclude_r_codes=True,
        embedder=embedder,
    )
    set8_p = _code_set_metrics(
        pred8.get("principal_codes") or [], gt_codes, gt_titles, embedder=embedder
    )
    set9 = (
        _code_set_metrics(
            pred9.get("all_codes") or [], gt_codes, gt_titles, embedder=embedder
        )
        if pred9
        else None
    )
    set9_nr = (
        _code_set_metrics(
            pred9.get("all_codes") or [],
            gt_codes,
            gt_titles,
            exclude_r_codes=True,
            embedder=embedder,
        )
        if pred9
        else None
    )

    return {
        "stage": 10,
        "type": "accuracy_vs_ground_truth",
        "patient_id": pid,
        "hadm_id": hid,
        "generated_at": datetime.now().isoformat(),
        "ground_truth": {
            "primary_icd_code": gt_primary_code,
            "primary_dx_title": gt_primary_title,
            "icd10_codes": gt_codes,
            "dx_titles": gt_titles,
            "n_diagnoses": gt.get("n_diagnoses") or len(gt_codes),
        },
        "stage8": {
            "diagnosis": dx8,
            "primary_icd": icd8_primary,
            "code_set": set8,
            "code_set_excluding_r": set8_nr,
            "principal_set": set8_p,
        },
        "stage9": {
            "diagnosis": dx9,
            "primary_icd": icd9_primary,
            "code_set": set9,
            "code_set_excluding_r": set9_nr,
        }
        if pred9
        else None,
        "has_stage8": bool(stage08),
        "has_stage9": bool(stage09),
        "has_ground_truth": bool(gt_codes or gt_primary_code),
    }


def _clip(text: Any, width: int = 42) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= width:
        return s
    return s[: max(0, width - 3)] + "..."


def _yes_no(flag: Any) -> str:
    return "YES" if flag else "NO"


def format_accuracy_comparison_txt(result: Dict[str, Any]) -> str:
    gt = result.get("ground_truth") or {}
    s8 = result.get("stage8") or {}
    s9 = result.get("stage9") or {}
    dx8 = s8.get("diagnosis") or {}
    dx9 = (s9 or {}).get("diagnosis") or {}
    p8 = s8.get("primary_icd") or {}
    p9 = (s9 or {}).get("primary_icd") or {}
    c8 = s8.get("code_set") or {}
    c9 = (s9 or {}).get("code_set") or {}
    gt_dx = dx8.get("ground_truth") or gt.get("primary_dx_title") or "(none)"
    gt_icd = p8.get("ground_truth") or _dot_icd_code(str(gt.get("primary_icd_code") or "")) or "(none)"

    def yn(v: Any) -> str:
        return "YES" if v else "NO"

    lines = [
        _format_section("ACCURACY vs GROUND TRUTH"),
        f"Patient ID       : {result.get('patient_id', 'N/A')}",
        f"HADM ID          : {result.get('hadm_id', 'N/A')}",
        f"Generated        : {result.get('generated_at', 'N/A')}",
        "",
        _format_section("PRIMARY DIFFDx — MATCH OR NO", "-"),
        "  MATCH = YES if rank-1 matches GT primary, GT primary is elsewhere in the DiffDx,",
        "  or rank-1 matches another billed GT diagnosis (same-condition rules apply)",
        "",
        f"  {'':<12} {'Predicted':<48} {'Ground truth'}",
        f"  {'S8 DiffDx':<12} {_clip(dx8.get('predicted'), 46):<48} {_clip(gt_dx, 46)}",
        f"  {'S9 DiffDx':<12} {_clip(dx9.get('predicted') or '(none)', 46):<48} {_clip(gt_dx, 46)}",
        "",
        f"  Full predicted (S8): {dx8.get('predicted') or '(none)'}",
        f"  Full predicted (S9): {dx9.get('predicted') or '(none)'}",
        f"  Full GT primary    : {gt_dx}",
        "",
        f"  MATCH (S8): {yn(dx8.get('semantic_match'))}"
        f"    reason={dx8.get('match_reason') or '-'}  sim={dx8.get('semantic_similarity', 0):.2f}",
        f"    matched: {_clip(dx8.get('matched_predicted'), 40)}  vs  "
        f"{_clip(dx8.get('matched_ground_truth'), 40)}",
        f"  MATCH (S9): {yn(dx9.get('semantic_match')) if dx9 else 'NO'}"
        f"    reason={(dx9.get('match_reason') if dx9 else '-') or '-'}  "
        f"sim={(dx9.get('semantic_similarity') if dx9 else 0) or 0:.2f}",
        f"    matched: {_clip((dx9 or {}).get('matched_predicted'), 40)}  vs  "
        f"{_clip((dx9 or {}).get('matched_ground_truth'), 40)}",
        "",
        _format_section("PRIMARY ICD — MATCH OR NO", "-"),
        "  MATCH = YES if rank-1 ICD matches GT primary, another package code matches GT primary,",
        "  or rank-1 ICD matches another billed GT code",
        "",
        f"  {'':<12} {'Predicted':<48} {'Ground truth'}",
        f"  {'S8 ICD':<12} {_clip((p8.get('predicted') or '') + ' ' + (p8.get('predicted_title') or ''), 46):<48} "
        f"{_clip(gt_icd + ' ' + (gt.get('primary_dx_title') or ''), 46)}",
        f"  {'S9 ICD':<12} {_clip(((p9 or {}).get('predicted') or '') + ' ' + ((p9 or {}).get('predicted_title') or ''), 46):<48} "
        f"{_clip(gt_icd + ' ' + (gt.get('primary_dx_title') or ''), 46)}",
        "",
        f"  Full predicted (S8): {p8.get('predicted') or '(none)'} — {p8.get('predicted_title') or ''}",
        f"  Full predicted (S9): {(p9 or {}).get('predicted') or '(none)'} — {(p9 or {}).get('predicted_title') or ''}",
        f"  Full ground truth  : {gt_icd} — {gt.get('primary_dx_title') or ''}",
        "",
        f"  MATCH (S8): {yn(p8.get('semantic_match'))}"
        f"    reason={p8.get('match_reason') or '-'}  family={yn(p8.get('family'))}  "
        f"sim={p8.get('semantic_similarity', 0):.2f}",
        f"  MATCH (S9): {yn((p9 or {}).get('semantic_match')) if p9 else 'NO'}"
        f"    reason={((p9 or {}).get('match_reason') if p9 else '-') or '-'}  "
        f"family={yn((p9 or {}).get('family')) if p9 else 'NO'}  "
        f"sim={((p9 or {}).get('semantic_similarity') if p9 else 0) or 0:.2f}",
    ]

    def dump_set(title: str, metrics: Dict[str, Any]) -> None:
        if not metrics:
            return
        lines.extend(
            [
                "",
                _format_section(title, "-"),
                f"Exact     P / R / F1 : "
                f"{metrics.get('precision', 0):.3f} / {metrics.get('recall', 0):.3f} / {metrics.get('f1', 0):.3f}",
                f"Semantic  P / R / F1 : "
                f"{metrics.get('semantic_precision', 0):.3f} / {metrics.get('semantic_recall', 0):.3f} / "
                f"{metrics.get('semantic_f1', 0):.3f}  (title MiniLM ≥ {metrics.get('semantic_threshold', 0.70)})",
                f"Exact hits {metrics.get('n_hits', 0)}  |  semantic hits {metrics.get('n_semantic_hits', 0)}  |  "
                f"predicted {metrics.get('n_predicted', 0)}  |  GT {metrics.get('n_ground_truth', 0)}",
            ]
        )
        if metrics.get("hits"):
            lines.append("Exact code hits:")
            for row in metrics["hits"]:
                lines.append(f"  ✓ {row.get('code')} — {row.get('title') or ''}")
        if metrics.get("semantic_pairs"):
            lines.append("Semantic hits (same meaning, different code):")
            for row in metrics["semantic_pairs"]:
                pred_dx = row.get("predicted_diagnosis") or ""
                extra = f"  [{pred_dx}]" if pred_dx else ""
                lines.append(
                    f"  ≈ {row.get('predicted_code')} — {row.get('predicted_title') or ''}{extra}"
                )
                lines.append(
                    f"      ↔ {row.get('ground_truth_code')} — {row.get('ground_truth_title') or ''}  "
                    f"(sim={row.get('similarity', 0):.2f})"
                )
        if metrics.get("related_pairs"):
            lines.append("Related (not counted; sim 0.50–0.70):")
            for row in metrics["related_pairs"]:
                pred_dx = row.get("predicted_diagnosis") or ""
                extra = f"  [{pred_dx}]" if pred_dx else ""
                lines.append(
                    f"  ~ {row.get('predicted_code')} — {row.get('predicted_title') or ''}{extra}"
                )
                lines.append(
                    f"      ↔ {row.get('ground_truth_code')} — {row.get('ground_truth_title') or ''}  "
                    f"(sim={row.get('similarity', 0):.2f})"
                )
        still_miss = metrics.get("semantic_misses")
        still_extra = metrics.get("semantic_extras")
        if still_miss:
            lines.append("Still missing after semantic match:")
            for row in still_miss:
                lines.append(f"  ✗ {row.get('code')} — {row.get('title') or ''}")
        if still_extra:
            lines.append("Still extra after semantic match:")
            for row in still_extra:
                lines.append(f"  + {row.get('code')} — {row.get('title') or ''}")

    dump_set("ICD CODE SET — Stage 8 vs GT", c8)
    dump_set("ICD CODE SET — Stage 9 vs GT", c9)

    lines.extend(
        [
            "",
            _format_section("GROUND TRUTH LIST", "-"),
        ]
    )
    for i, code in enumerate(gt.get("icd10_codes") or []):
        title = (gt.get("dx_titles") or [None])[i] if i < len(gt.get("dx_titles") or []) else ""
        mark = " (primary)" if i == 0 else ""
        lines.append(f"  {i+1:2d}. {_dot_icd_code(str(code))} — {title}{mark}")
    lines.append("")
    return "\n".join(lines)


def export_accuracy_to_admission(result: Dict[str, Any], admission_dir: Path) -> Path:
    acc_dir = Path(admission_dir) / "accuracy"
    acc_dir.mkdir(parents=True, exist_ok=True)
    _write_json(acc_dir / "accuracy.json", result)
    _write_text(acc_dir / "comparison.txt", format_accuracy_comparison_txt(result))
    return acc_dir


def _mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _rate(flags: List[bool]) -> float:
    return round(sum(1 for x in flags if x) / len(flags), 4) if flags else 0.0


def summarize_stage10(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def collect(path_stage: str, metric_group: str, field: str) -> List[float]:
        out = []
        for r in results:
            block = r.get(path_stage) or {}
            grp = block.get(metric_group) or {}
            if grp.get(field) is not None:
                out.append(float(grp[field]))
        return out

    def flags(path_stage: str, metric_group: str, field: str) -> List[bool]:
        out = []
        for r in results:
            block = r.get(path_stage) or {}
            grp = block.get(metric_group) or {}
            if field in grp:
                out.append(bool(grp.get(field)))
        return out

    def pack(stage_key: str) -> Dict[str, Any]:
        cs = "code_set"
        return {
            "dx_exact_rate": _rate(flags(stage_key, "diagnosis", "exact")),
            "dx_semantic_match_rate": _rate(flags(stage_key, "diagnosis", "semantic_match")),
            "dx_mean_similarity": _mean(collect(stage_key, "diagnosis", "semantic_similarity")),
            "primary_icd_exact_rate": _rate(flags(stage_key, "primary_icd", "exact")),
            "primary_icd_family_rate": _rate(flags(stage_key, "primary_icd", "family")),
            "primary_icd_semantic_rate": _rate(flags(stage_key, "primary_icd", "semantic_match")),
            "primary_icd_mean_similarity": _mean(collect(stage_key, "primary_icd", "semantic_similarity")),
            "mean_precision": _mean(collect(stage_key, cs, "precision")),
            "mean_recall": _mean(collect(stage_key, cs, "recall")),
            "mean_f1": _mean(collect(stage_key, cs, "f1")),
            "mean_semantic_precision": _mean(collect(stage_key, cs, "semantic_precision")),
            "mean_semantic_recall": _mean(collect(stage_key, cs, "semantic_recall")),
            "mean_semantic_f1": _mean(collect(stage_key, cs, "semantic_f1")),
            "mean_f1_excluding_r": _mean(collect(stage_key, "code_set_excluding_r", "f1")),
            "mean_semantic_f1_excluding_r": _mean(
                collect(stage_key, "code_set_excluding_r", "semantic_f1")
            ),
            "n_scored": sum(1 for r in results if r.get(stage_key)),
            "dx_match_n": sum(1 for x in flags(stage_key, "diagnosis", "semantic_match") if x),
            "dx_exact_n": sum(1 for x in flags(stage_key, "diagnosis", "exact") if x),
            "icd_match_n": sum(1 for x in flags(stage_key, "primary_icd", "semantic_match") if x),
            "icd_exact_n": sum(1 for x in flags(stage_key, "primary_icd", "exact") if x),
        }

    return {
        "n_admissions": len(results),
        "n_with_stage9": sum(1 for r in results if r.get("has_stage9")),
        "stage8": pack("stage8"),
        "stage9": pack("stage9"),
        "dx_semantic_threshold": DX_SEMANTIC_MATCH_THRESHOLD,
        "icd_title_semantic_threshold": ICD_TITLE_SEMANTIC_THRESHOLD,
    }


def format_primary_match_table(results: List[Dict[str, Any]], stage_key: str = "stage9") -> str:
    """Side-by-side predicted vs GT primary DiffDx + ICD with YES/NO match."""
    lines = [
        _format_section(f"PRIMARY DIFFDx TABLE ({stage_key} vs ground truth)"),
        "MATCH = YES if rank-1 matches GT primary, GT primary is elsewhere in the DiffDx,",
        "or rank-1 matches another billed GT diagnosis.",
        "",
        f"{'MATCH':<6} {'why':<18} {'sim':>5}  {'patient':<10} {'predicted':<36} {'ground truth'}",
        "-" * 124,
    ]
    n_yes = 0
    n = 0
    for r in results:
        block = ((r.get(stage_key) or {}) or {}).get("diagnosis") or {}
        if not block and stage_key == "stage9":
            block = (r.get("stage8") or {}).get("diagnosis") or {}
        pred = block.get("matched_predicted") or block.get("predicted") or "(none)"
        gt = block.get("matched_ground_truth") or block.get("ground_truth") or "(none)"
        sim = float(block.get("semantic_similarity") or 0)
        match = bool(block.get("semantic_match"))
        why = str(block.get("match_reason") or ("semantic" if match else "no_match"))
        n += 1
        if match:
            n_yes += 1
        lines.append(
            f"{'YES' if match else 'NO':<6} {why:<18} {sim:>5.2f}  {str(r.get('patient_id') or ''):<10} "
            f"{_clip(pred, 34):<36} {_clip(gt, 42)}"
        )
    pct = (100.0 * n_yes / n) if n else 0.0
    lines.extend(
        [
            "-" * 120,
            f"ACCURACY: {n_yes} YES / {n} = {pct:.1f}%",
            "",
            _format_section(f"PRIMARY ICD TABLE ({stage_key} vs ground truth)"),
            "MATCH = YES if rank-1 ICD matches GT primary, another package code matches GT primary,",
            "or rank-1 ICD matches another billed GT code.",
            "",
            f"{'MATCH':<6} {'sim':>5}  {'patient':<10} {'pred ICD':<12} {'GT ICD':<12} {'predicted title':<28} {'GT title'}",
            "-" * 120,
        ]
    )
    n_yes_icd = 0
    for r in results:
        block = ((r.get(stage_key) or {}) or {}).get("primary_icd") or {}
        if not block and stage_key == "stage9":
            block = (r.get("stage8") or {}).get("primary_icd") or {}
        pred = block.get("predicted") or "-"
        gt = block.get("ground_truth") or "-"
        sim = float(block.get("semantic_similarity") or 0)
        match = bool(block.get("semantic_match"))
        if match:
            n_yes_icd += 1
        lines.append(
            f"{'YES' if match else 'NO':<6} {sim:>5.2f}  {str(r.get('patient_id') or ''):<10} "
            f"{_clip(pred, 10):<12} {_clip(gt, 10):<12} "
            f"{_clip(block.get('predicted_title'), 26):<28} {_clip(block.get('ground_truth_title'), 36)}"
        )
    pct_i = (100.0 * n_yes_icd / n) if n else 0.0
    lines.extend(
        [
            "-" * 120,
            f"ACCURACY: {n_yes_icd} YES / {n} = {pct_i:.1f}%",
            "",
        ]
    )
    return "\n".join(lines)


def write_primary_match_csv(results: List[Dict[str, Any]], path: Union[str, Path] = None) -> Path:
    """Write two CSVs: diagnosis match, then ICD match for that same stay (no stacked table column)."""
    out_dir = Path(path).parent if path else Path(STAGE_10_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    dx_path = Path(EVAL_DIFFDX_CSV) if "EVAL_DIFFDX_CSV" in globals() else out_dir / "primary_diffdx_match.csv"
    icd_path = Path(EVAL_ICD_CSV) if "EVAL_ICD_CSV" in globals() else out_dir / "primary_icd_match.csv"

    def numbered(items: List[str], meta: Optional[List[Dict[str, Any]]] = None) -> str:
        parts = []
        by_name = {
            str(m.get("diagnosis") or "").strip().lower(): m for m in (meta or [])
        }
        for i, name in enumerate(items):
            text = str(name or "").strip()
            if not text:
                continue
            info = by_name.get(text.lower()) or {}
            score = info.get("score")
            conf = info.get("confidence") or ""
            extra = ""
            if score is not None or conf:
                extra = f" (score={score if score is not None else '-'}, {conf or '-'})"
            parts.append(f"{i + 1}: {text}{extra}")
        return " | ".join(parts)

    dx_fields = [
        "stage",
        "match",
        "match_type",
        "clinical_reason",
        "match_scope",
        "sim",
        "patient_id",
        "hadm_id",
        "n_diffdx",
        "matched_diffdx_rank",
        "rank1_diagnosis",
        "rank1_score",
        "rank1_confidence",
        "matched_diagnosis",
        "matched_score",
        "matched_confidence",
        "diffdx_list",
        "n_gt",
        "matched_gt_seq",
        "gt_primary",
        "matched_gt",
        "gt_list",
    ]
    icd_fields = [
        "stage",
        "diagnosis_match",
        "matched_diagnosis",
        "matched_diffdx_rank",
        "matched_score",
        "matched_confidence",
        "icd_match",
        "match_type",
        "clinical_reason",
        "match_scope",
        "sim",
        "patient_id",
        "hadm_id",
        "pred_icd",
        "pred_icd_title",
        "gt_icd",
        "gt_icd_title",
        "gt_primary",
    ]
    dx_rows: List[Dict[str, Any]] = []
    icd_rows: List[Dict[str, Any]] = []
    for stage_key in ("stage8", "stage9"):
        for r in results:
            dx = ((r.get(stage_key) or {}) or {}).get("diagnosis") or {}
            icd = ((r.get(stage_key) or {}) or {}).get("primary_icd") or {}
            if not dx and not icd:
                continue
            gt = r.get("ground_truth") or {}
            dx_match = bool(dx.get("semantic_match"))
            gt_index = dx.get("match_gt_index")
            pred_rank = dx.get("match_pred_rank")
            diagnoses = list(dx.get("diagnoses") or [])
            items = list(dx.get("diagnosis_items") or [])
            gt_titles = list(gt.get("dx_titles") or [])
            dx_list = numbered(diagnoses, items)
            matched_dx = dx.get("matched_predicted") or dx.get("predicted") or ""
            dx_rows.append(
                {
                    "stage": stage_key,
                    "match": "YES" if dx_match else "NO",
                    "match_type": dx.get("match_type") or ("no_match" if not dx_match else ""),
                    "clinical_reason": dx.get("clinical_reason") or dx.get("match_reason") or "",
                    "match_scope": dx.get("match_scope") or "",
                    "sim": round(float(dx.get("semantic_similarity") or 0), 4),
                    "patient_id": r.get("patient_id") or "",
                    "hadm_id": r.get("hadm_id") or "",
                    "n_diffdx": dx.get("n_diffdx") if dx.get("n_diffdx") is not None else len(diagnoses),
                    "matched_diffdx_rank": pred_rank if dx_match else "",
                    "rank1_diagnosis": dx.get("predicted") or "",
                    "rank1_score": dx.get("rank1_score") if dx.get("rank1_score") is not None else "",
                    "rank1_confidence": dx.get("rank1_confidence") or "",
                    "matched_diagnosis": matched_dx,
                    "matched_score": dx.get("matched_score") if dx.get("matched_score") is not None else "",
                    "matched_confidence": dx.get("matched_confidence") or "",
                    "diffdx_list": dx_list,
                    "n_gt": gt.get("n_diagnoses") or len(gt_titles),
                    "matched_gt_seq": (int(gt_index) + 1) if dx_match and gt_index is not None else "",
                    "gt_primary": dx.get("ground_truth") or gt.get("primary_dx_title") or "",
                    "matched_gt": dx.get("matched_ground_truth") or dx.get("ground_truth") or "",
                    "gt_list": numbered(gt_titles),
                }
            )
            icd_match = bool(icd.get("semantic_match"))
            icd_rows.append(
                {
                    "stage": stage_key,
                    "diagnosis_match": "YES" if dx_match else "NO",
                    "matched_diagnosis": matched_dx,
                    "matched_diffdx_rank": pred_rank if dx_match else "",
                    "matched_score": dx.get("matched_score") if dx.get("matched_score") is not None else "",
                    "matched_confidence": dx.get("matched_confidence") or "",
                    "icd_match": "YES" if icd_match else "NO",
                    "match_type": icd.get("match_type") or ("no_match" if not icd_match else ""),
                    "clinical_reason": icd.get("clinical_reason") or icd.get("match_reason") or "",
                    "match_scope": icd.get("match_scope") or "",
                    "sim": round(float(icd.get("semantic_similarity") or 0), 4),
                    "patient_id": r.get("patient_id") or "",
                    "hadm_id": r.get("hadm_id") or "",
                    "pred_icd": icd.get("predicted") or "",
                    "pred_icd_title": icd.get("predicted_title") or "",
                    "gt_icd": icd.get("ground_truth") or "",
                    "gt_icd_title": icd.get("ground_truth_title") or "",
                    "gt_primary": gt.get("primary_dx_title") or "",
                }
            )
    with dx_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dx_fields)
        writer.writeheader()
        writer.writerows(dx_rows)
    with icd_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=icd_fields)
        writer.writeheader()
        writer.writerows(icd_rows)
    old = out_dir / "primary_match_table.csv"
    if old.exists() and old.resolve() not in {dx_path.resolve(), icd_path.resolve()}:
        try:
            old.unlink()
        except OSError:
            pass
    return dx_path


def format_cohort_metrics_txt(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    s8 = summary.get("stage8") or {}
    s9 = summary.get("stage9") or {}
    n = int(summary.get("n_admissions") or 0)

    def acc_line(block: Dict[str, Any], n_key: str, rate_key: str) -> str:
        k = int(block.get(n_key) or 0)
        pct = 100.0 * float(block.get(rate_key) or 0)
        return f"{k} / {n}  =  {pct:.1f}%"

    lines = [
        _format_section("COHORT ACCURACY (Stage 10)"),
        f"Admissions scored : {n}",
        f"With Stage 9      : {summary.get('n_with_stage9')}",
        f"MATCH rule        : YES if rank-1 vs GT primary, GT primary elsewhere in DiffDx, "
        f"or rank-1 vs any billed GT dx (clinical match / MiniLM ≥ {summary.get('dx_semantic_threshold')})",
        "",
        "PRIMARY DIAGNOSIS ACCURACY (rank-1, other DiffDx, or other billed GT dx)",
        f"  Stage 8:  {acc_line(s8, 'dx_match_n', 'dx_semantic_match_rate')}",
        f"  Stage 9:  {acc_line(s9, 'dx_match_n', 'dx_semantic_match_rate')}",
        "",
        "PRIMARY ICD ACCURACY (rank-1, other package code vs GT primary, or rank-1 vs any billed GT code)",
        f"  Stage 8:  {acc_line(s8, 'icd_match_n', 'primary_icd_semantic_rate')}",
        f"  Stage 9:  {acc_line(s9, 'icd_match_n', 'primary_icd_semantic_rate')}",
        "",
        f"{'metric':<36} {'Stage 8':>10} {'Stage 9':>10}",
        "-" * 58,
    ]
    rows = [
        ("Top dx exact rate", "dx_exact_rate"),
        ("Top dx MATCH rate (accuracy)", "dx_semantic_match_rate"),
        ("Top dx mean similarity", "dx_mean_similarity"),
        ("Primary ICD exact rate", "primary_icd_exact_rate"),
        ("Primary ICD family rate", "primary_icd_family_rate"),
        ("Primary ICD MATCH rate (accuracy)", "primary_icd_semantic_rate"),
        ("Primary ICD mean similarity", "primary_icd_mean_similarity"),
        ("Mean ICD precision (exact)", "mean_precision"),
        ("Mean ICD recall (exact)", "mean_recall"),
        ("Mean ICD F1 (exact)", "mean_f1"),
        ("Mean ICD precision (semantic)", "mean_semantic_precision"),
        ("Mean ICD recall (semantic)", "mean_semantic_recall"),
        ("Mean ICD F1 (semantic)", "mean_semantic_f1"),
        ("Mean F1 excl. R (exact)", "mean_f1_excluding_r"),
        ("Mean F1 excl. R (semantic)", "mean_semantic_f1_excluding_r"),
    ]
    for label, key in rows:
        lines.append(f"{label:<36} {s8.get(key, 0):>10.3f} {s9.get(key, 0):>10.3f}")
    lines.append("")
    lines.append(format_primary_match_table(results, "stage8"))
    lines.append(format_primary_match_table(results, "stage9"))
    return "\n".join(lines).rstrip() + "\n"


def run_stage10_evaluation(
    export_dir: Union[str, Path] = None,
    use_embeddings: bool = True,
) -> Dict[str, Any]:
    """Score Stage 8 (and Stage 9 if present) against current-stay ground truth."""
    export_dir = Path(export_dir or EXPORT_DIR)
    STAGE_10_DIR.mkdir(parents=True, exist_ok=True)

    embedder = None
    if use_embeddings:
        try:
            from snomed_ct import TextEmbedder

            embedder = TextEmbedder(prefer_embeddings=True)
        except Exception as exc:  # noqa: BLE001
            print(f"MiniLM unavailable ({exc}); using char-ngram similarity.")
            embedder = None

    admissions = list_admission_export_dirs(export_dir)
    results: List[Dict[str, Any]] = []
    n_exported = 0
    texts: List[str] = []
    # Pre-encode diagnosis strings if embedder is available
    pending_pairs = []
    for adm in admissions:
        adm_dir = Path(adm["admission_dir"])
        if not (adm_dir / "ground_truth.json").exists():
            continue
        if not (adm_dir / "icd_coding.json").exists():
            continue
        pending_pairs.append(adm_dir)

    if embedder is not None:
        for adm_dir in pending_pairs:
            gt = _load_gt(adm_dir)
            texts.extend([t for t in (gt.get("dx_titles") or []) if t])
            if gt.get("primary_dx_title"):
                texts.append(gt["primary_dx_title"])
            s8 = json.loads((adm_dir / "icd_coding.json").read_text(encoding="utf-8"))
            pred8 = _stage8_predicted_codes(s8)
            if pred8.get("primary_dx"):
                texts.append(pred8["primary_dx"])
            texts.extend(pred8.get("diagnoses") or [])
            if pred8.get("primary_title"):
                texts.append(pred8["primary_title"])
            for row in pred8.get("all_codes") or []:
                texts.extend(_match_texts(row))
            s9p = adm_dir / "icd_coding_confirmed.json"
            if s9p.exists():
                s9 = json.loads(s9p.read_text(encoding="utf-8"))
                pred9 = _stage9_predicted_codes(s9)
                if pred9.get("primary_dx"):
                    texts.append(pred9["primary_dx"])
                texts.extend(pred9.get("diagnoses") or [])
                if pred9.get("primary_title"):
                    texts.append(pred9["primary_title"])
                for row in pred9.get("all_codes") or []:
                    texts.extend(_match_texts(row))
        expanded = [_expand_clinical_text(t) for t in texts if t]
        texts.extend(t for t in expanded if t)
        try:
            embedder.encode_many(texts, show_progress=False)
        except Exception:
            pass

    for adm_dir in pending_pairs:
        row = evaluate_admission_accuracy(adm_dir, embedder=embedder)
        export_accuracy_to_admission(row, adm_dir)
        results.append(row)
        n_exported += 1
        c8 = (row.get("stage8") or {}).get("code_set") or {}
        print(
            f"  {row.get('patient_id')} / {row.get('hadm_id')}: "
            f"S8 F1 exact/sem={c8.get('f1', 0):.2f}/{c8.get('semantic_f1', 0):.2f}  "
            f"dx_sim={(row.get('stage8') or {}).get('diagnosis', {}).get('semantic_similarity', 0):.2f}"
        )

    summary = summarize_stage10(results)
    payload = {
        "stage": 10,
        "description": (
            "Accuracy of top DiffDx + ICD packages vs current-stay ground truth "
            "(Stage 8 map-only and Stage 9 LLM-confirmed)"
        ),
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "results": results,
    }
    _write_json(Path(EVAL_SUMMARY_JSON), payload)
    _write_text(Path(EVAL_COHORT_TXT), format_cohort_metrics_txt(summary, results))
    table_path = Path(STAGE_10_DIR) / "primary_match_table.txt"
    _write_text(
        table_path,
        format_primary_match_table(results, "stage8")
        + "\n"
        + format_primary_match_table(results, "stage9"),
    )
    csv_path = write_primary_match_csv(results, Path(EVAL_PRIMARY_CSV))
    print(f"Saved Stage 10 → {EVAL_SUMMARY_JSON}")
    print(f"Cohort metrics → {EVAL_COHORT_TXT}")
    print(f"Primary match table → {table_path}")
    print(f"DiffDx CSV → {EVAL_DIFFDX_CSV}")
    print(f"ICD CSV → {EVAL_ICD_CSV}")
    print(f"Per-admission accuracy folders: {n_exported}")
    return payload
