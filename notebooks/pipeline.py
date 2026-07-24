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
        g["EXPORT_DIR"] = g["REPO_ROOT"] / "patient_records_test" if tm else g["REPO_ROOT"] / "patient_records"


def _refresh_artifact_paths() -> None:
    global COHORT_PICKLE, COHORT_INDEX_JSON, IE_RESULTS_JSON, IE_CHECKPOINT_JSON, SYMPTOM_TREE_RESULTS_JSON
    COHORT_PICKLE = COHORT_DIR / "cohort.pkl"
    COHORT_INDEX_JSON = COHORT_DIR / "cohort_index.json"
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
