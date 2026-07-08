"""
Stage 1 — MIMIC cohort selection.

Samples patients with:
  - ≥2 admissions each with a discharge note (≥ MIN_NOTE_CHARS)
  - ICD-10 ground-truth diagnoses on every included admission
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from pipeline_config import (
    COHORT_INDEX_JSON,
    COHORT_PICKLE,
    DISCHARGE_NOTES_PATH,
    MAX_NOTE_CHARS,
    MIMIC_BASE,
    MIN_ADMISSIONS_PER_PATIENT,
    MIN_NOTE_CHARS,
    N_PATIENTS,
    RANDOM_SEED,
    TEST_MODE,
)


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


def load_mimic_cohort(
    n_patients: int = N_PATIENTS,
    min_admissions: int = MIN_ADMISSIONS_PER_PATIENT,
    min_note_chars: int = MIN_NOTE_CHARS,
    seed: int = RANDOM_SEED,
    max_note_chars: int = MAX_NOTE_CHARS,
) -> pd.DataFrame:
    """
    Sample n MIMIC patients with multi-admission discharge notes and ICD-10 labels.
    Only admissions with ICD-10 ground truth are kept.
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

    patient_summary = (
        cohort_df.groupby("patient_id")
        .agg(
            subject_id=("subject_id", "first"),
            n_admissions=("hadm_id", "count"),
            admission_ids=("admission_id", list),
            primary_codes=("primary_icd_code", list),
        )
        .reset_index()
    )

    index: Dict[str, Any] = {
        "stage": 1,
        "description": "MIMIC cohort selection — 15 patients, ICD-10 ground truth",
        "test_mode": TEST_MODE,
        "generated_at": datetime.now().isoformat(),
        "random_seed": seed,
        "n_patients": int(cohort_df["patient_id"].nunique()),
        "n_admissions": int(len(cohort_df)),
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
    return pd.read_pickle(pickle_path)


def load_cohort_index(index_path: Union[str, Path] = COHORT_INDEX_JSON) -> Dict[str, Any]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Cohort index not found at {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))
