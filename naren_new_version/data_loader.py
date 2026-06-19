import streamlit as st
import pandas as pd
from pathlib import Path

VITAL_ITEMIDS = {220045, 220210, 220277, 220179, 220180, 223762}
VITAL_LABELS = {
    220045: "HR",
    220210: "RR",
    220277: "SpO2",
    220179: "SBP",
    220180: "DBP",
    223762: "Temp",
}

_TABLE_PATHS = {
    "patients": [
        "hosp/patients.csv", "hosp/patients.csv.gz",
        "patients.csv", "patients.csv.gz",
    ],
    "admissions": [
        "hosp/admissions.csv", "hosp/admissions.csv.gz",
        "admissions.csv", "admissions.csv.gz",
    ],
    "discharge_notes": [
        "hosp/note/discharge.csv", "hosp/note/discharge.csv.gz",
        "note/discharge.csv", "note/discharge.csv.gz",
        "discharge.csv", "discharge.csv.gz",
        "noteevents.csv", "noteevents.csv.gz",
    ],
    "radiology_notes": [
        "hosp/note/radiology.csv", "hosp/note/radiology.csv.gz",
        "note/radiology.csv", "note/radiology.csv.gz",
        "radiology.csv", "radiology.csv.gz",
    ],
    "chartevents": [
        "icu/chartevents.csv", "icu/chartevents.csv.gz",
        "chartevents.csv", "chartevents.csv.gz",
    ],
    "diagnoses_icd": [
        "hosp/diagnoses_icd.csv", "hosp/diagnoses_icd.csv.gz",
        "diagnoses_icd.csv", "diagnoses_icd.csv.gz",
    ],
}


def find_mimic_file(table_key: str, base_dir: str) -> Path | None:
    base = Path(base_dir)
    for rel in _TABLE_PATHS.get(table_key, []):
        p = base / rel
        if p.exists():
            return p

    # Also search in sibling mimic-iv-note dataset (separate PhysioNet download).
    # base_dir is typically physionet.org/files/mimiciv/<version>/
    # MIMIC-IV-Note lives at       physionet.org/files/mimic-iv-note/<version>/note/
    if table_key in ("discharge_notes", "radiology_notes"):
        files_root = base.parent.parent  # physionet.org/files/
        note_root = files_root / "mimic-iv-note"
        if note_root.exists():
            stem = "discharge" if table_key == "discharge_notes" else "radiology"
            for version_dir in sorted(note_root.iterdir()):
                for ext in (".csv.gz", ".csv"):
                    p = version_dir / "note" / f"{stem}{ext}"
                    if p.exists():
                        return p

    return None


def _read_csv_flex(path: Path, **kwargs) -> pd.DataFrame:
    compression = "gzip" if str(path).endswith(".gz") else "infer"
    return pd.read_csv(path, compression=compression, low_memory=False, **kwargs)


@st.cache_data(show_spinner=False)
def load_patients(base_dir: str) -> pd.DataFrame:
    path = find_mimic_file("patients", base_dir)
    if path is None:
        return pd.DataFrame(columns=["subject_id", "gender", "anchor_age", "dod"])
    df = _read_csv_flex(path, usecols=lambda c: c in {"subject_id", "gender", "anchor_age", "dod"})
    df["subject_id"] = df["subject_id"].astype("int32")
    if "anchor_age" in df.columns:
        df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce").fillna(0).astype("int16")
    return df


@st.cache_data(show_spinner=False)
def load_admissions(base_dir: str) -> pd.DataFrame:
    path = find_mimic_file("admissions", base_dir)
    if path is None:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "diagnosis"])
    want = {"subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "diagnosis"}
    df = _read_csv_flex(path, usecols=lambda c: c in want)
    for col in ("admittime", "dischtime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["subject_id"] = df["subject_id"].astype("int32")
    df["hadm_id"] = df["hadm_id"].astype("int32")
    return df


@st.cache_data(show_spinner=False)
def load_notes(base_dir: str, note_type: str = "discharge") -> pd.DataFrame:
    key = "discharge_notes" if note_type == "discharge" else "radiology_notes"
    path = find_mimic_file(key, base_dir)
    if path is None:
        return pd.DataFrame(columns=["note_id", "subject_id", "hadm_id", "charttime", "text"])
    want = {"note_id", "subject_id", "hadm_id", "charttime", "storetime", "text", "note_type", "category"}
    df = _read_csv_flex(path, usecols=lambda c: c in want, dtype={"text": str})
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    # Normalize: ensure note_id column exists
    if "note_id" not in df.columns:
        df["note_id"] = df.index.astype(str)
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").fillna(0).astype("int32")
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").fillna(0).astype("int32")
    return df


@st.cache_data(show_spinner=False)
def load_vitals_for_admission(base_dir: str, subject_id: int, hadm_id: int) -> pd.DataFrame:
    path = find_mimic_file("chartevents", base_dir)
    if path is None:
        return pd.DataFrame(columns=["charttime", "itemid", "valuenum", "valueuom"])

    want_cols = {"subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"}
    compression = "gzip" if str(path).endswith(".gz") else "infer"
    chunks = []
    for chunk in pd.read_csv(
        path,
        compression=compression,
        usecols=lambda c: c in want_cols,
        chunksize=100_000,
        low_memory=False,
    ):
        mask = (
            (chunk["subject_id"] == subject_id)
            & (chunk["hadm_id"] == hadm_id)
            & (chunk["itemid"].isin(VITAL_ITEMIDS))
        )
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=["charttime", "itemid", "valuenum", "valueuom"])

    df = pd.concat(chunks, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")
    df = df.sort_values("charttime")
    return df


@st.cache_data(show_spinner=False)
def load_diagnoses(base_dir: str) -> pd.DataFrame:
    path = find_mimic_file("diagnoses_icd", base_dir)
    if path is None:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    want = {"subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"}
    df = _read_csv_flex(path, usecols=lambda c: c in want)
    if "icd_version" in df.columns:
        df = df[df["icd_version"] == 10]
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").fillna(0).astype("int32")
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").fillna(0).astype("int32")
    return df


def get_patient_admissions(
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    subject_id: int,
) -> tuple[pd.Series | None, pd.DataFrame]:
    patient_rows = patients_df[patients_df["subject_id"] == subject_id]
    patient_row = patient_rows.iloc[0] if not patient_rows.empty else None
    adm = admissions_df[admissions_df["subject_id"] == subject_id].sort_values(
        "admittime", ascending=False, na_position="last"
    )
    return patient_row, adm


def get_notes_for_admission(
    notes_df: pd.DataFrame,
    subject_id: int,
    hadm_id: int,
) -> pd.DataFrame:
    return notes_df[
        (notes_df["subject_id"] == subject_id) & (notes_df["hadm_id"] == hadm_id)
    ].sort_values("charttime", ascending=False, na_position="last")
