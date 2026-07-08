from __future__ import annotations

import sys

import streamlit as st

if sys.version_info < (3, 10):
    st.error(
        "**Python 3.10+ is required.** "
        f"You are running Python {sys.version_info.major}.{sys.version_info.minor}. "
        "Please run: `brew install python@3.11` then restart with "
        "`python3.11 -m streamlit run app.py`"
    )
    st.stop()

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data_loader as dl
import llm_processor as llm
import icd10_client as icd_client
import highlighting as hl

# ── Config ──────────────────────────────────────────────────────────────────

MIMIC_BASE_DIR = str(Path(__file__).parent / "physionet.org" / "files" / "mimiciv" / "3.1")
TEST_MODE_PATIENT_LIMIT = 15

VITAL_NORMAL_RANGES = {
    "HR":   (60,   100),
    "RR":   (12,   20),
    "SpO2": (95,   100),
    "SBP":  (90,   140),
    "DBP":  (60,   90),
    "Temp": (36.1, 37.2),
}

st.set_page_config(
    page_title="Clinical ICD-10 Coding Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Fonts ────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App background ───────────────────────────────────────── */
    .stApp { background-color: #FFFDF7; }

    /* ── Sidebar ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF4E6 0%, #FFEFD6 100%);
        border-right: 1px solid #F5DEB3;
    }
    [data-testid="stSidebar"] h1 {
        color: #2A2218 !important;
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    /* ── Tabs ─────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: #FFF4E6;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
        color: #A07848;
        border-radius: 8px;
        padding: 8px 22px;
        border: none;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF8C42, #FFD060) !important;
        color: #1a0e04 !important;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(255,140,66,0.3);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"]    { display: none; }

    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF8C42, #FFD060);
        color: #1a0e04;
        border: none;
        box-shadow: 0 2px 8px rgba(255,140,66,0.25);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #FF7A2A, #FFC030);
        box-shadow: 0 4px 14px rgba(255,140,66,0.4);
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: #FFFDF7;
        color: #D4650A;
        border: 1.5px solid #F5C87A;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #FF8C42;
        background: #FFF4E6;
        color: #C45A06;
    }

    /* ── Metrics ──────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #F5DEB3;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 4px rgba(212,101,10,0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #A07848 !important;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #D4650A !important;
        font-weight: 700;
    }

    /* ── Inputs & Selects ─────────────────────────────────────── */
    .stTextInput > div > div > input {
        background: #FFFFFF !important;
        border: 1.5px solid #F5DEB3 !important;
        border-radius: 8px !important;
        color: #2C2416 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #FF8C42 !important;
        box-shadow: 0 0 0 3px rgba(255,140,66,0.15) !important;
    }

    /* ── Radio ────────────────────────────────────────────────── */
    .stRadio > div { gap: 8px; }
    .stRadio label {
        background: #FFFFFF;
        border: 1.5px solid #F5DEB3;
        border-radius: 8px;
        padding: 6px 16px;
        color: #A07848;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }
    .stRadio label:has(input:checked) {
        background: #FFF4E6;
        border-color: #FF8C42;
        color: #D4650A;
        font-weight: 600;
    }

    /* ── Dividers ─────────────────────────────────────────────── */
    hr { border-color: #F5DEB3 !important; }

    /* ── Expanders ────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #FFF4E6 !important;
        border: 1px solid #F5DEB3 !important;
        border-radius: 8px !important;
        color: #2C2416 !important;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background: #FFFDF7 !important;
        border: 1px solid #F5DEB3 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* ── DataFrames ───────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #F5DEB3;
    }

    /* ── Spinner ──────────────────────────────────────────────── */
    .stSpinner > div { border-top-color: #FF8C42 !important; }

    /* ── Alert boxes ──────────────────────────────────────────── */
    .stAlert { border-radius: 10px; border-left-width: 4px; }

    /* ── Scrollbar ────────────────────────────────────────────── */
    ::-webkit-scrollbar       { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #FFF4E6; }
    ::-webkit-scrollbar-thumb { background: #F5C87A; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF8C42; }

    /* ── Badges ───────────────────────────────────────────────── */
    .badge-green {
        background: #E6F4EA;
        color: #2D6A4F;
        border: 1px solid #A8D5B5;
        padding: 3px 12px;
        border-radius: 12px; font-size: 12px; font-weight: 600;
    }
    .badge-red {
        background: #FDECEA;
        color: #C0392B;
        border: 1px solid #F5ABAB;
        padding: 3px 12px;
        border-radius: 12px; font-size: 12px; font-weight: 600;
    }
    .badge-orange {
        background: #FFF4E6;
        color: #D4650A;
        border: 1px solid #F5C87A;
        padding: 3px 12px;
        border-radius: 12px; font-size: 12px; font-weight: 600;
    }

    /* ── Headings ─────────────────────────────────────────────── */
    h1 { color: #2A2218 !important; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #3D3530 !important; font-weight: 600; }
    h3 { color: #4A4038 !important; }
    .stSubheader { color: #3D3530 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "selected_patient_id": None,
        "selected_hadm_id": None,
        "selected_note_id": None,
        "extracted_entities": {},     # note_id -> list[dict]
        "processing_status": {},      # note_id -> "idle"|"running"|"done"|"error"|str(error)
        "suggested_codes": {},        # hadm_id -> list[dict]
        "code_decisions": {},         # hadm_id -> {code: "approved"|"rejected"|"pending"}
        "active_tab": 0,
        "test_mode": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Ollama health (cached 30s) ───────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def _ollama_status() -> tuple[bool, str]:
    return llm.check_ollama_health()


def _apply_test_mode(
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not st.session_state.get("test_mode", True):
        return patients_df, admissions_df

    limited_patients = patients_df.sort_values("subject_id").head(TEST_MODE_PATIENT_LIMIT)
    allowed_ids = set(limited_patients["subject_id"].tolist())
    limited_admissions = admissions_df[admissions_df["subject_id"].isin(allowed_ids)]
    return limited_patients, limited_admissions


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    total_patient_count: int,
):
    with st.sidebar:
        # Ollama status
        healthy, msg = _ollama_status()
        if healthy:
            st.markdown(f'<span class="badge-green">● Ollama</span> <small>{msg}</small>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-red">● Ollama offline</span>', unsafe_allow_html=True)
            st.caption(msg)
        st.divider()

        if patients_df.empty:
            st.warning("No patient data found in `mimic_dataset/`. Please add MIMIC-IV files.")
            return

        # Search filter
        search = st.text_input("Search by Subject ID or Diagnosis", placeholder="e.g. 10001 or sepsis")

        # Build display list
        merged = patients_df.copy()
        if not admissions_df.empty and "diagnosis" in admissions_df.columns:
            latest_dx = (
                admissions_df.sort_values("admittime", na_position="last")
                .groupby("subject_id", as_index=False)
                .last()[["subject_id", "diagnosis"]]
            )
            merged = merged.merge(latest_dx, on="subject_id", how="left")
        else:
            merged["diagnosis"] = ""

        if search.strip():
            q = search.strip().lower()
            mask = (
                merged["subject_id"].astype(str).str.contains(q)
                | merged.get("diagnosis", pd.Series(dtype=str)).fillna("").str.lower().str.contains(q)
            )
            filtered = merged[mask]
        else:
            filtered = merged

        if filtered.empty:
            st.info("No patients match your search.")
            return

        def _pat_label(row) -> str:
            gender = row.get("gender", "?")
            age = row.get("anchor_age", "?")
            dx = str(row.get("diagnosis", "") or "")[:40]
            label = f"ID {row['subject_id']} — {gender}, age {age}"
            if dx:
                label += f" | {dx}"
            return label

        patient_options = filtered["subject_id"].tolist()
        patient_labels = {row["subject_id"]: _pat_label(row) for _, row in filtered.iterrows()}

        if st.session_state.selected_patient_id not in patient_options:
            st.session_state.selected_patient_id = None
            st.session_state.selected_hadm_id = None
            st.session_state.selected_note_id = None

        current_pid = st.session_state.selected_patient_id
        default_idx = patient_options.index(current_pid) if current_pid in patient_options else 0

        selected_pid = st.selectbox(
            "Select Patient",
            options=patient_options,
            format_func=lambda pid: patient_labels.get(pid, str(pid)),
            index=default_idx,
            key="patient_select",
        )

        if selected_pid != st.session_state.selected_patient_id:
            st.session_state.selected_patient_id = selected_pid
            st.session_state.selected_hadm_id = None
            st.session_state.selected_note_id = None

        # Admission selector
        if not admissions_df.empty:
            pat_adms = admissions_df[admissions_df["subject_id"] == selected_pid].sort_values(
                "admittime", ascending=False, na_position="last"
            )
        else:
            pat_adms = pd.DataFrame()

        if pat_adms.empty:
            st.info("No admissions found for this patient.")
            st.session_state.selected_hadm_id = None
            return

        def _adm_label(row) -> str:
            dt = row.get("admittime")
            dt_str = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "Unknown date"
            adm_type = str(row.get("admission_type", "") or "")
            dx = str(row.get("diagnosis", "") or "")[:35]
            label = f"Adm {row['hadm_id']} ({dt_str}) — {adm_type}"
            if dx:
                label += f" | {dx}"
            return label

        hadm_options = pat_adms["hadm_id"].tolist()
        adm_labels = {row["hadm_id"]: _adm_label(row) for _, row in pat_adms.iterrows()}

        current_hadm = st.session_state.selected_hadm_id
        default_adm_idx = hadm_options.index(current_hadm) if current_hadm in hadm_options else 0

        selected_hadm = st.selectbox(
            "Select Admission",
            options=hadm_options,
            format_func=lambda hid: adm_labels.get(hid, str(hid)),
            index=default_adm_idx,
            key="admission_select",
        )

        if selected_hadm != st.session_state.selected_hadm_id:
            st.session_state.selected_hadm_id = selected_hadm
            st.session_state.selected_note_id = None

        st.divider()
        if st.session_state.test_mode:
            st.caption(
                f"Test mode: {len(filtered):,} of {total_patient_count:,} patients "
                f"(limited to first {TEST_MODE_PATIENT_LIMIT})"
            )
        else:
            st.caption(f"Total patients: {total_patient_count:,} | Shown: {len(filtered):,}")


# ── Demographics header ──────────────────────────────────────────────────────

def render_demographics(patient_row: pd.Series | None, admission_row: pd.Series | None):
    cols = st.columns(5)
    with cols[0]:
        pid = int(patient_row["subject_id"]) if patient_row is not None else "—"
        st.metric("Subject ID", pid)
    with cols[1]:
        gender = patient_row.get("gender", "—") if patient_row is not None else "—"
        st.metric("Gender", gender)
    with cols[2]:
        age = patient_row.get("anchor_age", "—") if patient_row is not None else "—"
        st.metric("Age (anchor)", age)
    with cols[3]:
        if admission_row is not None and pd.notna(admission_row.get("admittime")):
            admit_str = pd.Timestamp(admission_row["admittime"]).strftime("%Y-%m-%d")
        else:
            admit_str = "—"
        st.metric("Admission Date", admit_str)
    with cols[4]:
        adm_type = admission_row.get("admission_type", "—") if admission_row is not None else "—"
        st.metric("Admission Type", str(adm_type or "—"))


# ── Tab 1: Clinical Notes ────────────────────────────────────────────────────

def render_notes_tab(subject_id: int, hadm_id: int):
    note_type_choice = st.radio(
        "Note type",
        ["Discharge Summary", "Radiology", "All"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if note_type_choice == "Discharge Summary":
        notes_df = dl.load_notes(MIMIC_BASE_DIR, "discharge")
    elif note_type_choice == "Radiology":
        notes_df = dl.load_notes(MIMIC_BASE_DIR, "radiology")
    else:
        dis = dl.load_notes(MIMIC_BASE_DIR, "discharge")
        rad = dl.load_notes(MIMIC_BASE_DIR, "radiology")
        notes_df = pd.concat([dis, rad], ignore_index=True) if not (dis.empty and rad.empty) else pd.DataFrame()

    if notes_df.empty:
        st.info(
            "No clinical note files found. "
            "MIMIC-IV Clinical Notes are a **separate** PhysioNet download "
            "([mimic-iv-note](https://physionet.org/content/mimic-iv-note/)). "
            "Download it with `wget -r -N` and place it at "
            f"`physionet.org/files/mimic-iv-note/` alongside your existing MIMIC-IV files."
        )
        return

    adm_notes = dl.get_notes_for_admission(notes_df, subject_id, hadm_id)
    if adm_notes.empty:
        st.info("No notes found for this admission.")
        return

    def _note_label(row) -> str:
        dt = row.get("charttime")
        dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "Unknown"
        ntype = str(row.get("note_type", row.get("category", "Note")) or "Note")
        return f"{dt_str} — {ntype}"

    note_options = adm_notes["note_id"].tolist()
    note_labels = {row["note_id"]: _note_label(row) for _, row in adm_notes.iterrows()}

    current_note = st.session_state.selected_note_id
    default_note_idx = note_options.index(current_note) if current_note in note_options else 0

    selected_note_id = st.selectbox(
        "Select Note",
        options=note_options,
        format_func=lambda nid: note_labels.get(nid, str(nid)),
        index=default_note_idx,
    )
    st.session_state.selected_note_id = selected_note_id

    note_row = adm_notes[adm_notes["note_id"] == selected_note_id].iloc[0]
    note_text = str(note_row.get("text", "") or "")

    if not note_text.strip():
        st.warning("This note has no text content.")
        return

    st.caption(f"Note length: {len(note_text):,} characters")

    # ── Entity extraction ────────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 1])

    status = st.session_state.processing_status.get(str(selected_note_id), "idle")
    entities = st.session_state.extracted_entities.get(str(selected_note_id), [])

    with col_a:
        extract_btn = st.button(
            "Extract Entities" if status != "done" else "Re-extract Entities",
            type="primary",
            disabled=(status == "running"),
            key="extract_btn",
        )

    with col_b:
        suggest_btn = st.button(
            "Suggest ICD-10 Codes",
            disabled=(status != "done" or not entities),
            key="suggest_btn",
        )

    if extract_btn:
        healthy, health_msg = _ollama_status()
        if not healthy:
            st.error(f"Ollama unavailable: {health_msg}")
        else:
            st.session_state.processing_status[str(selected_note_id)] = "running"
            with st.spinner(f"Analyzing note with {health_msg}... (this may take ~30–90s)"):
                try:
                    result, debug_info = llm.extract_entities(note_text)
                    st.session_state.extracted_entities[str(selected_note_id)] = result
                    st.session_state.processing_status[str(selected_note_id)] = "done"
                    if not result:
                        st.warning(
                            "Extraction completed but no entities were found.  \n"
                            f"**Debug:** {debug_info}  \n"
                            "Try re-extracting. If this keeps happening, check that Ollama is "
                            "using a Qwen model (`ollama list`) and the note has enough clinical content."
                        )
                    else:
                        st.rerun()
                except llm.OllamaConnectionError as e:
                    st.session_state.processing_status[str(selected_note_id)] = f"error:{e}"
                    st.error(str(e))
                except llm.OllamaModelError as e:
                    st.session_state.processing_status[str(selected_note_id)] = f"error:{e}"
                    st.error(str(e))
                except llm.OllamaTimeoutError as e:
                    st.session_state.processing_status[str(selected_note_id)] = f"error:{e}"
                    st.error(str(e))

    if suggest_btn and entities:
        with st.spinner("Mapping entities to ICD-10 codes via MCP server..."):
            all_codes: dict[str, dict] = {}
            diagnosis_entities = [e for e in entities if e.get("type") == "diagnosis"]
            # Also include symptoms and procedures for broader coverage
            other_entities = [e for e in entities if e.get("type") in ("symptom", "procedure")]
            to_map = diagnosis_entities + other_entities[:5]

            for entity in to_map:
                query = entity.get("normalized") or entity.get("text", "")
                if not query.strip():
                    continue
                results = icd_client.search_icd10_codes(query, max_results=5)
                for r in results:
                    code = r.get("code", "")
                    if code and code not in all_codes:
                        all_codes[code] = {**r, "source_entity": entity.get("text", ""),
                                           "entity_type": entity.get("type", "")}

            codes_list = sorted(all_codes.values(), key=lambda x: x.get("score", 0), reverse=True)
            st.session_state.suggested_codes[hadm_id] = codes_list

            # Initialize decisions
            if hadm_id not in st.session_state.code_decisions:
                st.session_state.code_decisions[hadm_id] = {}
            for c in codes_list:
                code = c.get("code", "")
                if code and code not in st.session_state.code_decisions[hadm_id]:
                    st.session_state.code_decisions[hadm_id][code] = "pending"

            st.success(f"Found {len(codes_list)} ICD-10 suggestions. Review them in the ICD-10 Coding tab.")
            if not codes_list:
                st.warning(
                    "No ICD-10 matches were found for the extracted entities. "
                    "Try re-extracting entities or using a note with clearer diagnoses."
                )
            else:
                st.rerun()

    # ── Note display ─────────────────────────────────────────────────────────
    st.divider()
    if entities:
        st.markdown(hl.render_legend(), unsafe_allow_html=True)
        st.markdown(hl.render_entity_summary(entities), unsafe_allow_html=True)
        note_html = hl.render_highlighted_note(note_text, entities)
        st.components.v1.html(note_html, height=700, scrolling=True)

        with st.expander(f"Raw entities ({len(entities)} extracted)", expanded=False):
            st.dataframe(
                pd.DataFrame(entities)[["text", "type"]],
                use_container_width=True,
                height=300,
            )
    else:
        # Show plain note without highlights
        plain_html = hl.render_highlighted_note(note_text, [])
        st.components.v1.html(plain_html, height=700, scrolling=True)


# ── Tab 2: Vitals ────────────────────────────────────────────────────────────

def render_vitals_tab(subject_id: int, hadm_id: int):
    with st.spinner("Loading vitals..."):
        vitals_df = dl.load_vitals_for_admission(MIMIC_BASE_DIR, subject_id, hadm_id)

    if vitals_df.empty:
        st.info(
            "No vital signs found for this admission. "
            "Ensure the `icu/chartevents.csv` file is present in `mimic_dataset/`."
        )
        return

    vitals_df["label"] = vitals_df["itemid"].map(dl.VITAL_LABELS)
    available = vitals_df["label"].dropna().unique().tolist()

    if not available:
        st.info("No recognized vital sign items found in chartevents for this admission.")
        return

    # Build subplots
    layout_items = [
        ("HR", "SpO2"),
        ("SBP", "DBP"),
        ("RR", "Temp"),
    ]
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        subplot_titles=[v for pair in layout_items for v in pair],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    colors = {
        "HR": "#FF4B4B", "SpO2": "#1E88E5", "SBP": "#28A745",
        "DBP": "#6DB33F", "RR": "#FF8C00", "Temp": "#9C27B0",
    }

    for row_idx, (left, right) in enumerate(layout_items, start=1):
        for col_idx, vital in enumerate([left, right], start=1):
            subset = vitals_df[vitals_df["label"] == vital].dropna(subset=["valuenum"])
            if not subset.empty:
                fig.add_trace(
                    go.Scatter(
                        x=subset["charttime"],
                        y=subset["valuenum"],
                        mode="lines+markers",
                        name=vital,
                        line=dict(color=colors.get(vital, "#888"), width=2),
                        marker=dict(size=4),
                        showlegend=False,
                    ),
                    row=row_idx, col=col_idx,
                )
            # Normal range reference lines
            if vital in VITAL_NORMAL_RANGES and not subset.empty:
                lo, hi = VITAL_NORMAL_RANGES[vital]
                for val, dash in [(lo, "dash"), (hi, "dash")]:
                    fig.add_trace(
                        go.Scatter(
                            x=[subset["charttime"].min(), subset["charttime"].max()],
                            y=[val, val],
                            mode="lines",
                            line=dict(color="rgba(255,255,255,0.25)", dash=dash, width=1),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_idx, col=col_idx,
                    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
        title_text=f"Vital Signs — Admission {hadm_id}",
        title_x=0.5,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    with st.expander("Latest Values Summary", expanded=True):
        rows = []
        for vital in dl.VITAL_LABELS.values():
            subset = vitals_df[vitals_df["label"] == vital].dropna(subset=["valuenum"])
            if subset.empty:
                continue
            latest = subset.sort_values("charttime").iloc[-1]
            lo, hi = VITAL_NORMAL_RANGES.get(vital, (None, None))
            val = latest["valuenum"]
            status = "Normal" if (lo and hi and lo <= val <= hi) else "Abnormal" if (lo or hi) else "—"
            rows.append({
                "Vital": vital,
                "Latest Value": round(val, 1),
                "Unit": str(latest.get("valueuom", "") or ""),
                "Time": latest["charttime"].strftime("%Y-%m-%d %H:%M") if pd.notna(latest["charttime"]) else "—",
                "Normal Range": f"{lo}–{hi}" if lo and hi else "—",
                "Status": status,
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab 3: ICD-10 Coding ─────────────────────────────────────────────────────

def render_icd_coding_tab(subject_id: int, hadm_id: int):
    diagnoses_df = dl.load_diagnoses(MIMIC_BASE_DIR)

    # Existing codes from MIMIC
    existing = diagnoses_df[diagnoses_df["hadm_id"] == hadm_id] if not diagnoses_df.empty else pd.DataFrame()
    if not existing.empty:
        with st.expander("Existing ICD-10 Codes (from MIMIC record)", expanded=False):
            st.dataframe(existing[["icd_code", "seq_num"]].reset_index(drop=True),
                         use_container_width=True, hide_index=True)

    suggested = st.session_state.suggested_codes.get(hadm_id, [])
    decisions = st.session_state.code_decisions.get(hadm_id, {})

    if not suggested:
        st.info(
            "No ICD-10 codes suggested yet. "
            "Go to the **Clinical Notes** tab, extract entities, then click **Suggest ICD-10 Codes**."
        )
        return

    st.subheader(f"Suggested Codes ({len(suggested)} total)")

    approved_count = sum(1 for d in decisions.values() if d == "approved")
    rejected_count = sum(1 for d in decisions.values() if d == "rejected")
    pending_count = sum(1 for d in decisions.values() if d == "pending")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Approved", approved_count)
    mc2.metric("Rejected", rejected_count)
    mc3.metric("Pending", pending_count)

    st.divider()

    for idx, code_item in enumerate(suggested):
        code = code_item.get("code", "")
        if not code:
            continue
        desc = code_item.get("description", "No description")
        billable = code_item.get("billable", False)
        source = code_item.get("source_entity", "")
        entity_type = code_item.get("entity_type", "")
        decision = decisions.get(code, "pending")

        with st.container():
            row_cols = st.columns([3, 1, 1, 1])
            with row_cols[0]:
                bill_badge = "✓ Billable" if billable else "Non-billable"
                entity_info = f" ← *{source}* ({entity_type})" if source else ""
                st.markdown(
                    f"**`{code}`** — {desc}  \n"
                    f"<small style='color:#888'>{bill_badge}{entity_info}</small>",
                    unsafe_allow_html=True,
                )
            with row_cols[1]:
                if st.button("✅ Approve", key=f"approve_{code}_{idx}", use_container_width=True):
                    if hadm_id not in st.session_state.code_decisions:
                        st.session_state.code_decisions[hadm_id] = {}
                    st.session_state.code_decisions[hadm_id][code] = "approved"
                    st.rerun()
            with row_cols[2]:
                if st.button("❌ Reject", key=f"reject_{code}_{idx}", use_container_width=True):
                    if hadm_id not in st.session_state.code_decisions:
                        st.session_state.code_decisions[hadm_id] = {}
                    st.session_state.code_decisions[hadm_id][code] = "rejected"
                    st.rerun()
            with row_cols[3]:
                if decision == "approved":
                    st.markdown('<span class="badge-green">Approved</span>', unsafe_allow_html=True)
                elif decision == "rejected":
                    st.markdown('<span class="badge-red">Rejected</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-orange">Pending</span>', unsafe_allow_html=True)
        st.divider()

    # Export
    if any(d != "pending" for d in decisions.values()):
        export_rows = []
        for code_item in suggested:
            code = code_item.get("code", "")
            dec = decisions.get(code, "pending")
            export_rows.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "icd_code": code,
                "description": code_item.get("description", ""),
                "billable": code_item.get("billable", False),
                "source_entity": code_item.get("source_entity", ""),
                "entity_type": code_item.get("entity_type", ""),
                "decision": dec,
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        if export_rows:
            csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Decisions to CSV",
                data=csv_bytes,
                file_name=f"icd10_decisions_adm{hadm_id}.csv",
                mime="text/csv",
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()

    # Render test mode first so session_state is current before patient filtering.
    with st.sidebar:
        st.title("🏥 Patient Selection")
        st.checkbox(
            "Test mode (first 15 patients)",
            key="test_mode",
            help="Limits the patient list to the first 15 MIMIC subjects for faster local testing.",
        )
        st.divider()

    patients_df = dl.load_patients(MIMIC_BASE_DIR)
    admissions_df = dl.load_admissions(MIMIC_BASE_DIR)
    total_patient_count = len(patients_df)

    patients_df, admissions_df = _apply_test_mode(patients_df, admissions_df)

    render_sidebar(patients_df, admissions_df, total_patient_count)

    subject_id = st.session_state.selected_patient_id
    hadm_id = st.session_state.selected_hadm_id

    if subject_id is None:
        st.title("Clinical ICD-10 Coding Assistant")
        st.info("Select a patient from the sidebar to begin.")
        if patients_df.empty:
            st.warning(
                "**No MIMIC-IV data found.** Expected files at "
                f"`{MIMIC_BASE_DIR}/hosp/patients.csv.gz`. "
                "Ensure MIMIC-IV is downloaded to `physionet.org/files/mimiciv/3.1/` "
                "with the standard `hosp/` and `icu/` subdirectory layout."
            )
        return

    patient_row, pat_admissions = dl.get_patient_admissions(patients_df, admissions_df, subject_id)
    admission_row = None
    if hadm_id is not None and not pat_admissions.empty:
        rows = pat_admissions[pat_admissions["hadm_id"] == hadm_id]
        if not rows.empty:
            admission_row = rows.iloc[0]

    render_demographics(patient_row, admission_row)
    st.divider()

    if hadm_id is None:
        st.info("Select an admission from the sidebar.")
        return

    tab1, tab2, tab3 = st.tabs(["Clinical Notes", "Vitals", "ICD-10 Coding"])

    with tab1:
        render_notes_tab(subject_id, hadm_id)

    with tab2:
        render_vitals_tab(subject_id, hadm_id)

    with tab3:
        render_icd_coding_tab(subject_id, hadm_id)


if __name__ == "__main__":
    main()
