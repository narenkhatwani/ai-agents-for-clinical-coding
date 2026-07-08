"""
Physician Assistant — Streamlit UI
====================================
Clinical AI assistant for ICD-10 coding and diagnosis support.
Integrates with the 4-agent pipeline, UMLS normalization,
MCP ICD-10 server, and SQLite conversation memory.

Run:
    streamlit run app.py

Requirements:
    pip install streamlit requests
    Ollama running with qwen2.5:7b
    Node.js installed (for MCP server)
"""

import streamlit as st
import requests
import json
import time
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

# ── Page config — must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title  = "Physician Assistant",
    page_icon   = "🏥",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Import memory module (must be in same folder) ─────────────────────────────
try:
    from conversation_memory import ConversationMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# ── Import pipeline (must be in same folder) ──────────────────────────────────
try:
    from icd10_pipeline_with_umls import run_pipeline as run_icd10_pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_OLLAMA_BASE  = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL           = f"{DEFAULT_OLLAMA_BASE}/api/generate"
MODEL                = DEFAULT_OLLAMA_MODEL
UMLS_API_KEY         = "YOUR_UMLS_API_KEY_HERE"

# ── Navigation ────────────────────────────────────────────────────────────────
VIEW_LABELS = [
    "💬 Chat",
    "⚙️ ICD-10 Pipeline",
    "📋 Session History",
    "🗄️ Database Viewer",
    "ℹ️ About",
]


# ═══════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  :root {
    --bg-app:       #FAF8F5;
    --bg-sidebar:   #EDE8F5;
    --bg-card:      #FFFFFF;
    --accent-blue:  #9BB8D3;
    --accent-mint:  #A8D5BA;
    --accent-rose:  #E8A0A0;
    --accent-lav:   #C5B4E3;
    --text-body:    #4A5568;
    --text-muted:   #8B9AAB;
    --text-strong:  #2D3748;
    --border-soft:  #E8E0F0;
    --shadow-card:  0 2px 12px rgba(74, 85, 104, 0.08);
  }

  /* ── Base ── */
  [data-testid="stAppViewContainer"] {
    background: var(--bg-app);
  }
  [data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border-soft);
  }
  [data-testid="stSidebar"] .section-header {
    margin-top: 0.6rem;
  }

  /* ── Typography ── */
  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-body);
  }

  /* ── Page header ── */
  .page-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-soft);
  }
  .page-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-strong);
    letter-spacing: -0.5px;
    line-height: 1.2;
  }
  .page-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 0.35rem;
  }

  /* ── Content cards (bordered containers) ── */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--shadow-card);
    margin-bottom: 1rem;
  }

  /* ── Sidebar patient card ── */
  .patient-card {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-left: 3px solid var(--accent-blue);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-card);
  }
  .patient-card h4 {
    color: var(--accent-blue);
    margin: 0 0 0.5rem 0;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .patient-card p {
    color: var(--text-muted);
    font-size: 13px;
    margin: 3px 0;
  }
  .patient-card .pid {
    color: var(--text-strong);
    font-weight: 600;
    font-size: 15px;
  }
  .patient-card code {
    font-size: 10px;
    color: var(--text-muted);
  }

  /* ── Chat messages ── */
  .msg-physician {
    background: #F0F4FA;
    border: 1px solid #D4E0ED;
    border-left: 3px solid var(--accent-blue);
    border-radius: 12px 12px 12px 2px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    max-width: 85%;
  }
  .msg-assistant {
    background: #EDF7F1;
    border: 1px solid #C8E6D4;
    border-left: 3px solid var(--accent-mint);
    border-radius: 12px 12px 2px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 85%;
  }
  .msg-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
    font-weight: 600;
  }
  .msg-physician .msg-label { color: #6A9BC3; }
  .msg-assistant .msg-label { color: #6BA888; }
  .msg-time {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 0.4rem;
  }

  /* ── ICD-10 code blocks ── */
  .code-block {
    background: #F5F0FA;
    border: 1px solid #E0D4F0;
    border-left: 3px solid var(--accent-lav);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
  }
  .code-primary {
    color: #C97A7A;
    font-weight: 700;
    font-size: 16px;
  }
  .code-secondary {
    color: #6A9BC3;
    font-weight: 700;
    font-size: 16px;
  }
  .code-desc {
    color: var(--text-muted);
    margin-left: 12px;
  }
  .code-justification {
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 4px;
    display: block;
  }
  .code-label {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .code-tag {
    background: #EDF2F8;
    color: #6A9BC3;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
    font-size: 12px;
  }
  .sidebar-code-line {
    font-size: 12px;
    padding: 3px 0;
    color: var(--text-muted);
  }
  .sidebar-code-line code {
    color: #6A9BC3;
  }

  /* ── Status badges ── */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .badge-ready    { background: #D4EDDA; color: #3D7A5A; }
  .badge-hold     { background: #FADADD; color: #A05050; }
  .badge-active   { background: #D6E4F0; color: #4A7A9B; }
  .badge-thinking { background: #EDE8F5; color: var(--text-muted); }

  /* ── Section headers ── */
  .section-header {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    border-bottom: 1px solid var(--border-soft);
    padding-bottom: 0.4rem;
    padding-left: 0.6rem;
    border-left: 3px solid var(--accent-blue);
    margin: 0.8rem 0 0.8rem 0;
  }
  .section-header--mint    { border-left-color: var(--accent-mint); }
  .section-header--lavender { border-left-color: var(--accent-lav); }
  .section-header--rose    { border-left-color: var(--accent-rose); }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
  }
  .empty-state-icon {
    font-size: 48px;
    margin-bottom: 1rem;
  }
  .empty-state-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-body);
  }
  .empty-state-desc {
    font-size: 13px;
    margin-top: 0.5rem;
  }

  /* ── History message rows ── */
  .history-msg {
    font-size: 12px;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-soft);
  }
  .history-msg-role-physician { color: #6A9BC3; font-weight: 600; }
  .history-msg-role-assistant { color: #6BA888; font-weight: 600; }
  .history-msg-content { color: var(--text-muted); }
  .history-session-meta {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 1.8;
  }

  /* ── About content ── */
  .about-body {
    font-size: 13px;
    color: var(--text-muted);
    line-height: 2;
  }
  .about-body b {
    color: var(--text-strong);
  }
  .about-refs {
    font-size: 12px;
    color: var(--text-muted);
    line-height: 2;
  }

  /* ── Sidebar navigation radio ── */
  [data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-size: 13px;
    padding: 0.35rem 0;
  }
  [data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 13px !important;
  }

  /* ── Input area ── */
  textarea, .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text-body) !important;
    font-size: 14px !important;
  }
  textarea:focus, .stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(155, 184, 211, 0.35) !important;
  }
  .stTextInput input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-soft) !important;
    color: var(--text-body) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #E8F0F8;
    color: #4A7A9B;
    border: 1px solid var(--accent-blue);
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    background: var(--accent-blue);
    color: #FFFFFF;
    border-color: #7AA3C4;
  }
  .stButton > button[kind="primary"] {
    background: var(--accent-blue);
    color: #FFFFFF;
    border-color: #7AA3C4;
  }
  .stButton > button[kind="primary"]:hover {
    background: #7AA3C4;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
  }

  /* ── Metric boxes ── */
  [data-testid="metric-container"] {
    background: #F5F0FA;
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 0.75rem;
  }
  [data-testid="metric-container"] label {
    color: var(--text-muted) !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-strong) !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    overflow: hidden;
  }

  /* ── Row count caption ── */
  .row-count {
    font-size: 12px;
    color: var(--text-muted);
  }

  /* ── Hide streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "patient_id":       "",
        "physician_id":     "",
        "session_id":       None,
        "messages":         [],
        "last_codes":       [],
        "last_diagnoses":   [],
        "last_queries":     [],
        "claim_ready":      None,
        "memory":           None,
        "pipeline_running": False,
        "active_view":      VIEW_LABELS[0],
        "ollama_base_url":  DEFAULT_OLLAMA_BASE,
        "ollama_model":     DEFAULT_OLLAMA_MODEL,
        "ollama_models":    [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def render_page_header(title: str, subtitle: str = ""):
    """Renders a consistent page title and subtitle in the main area."""
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">{title}</div>
      <div class="page-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def get_memory() -> ConversationMemory:
    """Initialize memory once, reuse across reruns."""
    if not MEMORY_AVAILABLE:
        return None
    if st.session_state.memory is None:
        st.session_state.memory = ConversationMemory()
    return st.session_state.memory


def normalize_ollama_base(url: str) -> str:
    return (url or DEFAULT_OLLAMA_BASE).strip().rstrip("/")


def ollama_generate_url(base_url: str = None) -> str:
    base = normalize_ollama_base(base_url or st.session_state.ollama_base_url)
    if base.endswith("/api/generate"):
        return base
    return f"{base}/api/generate"


def check_ollama_running(base_url: str = None) -> bool:
    try:
        requests.get(normalize_ollama_base(base_url or st.session_state.ollama_base_url), timeout=2)
        return True
    except Exception:
        return False


def fetch_ollama_models(base_url: str = None) -> Tuple[List[str], Optional[str]]:
    """Returns (model_names, error_message)."""
    try:
        base = normalize_ollama_base(base_url or st.session_state.ollama_base_url)
        response = requests.get(f"{base}/api/tags", timeout=5)
        response.raise_for_status()
        models = sorted(m["name"] for m in response.json().get("models", []))
        return models, None
    except Exception as e:
        return [], str(e)


def call_ollama_chat(messages: list, system_prompt: str = None) -> str:
    """
    Calls Ollama with a conversation history.
    Formats messages into a single prompt string since Ollama's
    /api/generate endpoint takes a single prompt, not a messages array.
    """
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n\n"
    for msg in messages:
        role    = "Physician" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n\n"
    prompt += "Assistant:"

    try:
        response = requests.post(ollama_generate_url(), json={
            "model":  st.session_state.ollama_model,
            "prompt": prompt,
            "stream": False
        }, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Make sure it is running: `ollama serve`"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def format_time(iso_string: str) -> str:
    """Converts ISO timestamp to readable HH:MM."""
    try:
        return iso_string[11:16]
    except Exception:
        return ""


def add_message(role: str, content: str, message_type: str = "chat"):
    """Adds message to display state and saves to database."""
    msg = {
        "role":         role,
        "content":      content,
        "time":         datetime.now().strftime("%H:%M"),
        "message_type": message_type
    }
    st.session_state.messages.append(msg)

    if st.session_state.session_id and MEMORY_AVAILABLE:
        memory = get_memory()
        if memory:
            memory.add(role, content, st.session_state.session_id, message_type)


def render_message(msg: dict):
    """Renders a single chat message with appropriate styling."""
    role    = msg["role"]
    content = msg["content"]
    time_   = msg.get("time", "")
    mtype   = msg.get("message_type", "chat")

    if role == "physician":
        st.markdown(f"""
        <div class="msg-physician">
          <div class="msg-label">👨‍⚕️ Physician</div>
          <div>{content}</div>
          <div class="msg-time">{time_}</div>
        </div>
        """, unsafe_allow_html=True)

    elif role == "assistant":
        if mtype == "icd10_result":
            st.markdown(f"""
            <div class="msg-assistant">
              <div class="msg-label">🤖 Assistant — ICD-10 Coding Result</div>
              <div class="code-block">{content}</div>
              <div class="msg-time">{time_}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-assistant">
              <div class="msg-label">🤖 Assistant</div>
              <div>{content}</div>
              <div class="msg-time">{time_}</div>
            </div>
            """, unsafe_allow_html=True)


def format_icd10_result(final_result: dict) -> str:
    """Formats the pipeline final result as a readable string for chat."""
    codes   = final_result.get("final_codes", [])
    summary = final_result.get("coding_summary", "")
    queries = final_result.get("queries", [])
    ready   = final_result.get("claim_ready", False)

    lines = []
    for c in codes:
        role = c.get("role", "")
        icon = "🔴" if role == "PRIMARY" else "🔵"
        lines.append(f"{icon} [{c.get('sequence')}] {c.get('code')} — {c.get('description')} ({role})")

    result = "\n".join(lines)
    result += f"\n\n📝 {summary}"

    if queries:
        result += "\n\n⚠️ Physician queries needed:\n"
        for q in queries:
            result += f"• {q}\n"

    status = "✅ CLAIM READY" if ready else "⛔ HOLD — queries needed"
    result += f"\n\nStatus: {status}"
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏥 Physician Assistant")

    st.markdown('<div class="section-header section-header--lavender">Navigation</div>', unsafe_allow_html=True)
    active_view = st.radio(
        "Navigation",
        VIEW_LABELS,
        index=VIEW_LABELS.index(st.session_state.active_view) if st.session_state.active_view in VIEW_LABELS else 0,
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.active_view = active_view

    st.markdown('<div class="section-header">Patient Session</div>', unsafe_allow_html=True)

    patient_id_input = st.text_input(
        "Patient ID",
        value       = st.session_state.patient_id,
        placeholder = "e.g. P001",
        key         = "patient_id_input"
    )
    physician_input = st.text_input(
        "Physician",
        value       = st.session_state.physician_id,
        placeholder = "e.g. Dr. Wang",
        key         = "physician_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start / Resume", use_container_width=True):
            if patient_id_input:
                st.session_state.patient_id   = patient_id_input
                st.session_state.physician_id = physician_input
                memory = get_memory()
                if memory:
                    session_id = memory.new_or_resume(
                        patient_id   = patient_id_input,
                        physician_id = physician_input or "unknown"
                    )
                    st.session_state.session_id = session_id

                    history = memory.get_history(session_id)
                    st.session_state.messages = []
                    for h in history:
                        role = "physician" if h["role"] == "user" else "assistant"
                        st.session_state.messages.append({
                            "role":         role,
                            "content":      h["content"],
                            "time":         "",
                            "message_type": "chat"
                        })
                    st.rerun()
            else:
                st.error("Enter a patient ID first.")

    with col2:
        if st.button("＋ New Session", use_container_width=True):
            if patient_id_input:
                st.session_state.patient_id   = patient_id_input
                st.session_state.physician_id = physician_input
                memory = get_memory()
                if memory:
                    session_id = memory.new_or_resume(
                        patient_id   = patient_id_input,
                        physician_id = physician_input or "unknown",
                        force_new    = True
                    )
                    st.session_state.session_id = session_id
                    st.session_state.messages   = []
                    st.rerun()

    if st.session_state.session_id:
        st.markdown(f"""
        <div class="patient-card">
          <h4>Active Session</h4>
          <p>Patient: <span class="pid">{st.session_state.patient_id}</span></p>
          <p>Physician: {st.session_state.physician_id or '—'}</p>
          <p>Session: <code>{st.session_state.session_id}</code></p>
          <span class="badge badge-active">Active</span>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.last_codes:
        st.markdown('<div class="section-header section-header--mint">Last Coding Result</div>', unsafe_allow_html=True)
        for c in st.session_state.last_codes:
            role = c.get("role", "")
            icon = "🔴" if role == "PRIMARY" else "🔵"
            code = c.get("code", "")
            desc = c.get("description", "")[:30]
            st.markdown(f"""
            <div class="sidebar-code-line">
              {icon} <code>{code}</code> {desc}
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.claim_ready is not None:
            badge = "badge-ready" if st.session_state.claim_ready else "badge-hold"
            label = "Claim Ready" if st.session_state.claim_ready else "Hold — Query Needed"
            st.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    if st.session_state.session_id and MEMORY_AVAILABLE:
        if st.button("⬇ Export session as .txt", use_container_width=True):
            memory   = get_memory()
            filepath = memory.export_txt(st.session_state.session_id)
            st.success(f"Saved: {filepath}")

        if st.button("⬇ Export patient history", use_container_width=True):
            memory   = get_memory()
            filepath = memory.export_patient_txt(st.session_state.patient_id)
            st.success(f"Saved: {filepath}")

    st.markdown('<div class="section-header section-header--mint">Ollama Model</div>', unsafe_allow_html=True)

    ollama_base_input = st.text_input(
        "Ollama URL",
        value       = st.session_state.ollama_base_url,
        placeholder = DEFAULT_OLLAMA_BASE,
        help        = "Base URL for your local Ollama server",
        key         = "ollama_base_input",
    )
    st.session_state.ollama_base_url = normalize_ollama_base(ollama_base_input)

    if st.button("↻ Refresh models", use_container_width=True, key="refresh_ollama_models"):
        models, err = fetch_ollama_models()
        if err:
            st.session_state.ollama_models = []
            st.error(f"Could not reach Ollama: {err}")
        else:
            st.session_state.ollama_models = models
            if models and st.session_state.ollama_model not in models:
                st.session_state.ollama_model = models[0]
            st.rerun()

    if not st.session_state.ollama_models and check_ollama_running():
        models, _ = fetch_ollama_models()
        st.session_state.ollama_models = models

    model_options = list(st.session_state.ollama_models)
    if st.session_state.ollama_model and st.session_state.ollama_model not in model_options:
        model_options = [st.session_state.ollama_model] + model_options

    if model_options:
        current_idx = (
            model_options.index(st.session_state.ollama_model)
            if st.session_state.ollama_model in model_options
            else 0
        )
        st.session_state.ollama_model = st.selectbox(
            "Model",
            model_options,
            index = current_idx,
            help  = "Local models pulled via `ollama pull`",
            key   = "ollama_model_select",
        )
    else:
        st.session_state.ollama_model = st.text_input(
            "Model name",
            value       = st.session_state.ollama_model,
            placeholder = DEFAULT_OLLAMA_MODEL,
            help        = "Enter a model name (e.g. qwen2.5:7b). Refresh when Ollama is running to list installed models.",
            key         = "ollama_model_manual",
        )

    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
    if check_ollama_running():
        st.markdown('<span class="badge badge-ready">Ollama Running</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-hold">Ollama Offline</span>', unsafe_allow_html=True)
        st.caption("Run: `ollama serve`")

    st.markdown(
        f'<p class="row-count" style="margin-top:4px">Active model: {st.session_state.ollama_model}</p>',
        unsafe_allow_html=True,
    )
    mem_status = "badge-ready" if MEMORY_AVAILABLE else "badge-hold"
    mem_label  = "Memory Active" if MEMORY_AVAILABLE else "Memory Unavailable"
    st.markdown(f'<span class="badge {mem_status}">{mem_label}</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA — SIDEBAR-DRIVEN VIEWS
# ═══════════════════════════════════════════════════════════════════════════

view = st.session_state.active_view


# ── VIEW: CHAT ─────────────────────────────────────────────────────────────
if view == "💬 Chat":
    render_page_header(
        "Physician Chat",
        "Clinical Q&A powered by Ollama — use the Pipeline view for verified ICD-10 coding"
    )

    if not st.session_state.session_id:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-state-icon">🏥</div>
          <div class="empty-state-title">No active session</div>
          <div class="empty-state-desc">
            Enter a Patient ID in the sidebar and click Start / Resume
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.container(border=True):
            st.markdown('<div class="section-header section-header--lavender">Conversation</div>', unsafe_allow_html=True)
            if not st.session_state.messages:
                st.markdown("""
                <div class="empty-state" style="padding:2rem">
                  <div class="empty-state-desc">Session started. Enter a clinical note or question below.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    render_message(msg)

        with st.container(border=True):
            st.markdown('<div class="section-header section-header--mint">Send Message</div>', unsafe_allow_html=True)
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Message",
                    placeholder      = "Ask a clinical question, or paste a clinical note...",
                    height             = 100,
                    label_visibility   = "collapsed"
                )
                col_send, col_clear = st.columns([6, 1])
                with col_send:
                    submitted = st.form_submit_button(
                        "Send →",
                        use_container_width = True,
                        type = "primary"
                    )
                with col_clear:
                    cleared = st.form_submit_button("Clear", use_container_width=True)

        if cleared:
            st.session_state.messages = []
            st.rerun()

        if submitted and user_input.strip():
            add_message("physician", user_input.strip())

            history = []
            for m in st.session_state.messages[:-1]:
                role = "user" if m["role"] == "physician" else "assistant"
                history.append({"role": role, "content": m["content"]})

            system = """You are a clinical coding assistant helping physicians with:
1. ICD-10-CM code identification and explanation
2. Clinical question answering
3. Diagnosis support based on clinical notes

Be concise, accurate, and always cite coding guidelines when relevant.
If asked to extract ICD-10 codes, suggest the physician uses the ICD-10 Pipeline view for the full verified pipeline."""

            history.append({"role": "user", "content": user_input.strip()})

            with st.spinner("Thinking..."):
                response = call_ollama_chat(history, system_prompt=system)

            add_message("assistant", response)
            st.rerun()


# ── VIEW: ICD-10 PIPELINE ──────────────────────────────────────────────────
elif view == "⚙️ ICD-10 Pipeline":
    render_page_header(
        "ICD-10 Coding Pipeline",
        "Evidence Extraction → UMLS Normalization → MCP Code Lookup → Validation → Reconciliation"
    )

    if not st.session_state.session_id:
        st.info("Start a session in the sidebar first.")
    else:
        with st.container(border=True):
            st.markdown('<div class="section-header section-header--lavender">Clinical Note</div>', unsafe_allow_html=True)
            note_input = st.text_area(
                "Clinical note",
                placeholder = "Paste the full clinical note here...",
                height      = 200,
                key         = "pipeline_note",
                label_visibility = "collapsed"
            )

        with st.container(border=True):
            st.markdown('<div class="section-header section-header--mint">Configuration</div>', unsafe_allow_html=True)
            umls_key_input = st.text_input(
                "UMLS API Key (optional — leave blank to skip normalization)",
                value = "" if UMLS_API_KEY == "YOUR_UMLS_API_KEY_HERE" else UMLS_API_KEY,
                type  = "password",
                key   = "umls_key"
            )

        run_btn = st.button("▶ Run ICD-10 Pipeline", type="primary", use_container_width=True)

        if run_btn and note_input.strip():
            if not PIPELINE_AVAILABLE:
                st.error("Pipeline file (icd10_pipeline_with_umls.py) not found in the same folder.")
            else:
                add_message("physician", f"[Pipeline run requested]\n\n{note_input.strip()}")

                progress = st.progress(0)
                status   = st.empty()

                try:
                    status.markdown('<span class="badge badge-thinking">Stage 1 — Extracting evidence...</span>', unsafe_allow_html=True)
                    progress.progress(20)

                    api_key = umls_key_input.strip() or UMLS_API_KEY
                    results = run_icd10_pipeline(
                        note_input.strip(),
                        umls_api_key    = api_key,
                        model           = st.session_state.ollama_model,
                        ollama_base_url = st.session_state.ollama_base_url,
                    )

                    progress.progress(100)
                    status.markdown('<span class="badge badge-ready">Pipeline complete</span>', unsafe_allow_html=True)

                    final = results.get("final", {})
                    codes = final.get("final_codes", [])

                    st.session_state.last_codes   = codes
                    st.session_state.claim_ready  = final.get("claim_ready", False)
                    st.session_state.last_queries = final.get("queries", [])

                    code_list = [c["code"] for c in codes]
                    if MEMORY_AVAILABLE:
                        memory = get_memory()
                        if memory:
                            memory.save_results(
                                st.session_state.session_id,
                                icd10_codes = code_list
                            )

                    formatted = format_icd10_result(final)
                    add_message("assistant", formatted, message_type="icd10_result")

                    with st.container(border=True):
                        st.markdown('<div class="section-header section-header--rose">Code Breakdown</div>', unsafe_allow_html=True)
                        for c in codes:
                            role_class = "code-primary" if c.get("role") == "PRIMARY" else "code-secondary"
                            st.markdown(f"""
                            <div class="code-block">
                              <span class="code-label">Seq {c.get('sequence')} · {c.get('role')}</span><br>
                              <span class="{role_class}">{c.get('code')}</span>
                              <span class="code-desc">{c.get('description')}</span><br>
                              <span class="code-justification">{c.get('justification','')}</span>
                            </div>
                            """, unsafe_allow_html=True)

                        queries = final.get("queries", [])
                        if queries:
                            st.markdown('<div class="section-header section-header--rose">Physician Queries Required</div>', unsafe_allow_html=True)
                            for q in queries:
                                st.warning(q)

                        if final.get("claim_ready"):
                            st.success("✅ Claim ready for submission")
                        else:
                            st.error("⛔ Hold claim — physician queries must be resolved first")

                        st.markdown('<div class="section-header section-header--mint">Billing Summary</div>', unsafe_allow_html=True)
                        st.info(final.get("coding_summary", ""))

                        with st.expander("Full pipeline output (JSON)"):
                            st.json(results)

                except Exception as e:
                    progress.progress(0)
                    status.error(f"Pipeline error: {str(e)}")
                    st.exception(e)


# ── VIEW: SESSION HISTORY ──────────────────────────────────────────────────
elif view == "📋 Session History":
    render_page_header(
        "Session History",
        "Review past sessions, assigned codes, and message previews for the current patient"
    )

    if not MEMORY_AVAILABLE:
        st.error("conversation_memory.py not found. Place it in the same folder as this app.")
    elif not st.session_state.patient_id:
        st.info("Start a session in the sidebar to view history.")
    else:
        memory = get_memory()
        if memory:
            summary = memory.patient_summary(st.session_state.patient_id)

            with st.container(border=True):
                st.markdown('<div class="section-header section-header--lavender">Patient Summary</div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sessions",       summary["total_sessions"])
                col2.metric("Messages",       summary["total_messages"])
                col3.metric("Codes assigned", len(summary["all_icd10_codes"]))
                col4.metric("Last active",    summary["last_active"][8:10] + "/" +
                            summary["last_active"][5:7] if summary["last_active"] else "—")

            if summary["all_icd10_codes"]:
                with st.container(border=True):
                    st.markdown('<div class="section-header section-header--mint">All ICD-10 Codes (this patient)</div>', unsafe_allow_html=True)
                    codes_str = "  ".join([
                        f'<code class="code-tag">{c}</code>'
                        for c in summary["all_icd10_codes"]
                    ])
                    st.markdown(codes_str, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown('<div class="section-header section-header--lavender">Past Sessions</div>', unsafe_allow_html=True)
                for s in summary["sessions"]:
                    status_badge = "badge-active" if s["status"] == "active" else "badge-hold"
                    with st.expander(f"Session: {s['session_id']}"):
                        st.markdown(f"""
                        <div class="history-session-meta">
                          Started: {s['started_at'][:16]}<br>
                          Last active: {s['last_active'][:16]}<br>
                          Status: <span class="badge {status_badge}">{s['status']}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        if s.get("icd10_codes"):
                            try:
                                codes = json.loads(s["icd10_codes"])
                                if codes:
                                    st.caption(f"Codes: {', '.join(codes)}")
                            except Exception:
                                pass

                        conn = __import__('sqlite3').connect("physician_conversations.db")
                        conn.row_factory = __import__('sqlite3').Row
                        msgs = conn.execute(
                            "SELECT role, content, timestamp FROM messages WHERE session_id=? ORDER BY timestamp ASC",
                            (s["session_id"],)
                        ).fetchall()
                        conn.close()

                        for m in msgs:
                            role_class = "history-msg-role-physician" if m["role"] == "physician" else "history-msg-role-assistant"
                            time_ = m["timestamp"][11:16]
                            preview = m['content'][:120] + ('...' if len(m['content']) > 120 else '')
                            st.markdown(f"""
                            <div class="history-msg">
                              <span class="{role_class}">[{time_}] {m["role"].upper()}:</span>
                              <span class="history-msg-content"> {preview}</span>
                            </div>
                            """, unsafe_allow_html=True)


# ── VIEW: DATABASE VIEWER ─────────────────────────────────────────────────
elif view == "🗄️ Database Viewer":
    import sqlite3 as _sqlite3
    import pandas as _pd

    render_page_header(
        "Database Viewer",
        f"Browse and query local SQLite data — {os.path.abspath('physician_conversations.db')}"
    )

    DB_FILE = "physician_conversations.db"

    if not os.path.exists(DB_FILE):
        st.info("No database found yet. Start a session and send a message to create it.")
    else:
        with st.container(border=True):
            st.markdown('<div class="section-header section-header--lavender">Browse Tables</div>', unsafe_allow_html=True)

            db_table = st.radio(
                "Select table",
                ["sessions", "messages"],
                horizontal = True,
                key        = "db_table_select"
            )

            conn_view = _sqlite3.connect(DB_FILE)

            col_f1, col_f2 = st.columns([2, 3])
            with col_f1:
                filter_patient = st.text_input(
                    "Filter by Patient ID",
                    placeholder = "e.g. P001 (leave blank for all)",
                    key         = "db_filter_patient"
                )
            with col_f2:
                if db_table == "messages":
                    filter_role = st.selectbox(
                        "Filter by role",
                        ["All", "physician", "assistant", "system"],
                        key = "db_filter_role"
                    )
                else:
                    filter_status = st.selectbox(
                        "Filter by status",
                        ["All", "active", "closed"],
                        key = "db_filter_status"
                    )

            if db_table == "sessions":
                query  = "SELECT session_id, patient_id, physician_id, started_at, last_active, status, icd10_codes FROM sessions WHERE 1=1"
                params = []
                if filter_patient.strip():
                    query  += " AND patient_id = ?"
                    params.append(filter_patient.strip())
                if filter_status != "All":
                    query  += " AND status = ?"
                    params.append(filter_status)
                query += " ORDER BY last_active DESC"
            else:
                query  = "SELECT id, timestamp, patient_id, session_id, role, message_type, content FROM messages WHERE 1=1"
                params = []
                if filter_patient.strip():
                    query  += " AND patient_id = ?"
                    params.append(filter_patient.strip())
                if filter_role != "All":
                    query  += " AND role = ?"
                    params.append(filter_role)
                query += " ORDER BY timestamp DESC LIMIT 200"

            try:
                df = _pd.read_sql_query(query, conn_view, params=params)
                conn_view.close()
                st.markdown(f'<p class="row-count">{len(df)} rows</p>', unsafe_allow_html=True)
                if "content" in df.columns:
                    df["content"] = df["content"].str[:80] + "..."
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label     = f"Download {db_table} as CSV",
                    data      = csv,
                    file_name = f"{db_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime      = "text/csv",
                    key       = f"download_{db_table}"
                )
            except Exception as e:
                conn_view.close()
                st.error(f"Query error: {e}")

        with st.container(border=True):
            st.markdown('<div class="section-header section-header--mint">Database Statistics</div>', unsafe_allow_html=True)
            try:
                conn_stats = _sqlite3.connect(DB_FILE)
                n_sessions = conn_stats.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                n_messages = conn_stats.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                n_patients = conn_stats.execute("SELECT COUNT(DISTINCT patient_id) FROM sessions").fetchone()[0]
                db_size    = os.path.getsize(DB_FILE) / 1024
                conn_stats.close()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total sessions",  n_sessions)
                c2.metric("Total messages",  n_messages)
                c3.metric("Unique patients", n_patients)
                c4.metric("DB size",         f"{db_size:.1f} KB")
            except Exception as e:
                st.error(f"Stats error: {e}")

        with st.container(border=True):
            st.markdown('<div class="section-header section-header--lavender">Custom SQL Query</div>', unsafe_allow_html=True)
            st.caption("Run any SELECT query directly against the database")
            custom_sql = st.text_input(
                "SQL",
                placeholder = "SELECT * FROM messages WHERE role='physician' LIMIT 10",
                key         = "custom_sql"
            )
            if st.button("Run SQL", key="run_sql"):
                if custom_sql.strip().lower().startswith("select"):
                    try:
                        conn_custom = _sqlite3.connect(DB_FILE)
                        result_df   = _pd.read_sql_query(custom_sql, conn_custom)
                        conn_custom.close()
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                        csv2 = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download result",
                            data      = csv2,
                            file_name = "query_result.csv",
                            mime      = "text/csv",
                            key       = "download_custom"
                        )
                    except Exception as e:
                        st.error(f"SQL error: {e}")
                else:
                    st.warning("Only SELECT queries are allowed for safety.")


# ── VIEW: ABOUT ────────────────────────────────────────────────────────────
elif view == "ℹ️ About":
    render_page_header(
        "About This System",
        "Local-first clinical coding assistant — architecture, privacy, and references"
    )

    with st.container(border=True):
        st.markdown('<div class="section-header section-header--lavender">System Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-body">
        <b>Agent 1</b> — Evidence Extractor: reads the clinical note, pulls all diagnoses,
        symptoms, and findings with certainty levels.<br>
        <b>Stage 1.5</b> — UMLS Normalizer: maps clinical shorthand to canonical terms,
        retrieves CUIs, ICD-10 hints, and SNOMED codes.<br>
        <b>Agent 2</b> — Code Candidate Generator: uses MCP ICD-10 server for verified
        real-time code lookup. Qwen reasons over verified candidates.<br>
        <b>Agent 3</b> — Validator: audits codes against ICD-10-CM guidelines,
        flags exclusions, missing codes, and physician queries.<br>
        <b>Agent 4</b> — Reconciler: sequences final codes, assigns PRIMARY/SECONDARY
        roles, produces claim-ready output and billing summary.
        </div>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-header section-header--mint">Data & Privacy</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-body">
        All processing is <b>local</b> — Qwen2.5 runs via Ollama on your machine.
        No patient data is sent to external LLM servers.<br>
        Conversations are stored in a local SQLite database:
        <code>physician_conversations.db</code>
        </div>
        """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-header section-header--rose">References</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-refs">
        Corti — Code Like Humans (CLH), EMNLP 2025<br>
        DR.KNOWS — JMIR AI 2025<br>
        PLM-ICD + Qwen2.5 hybrid, Preprints.org 2025<br>
        Benchmarking LLMs for ICD-10, MedRxiv 2024
        </div>
        """, unsafe_allow_html=True)
