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
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL        = "qwen2.5:7b"
UMLS_API_KEY = "YOUR_UMLS_API_KEY_HERE"


# ═══════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  /* ── Base ── */
  [data-testid="stAppViewContainer"] {
    background: #0F1117;
  }
  [data-testid="stSidebar"] {
    background: #161B27;
    border-right: 1px solid #1E2A3A;
  }

  /* ── Typography ── */
  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E2E8F0;
  }

  /* ── Sidebar patient card ── */
  .patient-card {
    background: #1A2235;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  .patient-card h4 {
    color: #63B3ED;
    margin: 0 0 0.5rem 0;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .patient-card p {
    color: #A0AEC0;
    font-size: 13px;
    margin: 3px 0;
  }
  .patient-card .pid {
    color: #E2E8F0;
    font-weight: 600;
    font-size: 15px;
  }

  /* ── Chat messages ── */
  .msg-physician {
    background: #1A2235;
    border: 1px solid #1E3A5F;
    border-radius: 12px 12px 12px 2px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    max-width: 85%;
  }
  .msg-assistant {
    background: #0D2137;
    border: 1px solid #0E4272;
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
  .msg-physician .msg-label { color: #63B3ED; }
  .msg-assistant .msg-label { color: #48BB78; }
  .msg-time {
    font-size: 10px;
    color: #4A5568;
    margin-top: 0.4rem;
  }

  /* ── ICD-10 code badge ── */
  .code-block {
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 13px;
  }
  .code-primary {
    color: #FC8181;
    font-weight: 700;
  }
  .code-secondary {
    color: #63B3ED;
  }
  .code-label {
    color: #4A5568;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
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
  .badge-ready    { background: #1C4532; color: #68D391; }
  .badge-hold     { background: #742A2A; color: #FC8181; }
  .badge-active   { background: #1A365D; color: #63B3ED; }
  .badge-thinking { background: #2D3748; color: #A0AEC0; }

  /* ── Section headers ── */
  .section-header {
    color: #4A5568;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    border-bottom: 1px solid #1E2A3A;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
  }

  /* ── Tab content ── */
  [data-testid="stTabs"] [role="tab"] {
    color: #4A5568;
    font-size: 13px;
  }
  [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #63B3ED;
    border-bottom-color: #63B3ED;
  }

  /* ── Input area ── */
  textarea, .stTextArea textarea {
    background: #161B27 !important;
    border: 1px solid #1E2A3A !important;
    color: #E2E8F0 !important;
    font-size: 14px !important;
  }
  textarea:focus, .stTextArea textarea:focus {
    border-color: #2B6CB0 !important;
    box-shadow: 0 0 0 2px rgba(43,108,176,0.25) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #1A365D;
    color: #63B3ED;
    border: 1px solid #2B6CB0;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    background: #2B6CB0;
    color: #EBF8FF;
    border-color: #3182CE;
  }
  .stButton > button[kind="primary"] {
    background: #2B6CB0;
    color: #EBF8FF;
    border-color: #3182CE;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] {
    background: #161B27;
    border: 1px solid #1E2A3A;
    border-radius: 8px;
  }

  /* ── Metric boxes ── */
  [data-testid="metric-container"] {
    background: #161B27;
    border: 1px solid #1E2A3A;
    border-radius: 8px;
    padding: 0.75rem;
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
        "patient_id":    "",
        "physician_id":  "",
        "session_id":    None,
        "messages":      [],        # displayed chat messages
        "last_codes":    [],        # last ICD-10 result
        "last_diagnoses":[],
        "last_queries":  [],
        "claim_ready":   None,
        "memory":        None,
        "pipeline_running": False,
        "active_tab":    "chat"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_memory() -> ConversationMemory:
    """Initialize memory once, reuse across reruns."""
    if not MEMORY_AVAILABLE:
        return None
    if st.session_state.memory is None:
        st.session_state.memory = ConversationMemory()
    return st.session_state.memory


def call_ollama_chat(messages: list, system_prompt: str = None) -> str:
    """
    Calls Ollama with a conversation history.
    Formats messages into a single prompt string since Ollama's
    /api/generate endpoint takes a single prompt, not a messages array.
    For the /api/chat endpoint (if available) you could pass messages directly.
    """
    # Build prompt from message history
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n\n"
    for msg in messages:
        role    = "Physician" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n\n"
    prompt += "Assistant:"

    try:
        response = requests.post(OLLAMA_URL, json={
            "model":  MODEL,
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

    # Persist to SQLite
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
            # Render ICD-10 results in a structured block
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
    st.markdown('<div class="section-header">Patient Session</div>', unsafe_allow_html=True)

    # Patient ID input
    patient_id_input = st.text_input(
        "Patient ID",
        value    = st.session_state.patient_id,
        placeholder = "e.g. P001",
        key      = "patient_id_input"
    )
    physician_input = st.text_input(
        "Physician",
        value    = st.session_state.physician_id,
        placeholder = "e.g. Dr. Wang",
        key      = "physician_input"
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

                    # Load existing messages from database
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

    # Active session info
    if st.session_state.session_id:
        st.markdown(f"""
        <div class="patient-card">
          <h4>Active Session</h4>
          <p>Patient: <span class="pid">{st.session_state.patient_id}</span></p>
          <p>Physician: {st.session_state.physician_id or '—'}</p>
          <p>Session: <code style="font-size:10px;color:#4A5568">{st.session_state.session_id}</code></p>
          <span class="badge badge-active">Active</span>
        </div>
        """, unsafe_allow_html=True)

    # Last ICD-10 result summary
    if st.session_state.last_codes:
        st.markdown('<div class="section-header">Last Coding Result</div>', unsafe_allow_html=True)
        for c in st.session_state.last_codes:
            role  = c.get("role", "")
            icon  = "🔴" if role == "PRIMARY" else "🔵"
            code  = c.get("code", "")
            desc  = c.get("description", "")[:30]
            st.markdown(f"""
            <div style="font-size:12px; padding:3px 0; color:#A0AEC0">
              {icon} <code style="color:#63B3ED">{code}</code> {desc}
            </div>
            """, unsafe_allow_html=True)

        if st.session_state.claim_ready is not None:
            badge = "badge-ready" if st.session_state.claim_ready else "badge-hold"
            label = "Claim Ready" if st.session_state.claim_ready else "Hold — Query Needed"
            st.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)

    # Export section
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

    # Model status
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
    try:
        r = requests.get("http://localhost:11434", timeout=2)
        st.markdown('<span class="badge badge-ready">Ollama Running</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="badge badge-hold">Ollama Offline</span>', unsafe_allow_html=True)
        st.caption("Run: `ollama serve`")

    st.markdown(f'<p style="font-size:11px;color:#4A5568;margin-top:4px">Model: {MODEL}</p>', unsafe_allow_html=True)
    mem_status = "badge-ready" if MEMORY_AVAILABLE else "badge-hold"
    mem_label  = "Memory Active" if MEMORY_AVAILABLE else "Memory Unavailable"
    st.markdown(f'<span class="badge {mem_status}">{mem_label}</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AREA — TABS
# ═══════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:1.5rem;">
  <div style="font-size:28px; font-weight:700; color:#E2E8F0; letter-spacing:-0.5px">
    Clinical Coding Assistant
  </div>
  <div style="font-size:12px; color:#4A5568; padding-top:6px">
    ICD-10 extraction · Diagnosis support · Physician Q&A
  </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_pipeline, tab_history, tab_db, tab_about = st.tabs([
    "💬 Chat",
    "⚙️ ICD-10 Pipeline",
    "📋 Session History",
    "🗄️ Database Viewer",
    "ℹ️ About"
])


# ── TAB 1: CHAT ────────────────────────────────────────────────────────────
with tab_chat:
    if not st.session_state.session_id:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#4A5568">
          <div style="font-size:48px; margin-bottom:1rem">🏥</div>
          <div style="font-size:16px; font-weight:600; color:#718096">No active session</div>
          <div style="font-size:13px; margin-top:0.5rem">
            Enter a Patient ID in the sidebar and click Start / Resume
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center; padding:2rem; color:#4A5568; font-size:13px">
                  Session started. Enter a clinical note or question below.
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    render_message(msg)

        st.markdown("<div style='margin:1rem 0'></div>", unsafe_allow_html=True)

        # Message input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Message",
                placeholder = "Ask a clinical question, or paste a clinical note to extract ICD-10 codes...",
                height      = 100,
                label_visibility = "collapsed"
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
            # Save physician message
            add_message("physician", user_input.strip())

            # Build conversation history for Qwen
            history = []
            for m in st.session_state.messages[:-1]:  # exclude the one just added
                role = "user" if m["role"] == "physician" else "assistant"
                history.append({"role": role, "content": m["content"]})

            # System prompt for the chat assistant
            system = """You are a clinical coding assistant helping physicians with:
1. ICD-10-CM code identification and explanation
2. Clinical question answering
3. Diagnosis support based on clinical notes

Be concise, accurate, and always cite coding guidelines when relevant.
If asked to extract ICD-10 codes, suggest the physician uses the ICD-10 Pipeline tab for the full verified pipeline."""

            # Add current message
            history.append({"role": "user", "content": user_input.strip()})

            with st.spinner("Thinking..."):
                response = call_ollama_chat(history, system_prompt=system)

            add_message("assistant", response)
            st.rerun()


# ── TAB 2: ICD-10 PIPELINE ─────────────────────────────────────────────────
with tab_pipeline:
    st.markdown('<div class="section-header">Run Full Pipeline</div>', unsafe_allow_html=True)
    st.caption("Runs all 4 agents: Evidence Extraction → UMLS Normalization → MCP Code Lookup → Validation → Reconciliation")

    if not st.session_state.session_id:
        st.info("Start a session in the sidebar first.")
    else:
        note_input = st.text_area(
            "Clinical note",
            placeholder = "Paste the full clinical note here...",
            height      = 200,
            key         = "pipeline_note"
        )

        umls_key_input = st.text_input(
            "UMLS API Key (optional — leave blank to skip normalization)",
            value       = "" if UMLS_API_KEY == "YOUR_UMLS_API_KEY_HERE" else UMLS_API_KEY,
            type        = "password",
            key         = "umls_key"
        )

        run_btn = st.button("▶ Run ICD-10 Pipeline", type="primary", use_container_width=True)

        if run_btn and note_input.strip():
            if not PIPELINE_AVAILABLE:
                st.error("Pipeline file (icd10_pipeline_with_umls.py) not found in the same folder.")
            else:
                add_message("physician", f"[Pipeline run requested]\n\n{note_input.strip()}")

                # Progress indicators
                progress = st.progress(0)
                status   = st.empty()

                try:
                    status.markdown('<span class="badge badge-thinking">Stage 1 — Extracting evidence...</span>', unsafe_allow_html=True)
                    progress.progress(20)

                    api_key = umls_key_input.strip() or UMLS_API_KEY
                    results = run_icd10_pipeline(note_input.strip(), umls_api_key=api_key)

                    progress.progress(100)
                    status.markdown('<span class="badge badge-ready">Pipeline complete</span>', unsafe_allow_html=True)

                    final = results.get("final", {})
                    codes = final.get("final_codes", [])

                    # Save to session state for sidebar display
                    st.session_state.last_codes    = codes
                    st.session_state.claim_ready   = final.get("claim_ready", False)
                    st.session_state.last_queries  = final.get("queries", [])

                    # Save to memory
                    code_list = [c["code"] for c in codes]
                    if MEMORY_AVAILABLE:
                        memory = get_memory()
                        if memory:
                            memory.save_results(
                                st.session_state.session_id,
                                icd10_codes = code_list
                            )

                    # Format and display results
                    formatted = format_icd10_result(final)
                    add_message("assistant", formatted, message_type="icd10_result")

                    # Show detailed breakdown
                    st.markdown('<div class="section-header">Code Breakdown</div>', unsafe_allow_html=True)
                    for c in codes:
                        role_color = "#FC8181" if c.get("role") == "PRIMARY" else "#63B3ED"
                        st.markdown(f"""
                        <div class="code-block">
                          <span class="code-label">Seq {c.get('sequence')} · {c.get('role')}</span><br>
                          <span style="color:{role_color}; font-size:16px; font-weight:700">{c.get('code')}</span>
                          <span style="color:#A0AEC0; margin-left:12px">{c.get('description')}</span><br>
                          <span style="color:#4A5568; font-size:11px; margin-top:4px; display:block">{c.get('justification','')}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Physician queries
                    queries = final.get("queries", [])
                    if queries:
                        st.markdown('<div class="section-header">⚠️ Physician Queries Required</div>', unsafe_allow_html=True)
                        for q in queries:
                            st.warning(q)

                    # Claim status
                    if final.get("claim_ready"):
                        st.success("✅ Claim ready for submission")
                    else:
                        st.error("⛔ Hold claim — physician queries must be resolved first")

                    # Coding summary
                    st.markdown('<div class="section-header">Billing Summary</div>', unsafe_allow_html=True)
                    st.info(final.get("coding_summary", ""))

                    # Full JSON expander
                    with st.expander("Full pipeline output (JSON)"):
                        st.json(results)

                except Exception as e:
                    progress.progress(0)
                    status.error(f"Pipeline error: {str(e)}")
                    st.exception(e)


# ── TAB 3: SESSION HISTORY ─────────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="section-header">Session History</div>', unsafe_allow_html=True)

    if not MEMORY_AVAILABLE:
        st.error("conversation_memory.py not found. Place it in the same folder as this app.")
    elif not st.session_state.patient_id:
        st.info("Start a session in the sidebar to view history.")
    else:
        memory = get_memory()
        if memory:
            summary = memory.patient_summary(st.session_state.patient_id)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sessions",   summary["total_sessions"])
            col2.metric("Messages",   summary["total_messages"])
            col3.metric("Codes assigned", len(summary["all_icd10_codes"]))
            col4.metric("Last active", summary["last_active"][8:10] + "/" +
                        summary["last_active"][5:7] if summary["last_active"] else "—")

            # All ICD-10 codes ever assigned
            if summary["all_icd10_codes"]:
                st.markdown('<div class="section-header">All ICD-10 Codes (this patient)</div>', unsafe_allow_html=True)
                codes_str = "  ".join([
                    f'<code style="background:#0A1628;color:#63B3ED;padding:2px 8px;border-radius:4px;margin:2px">{c}</code>'
                    for c in summary["all_icd10_codes"]
                ])
                st.markdown(codes_str, unsafe_allow_html=True)

            # Session list
            st.markdown('<div class="section-header">Past Sessions</div>', unsafe_allow_html=True)
            for s in summary["sessions"]:
                status_badge = "badge-active" if s["status"] == "active" else "badge-hold"
                with st.expander(f"Session: {s['session_id']}"):
                    st.markdown(f"""
                    <div style="font-size:12px; color:#A0AEC0; line-height:1.8">
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

                    # Load this session's messages
                    conn = __import__('sqlite3').connect("physician_conversations.db")
                    conn.row_factory = __import__('sqlite3').Row
                    msgs = conn.execute(
                        "SELECT role, content, timestamp FROM messages WHERE session_id=? ORDER BY timestamp ASC",
                        (s["session_id"],)
                    ).fetchall()
                    conn.close()

                    for m in msgs:
                        role  = m["role"].upper()
                        color = "#63B3ED" if m["role"] == "physician" else "#68D391"
                        time_ = m["timestamp"][11:16]
                        st.markdown(f"""
                        <div style="font-size:12px; padding:4px 0; border-bottom:1px solid #1E2A3A">
                          <span style="color:{color}; font-weight:600">[{time_}] {role}:</span>
                          <span style="color:#A0AEC0"> {m['content'][:120]}{'...' if len(m['content'])>120 else ''}</span>
                        </div>
                        """, unsafe_allow_html=True)


# ── TAB 4: DATABASE VIEWER ────────────────────────────────────────────────
with tab_db:
    import sqlite3 as _sqlite3
    import pandas as _pd

    st.markdown('<div class="section-header">Database Viewer</div>', unsafe_allow_html=True)
    st.caption(f"Reading from: {os.path.abspath('physician_conversations.db')}")

    DB_FILE = "physician_conversations.db"

    if not os.path.exists(DB_FILE):
        st.info("No database found yet. Start a session and send a message to create it.")
    else:
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
            st.markdown(f'<p style="font-size:12px;color:#4A5568">{len(df)} rows</p>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-header">Database Stats</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="section-header">Custom SQL Query</div>', unsafe_allow_html=True)
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


# ── TAB 5: ABOUT ───────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    <div style="max-width:600px">

    <div class="section-header">System Architecture</div>

    <div style="font-size:13px; color:#A0AEC0; line-height:2">
    <b style="color:#E2E8F0">Agent 1</b> — Evidence Extractor: reads the clinical note, pulls all diagnoses,
    symptoms, and findings with certainty levels.<br>
    <b style="color:#E2E8F0">Stage 1.5</b> — UMLS Normalizer: maps clinical shorthand to canonical terms,
    retrieves CUIs, ICD-10 hints, and SNOMED codes.<br>
    <b style="color:#E2E8F0">Agent 2</b> — Code Candidate Generator: uses MCP ICD-10 server for verified
    real-time code lookup. Qwen reasons over verified candidates.<br>
    <b style="color:#E2E8F0">Agent 3</b> — Validator: audits codes against ICD-10-CM guidelines,
    flags exclusions, missing codes, and physician queries.<br>
    <b style="color:#E2E8F0">Agent 4</b> — Reconciler: sequences final codes, assigns PRIMARY/SECONDARY
    roles, produces claim-ready output and billing summary.
    </div>

    <div class="section-header" style="margin-top:1.5rem">Data & Privacy</div>
    <div style="font-size:13px; color:#A0AEC0; line-height:2">
    All processing is <b style="color:#E2E8F0">local</b> — Qwen2.5 runs via Ollama on your machine.
    No patient data is sent to external servers.<br>
    Conversations are stored in a local SQLite database: <code>physician_conversations.db</code>
    </div>

    <div class="section-header" style="margin-top:1.5rem">References</div>
    <div style="font-size:12px; color:#4A5568; line-height:2">
    Corti — Code Like Humans (CLH), EMNLP 2025<br>
    DR.KNOWS — JMIR AI 2025<br>
    PLM-ICD + Qwen2.5 hybrid, Preprints.org 2025<br>
    Benchmarking LLMs for ICD-10, MedRxiv 2024
    </div>

    </div>
    """, unsafe_allow_html=True)
