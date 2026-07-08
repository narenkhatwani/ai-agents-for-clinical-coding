"""
Conversation Memory Module — SQLite
====================================
Handles saving, loading, resuming, and exporting physician conversations.
No installation needed — sqlite3 is built into Python.

Usage:
    from conversation_memory import ConversationMemory
    
    memory = ConversationMemory()
    
    # Start or resume a session
    session_id = memory.start_session(patient_id="P001", physician_id="DR_WANG")
    
    # Save messages as the conversation happens
    memory.save_message(session_id, role="physician", content="Patient has AFib...")
    memory.save_message(session_id, role="assistant", content="Based on the note...")
    
    # Load history for context window
    history = memory.load_session_for_prompt(session_id, max_tokens=3000)
    
    # Export to txt
    memory.export_to_txt(patient_id="P001")
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH          = "physician_conversations.db"   # single local file, no server
MAX_CHARS        = 12000                          # ~3000 tokens, fits in Qwen context
EXPORT_FOLDER    = "conversation_exports"         # where txt/csv exports go


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    """
    Opens a connection to the SQLite database.
    Creates the database file automatically if it doesn't exist.
    row_factory lets us access columns by name (row['content'])
    instead of by index (row[3]).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    """
    Creates all tables if they don't already exist.
    Safe to call every time the app starts — IF NOT EXISTS means
    it won't overwrite data that's already there.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Sessions table — one row per conversation session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id    TEXT PRIMARY KEY,
            patient_id    TEXT NOT NULL,
            physician_id  TEXT,
            started_at    TEXT NOT NULL,
            last_active   TEXT NOT NULL,
            note_context  TEXT,        -- the clinical note being discussed
            icd10_codes   TEXT,        -- JSON list of codes assigned in this session
            diagnoses     TEXT,        -- JSON list of diagnoses predicted
            status        TEXT DEFAULT 'active'   -- active | closed
        )
    """)

    # Messages table — one row per message in a conversation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT NOT NULL,
            patient_id    TEXT NOT NULL,
            timestamp     TEXT NOT NULL,
            role          TEXT NOT NULL,   -- 'physician' | 'assistant' | 'system'
            content       TEXT NOT NULL,
            message_type  TEXT DEFAULT 'chat',  -- 'chat' | 'icd10_result' | 'diagnosis'
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # Index for fast patient lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_patient
        ON messages(patient_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id)
    """)

    conn.commit()
    conn.close()
    print(f"  [Memory] Database ready: {DB_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def start_session(
    patient_id:   str,
    physician_id: str  = "unknown",
    note_context: str  = None
) -> str:
    """
    Creates a new conversation session and returns its session_id.
    
    session_id format: P001_20260609_143022
    — easy to read, sortable by date, unique per patient per day-time.
    """
    now        = datetime.now()
    session_id = f"{patient_id}_{now.strftime('%Y%m%d_%H%M%S')}"

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions
            (session_id, patient_id, physician_id, started_at, last_active, note_context)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        patient_id,
        physician_id,
        now.isoformat(),
        now.isoformat(),
        note_context
    ))
    conn.commit()
    conn.close()

    print(f"  [Memory] New session started: {session_id}")
    return session_id


def resume_latest_session(patient_id: str) -> Optional[str]:
    """
    Returns the session_id of the most recent active session for a patient.
    Returns None if no previous session exists.
    
    Used when physician opens a patient's record and wants to
    continue where they left off.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id FROM sessions
        WHERE patient_id = ? AND status = 'active'
        ORDER BY last_active DESC
        LIMIT 1
    """, (patient_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        session_id = row['session_id']
        print(f"  [Memory] Resuming session: {session_id}")
        return session_id
    else:
        print(f"  [Memory] No previous session found for patient {patient_id}")
        return None


def close_session(session_id: str):
    """
    Marks a session as closed. Closed sessions are kept in the
    database for history but won't show up in resume searches.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE sessions SET status = 'closed'
        WHERE session_id = ?
    """, (session_id,))
    conn.commit()
    conn.close()


def update_session_results(
    session_id: str,
    icd10_codes: list = None,
    diagnoses:   list = None
):
    """
    Saves the ICD-10 codes and diagnoses from a pipeline run
    back into the session record. Stored as JSON strings.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE sessions SET
            icd10_codes = ?,
            diagnoses   = ?,
            last_active = ?
        WHERE session_id = ?
    """, (
        json.dumps(icd10_codes or []),
        json.dumps(diagnoses   or []),
        datetime.now().isoformat(),
        session_id
    ))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# MESSAGE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def save_message(
    session_id:   str,
    role:         str,
    content:      str,
    message_type: str = "chat"
):
    """
    Saves one message to the database.
    
    role:         'physician', 'assistant', or 'system'
    message_type: 'chat'         — regular conversation turn
                  'icd10_result' — pipeline output with codes
                  'diagnosis'    — diagnosis prediction result
    """
    # Get patient_id from session (needed for the messages table index)
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT patient_id FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Session {session_id} not found")
    patient_id = row['patient_id']

    # Save the message
    cursor.execute("""
        INSERT INTO messages
            (session_id, patient_id, timestamp, role, content, message_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        patient_id,
        datetime.now().isoformat(),
        role,
        content,
        message_type
    ))

    # Update last_active on the session so resume() finds the most recent one
    cursor.execute("""
        UPDATE sessions SET last_active = ? WHERE session_id = ?
    """, (datetime.now().isoformat(), session_id))

    conn.commit()
    conn.close()


def load_session_for_prompt(session_id: str, max_chars: int = MAX_CHARS) -> list:
    """
    Loads conversation history formatted for the Ollama prompt.
    
    Returns a list of dicts: [{"role": "user", "content": "..."}, ...]
    
    max_chars controls how much history to include — this prevents
    the context window from overflowing. When the conversation is
    very long, it keeps the most recent messages (the ones most
    relevant to continuing the conversation).
    
    Dr. Wang's requirement: "how much can it remember? save as txt,
    use when resuming" — this is that mechanism.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content FROM messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()

    # Map our role names to Ollama's expected format
    role_map = {
        "physician": "user",
        "assistant": "assistant",
        "system":    "system"
    }

    # Build message list, truncating from oldest if over char limit
    messages   = []
    total_chars = 0
    for row in reversed(rows):       # start from most recent
        role    = role_map.get(row['role'], row['role'])
        content = row['content']
        total_chars += len(content)
        if total_chars > max_chars:
            break                    # stop adding older messages
        messages.insert(0, {"role": role, "content": content})

    return messages


def get_patient_summary(patient_id: str) -> dict:
    """
    Returns a summary of all sessions for a patient:
    total sessions, all ICD-10 codes ever assigned,
    all diagnoses ever predicted, last active date.
    
    Useful for the physician assistant agent — "what do we
    know about this patient from previous sessions?"
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # Get all sessions
    cursor.execute("""
        SELECT session_id, started_at, last_active, icd10_codes, diagnoses, status
        FROM sessions
        WHERE patient_id = ?
        ORDER BY started_at DESC
    """, (patient_id,))
    sessions = cursor.fetchall()

    # Get message count
    cursor.execute("""
        SELECT COUNT(*) as count FROM messages WHERE patient_id = ?
    """, (patient_id,))
    msg_count = cursor.fetchone()['count']

    conn.close()

    # Aggregate all codes and diagnoses across all sessions
    all_codes     = []
    all_diagnoses = []
    for s in sessions:
        if s['icd10_codes']:
            all_codes.extend(json.loads(s['icd10_codes']))
        if s['diagnoses']:
            all_diagnoses.extend(json.loads(s['diagnoses']))

    return {
        "patient_id":      patient_id,
        "total_sessions":  len(sessions),
        "total_messages":  msg_count,
        "last_active":     sessions[0]['last_active'] if sessions else None,
        "all_icd10_codes": list(set(all_codes)),        # deduplicated
        "all_diagnoses":   list(set(all_diagnoses)),    # deduplicated
        "sessions":        [dict(s) for s in sessions]
    }


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def export_session_to_txt(session_id: str) -> str:
    """
    Exports a single session to a readable txt file.
    Returns the file path.
    
    Format:
    ─────────────────────────────────────
    Session: P001_20260609_143022
    Patient: P001
    Started: 2026-06-09 14:30:22
    ─────────────────────────────────────
    [14:30:25] PHYSICIAN:
    Patient has AFib with RVR...
    
    [14:30:48] ASSISTANT:
    Based on the clinical note, the following ICD-10 codes...
    """
    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    conn   = get_connection()
    cursor = conn.cursor()

    # Get session info
    cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    session = cursor.fetchone()

    # Get all messages
    cursor.execute("""
        SELECT timestamp, role, content FROM messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = cursor.fetchall()
    conn.close()

    filepath = os.path.join(EXPORT_FOLDER, f"{session_id}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Session ID  : {session['session_id']}\n")
        f.write(f"Patient ID  : {session['patient_id']}\n")
        f.write(f"Physician   : {session['physician_id']}\n")
        f.write(f"Started     : {session['started_at']}\n")
        f.write(f"Last active : {session['last_active']}\n")
        if session['icd10_codes']:
            codes = json.loads(session['icd10_codes'])
            f.write(f"ICD-10 codes: {', '.join(codes)}\n")
        if session['diagnoses']:
            diag = json.loads(session['diagnoses'])
            f.write(f"Diagnoses   : {', '.join(diag)}\n")
        f.write("=" * 60 + "\n\n")

        for msg in messages:
            # Format timestamp to just HH:MM:SS for readability
            ts   = msg['timestamp'][11:19]
            role = msg['role'].upper()
            f.write(f"[{ts}] {role}:\n")
            f.write(f"{msg['content']}\n\n")

    print(f"  [Memory] Exported to: {filepath}")
    return filepath


def export_patient_to_txt(patient_id: str) -> str:
    """
    Exports ALL sessions for a patient into one txt file.
    Sessions are separated by a divider.
    """
    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id FROM sessions
        WHERE patient_id = ?
        ORDER BY started_at ASC
    """, (patient_id,))
    sessions = cursor.fetchall()
    conn.close()

    filepath = os.path.join(EXPORT_FOLDER, f"patient_{patient_id}_full_history.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"FULL CONVERSATION HISTORY — Patient {patient_id}\n")
        f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for s in sessions:
            # Write each session inline
            conn2   = get_connection()
            cursor2 = conn2.cursor()
            cursor2.execute("""
                SELECT timestamp, role, content FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (s['session_id'],))
            messages = cursor2.fetchall()
            conn2.close()

            f.write(f"SESSION: {s['session_id']}\n")
            f.write("-" * 40 + "\n")
            for msg in messages:
                ts   = msg['timestamp'][11:19]
                role = msg['role'].upper()
                f.write(f"[{ts}] {role}: {msg['content']}\n\n")
            f.write("\n")

    print(f"  [Memory] Full history exported to: {filepath}")
    return filepath


def export_patient_to_csv(patient_id: str) -> str:
    """
    Exports all messages for a patient to a CSV file.
    Useful for analysis, sharing with the team, or loading into Excel.
    Requires pandas (pip install pandas).
    """
    try:
        import pandas as pd
    except ImportError:
        print("  [Memory] pandas not installed. Run: pip install pandas")
        return None

    os.makedirs(EXPORT_FOLDER, exist_ok=True)

    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT
            m.timestamp,
            m.patient_id,
            m.session_id,
            m.role,
            m.content,
            m.message_type
        FROM messages m
        WHERE m.patient_id = ?
        ORDER BY m.timestamp ASC
    """, conn, params=(patient_id,))
    conn.close()

    filepath = os.path.join(EXPORT_FOLDER, f"patient_{patient_id}_messages.csv")
    df.to_csv(filepath, index=False)
    print(f"  [Memory] CSV exported to: {filepath}")
    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER
# Wraps everything into one class for easy use in the pipeline
# ═══════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Single entry point for all memory operations.
    Initialize once, use throughout the pipeline.

    Example:
        memory     = ConversationMemory()
        session_id = memory.new_or_resume("P001", physician_id="DR_WANG")

        # Save physician's note
        memory.add("physician", "Patient has AFib with RVR...", session_id)

        # Save pipeline result
        memory.add("assistant", "ICD-10 codes: I48.91, E11.22...", session_id,
                   message_type="icd10_result")

        # Get history for next Qwen call
        history = memory.get_history(session_id)

        # Save codes to session record
        memory.save_results(session_id, icd10_codes=["I48.91", "E11.22"])

        # Export for download
        memory.export_txt(session_id)
    """

    def __init__(self, db_path: str = DB_PATH):
        global DB_PATH
        DB_PATH = db_path
        initialize_database()

    def new_or_resume(
        self,
        patient_id:   str,
        physician_id: str = "unknown",
        note_context: str = None,
        force_new:    bool = False
    ) -> str:
        """
        Returns an existing session_id if one exists for this patient,
        or creates a new one. 
        
        force_new=True always creates a new session even if one exists.
        """
        if not force_new:
            existing = resume_latest_session(patient_id)
            if existing:
                return existing
        return start_session(patient_id, physician_id, note_context)

    def add(
        self,
        role:         str,
        content:      str,
        session_id:   str,
        message_type: str = "chat"
    ):
        save_message(session_id, role, content, message_type)

    def get_history(self, session_id: str, max_chars: int = MAX_CHARS) -> list:
        return load_session_for_prompt(session_id, max_chars)

    def save_results(
        self,
        session_id:  str,
        icd10_codes: list = None,
        diagnoses:   list = None
    ):
        update_session_results(session_id, icd10_codes, diagnoses)

    def patient_summary(self, patient_id: str) -> dict:
        return get_patient_summary(patient_id)

    def export_txt(self, session_id: str) -> str:
        return export_session_to_txt(session_id)

    def export_patient_txt(self, patient_id: str) -> str:
        return export_patient_to_txt(patient_id)

    def export_patient_csv(self, patient_id: str) -> str:
        return export_patient_to_csv(patient_id)

    def close(self, session_id: str):
        close_session(session_id)


# ═══════════════════════════════════════════════════════════════════════════
# DEMO — run this file directly to see it working
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  CONVERSATION MEMORY MODULE — DEMO")
    print("=" * 55)

    memory = ConversationMemory()

    # Simulate a physician opening a patient record
    print("\n--- Physician opens Patient P001 ---")
    session_id = memory.new_or_resume(
        patient_id   = "P001",
        physician_id = "Dr_Wang",
        note_context = "67yo male, AFib, T2DM, CKD stage 3"
    )

    # Simulate conversation
    memory.add("physician",
               "Patient admitted with chest pain. History of AFib and T2DM with CKD stage 3.",
               session_id)

    memory.add("assistant",
               "Based on the note, I suggest the following ICD-10 codes:\n"
               "PRIMARY: I48.91 — Unspecified atrial fibrillation\n"
               "SECONDARY: E11.22 — T2DM with diabetic CKD stage 3\n"
               "SECONDARY: I10 — Essential hypertension",
               session_id,
               message_type="icd10_result")

    memory.add("physician", "Can you also check for heart failure?", session_id)

    memory.add("assistant",
               "The chest X-ray shows pulmonary edema which may indicate heart failure. "
               "I recommend a physician query: please clarify whether heart failure is "
               "present and specify type if confirmed.",
               session_id)

    # Save the pipeline results to the session record
    memory.save_results(
        session_id,
        icd10_codes = ["I48.91", "E11.22", "I10"],
        diagnoses   = ["Atrial fibrillation", "Type 2 diabetes with CKD", "Possible heart failure"]
    )

    # Show what history looks like for the prompt
    print("\n--- History loaded for next Qwen call ---")
    history = memory.get_history(session_id)
    for msg in history:
        print(f"  [{msg['role']}]: {msg['content'][:60]}...")

    # Show patient summary
    print("\n--- Patient summary ---")
    summary = memory.patient_summary("P001")
    print(f"  Total sessions : {summary['total_sessions']}")
    print(f"  Total messages : {summary['total_messages']}")
    print(f"  ICD-10 codes   : {summary['all_icd10_codes']}")
    print(f"  Diagnoses      : {summary['all_diagnoses']}")

    # Export
    print("\n--- Exporting ---")
    txt_path = memory.export_txt(session_id)
    print(f"  Single session : {txt_path}")

    full_path = memory.export_patient_txt("P001")
    print(f"  Full history   : {full_path}")

    print("\n  Done. Database saved to:", DB_PATH)
    print("=" * 55)
