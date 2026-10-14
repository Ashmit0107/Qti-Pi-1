"""
backend/database.py — Qti pi

SQLite storage layer, ported/extended from Ashmit0107/Health-Companion's
backend/db.py (local copy audited at C:\\SIH\\health_companion\\backend\\db.py).

Extends the original vitals/sos/assistant_log/emergency_contacts schema with
a new `neuro_readings` table + helper functions for the ear-EEG vertical,
and a `risk_history` table so the dashboard can chart risk trends over time.
"""
import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qti_pi.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # --- Cardiac / general vitals (ported from Health-Companion) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            heart_rate REAL,
            spo2 REAL,
            body_temp REAL,
            activity TEXT,
            sleep_hours REAL,
            accel_peak_g REAL,
            accel_post_variance_g REAL,
            bp_systolic REAL,
            age REAL,
            sex INTEGER,
            height REAL,
            weight REAL,
            qrs REAL,
            qt REAL,
            t_wave REAL,
            pr REAL,
            p_wave REAL,
            created_at TEXT
        )
    """)

    # --- Neuro / ear-EEG readings (new for Qti pi) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS neuro_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            power_delta REAL,
            power_theta REAL,
            power_alpha REAL,
            power_beta REAL,
            power_gamma REAL,
            dominant_tremor_freq_hz REAL,
            tremor_amplitude_g REAL,
            seizure_risk_score REAL,
            seizure_risk_tier TEXT,
            model_source TEXT,
            created_at TEXT
        )
    """)

    # --- Risk history (new) — one row per computed /risk-score call, so the
    # dashboard can plot a trend line across cardiac + neuro + rule-based risk. ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            overall_score REAL,
            overall_tier TEXT,
            breakdown_json TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sos_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reason TEXT,
            vitals_snapshot TEXT,
            narration TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS assistant_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            response TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS emergency_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            relation TEXT,
            email TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


# ---------------- Vitals ----------------

def insert_vitals(data: dict) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cols = list(data.keys()) + ["created_at"]
    vals = list(data.values()) + [datetime.utcnow().isoformat()]
    placeholders = ",".join(["?"] * len(vals))
    cur.execute(f"INSERT INTO vitals ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def get_latest_vitals() -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vitals ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_vitals_history(limit: int = 50) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM vitals ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows][::-1]


# ---------------- Neuro readings ----------------

def insert_neuro_reading(data: dict) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cols = list(data.keys()) + ["created_at"]
    vals = list(data.values()) + [datetime.utcnow().isoformat()]
    placeholders = ",".join(["?"] * len(vals))
    cur.execute(f"INSERT INTO neuro_readings ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def get_latest_neuro_reading() -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM neuro_readings ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_neuro_history(limit: int = 50) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM neuro_readings ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows][::-1]


# ---------------- Risk history ----------------

def insert_risk_snapshot(overall_score: float, overall_tier: str, breakdown: dict) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO risk_history (overall_score, overall_tier, breakdown_json, created_at) VALUES (?, ?, ?, ?)",
        (overall_score, overall_tier, json.dumps(breakdown), datetime.utcnow().isoformat()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def get_risk_history(limit: int = 50) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM risk_history ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows[::-1]:
        d = dict(r)
        d["breakdown"] = json.loads(d.pop("breakdown_json") or "{}")
        out.append(d)
    return out


# ---------------- SOS ----------------

def insert_sos(reason: str, vitals_snapshot: str, narration: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sos_events (reason, vitals_snapshot, narration, created_at) VALUES (?, ?, ?, ?)",
        (reason, vitals_snapshot, narration, datetime.utcnow().isoformat()),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def get_sos_history(limit: int = 20) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM sos_events ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- Assistant log ----------------

def log_assistant(query: str, response: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO assistant_log (query, response, created_at) VALUES (?, ?, ?)",
        (query, response, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ---------------- Emergency contacts (multi-contact, ported) ----------------

def add_emergency_contact(name: str, phone: str, relation: str = "", email: str = "") -> dict:
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO emergency_contacts (name, phone, relation, email, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, phone, relation, email, now, now),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return {"id": rid, "name": name, "phone": phone, "relation": relation, "email": email}


def list_emergency_contacts() -> list[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, phone, relation, email, created_at, updated_at FROM emergency_contacts ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_emergency_contact(contact_id: int, **fields) -> bool:
    if not fields:
        return False
    conn = get_conn()
    cur = conn.cursor()
    fields["updated_at"] = datetime.utcnow().isoformat()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    cur.execute(f"UPDATE emergency_contacts SET {set_clause} WHERE id = ?", (*fields.values(), contact_id))
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def delete_emergency_contact(contact_id: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM emergency_contacts WHERE id = ?", (contact_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted
