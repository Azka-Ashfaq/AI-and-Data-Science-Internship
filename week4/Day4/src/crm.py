"""
Day 4 - Task 5: CRM Logging.
Stores call transcripts, client preferences, appointment history (via
appointments.py, same crm.db), and follow-up reminders -- the persistent
record a human agent would see if they pulled up a client's file.
"""
import sqlite3
import os
import json
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "..", "data", "crm.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            call_id TEXT PRIMARY KEY,
            client_phone TEXT,
            client_name TEXT,
            transcript_json TEXT,   -- list of {"role","text"} turns
            preferences_json TEXT,  -- final slot state (budget, city, bedrooms, ...)
            outcome TEXT,           -- 'booked' | 'no_action' | 'escalated' | 'cancelled'
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS follow_up_reminders (
            reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_phone TEXT,
            client_name TEXT,
            due_at TEXT,
            reason TEXT,
            done INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


def log_call(call_id, client_phone, client_name, transcript, preferences, outcome):
    conn = _conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO call_logs (call_id, client_phone, client_name, "
            "transcript_json, preferences_json, outcome, created_at) VALUES (?,?,?,?,?,?,?)",
            (call_id, client_phone, client_name, json.dumps(transcript),
             json.dumps(preferences), outcome, datetime.now().isoformat())
        )
        conn.commit()
    finally:
        conn.close()


def get_client_history(client_phone):
    """Everything we know about a returning caller -- this is what makes
    Day 1's 'Returning Customer' flow actually work: look them up by phone,
    greet by name, recall preferences, instead of starting from zero."""
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT call_id, client_name, preferences_json, outcome, created_at "
            "FROM call_logs WHERE client_phone = ? ORDER BY created_at DESC",
            (client_phone,)
        ).fetchall()
        return [
            {"call_id": r[0], "client_name": r[1], "preferences": json.loads(r[2]),
             "outcome": r[3], "created_at": r[4]}
            for r in rows
        ]
    finally:
        conn.close()


def add_follow_up(client_phone, client_name, due_at, reason):
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO follow_up_reminders (client_phone, client_name, due_at, reason, done, created_at) "
            "VALUES (?,?,?,?,0,?)",
            (client_phone, client_name, due_at, reason, datetime.now().isoformat())
        )
        conn.commit()
    finally:
        conn.close()


def get_due_follow_ups(as_of_iso=None):
    as_of_iso = as_of_iso or datetime.now().isoformat()
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT reminder_id, client_phone, client_name, due_at, reason FROM follow_up_reminders "
            "WHERE done = 0 AND due_at <= ? ORDER BY due_at ASC",
            (as_of_iso,)
        ).fetchall()
        return [{"reminder_id": r[0], "client_phone": r[1], "client_name": r[2],
                 "due_at": r[3], "reason": r[4]} for r in rows]
    finally:
        conn.close()


def mark_follow_up_done(reminder_id):
    conn = _conn()
    try:
        conn.execute("UPDATE follow_up_reminders SET done = 1 WHERE reminder_id = ?", (reminder_id,))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    log_call(
        call_id="call-0001", client_phone="0300-1234567", client_name="Ahmed Raza",
        transcript=[{"role": "caller", "text": "Budget 3 crore hai, DHA mein 3 bedroom chahiye."},
                    {"role": "agent", "text": "Ji sir, 3 options hain..."}],
        preferences={"budget": 30000000, "city": "Lahore", "location_hint": "DHA", "bedrooms": 3},
        outcome="booked",
    )
    add_follow_up("0300-1234567", "Ahmed Raza", "2026-10-05T10:00:00",
                  "Confirm if visit went well, ask about decision")

    print("Client history for 0300-1234567:")
    for h in get_client_history("0300-1234567"):
        print(" ", h)

    print("\nDue follow-ups as of 2026-10-06:")
    for f in get_due_follow_ups("2026-10-06T00:00:00"):
        print(" ", f)
