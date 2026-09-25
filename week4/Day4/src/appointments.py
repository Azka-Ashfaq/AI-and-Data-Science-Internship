"""
Day 4 - Task 1 & 3 (data layer): Appointment Management.

SQLite-backed appointment store shared by calendar sync, email
notifications, and CRM logging. This is the single source of truth for
"is this slot free" -- Calendar/Email are notified AFTER a slot is
reserved here, never the other way round, so we never double-book while
waiting on a slower external API call.
"""
import sqlite3
import os
import uuid
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "..", "data", "crm.db")


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id TEXT PRIMARY KEY,
            property_id INTEGER,
            client_name TEXT,
            client_phone TEXT,
            employee_name TEXT,
            scheduled_at TEXT,     -- ISO 8601, e.g. 2026-10-02T11:00:00
            duration_minutes INTEGER DEFAULT 30,
            status TEXT,           -- 'booked' | 'rescheduled' | 'cancelled'
            notes TEXT,
            calendar_event_id TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    return conn


def _overlaps(conn, employee_name, scheduled_at, duration_minutes, exclude_id=None):
    """True if this employee already has a non-cancelled appointment whose
    window overlaps [scheduled_at, scheduled_at + duration]."""
    start = datetime.fromisoformat(scheduled_at)
    end = start + timedelta(minutes=duration_minutes)
    rows = conn.execute(
        "SELECT appointment_id, scheduled_at, duration_minutes FROM appointments "
        "WHERE employee_name = ? AND status != 'cancelled'", (employee_name,)
    ).fetchall()
    for appt_id, other_start_s, other_dur in rows:
        if exclude_id and appt_id == exclude_id:
            continue
        other_start = datetime.fromisoformat(other_start_s)
        other_end = other_start + timedelta(minutes=other_dur or 30)
        if start < other_end and other_start < end:
            return True
    return False


def check_availability(employee_name, scheduled_at, duration_minutes=30):
    """Task 5 (Day 5 tool) lives here in Day 4 as the underlying check:
    'Never book unavailable slots' starts with having a real availability
    checker, not just trusting the caller's requested time."""
    conn = _conn()
    try:
        return not _overlaps(conn, employee_name, scheduled_at, duration_minutes)
    finally:
        conn.close()


def book_appointment(property_id, client_name, client_phone, employee_name,
                      scheduled_at, duration_minutes=30, notes=""):
    """Returns (appointment_id, ok, message). Refuses to book an
    overlapping slot -- this is the 'never book unavailable slots' rule
    enforced at the data layer, not just hoped for by the LLM."""
    conn = _conn()
    try:
        if _overlaps(conn, employee_name, scheduled_at, duration_minutes):
            return None, False, f"{employee_name} is already booked around {scheduled_at}."
        appt_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO appointments (appointment_id, property_id, client_name, "
            "client_phone, employee_name, scheduled_at, duration_minutes, status, "
            "notes, calendar_event_id, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (appt_id, property_id, client_name, client_phone, employee_name,
             scheduled_at, duration_minutes, "booked", notes, None, now, now)
        )
        conn.commit()
        return appt_id, True, "Booked."
    finally:
        conn.close()


def reschedule_appointment(appointment_id, new_scheduled_at):
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT employee_name, duration_minutes, status FROM appointments WHERE appointment_id = ?",
            (appointment_id,)
        ).fetchone()
        if not row:
            return False, "Appointment not found."
        employee_name, duration_minutes, status = row
        if status == "cancelled":
            return False, "Cannot reschedule a cancelled appointment."
        if _overlaps(conn, employee_name, new_scheduled_at, duration_minutes, exclude_id=appointment_id):
            return False, f"{employee_name} is already booked around {new_scheduled_at}."
        conn.execute(
            "UPDATE appointments SET scheduled_at = ?, status = 'rescheduled', updated_at = ? "
            "WHERE appointment_id = ?",
            (new_scheduled_at, datetime.now().isoformat(), appointment_id)
        )
        conn.commit()
        return True, "Rescheduled."
    finally:
        conn.close()


def cancel_appointment(appointment_id, reason=""):
    conn = _conn()
    try:
        row = conn.execute("SELECT status FROM appointments WHERE appointment_id = ?",
                            (appointment_id,)).fetchone()
        if not row:
            return False, "Appointment not found."
        if row[0] == "cancelled":
            return False, "Already cancelled."
        conn.execute(
            "UPDATE appointments SET status = 'cancelled', notes = notes || ' | cancel reason: ' || ?, "
            "updated_at = ? WHERE appointment_id = ?",
            (reason, datetime.now().isoformat(), appointment_id)
        )
        conn.commit()
        return True, "Cancelled."
    finally:
        conn.close()


def set_calendar_event_id(appointment_id, calendar_event_id):
    conn = _conn()
    try:
        conn.execute("UPDATE appointments SET calendar_event_id = ? WHERE appointment_id = ?",
                     (calendar_event_id, appointment_id))
        conn.commit()
    finally:
        conn.close()


def get_appointment(appointment_id):
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM appointments WHERE appointment_id = ?",
                            (appointment_id,)).fetchone()
        if not row:
            return None
        cols = [c[1] for c in conn.execute("PRAGMA table_info(appointments)").fetchall()]
        return dict(zip(cols, row))
    finally:
        conn.close()


if __name__ == "__main__":
    appt_id, ok, msg = book_appointment(
        property_id=14436786, client_name="Ahmed Raza", client_phone="0300-1234567",
        employee_name="Asad Mirza", scheduled_at="2026-10-02T11:00:00",
        notes="Interested in 4 bed DHA house, budget 3 crore"
    )
    print(f"Book: {ok} | {msg} | id={appt_id}")

    # try double-booking the same employee at an overlapping time -> should fail
    _, ok2, msg2 = book_appointment(
        property_id=16487087, client_name="Sara Khan", client_phone="0301-9998888",
        employee_name="Asad Mirza", scheduled_at="2026-10-02T11:15:00",
        notes="Different client, overlapping slot"
    )
    print(f"Double-book attempt: {ok2} | {msg2}")

    ok3, msg3 = reschedule_appointment(appt_id, "2026-10-03T15:00:00")
    print(f"Reschedule: {ok3} | {msg3}")

    print("Availability check for new slot:", check_availability("Asad Mirza", "2026-10-04T09:00:00"))

    ok4, msg4 = cancel_appointment(appt_id, reason="client asked to cancel")
    print(f"Cancel: {ok4} | {msg4}")

    print(get_appointment(appt_id))
