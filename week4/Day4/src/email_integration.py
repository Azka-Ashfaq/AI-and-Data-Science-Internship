"""
Day 4 - Task 2: Email Automation.

Notifies the assigned employee when a visit is booked/rescheduled/
cancelled. Same mock-by-default pattern as calendar_integration.py --
real SMTP/Gmail API sending requires either an app password or full
Gmail API OAuth (same complexity as Calendar), so this defaults to
logging exactly what would be sent to data/email_log.jsonl, and only
sends for real if SMTP_* env vars are set.

Real sending uses plain SMTP (works with Gmail's "App Passwords" feature,
which is much simpler to set up than the full Gmail API OAuth flow used
in calendar_integration.py) rather than the Gmail API, specifically to
avoid putting a second OAuth setup in this student's path.
"""
import os
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "..", "data", "email_log.jsonl")


def is_live():
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER")
                and os.environ.get("SMTP_PASSWORD"))


def _mock_send(to_addr, subject, body):
    record = {"to": to_addr, "subject": subject, "body": body,
              "sent_at": datetime.now().isoformat(), "mode": "mock"}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return True


def _real_send(to_addr, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = to_addr
    with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                       int(os.environ.get("SMTP_PORT", 587))) as server:
        server.starttls()
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(msg)
    return True


def notify_employee(employee_email, employee_name, client_name, client_phone,
                     property_summary, scheduled_at, action="booked", notes=""):
    """action: 'booked' | 'rescheduled' | 'cancelled'"""
    verb = {"booked": "New visit booked", "rescheduled": "Visit rescheduled",
            "cancelled": "Visit cancelled"}[action]
    subject = f"{verb}: {client_name} - {scheduled_at}"
    body = (f"Hi {employee_name},\n\n"
            f"{verb} for one of your assigned properties.\n\n"
            f"Client: {client_name} ({client_phone})\n"
            f"Property: {property_summary}\n"
            f"Time: {scheduled_at}\n"
            f"Notes: {notes}\n\n"
            f"- RealEstate Hub Voice Agent (Ayesha)")

    if not employee_email:
        # agent unassigned in the data -- log it as a skipped notification
        # rather than silently failing or crashing
        record = {"to": None, "subject": subject, "body": body,
                   "sent_at": datetime.now().isoformat(), "mode": "skipped_no_email"}
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        return False, "No employee email on file -- notification skipped and logged."

    try:
        if is_live():
            _real_send(employee_email, subject, body)
            return True, "Sent live."
        else:
            _mock_send(employee_email, subject, body)
            return True, "Logged (mock mode)."
    except Exception as e:
        # Task 4 requires handling failures/retries -- surfaced here as a
        # clear failure the caller (workflow.py) can retry or escalate on.
        return False, f"Send failed: {e}"


if __name__ == "__main__":
    print("Live mode:", is_live())
    ok, msg = notify_employee(
        employee_email="asad.mirza@realestatehub.example.pk",
        employee_name="Asad Mirza", client_name="Ahmed Raza", client_phone="0300-1234567",
        property_summary="House in DHA Defence, Lahore, PKR 15,000,000",
        scheduled_at="2026-10-02T11:00:00", action="booked",
        notes="Budget 3 crore, wants 3 bed"
    )
    print(ok, msg)

    ok2, msg2 = notify_employee(
        employee_email=None, employee_name="Unassigned", client_name="Sara Khan",
        client_phone="0301-9998888", property_summary="Flat in Gulberg, Lahore",
        scheduled_at="2026-10-04T14:00:00", action="booked"
    )
    print(ok2, msg2)

    print(f"\nSee {LOG_PATH} for the full mock log.")
