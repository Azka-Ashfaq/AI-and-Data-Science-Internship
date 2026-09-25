"""
Day 4 - Task 1: Google Calendar Integration.

Real Google Calendar API support (OAuth) is included, but requires a
Google Cloud project + OAuth consent screen setup, which is a much bigger
lift than an API key (and given how much time was already spent on
LLM-key issues in Day 2, this defaults to a MOCK mode that needs zero
setup and is honest about it). Mock mode logs exactly what a real
calendar event would contain to data/calendar_log.jsonl -- same fields,
same structure, just not actually hitting Google's servers.

To turn on the real API: see README.md "Enabling real Google Calendar".
"""
import os
import json
import uuid
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "..", "data", "calendar_log.jsonl")


def _mock_create_event(summary, description, start_iso, duration_minutes, attendees):
    event_id = f"mock-evt-{uuid.uuid4().hex[:10]}"
    end_iso = (datetime.fromisoformat(start_iso) + timedelta(minutes=duration_minutes)).isoformat()
    record = {
        "action": "create", "event_id": event_id, "summary": summary,
        "description": description, "start": start_iso, "end": end_iso,
        "attendees": attendees, "logged_at": datetime.now().isoformat(),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return event_id


def _mock_update_event(event_id, start_iso, duration_minutes):
    end_iso = (datetime.fromisoformat(start_iso) + timedelta(minutes=duration_minutes)).isoformat()
    record = {"action": "update", "event_id": event_id, "start": start_iso,
              "end": end_iso, "logged_at": datetime.now().isoformat()}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _mock_delete_event(event_id):
    record = {"action": "delete", "event_id": event_id, "logged_at": datetime.now().isoformat()}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _real_calendar_service():
    """Only imported/used if GOOGLE_CALENDAR_CREDENTIALS_JSON is set --
    see README for the one-time OAuth setup this requires."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    token_path = os.path.join(HERE, "..", "data", "calendar_token.json")
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.environ["GOOGLE_CALENDAR_CREDENTIALS_JSON"], SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def is_live():
    return bool(os.environ.get("GOOGLE_CALENDAR_CREDENTIALS_JSON"))


def create_event(client_name, client_phone, employee_name, property_summary,
                  start_iso, duration_minutes=30, notes=""):
    """Create a property-visit calendar event. Returns event_id."""
    summary = f"Property Visit: {client_name} x {employee_name}"
    description = (f"Client: {client_name} ({client_phone})\n"
                    f"Employee: {employee_name}\n"
                    f"Property: {property_summary}\n"
                    f"Notes: {notes}")
    if not is_live():
        return _mock_create_event(summary, description, start_iso, duration_minutes,
                                   attendees=[client_phone, employee_name])
    service = _real_calendar_service()
    event = {
        "summary": summary, "description": description,
        "start": {"dateTime": start_iso, "timeZone": "Asia/Karachi"},
        "end": {"dateTime": (datetime.fromisoformat(start_iso) +
                             timedelta(minutes=duration_minutes)).isoformat(),
                "timeZone": "Asia/Karachi"},
    }
    created = service.events().insert(calendarId="primary", body=event).execute()
    return created["id"]


def update_event(event_id, new_start_iso, duration_minutes=30):
    if not is_live():
        _mock_update_event(event_id, new_start_iso, duration_minutes)
        return True
    service = _real_calendar_service()
    event = service.events().get(calendarId="primary", eventId=event_id).execute()
    event["start"] = {"dateTime": new_start_iso, "timeZone": "Asia/Karachi"}
    event["end"] = {"dateTime": (datetime.fromisoformat(new_start_iso) +
                                 timedelta(minutes=duration_minutes)).isoformat(),
                     "timeZone": "Asia/Karachi"}
    service.events().update(calendarId="primary", eventId=event_id, body=event).execute()
    return True


def delete_event(event_id):
    if not is_live():
        _mock_delete_event(event_id)
        return True
    service = _real_calendar_service()
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return True


if __name__ == "__main__":
    print("Live mode:", is_live())
    eid = create_event("Ahmed Raza", "0300-1234567", "Asad Mirza",
                        "House in DHA Defence, Lahore, PKR 15,000,000",
                        "2026-10-02T11:00:00", notes="Budget 3 crore, wants 3 bed")
    print("Created event:", eid)
    update_event(eid, "2026-10-03T15:00:00")
    print("Updated event start time.")
    delete_event(eid)
    print("Deleted event.")
    print(f"\nSee {LOG_PATH} for the full mock log.")
