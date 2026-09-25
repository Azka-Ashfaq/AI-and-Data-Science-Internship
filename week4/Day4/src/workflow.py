"""
Day 4 - Task 4: Workflow Automation.
Call -> Intent -> Property Match -> Appointment -> Calendar -> Email -> CRM Update

This is the Python-native version of the n8n workflow (see n8n_workflow.json
for the visual/importable equivalent -- same steps, same failure handling).
Each external-facing step (Calendar, Email) is retried with backoff; a step
that ultimately fails does NOT roll back the appointment booking (the slot
is real and held), it's logged as a degraded outcome so a human can follow
up -- silently losing a real booking because a notification failed would
be worse than a booking whose calendar sync needs a manual nudge.
"""
import time
import uuid
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "day2", "src"))

import appointments
import calendar_integration
import email_integration
import crm

try:
    from retriever import structured_search
    DAY2_AVAILABLE = True
except Exception:
    DAY2_AVAILABLE = False


def with_retry(fn, max_attempts=3, backoff_seconds=0.3, step_name="step"):
    """Task 4's 'handle failures and retries' requirement. Returns
    (ok, result_or_error, attempts_used)."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            return True, result, attempt
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)  # linear backoff
    return False, str(last_error), max_attempts


def run_booking_workflow(client_name, client_phone, city, purpose, max_price,
                          min_bedrooms, scheduled_at, transcript=None):
    """The full 7-node flow. Returns a dict describing what happened at
    every stage, for logging/debugging/grading."""
    log = {"steps": []}
    call_id = f"call-{uuid.uuid4().hex[:8]}"

    # 1. Intent (already classified by the caller passing purpose/bedrooms in --
    #    Day 3's conversation_state.py is the real intent-detection layer;
    #    this workflow starts from its output)
    log["steps"].append({"step": "intent", "status": "ok",
                          "detail": {"purpose": purpose, "city": city}})

    # 2. Property Match (Day 2's structured search)
    if not DAY2_AVAILABLE:
        log["steps"].append({"step": "property_match", "status": "failed",
                              "detail": "Day 2 pipeline not importable"})
        return {**log, "outcome": "failed_no_property_data"}

    matches = structured_search(city=city, purpose=purpose, max_price=max_price,
                                 min_bedrooms=min_bedrooms, limit=1)
    if matches.empty:
        log["steps"].append({"step": "property_match", "status": "no_match"})
        crm.log_call(call_id, client_phone, client_name, transcript or [],
                      {"city": city, "purpose": purpose, "max_price": max_price,
                       "min_bedrooms": min_bedrooms}, outcome="no_match")
        return {**log, "outcome": "no_match", "call_id": call_id}

    prop = matches.iloc[0]
    property_summary = (f"{prop['property_type']} in {prop['location']}, {prop['city']}, "
                         f"PKR {prop['price']:,}, {prop['bedrooms']} bed")
    employee_name = prop["agent"] if prop["agent"] != "Unassigned" else "RealEstate Hub Direct"
    log["steps"].append({"step": "property_match", "status": "ok",
                          "detail": {"property_id": int(prop["property_id"]), "summary": property_summary}})

    # 3. Appointment (data layer -- the real availability check + booking)
    appt_id, booked_ok, booked_msg = appointments.book_appointment(
        property_id=int(prop["property_id"]), client_name=client_name,
        client_phone=client_phone, employee_name=employee_name,
        scheduled_at=scheduled_at, notes=f"Auto-booked via voice agent. {property_summary}",
    )
    log["steps"].append({"step": "appointment", "status": "ok" if booked_ok else "failed",
                          "detail": booked_msg})
    if not booked_ok:
        crm.log_call(call_id, client_phone, client_name, transcript or [],
                      {"city": city, "purpose": purpose}, outcome="booking_failed")
        return {**log, "outcome": "booking_failed", "call_id": call_id}

    # 4. Calendar (external call -- retried on failure)
    def _do_calendar():
        return calendar_integration.create_event(
            client_name, client_phone, employee_name, property_summary,
            scheduled_at, notes=f"Booked via voice agent, appointment_id={appt_id}")

    cal_ok, cal_result, cal_attempts = with_retry(_do_calendar, step_name="calendar")
    log["steps"].append({"step": "calendar", "status": "ok" if cal_ok else "failed_after_retries",
                          "attempts": cal_attempts, "detail": cal_result})
    if cal_ok:
        appointments.set_calendar_event_id(appt_id, cal_result)

    # 5. Email (external call -- retried on failure; does not block the booking)
    employee_email = f"{employee_name.lower().replace(' ', '.')}@realestatehub.example.pk" \
        if employee_name != "RealEstate Hub Direct" else None

    def _do_email():
        ok, msg = email_integration.notify_employee(
            employee_email, employee_name, client_name, client_phone,
            property_summary, scheduled_at, action="booked",
            notes=f"appointment_id={appt_id}")
        if not ok:
            raise RuntimeError(msg)
        return msg

    email_ok, email_result, email_attempts = with_retry(_do_email, step_name="email")
    log["steps"].append({"step": "email", "status": "ok" if email_ok else "failed_after_retries",
                          "attempts": email_attempts, "detail": email_result})

    # 6. CRM Update (always runs -- this is our own database, not an external
    #    dependency, so it's the one step that should never be allowed to fail
    #    silently; log it even if Calendar/Email degraded)
    crm.log_call(call_id, client_phone, client_name, transcript or [],
                 {"city": city, "purpose": purpose, "max_price": max_price,
                  "min_bedrooms": min_bedrooms, "appointment_id": appt_id},
                 outcome="booked")
    log["steps"].append({"step": "crm_update", "status": "ok"})

    degraded = not cal_ok or not email_ok
    return {**log, "outcome": "booked_degraded" if degraded else "booked",
            "call_id": call_id, "appointment_id": appt_id}


if __name__ == "__main__":
    result = run_booking_workflow(
        client_name="Ahmed Raza", client_phone="0300-1234567",
        city="Lahore", purpose="For Sale", max_price=30_000_000, min_bedrooms=3,
        scheduled_at="2026-10-02T11:00:00",
        transcript=[{"role": "caller", "text": "Budget 3 crore hai, DHA mein 3 bedroom chahiye."}],
    )
    import json
    print(json.dumps(result, indent=2, default=str))
