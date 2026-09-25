"""
Day 4 - Task 3: Appointment Management (booking / rescheduling /
cancellation), demonstrated end-to-end with Calendar + Email kept in sync
at every step, using the same workflow.py components rather than calling
appointments.py in isolation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appointments
import calendar_integration
import email_integration


def book_with_notifications(property_id, client_name, client_phone, employee_name,
                             property_summary, scheduled_at, employee_email=None):
    appt_id, ok, msg = appointments.book_appointment(
        property_id, client_name, client_phone, employee_name, scheduled_at)
    if not ok:
        print(f"BOOK FAILED: {msg}")
        return None
    event_id = calendar_integration.create_event(
        client_name, client_phone, employee_name, property_summary, scheduled_at)
    appointments.set_calendar_event_id(appt_id, event_id)
    email_integration.notify_employee(employee_email, employee_name, client_name,
                                       client_phone, property_summary, scheduled_at, action="booked")
    print(f"BOOKED: appointment_id={appt_id}, calendar_event={event_id}")
    return appt_id


def reschedule_with_notifications(appt_id, new_scheduled_at):
    appt = appointments.get_appointment(appt_id)
    ok, msg = appointments.reschedule_appointment(appt_id, new_scheduled_at)
    if not ok:
        print(f"RESCHEDULE FAILED: {msg}")
        return False
    calendar_integration.update_event(appt["calendar_event_id"], new_scheduled_at)
    email_integration.notify_employee(
        None, appt["employee_name"], appt["client_name"], appt["client_phone"],
        f"property_id={appt['property_id']}", new_scheduled_at, action="rescheduled")
    print(f"RESCHEDULED: {appt_id} -> {new_scheduled_at}")
    return True


def cancel_with_notifications(appt_id, reason=""):
    appt = appointments.get_appointment(appt_id)
    ok, msg = appointments.cancel_appointment(appt_id, reason)
    if not ok:
        print(f"CANCEL FAILED: {msg}")
        return False
    if appt["calendar_event_id"]:
        calendar_integration.delete_event(appt["calendar_event_id"])
    email_integration.notify_employee(
        None, appt["employee_name"], appt["client_name"], appt["client_phone"],
        f"property_id={appt['property_id']}", appt["scheduled_at"], action="cancelled", notes=reason)
    print(f"CANCELLED: {appt_id} ({reason})")
    return True


if __name__ == "__main__":
    print("--- Booking ---")
    appt_id = book_with_notifications(
        property_id=14436786, client_name="Bilal Ahmed", client_phone="0333-1112222",
        employee_name="Zubair Javed", property_summary="House in DHA Defence, Lahore, PKR 30,000,000",
        scheduled_at="2026-10-06T10:00:00")

    print("\n--- Rescheduling ---")
    reschedule_with_notifications(appt_id, "2026-10-07T16:00:00")

    print("\n--- Cancelling ---")
    cancel_with_notifications(appt_id, reason="Client found another property")

    print("\n--- Final state ---")
    print(appointments.get_appointment(appt_id))
