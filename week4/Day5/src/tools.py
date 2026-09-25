"""
Day 5 - Task 3: Tool Integration.
Wraps the already-built, already-tested Day 2 (search/RAG) and Day 4
(calendar/email/CRM/availability) components as plain callables the graph
nodes invoke. Kept as simple functions rather than @tool-decorated LLM
function-calling tools -- the brief's own routing (Task 2's graph) decides
which tool runs, so the LLM doesn't need to pick tools itself here (that
model is available too, see README "Swapping in LLM-driven tool calls").
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "day2", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "day4", "src"))

from retriever import structured_search, semantic_search
import appointments
import calendar_integration
import email_integration
import crm


def search_property(city=None, purpose="For Sale", max_price=None, min_bedrooms=None,
                     location=None, limit=3):
    """Task 3 tool: Search Property (Day 2's structured search)."""
    df = structured_search(city=city, purpose=purpose, max_price=max_price,
                            min_bedrooms=min_bedrooms, location_contains=location, limit=limit)
    return df.to_dict("records")


def rag_search(query, n_results=3):
    """Task 3 tool: RAG Search (Day 2's semantic search, for fuzzy questions
    like amenities/FAQs that don't map to a single SQL column)."""
    return semantic_search(query, collection="property_docs", n_results=n_results)


def availability_checker(employee_name, scheduled_at, duration_minutes=30):
    """Task 3 tool: Availability Checker (Day 4). Used BEFORE booking --
    Task 4's 'never book unavailable slots' rule starts here."""
    return appointments.check_availability(employee_name, scheduled_at, duration_minutes)


def book_appointment_tool(property_id, client_name, client_phone, employee_name,
                           scheduled_at, notes=""):
    """Task 3 tool: Calendar (booking half). Refuses an unavailable slot at
    the data layer -- this IS the Task 4 validation, not a separate check
    the graph has to remember to run."""
    appt_id, ok, msg = appointments.book_appointment(
        property_id, client_name, client_phone, employee_name, scheduled_at, notes=notes)
    if not ok:
        return {"ok": False, "message": msg}
    return {"ok": True, "appointment_id": appt_id, "message": msg}


def reschedule_appointment_tool(appointment_id, new_scheduled_at):
    ok, msg = appointments.reschedule_appointment(appointment_id, new_scheduled_at)
    return {"ok": ok, "message": msg}


def cancel_appointment_tool(appointment_id, reason=""):
    ok, msg = appointments.cancel_appointment(appointment_id, reason)
    return {"ok": ok, "message": msg}


def calendar_tool(action, **kwargs):
    """Task 3 tool: Calendar (create/update/delete, Day 4's mock-by-default wrapper)."""
    if action == "create":
        return calendar_integration.create_event(**kwargs)
    elif action == "update":
        return calendar_integration.update_event(**kwargs)
    elif action == "delete":
        return calendar_integration.delete_event(**kwargs)
    raise ValueError(f"Unknown calendar action: {action}")


def email_tool(**kwargs):
    """Task 3 tool: Email (Day 4's employee-notification wrapper)."""
    ok, msg = email_integration.notify_employee(**kwargs)
    return {"ok": ok, "message": msg}


def crm_tool(action, **kwargs):
    """Task 3 tool: CRM (Day 4's call-logging / history / follow-up wrapper)."""
    if action == "log_call":
        return crm.log_call(**kwargs)
    elif action == "get_history":
        return crm.get_client_history(**kwargs)
    elif action == "add_follow_up":
        return crm.add_follow_up(**kwargs)
    raise ValueError(f"Unknown crm action: {action}")


if __name__ == "__main__":
    print("search_property:", search_property(city="Lahore", purpose="For Sale",
                                               max_price=30_000_000, min_bedrooms=3, limit=2))
    print("\navailability_checker (free slot):",
          availability_checker("Test Agent", "2026-11-01T10:00:00"))
