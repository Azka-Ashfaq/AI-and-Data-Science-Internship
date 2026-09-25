"""
Day 5 - Task 2 (nodes) + Task 4 (validation, built into booking/reschedule/
recommendation) + Task 3 (each node calls a tool from tools.py).
"""
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from state_logger import logged_node
import tools

CITY_KEYWORDS = ["lahore", "karachi", "islamabad", "rawalpindi", "faisalabad"]
LOCATION_HINTS = ["dha", "bahria", "johar town", "gulberg", "askari", "f-10", "f-11",
                   "g-10", "g-11", "clifton", "gulshan", "north nazimabad"]


def _crore_lakh_to_pkr(text):
    m = re.search(r"([\d.]+)\s*crore", text, re.I)
    if m:
        return int(float(m.group(1)) * 10_000_000)
    m = re.search(r"([\d.]+)\s*lakh", text, re.I)
    if m:
        return int(float(m.group(1)) * 100_000)
    return None


# ---------------------------------------------------------------
# GREETING
# ---------------------------------------------------------------
@logged_node("greeting")
def greeting_node(state):
    if state["conversation_history"]:
        return {}  # only greet on the very first turn
    return {"conversation_history": [{"role": "agent",
             "text": "Assalam-o-Alaikum! RealEstate Hub se baat ho rahi hai, main Ayesha. Aap ki kis tarah madad kar sakti hoon?"}]}


# ---------------------------------------------------------------
# INTENT DETECTION (also does Day 3-style slot-filling)
# ---------------------------------------------------------------
@logged_node("intent_detection")
def intent_detection_node(state):
    text = state["current_user_text"]
    t = text.lower()
    update = {"conversation_history": [{"role": "caller", "text": text}]}

    budget = _crore_lakh_to_pkr(text)
    if budget:
        update["budget"] = budget
    if any(p in t for p in ["sasti", "cheaper", "kam mein", "less than that"]) and state.get("budget"):
        update["budget"] = int(state["budget"] * 0.85)

    for c in CITY_KEYWORDS:
        if c in t:
            update["city"] = c.title()
            break
    for loc in LOCATION_HINTS:
        if loc in t:
            update["location_hint"] = loc.upper() if loc == "dha" else loc.title()
            break
    m = re.search(r"(\d+)\s*(bed|bedroom)", t)
    if m:
        update["bedrooms"] = int(m.group(1))

    # --- intent classification ---
    if any(w in t for w in ["reschedule", "date change", "waqt change"]):
        intent = "reschedule"
    elif any(w in t for w in ["cancel", "cancell"]):
        intent = "cancel"
    elif any(w in t for w in ["book kar", "book kr", "book dein", "book karo",
                               "book kardo", "book kardein", "booking", "schedule kar",
                               "visit book", "visit schedule", "confirm kar", "book it", "yes book"]):
        intent = "booking"
    elif any(w in t for w in ["bye", "shukriya", "thank you", "khuda hafiz", "allah hafiz"]):
        intent = "goodbye"
    elif any(w in t for w in ["invest", "roi", "appreciation"]):
        intent = "investment"
    elif any(w in t for w in ["commercial", "shop", "office", "plaza"]):
        intent = "commercial"
    elif any(w in t for w in ["rent", "kiraya", "rental"]):
        update["purpose"] = "For Rent"
        intent = "rental"
    elif any(w in t for w in ["fee", "commission", "document", "negotiable", "possession", "maintenance"]):
        intent = "faq"
    elif any(w in t for w in ["khareed", "buy", "purchase", "ghar", "flat", "house", "property", "bedroom", "crore", "lakh"]):
        intent = "buyer"
    else:
        intent = "unclear"
    update["intent"] = intent

    # --- validation (Task 4): ask clarification instead of guessing ---
    if intent in ("buyer", "rental", "investment", "commercial") and not (state.get("city") or update.get("city")):
        update["needs_clarification"] = True
        update["clarification_reason"] = "city"
    else:
        update["needs_clarification"] = False
        update["clarification_reason"] = None

    return update


def route_after_intent(state):
    if state["needs_clarification"]:
        return "clarification"
    intent = state["intent"]
    return {
        "buyer": "recommendation", "rental": "recommendation",
        "investment": "recommendation", "commercial": "recommendation",
        "booking": "booking", "reschedule": "rescheduling", "cancel": "cancellation",
        "faq": "rag", "goodbye": "goodbye",
    }.get(intent, "clarification")


# ---------------------------------------------------------------
# CLARIFICATION
# ---------------------------------------------------------------
@logged_node("clarification")
def clarification_node(state):
    reason = state["clarification_reason"]
    if reason == "city":
        msg = "Ji, kaunse city mein property dekh rahe hain aap -- Lahore, Karachi, ya Islamabad?"
    elif reason == "no_property_selected":
        msg = "Ji, pehle main kuch options dikhati hoon, phir aap batayein kaunsa book karna hai."
    else:
        msg = "Sorry, thora clear kar dein aap kya dhoond rahe hain?"
    return {"conversation_history": [{"role": "agent", "text": msg}]}


# ---------------------------------------------------------------
# RECOMMENDATION
# ---------------------------------------------------------------
@logged_node("recommendation")
def recommendation_node(state):
    results = tools.search_property(
        city=state.get("city"), purpose=state.get("purpose", "For Sale"),
        max_price=state.get("budget"), min_bedrooms=state.get("bedrooms"),
        location=state.get("location_hint"), limit=3,
    )
    tool_outputs = dict(state.get("tool_outputs", {}))
    tool_outputs["last_search_results"] = results

    if not results:
        msg = "Sorry sir, is criteria mein abhi koi property available nahi hai -- budget ya area thora adjust karna chahenge?"
    else:
        lines = [f"property_id {r['property_id']}: {r['location']}, PKR {r['price']:,}, "
                 f"{r['bedrooms']} bed" for r in results]
        msg = "Ji sir, yeh options hain:\n" + "\n".join(lines)

    return {"conversation_history": [{"role": "agent", "text": msg}], "tool_outputs": tool_outputs}


# ---------------------------------------------------------------
# RAG (FAQ)
# ---------------------------------------------------------------
@logged_node("rag")
def rag_node(state):
    faq_hits = tools.semantic_search(state["current_user_text"], collection="faqs", n_results=1)
    if faq_hits:
        raw = faq_hits[0]["text"]
        text = raw.split("A: ", 1)[-1] if "A: " in raw else raw
    else:
        text = "Is bare mein exact detail abhi available nahi, main confirm kar ke batati hoon."
    return {"conversation_history": [{"role": "agent", "text": text}]}


# ---------------------------------------------------------------
# BOOKING
# ---------------------------------------------------------------
@logged_node("booking")
def booking_node(state):
    results = state.get("tool_outputs", {}).get("last_search_results", [])

    if not results:
        results = tools.search_property(
            city=state.get("city"),
            purpose=state.get("purpose", "For Sale"),
            max_price=state.get("budget"),
            min_bedrooms=state.get("bedrooms"),
            location=state.get("location_hint"),
            limit=3,
        )
        tool_outputs = dict(state.get("tool_outputs", {}))
        tool_outputs["last_search_results"] = results
    else:
        tool_outputs = state.get("tool_outputs", {})

    if not results:
        return {
            "conversation_history": [{"role": "agent", "text": "Pehle property select karein please."}],
            "tool_outputs": tool_outputs,
        }

    prop = results[0]
    employee = prop["agent"] if prop["agent"] and prop["agent"] != "Unassigned" else "RealEstate Hub Direct"

    scheduled_at = (datetime.now() + timedelta(days=7)).replace(microsecond=0).isoformat()

    is_named_agent = employee != "RealEstate Hub Direct"
    if is_named_agent and not tools.availability_checker(employee, scheduled_at):
        return {
            "conversation_history": [{"role": "agent",
             "text": f"Sorry, {employee} us waqt available nahi hain -- doosra time try karein?"}],
            "appointment_status": "none",
            "tool_outputs": tool_outputs,
        }

    result = tools.book_appointment_tool(
        property_id=prop["property_id"],
        client_name=state.get("client_name") or "Caller",
        client_phone=state.get("client_phone") or "unknown",
        employee_name=employee,
        scheduled_at=scheduled_at,
        notes=f"Booked via LangGraph agent, property_id={prop['property_id']}")

    if not result["ok"]:
        return {
            "conversation_history": [{"role": "agent", "text": f"Booking mein masla aaya: {result['message']}"}],
            "appointment_status": "none",
            "tool_outputs": tool_outputs,
        }

    msg = f"Zaroor! Aapka visit {scheduled_at} ke liye book ho gaya hai, appointment ID {result['appointment_id']}."
    return {
        "conversation_history": [{"role": "agent", "text": msg}],
        "appointment_id": result["appointment_id"],
        "appointment_status": "booked",
        "tool_outputs": tool_outputs,
    }


# ---------------------------------------------------------------
# RESCHEDULING
# ---------------------------------------------------------------
@logged_node("rescheduling")
def rescheduling_node(state):
    appt_id = state.get("appointment_id")
    if not appt_id:
        return {"conversation_history": [{"role": "agent", "text": "Mujhe pehle apni booking ID ya detail batayein please."}]}
    new_time = (datetime.now() + timedelta(days=9)).replace(microsecond=0).isoformat()
    result = tools.reschedule_appointment_tool(appt_id, new_time)
    msg = f"Aapki visit {new_time} par reschedule kar di hai." if result["ok"] else f"Reschedule nahi ho saka: {result['message']}"
    return {"conversation_history": [{"role": "agent", "text": msg}],
            "appointment_status": "rescheduled" if result["ok"] else state.get("appointment_status")}


# ---------------------------------------------------------------
# CANCELLATION
# ---------------------------------------------------------------
@logged_node("cancellation")
def cancellation_node(state):
    appt_id = state.get("appointment_id")
    if not appt_id:
        return {"conversation_history": [{"role": "agent", "text": "Mujhe pehle apni booking ID ya detail batayein please."}]}
    result = tools.cancel_appointment_tool(appt_id, reason="caller requested cancellation")
    msg = "Aapki visit cancel kar di hai." if result["ok"] else f"Cancel nahi ho saka: {result['message']}"
    return {"conversation_history": [{"role": "agent", "text": msg}],
            "appointment_status": "cancelled" if result["ok"] else state.get("appointment_status")}


# ---------------------------------------------------------------
# GOODBYE
# ---------------------------------------------------------------
@logged_node("goodbye")
def goodbye_node(state):
    msg = "Shukriya RealEstate Hub se rabta karne ke liye, Allah Hafiz!"
    return {"conversation_history": [{"role": "agent", "text": msg}]}