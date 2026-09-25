"""
Day 5 - Task 1: LangGraph State Design.

The state that flows through every node of the graph. Built as a
TypedDict (LangGraph's standard state shape) rather than a plain class,
so LangGraph can merge partial updates from each node automatically.

Covers every field the brief lists:
  conversation history, user profile, property preferences, budget,
  intent, tool outputs, appointment status.
"""
from typing import TypedDict, Optional, Any
import operator
from typing import Annotated


class AgentState(TypedDict):
    # --- conversation history ---
    # Annotated with operator.add so LangGraph APPENDS new turns instead of
    # overwriting the whole history on every node's return value.
    conversation_history: Annotated[list[dict], operator.add]

    # --- user profile (persists across calls once CRM lookup runs) ---
    client_name: Optional[str]
    client_phone: Optional[str]
    is_returning_customer: bool

    # --- property preferences / budget (Day 3's slot-filling, reused) ---
    city: Optional[str]
    location_hint: Optional[str]
    budget: Optional[int]
    bedrooms: Optional[int]
    purpose: str  # "For Sale" | "For Rent"

    # --- intent routing ---
    intent: Optional[str]  # "buyer" | "rental" | "investment" | "commercial" |
                            # "reschedule" | "cancel" | "returning" | "goodbye"
    current_user_text: str

    # --- tool outputs (each node writes here, later nodes read it) ---
    tool_outputs: dict[str, Any]

    # --- appointment status ---
    appointment_id: Optional[str]
    appointment_status: Optional[str]  # "none" | "booked" | "rescheduled" | "cancelled"

    # --- control flow / logging (Task 5) ---
    current_node: str
    execution_trace: Annotated[list[dict], operator.add]
    needs_clarification: bool
    clarification_reason: Optional[str]


def new_state(user_text="", client_phone=None) -> AgentState:
    """Fresh state for a new call."""
    return AgentState(
        conversation_history=[],
        client_name=None, client_phone=client_phone, is_returning_customer=False,
        city=None, location_hint=None, budget=None, bedrooms=None, purpose="For Sale",
        intent=None, current_user_text=user_text,
        tool_outputs={},
        appointment_id=None, appointment_status="none",
        current_node="start", execution_trace=[],
        needs_clarification=False, clarification_reason=None,
    )
