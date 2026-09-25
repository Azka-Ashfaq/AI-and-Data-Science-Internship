"""
Day 5 - full demo: runs several multi-turn conversations through the
compiled LangGraph, printing the conversation plus the annotated
execution trace for each -- the same conversations style as Day 3, now
routed through an actual LangGraph state machine with real tool calls
(Day 2 search/RAG, Day 4 calendar/email/CRM/appointments) instead of the
Day 3 hand-rolled pipeline.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import build_graph
from state import new_state
from state_logger import print_trace

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")


CONVERSATIONS = [
    {
        "id": "buyer_full_flow",
        "turns": ["Assalam o Alaikum, Lahore mein 3 bedroom ghar chahiye, budget 3 crore, DHA mein.",
                  "Book kar dein visit."],
    },
    {
        "id": "rental_reschedule_cancel",
        "turns": ["Karachi mein 2 bedroom flat rent par chahiye, budget 50000.",
                  "Book kar dein.", "Reschedule karna hai.", "Cancel kar dein."],
    },
    {
        "id": "vague_then_clarify",
        "turns": ["Mujhe ghar chahiye.", "Lahore mein."],
    },
    {
        "id": "impossible_budget",
        "turns": ["Islamabad mein ghar chahiye, budget 10 lakh."],
    },
    {
        "id": "faq_then_goodbye",
        "turns": ["Kya aap buyers se commission charge karte hain?", "Shukriya, Allah Hafiz."],
    },
]


def run_all():
    graph = build_graph()
    all_traces = []

    for convo in CONVERSATIONS:
        print(f"\n{'='*70}\n{convo['id']}\n{'='*70}")
        state = new_state(client_phone=f"0300-{convo['id'][:7]}")
        for t in convo["turns"]:
            state["current_user_text"] = t
            state = graph.invoke(state)
            print(f"Caller: {t}")
            print(f"Agent:  {state['conversation_history'][-1]['text']}\n")

        print_trace(state)
        all_traces.append({"conversation_id": convo["id"],
                            "final_intent": state["intent"],
                            "appointment_status": state["appointment_status"],
                            "trace": state["execution_trace"]})

    trace_path = os.path.join(DATA_DIR, "execution_traces.json")
    with open(trace_path, "w") as f:
        json.dump(all_traces, f, indent=2, default=str)
    print(f"\n\nAll execution traces written to {trace_path}")


if __name__ == "__main__":
    run_all()
