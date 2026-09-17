"""
e2e_tests.py — 10+ full conversation tests with state trace logging (Task 5).
"""

import json
from graph_app import app


CONVERSATIONS = [
    "who will win the Pies vs Cats this week",
    "who will top-score for Carlton",
    "what were Dustin Martin's 2017 stats",
    "how many disposals did Crippa get last round",
    "career average for Patrick Cripps",
    "head to head between Richmond and Carlton",
    "how many teams are in the AFL",
    "what's the weather in Sydney",
    "will Smith play well this week",
    "predict total tackles for Geelong",
    "who will win",  # ambiguous: no teams
]


def log_trace(query: str):
    print("=" * 80)
    print(f"USER: {query}")
    state = {
        "user_query": query,
        "conversation_history": [],
        "messages": [],
    }
    final = app.invoke(state)
    print(f"[Router] intent={final.get('detected_intent')} "
          f"confidence={final.get('intent_confidence')}")
    if final.get("resolved_entities"):
        print(f"[Resolve] {final['resolved_entities']}")
    print(f"[Tool] {final.get('tool_called')}")
    print(f"[Validate] {final.get('validation_status')}")
    if final.get("feature_drivers"):
        print(f"[Drivers] {len(final['feature_drivers'])} features")
    print(f"[Response] {final.get('final_response')}")
    print()
    return final


def run_all():
    traces = []
    for q in CONVERSATIONS:
        final = log_trace(q)
        traces.append({
            "query": q,
            "intent": final.get("detected_intent"),
            "tool": final.get("tool_called"),
            "validation": final.get("validation_status"),
            "response": final.get("final_response"),
        })
    with open("e2e_traces.json", "w") as f:
        json.dump(traces, f, indent=2, default=str)
    print(f"Wrote {len(traces)} traces to e2e_traces.json")


if __name__ == "__main__":
    run_all()