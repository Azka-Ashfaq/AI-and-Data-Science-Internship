"""
graph_app.py — Full AFL LangGraph orchestration (Day 4 deliverable).

Builds and compiles the StateGraph connecting router -> prediction/retrieval/
factual/refusal -> validation -> formatter/clarify/fallback.
"""

from langgraph.graph import StateGraph, START, END

from state import AFLGraphState
from nodes_router import router_node, route_from_router
from nodes_prediction import prediction_node
from nodes_retrieval import retrieval_node
from nodes_validation import (
    validate_node,
    route_from_validation,
    clarify_node,
    fallback_node,
    refusal_node,
    direct_answer_node,
)
from nodes_formatter import formatter_node


def build_graph():
    g = StateGraph(AFLGraphState)

    # Nodes
    g.add_node("router_node", router_node)
    g.add_node("prediction_node", prediction_node)
    g.add_node("retrieval_node", retrieval_node)
    g.add_node("direct_answer_node", direct_answer_node)
    g.add_node("refusal_node", refusal_node)
    g.add_node("validate_node", validate_node)
    g.add_node("formatter_node", formatter_node)
    g.add_node("clarify_node", clarify_node)
    g.add_node("fallback_node", fallback_node)

    # Entry
    g.add_edge(START, "router_node")

    # Router branching
    g.add_conditional_edges(
        "router_node",
        route_from_router,
        {
            "prediction_node": "prediction_node",
            "retrieval_node": "retrieval_node",
            "direct_answer_node": "direct_answer_node",
            "refusal_node": "refusal_node",
        },
    )

    # Tool nodes -> validation
    g.add_edge("prediction_node", "validate_node")
    g.add_edge("retrieval_node", "validate_node")

    # Direct answer and refusal are terminal
    g.add_edge("direct_answer_node", END)
    g.add_edge("refusal_node", END)

    # Validation branching
    g.add_conditional_edges(
        "validate_node",
        route_from_validation,
        {
            "formatter_node": "formatter_node",
            "clarify_node": "clarify_node",
            "fallback_node": "fallback_node",
        },
    )

    g.add_edge("formatter_node", END)
    g.add_edge("clarify_node", END)
    g.add_edge("fallback_node", END)

    return g.compile()


# Compiled app
app = build_graph()


def run_conversation(query: str, history: list = None) -> dict:
    """Run one turn through the graph and return the final state."""
    state = {
        "user_query": query,
        "conversation_history": history or [],
        "messages": [],
    }
    final_state = app.invoke(state)
    return final_state


if __name__ == "__main__":
    import json
    test_queries = [
        "who will win the Pies vs Cats this week",
        "who will top-score for Carlton",
        "what were Dustin Martin's 2017 stats",
        "how many disposals did Crippa get last round",
        "head to head between Richmond and Carlton",
        "how many teams are in the AFL",
        "what's the weather in Sydney",
        "will Smith play well this week",
    ]
    for q in test_queries:
        print("=" * 70)
        print(f"Q: {q}")
        result = run_conversation(q)
        print(f"Intent: {result.get('detected_intent')}")
        print(f"Tool: {result.get('tool_called')}")
        print(f"Validation: {result.get('validation_status')}")
        print(f"A: {result.get('final_response')}")