"""
Day 5 - Task 2: Graph Design.
Wires the nodes (nodes.py) into an actual LangGraph StateGraph, with
conditional routing driven by intent (Task 4's validation lives inside
the node functions themselves -- the graph's job is just to route).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langgraph.graph import StateGraph, END
from state import AgentState
import nodes


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("greeting", nodes.greeting_node)
    g.add_node("intent_detection", nodes.intent_detection_node)
    g.add_node("clarification", nodes.clarification_node)
    g.add_node("recommendation", nodes.recommendation_node)
    g.add_node("rag", nodes.rag_node)
    g.add_node("booking", nodes.booking_node)
    g.add_node("rescheduling", nodes.rescheduling_node)
    g.add_node("cancellation", nodes.cancellation_node)
    g.add_node("goodbye", nodes.goodbye_node)

    g.set_entry_point("greeting")
    g.add_edge("greeting", "intent_detection")

    g.add_conditional_edges("intent_detection", nodes.route_after_intent, {
        "clarification": "clarification", "recommendation": "recommendation",
        "rag": "rag", "booking": "booking", "rescheduling": "rescheduling",
        "cancellation": "cancellation", "goodbye": "goodbye",
    })

    # Every terminal node ends the turn -- the graph is invoked once per
    # caller turn (see run_demo.py), with state carried forward turn to turn.
    for terminal in ["clarification", "recommendation", "rag", "booking",
                      "rescheduling", "cancellation", "goodbye"]:
        g.add_edge(terminal, END)

    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    print("Graph compiled successfully.")
    print("Nodes:", list(graph.get_graph().nodes.keys()))
