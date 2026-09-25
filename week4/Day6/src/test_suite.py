"""
Day 6 - Task 1 & 3: Evaluation Suite + Performance Evaluation.
Runs 48 test conversations through the Day 5 LangGraph agent, measures
success/failure per category, and writes results to data/test_results.json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "day5", "src"))

from graph import build_graph
from state import new_state

DATA_DIR = os.path.join(HERE, "..", "data")
CONVOS_PATH = os.path.join(DATA_DIR, "test_conversations.json")
RESULTS_PATH = os.path.join(DATA_DIR, "test_results.json")


def run_conversation(graph, convo):
    """Run one multi-turn conversation, return summary."""
    state = new_state(client_phone=f"0300-{convo['id'][:7]}")
    turn_results = []
    t_start = time.time()

    for turn in convo["turns"]:
        state["current_user_text"] = turn
        try:
            state = graph.invoke(state)
            turn_results.append({
                "turn": turn,
                "reply": state["conversation_history"][-1]["text"][:200],
                "intent": state.get("intent"),
                "status": "ok",
            })
        except Exception as e:
            turn_results.append({
                "turn": turn, "status": "error", "error": str(e)
            })

    total_ms = round((time.time() - t_start) * 1000, 1)
    return {
        "conversation_id": convo["id"],
        "category": convo["category"],
        "turns_count": len(convo["turns"]),
        "turns": turn_results,
        "final_intent": state.get("intent"),
        "appointment_status": state.get("appointment_status"),
        "total_ms": total_ms,
        "avg_turn_ms": round(total_ms / len(convo["turns"]), 1),
    }


def main():
    with open(CONVOS_PATH, encoding="utf-8") as f:
        conversations = json.load(f)

    print(f"Loaded {len(conversations)} test conversations")
    print(f"Categories: {sorted(set(c['category'] for c in conversations))}\n")

    graph = build_graph()
    results = []
    t_all_start = time.time()

    for i, convo in enumerate(conversations, 1):
        print(f"[{i:>2}/{len(conversations)}] {convo['id']} ({convo['category']})...", end=" ")
        try:
            result = run_conversation(graph, convo)
            results.append(result)
            print(f"OK ({result['total_ms']}ms)")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({"conversation_id": convo["id"], "category": convo["category"],
                             "error": str(e), "status": "failed"})

    # aggregate per category
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, {"count": 0, "ok": 0, "errors": 0,
                                     "total_ms": 0, "max_turn_ms": 0})
        categories[cat]["count"] += 1
        if "error" in r or any(t.get("status") == "error" for t in r.get("turns", [])):
            categories[cat]["errors"] += 1
        else:
            categories[cat]["ok"] += 1
        categories[cat]["total_ms"] += r.get("total_ms", 0)
        categories[cat]["max_turn_ms"] = max(categories[cat]["max_turn_ms"],
                                              r.get("total_ms", 0) / max(r.get("turns_count", 1), 1))

    summary = {
        "total_conversations": len(results),
        "total_ok": sum(c["ok"] for c in categories.values()),
        "total_errors": sum(c["errors"] for c in categories.values()),
        "per_category": {
            cat: {
                "count": c["count"], "ok": c["ok"], "errors": c["errors"],
                "avg_total_ms": round(c["total_ms"] / max(c["count"], 1), 1),
                "max_turn_ms": round(c["max_turn_ms"], 1),
            } for cat, c in categories.items()
        },
        "wall_clock_ms": round((time.time() - t_all_start) * 1000, 1),
        "results": results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {summary['total_conversations']} conversations")
    print(f"  OK: {summary['total_ok']}  Errors: {summary['total_errors']}")
    print(f"  Success rate: {100 * summary['total_ok'] / summary['total_conversations']:.1f}%")
    print(f"\nPer category:")
    for cat, c in sorted(summary["per_category"].items()):
        rate = 100 * c["ok"] / c["count"]
        print(f"  {cat:<22} {c['ok']}/{c['count']} ({rate:.0f}%)  avg={c['avg_total_ms']}ms")

    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()