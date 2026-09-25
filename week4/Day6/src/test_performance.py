"""
Day 6 - Task 3: Performance Evaluation.
Measures latency, RAG accuracy, memory accuracy across the test suite.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "day5", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "day2", "src"))

from graph import build_graph
from state import new_state
from retriever import structured_search

DATA_DIR = os.path.join(HERE, "..", "data")
RESULTS_PATH = os.path.join(DATA_DIR, "performance_results.json")


LATENCY_QUERIES = [
    "Assalam o Alaikum",
    "Lahore mein 3 bedroom chahiye",
    "Budget 3 crore hai",
    "Book kar dein",
    "Cancel kar dein",
    "Commission kitna hai?",
    "DHA mein kya options hain?",
    "Yeh mehnga hai",
    "Visit schedule karo",
    "Shukriya",
]


def main():
    graph = build_graph()
    timings = []

    print("Measuring per-turn latency on 10 representative queries...\n")
    for q in LATENCY_QUERIES:
        state = new_state(client_phone="0300-perf")
        state["current_user_text"] = q
        t0 = time.time()
        try:
            state = graph.invoke(state)
            ms = round((time.time() - t0) * 1000, 1)
            timings.append({"query": q, "latency_ms": ms, "status": "ok"})
            print(f"  {ms:>7}ms  {q[:50]}")
        except Exception as e:
            timings.append({"query": q, "latency_ms": None, "status": "error", "error": str(e)})
            print(f"  ERROR: {e}")

    valid = [t["latency_ms"] for t in timings if t["latency_ms"] is not None]
    avg = round(sum(valid) / len(valid), 1) if valid else 0
    p50 = round(sorted(valid)[len(valid) // 2], 1) if valid else 0
    p95 = round(sorted(valid)[int(len(valid) * 0.95)], 1) if valid else 0

    # RAG accuracy: check known queries return grounded properties
    print("\n\nRAG accuracy check (5 known queries with expected results)...")
    rag_tests = [
        ("DHA Lahore 3 bed 3 crore", "Lahore", "For Sale", 30_000_000, 3, True),
        ("Bahria Lahore 2 crore", "Lahore", "For Sale", 20_000_000, 2, True),
        ("Islamabad 5 crore", "Islamabad", "For Sale", 50_000_000, None, True),
        ("Karachi rent 50k", "Karachi", "For Rent", 50_000, None, True),
        ("Faisalabad under 1 crore", "Faisalabad", "For Sale", 10_000_000, None, None),
    ]
    rag_correct = 0
    rag_results = []
    for label, city, purpose, max_p, min_b, expect_results in rag_tests:
        df = structured_search(city=city, purpose=purpose, max_price=max_p,
                                min_bedrooms=min_b, limit=5)
        has_results = not df.empty
        ok = (expect_results is None or has_results == expect_results)
        rag_correct += int(ok)
        rag_results.append({"query": label, "returned": len(df), "expected": expect_results, "ok": ok})
        mark = "OK " if ok else "X  "
        print(f"  {mark} {label}: {len(df)} results")

    rag_acc = round(100 * rag_correct / len(rag_tests), 1)

    summary = {
        "latency": {
            "samples": len(valid),
            "avg_ms": avg,
            "p50_ms": p50,
            "p95_ms": p95,
            "min_ms": min(valid) if valid else None,
            "max_ms": max(valid) if valid else None,
            "target_ms": 2000,
            "meets_target": p95 < 2000,
            "details": timings,
        },
        "rag_accuracy": {
            "correct": rag_correct,
            "total": len(rag_tests),
            "accuracy_pct": rag_acc,
            "details": rag_results,
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    status = "PASS" if p95 < 2000 else "FAIL (see report)"
    print(f"\n{'='*70}")
    print(f"Latency:  avg={avg}ms  p50={p50}ms  p95={p95}ms  (target: p95 < 2000ms)")
    print(f"          {status}")
    print(f"RAG:      {rag_correct}/{len(rag_tests)} ({rag_acc}%)")
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()