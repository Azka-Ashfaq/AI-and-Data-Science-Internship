"""
router_test.py — Tests router accuracy on 20 labeled queries (Task 2).
"""

from nodes_router import classify_query

TEST_SET = [
    ("who will win the Pies vs Cats this week", "prediction"),
    ("will the Pies beat the Cats", "prediction"),
    ("who will top-score for Carlton", "prediction"),
    ("predict the winner of Richmond vs Essendon", "prediction"),
    ("who is going to win between Sydney and Geelong", "prediction"),
    ("who will be the top player for Brisbane", "prediction"),
    ("what were Dustin Martin's 2017 stats", "retrieval"),
    ("how many disposals did Crippa get last round", "retrieval"),
    ("head to head between Richmond and Carlton", "retrieval"),
    ("career average for Patrick Cripps", "retrieval"),
    ("how did Dusty play last game", "retrieval"),
    ("Richmond vs Carlton record", "retrieval"),
    ("how many teams are in the AFL", "factual"),
    ("what is a behind in AFL", "factual"),
    ("explain the rules of AFL", "factual"),
    ("how does the AFL season work", "factual"),
    ("what's the weather in Sydney", "off_topic"),
    ("what's the stock market doing today", "off_topic"),
    ("who is the president of the USA", "off_topic"),
    ("how do I cook pasta", "off_topic"),
]


def run_accuracy_test():
    correct = 0
    print(f"{'#':<3} {'Query':<50} {'Expected':<12} {'Predicted':<12} {'OK'}")
    print("-" * 95)
    for i, (query, expected) in enumerate(TEST_SET, 1):
        result = classify_query(query)
        ok = result.intent == expected
        correct += int(ok)
        print(
            f"{i:<3} {query[:48]:<50} {expected:<12} "
            f"{result.intent:<12} {'✅' if ok else '❌'}"
        )
    print("-" * 95)
    acc = correct / len(TEST_SET) * 100
    print(f"Accuracy: {correct}/{len(TEST_SET)} = {acc:.1f}%")


if __name__ == "__main__":
    run_accuracy_test()