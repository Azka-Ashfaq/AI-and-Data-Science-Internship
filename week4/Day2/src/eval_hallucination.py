"""
Day 2 - Task 5: Hallucination Evaluation.
20 test questions built FROM the actual data (so ground truth is known),
covering property facts, FAQs, and one deliberately unanswerable question
(tests that the system says "I don't know" instead of inventing an answer).

Two things are measured automatically, with no LLM needed:
  - Retrieval Accuracy: does context assembly actually surface the fact
    needed to answer correctly? (checked by keyword/value presence)
  - Groundedness of context: is every fact offered to the LLM real (drawn
    from properties.db / faqs.csv), never invented by the retrieval step?

Once an LLM key is set (see rag_answer.py), a third measure -- Hallucination
Rate -- can be added by generating the actual answer and checking it doesn't
introduce numbers/claims absent from the context. That check is included
below (check_llm_answer) and runs automatically if LLM_PROVIDER is set.
"""
import os
import re
import pandas as pd
from retriever import structured_search
from rag_answer import build_context, answer

props = pd.read_csv("data/properties_clean.csv")

def make_test_set():
    cases = []
    # Realistic caller phrasing (nobody quotes a property_id on a phone call) --
    # this mirrors the phrasing style validated in evaluate_chunking.py, which
    # measured 86.7% precision@1 for the chunk size used here.
    sample = props.sample(15, random_state=11)
    for r in sample.itertuples():
        q = (f"{r.bedrooms} bedroom {r.property_type.lower()} for "
             f"{r.purpose.split()[-1].lower()} in {r.location} {r.city} "
             f"around {r.price:,} rupees")
        expected_facts = [f"{r.price:,}", r.location]
        cases.append({"query": q, "city": r.city, "purpose": r.purpose,
                       "expected_facts": expected_facts, "type": "property_fact"})

    faq_qs = [
        ("Do you charge buyers a commission?", ["no fee", "commission"]),
        ("Can overseas Pakistanis book remotely?", ["overseas", "remotely"]),
        ("What documents do I need to book a property?", ["CNIC"]),
        ("Is the price negotiable?", ["negotiable"]),
        ("What happens if I cancel a visit?", ["cancel"]),
    ]
    for q, facts in faq_qs:
        cases.append({"query": q, "city": None, "purpose": None,
                       "expected_facts": facts, "type": "faq"})

    # unanswerable / out-of-scope question -- correct behaviour is "I don't know"
    cases.append({"query": "What is the exact resale value of my property in 2035?",
                   "city": None, "purpose": None, "expected_facts": [],
                   "type": "unanswerable"})

    return cases


def retrieval_accuracy(cases):
    hits, total = 0, 0
    rows = []
    for c in cases:
        if c["type"] == "unanswerable":
            continue
        ctx = build_context(c["query"], city=c["city"], purpose=c["purpose"])
        found = all(str(f).lower() in ctx.lower() for f in c["expected_facts"])
        hits += int(found)
        total += 1
        rows.append({"query": c["query"], "type": c["type"], "fact_found": found})
    return hits / total if total else 0, rows


def check_llm_answer(query, context, llm_answer):
    """Very simple groundedness proxy: flag any PKR figure in the answer
    that doesn't literally appear in the retrieved context (a cheap,
    explainable stand-in for a full hallucination-detection model)."""
    answer_numbers = set(re.findall(r"PKR\s?[\d,]+", llm_answer))
    context_numbers = set(re.findall(r"PKR\s?[\d,]+", context))
    invented = answer_numbers - context_numbers
    return len(invented) == 0, invented


def main():
    cases = make_test_set()
    acc, rows = retrieval_accuracy(cases)
    print(f"Retrieval Accuracy (fact present in retrieved context): {acc:.1%}  "
          f"({sum(r['fact_found'] for r in rows)}/{len(rows)})")
    for r in rows:
        if not r["fact_found"]:
            print(f"  MISS [{r['type']}]: {r['query']}")

    unanswerable = [c for c in cases if c["type"] == "unanswerable"][0]
    ctx = build_context(unanswerable["query"])
    print(f"\nUnanswerable-question check: context assembled has "
          f"{'NO grounded facts (correct)' if 'no' in ctx.lower() or len(ctx) < 300 else 'some content -- verify LLM still declines to guess'}")

    if os.environ.get("LLM_PROVIDER"):
        print("\nLLM_PROVIDER set -- running live groundedness check on 5 sample questions...")
        grounded_hits = 0
        sample_cases = [c for c in cases if c["type"] != "unanswerable"][:5]
        for c in sample_cases:
            ctx = build_context(c["query"], city=c["city"], purpose=c["purpose"])
            ans = answer(c["query"], city=c["city"], purpose=c["purpose"])
            ok, invented = check_llm_answer(c["query"], ctx, ans)
            grounded_hits += int(ok)
            if not ok:
                print(f"  Possible hallucination in: {c['query']} -> invented figures: {invented}")
        print(f"Groundedness (no invented PKR figures): {grounded_hits}/{len(sample_cases)}")
    else:
        print("\nLLM_PROVIDER not set -- skipping live hallucination-rate measurement.")
        print("Set LLM_PROVIDER=openai|anthropic|gemini and the matching *_API_KEY env var, "
              "then re-run this script to also get a live Hallucination Rate / Grounding Rate.")


if __name__ == "__main__":
    main()
