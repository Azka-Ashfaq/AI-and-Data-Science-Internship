"""
Day 3 - Task 2: Natural Speech Behaviors.
Wraps a plain LLM answer with the UrduLish filler/hesitation/acknowledgement
phrases designed in Week 4 Day 1, so spoken output sounds like a person
thinking and responding in real time, not a chatbot reading text.
"""
import random

random.seed()  # keep natural run-to-run variety; deterministic seeding would
                # make every call sound identical, which is itself unnatural

THINKING_PAUSES = [
    "Ek second sir, main abhi check karti hoon...",
    "Hmm... acha, dekhte hain...",
    "Thori dair dijiye, main latest availability nikal rahi hoon...",
]

ACKNOWLEDGEMENTS = [
    "Acha, samajh gayi.",
    "Ji ji, bilkul theek keh rahe hain aap.",
    "Ji bilkul.",
    "Theek hai, noted.",
]

FILLERS_MIDSENTENCE = ["you know", "basically", "matlab", "waisay"]

LAUGHTER_LIGHT = ["haha", "hehe"]


def needs_lookup(user_text):
    """Heuristic: does this turn require a retrieval/tool call (and so
    should get a 'thinking pause' before the answer, the way a human agent
    checking a system would)?"""
    triggers = ["option", "available", "price", "kitna", "kitni", "book",
                "schedule", "visit", "reschedule", "cancel", "kya hai"]
    t = user_text.lower()
    return any(trig in t for trig in triggers)


def wants_acknowledgement(user_text):
    """Heuristic: did the caller just state a fact/preference (budget,
    city, bedrooms) that deserves a short ack before moving on?"""
    triggers = ["budget", "crore", "lakh", "bedroom", "chahiye", "hai"]
    t = user_text.lower()
    return any(trig in t for trig in triggers)


def add_natural_behaviors(user_text, llm_answer, turn_index=0):
    """Compose the final spoken-style output: optional thinking pause before
    a lookup, optional short acknowledgement, then the LLM's answer.
    Returns (spoken_text, behavior_tags) -- behavior_tags is for the eval
    harness / transcript log, not spoken aloud."""
    parts = []
    tags = []

    if turn_index > 0 and wants_acknowledgement(user_text) and random.random() < 0.7:
        ack = random.choice(ACKNOWLEDGEMENTS)
        parts.append(ack)
        tags.append("acknowledgement")

    if needs_lookup(user_text):
        pause = random.choice(THINKING_PAUSES)
        parts.append(pause)
        tags.append("thinking_pause")

    parts.append(llm_answer.strip())
    spoken = " ".join(parts)
    return spoken, tags


if __name__ == "__main__":
    tests = [
        (0, "Budget 3 crore hai, DHA mein kya options hain?", "Ji sir, DHA mein 3 options hain..."),
        (1, "Us se sasti koi option?", "Ji, yeh option 2.5 crore mein hai..."),
        (2, "Theek hai, visit book kar dein.", "Zaroor, main aapka visit Saturday ke liye book kar deti hoon."),
    ]
    for idx, user_text, answer in tests:
        spoken, tags = add_natural_behaviors(user_text, answer, idx)
        print(f"Caller: {user_text}")
        print(f"Agent:  {spoken}   [{', '.join(tags) or 'no behaviors'}]\n")
