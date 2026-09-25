"""
Day 3 - Task 3: Context Memory.
Tracks slots (budget, city, area, bedrooms, purpose) across turns of one
call, so the agent handles conversations like:
    "Budget 3 crore hai." -> "DHA mein kya options hain?" -> "Us se sasti koi option?"
without re-asking anything the caller already said.
"""
import re
import time

CITY_KEYWORDS = ["lahore", "karachi", "islamabad", "rawalpindi", "faisalabad"]
LOCATION_HINTS = ["dha", "bahria", "johar town", "gulberg", "askari", "f-10", "f-11",
                   "g-10", "g-11", "clifton", "gulshan", "north nazimabad"]


def _crore_lakh_to_pkr(text):
    """Parses amounts like '3 crore', '50 lakh', '3.5 crore' into PKR int."""
    m = re.search(r"([\d.]+)\s*crore", text, re.I)
    if m:
        return int(float(m.group(1)) * 10_000_000)
    m = re.search(r"([\d.]+)\s*lakh", text, re.I)
    if m:
        return int(float(m.group(1)) * 100_000)
    m = re.search(r"PKR?\s?([\d,]{6,})", text, re.I)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


class ConversationState:
    def __init__(self, caller_id=None):
        self.caller_id = caller_id
        self.slots = {
            "budget": None, "city": None, "location_hint": None,
            "bedrooms": None, "purpose": "For Sale", "intent": None,
        }
        self.history = []          # list of {"role", "text", "ts"}
        self.last_result_ids = []  # property_ids shown in the most recent turn
        self.started_at = time.time()

    def update_from_utterance(self, text):
        """Heuristic slot-filling -- lightweight regex/keyword extraction.
        In production this would be the LLM's structured-output tool call;
        kept rule-based here so Day 3 has zero extra LLM round-trips just
        for memory bookkeeping."""
        t = text.lower()

        budget = _crore_lakh_to_pkr(text)
        if budget:
            self.slots["budget"] = budget

        for c in CITY_KEYWORDS:
            if c in t:
                self.slots["city"] = c.title()
                break

        for loc in LOCATION_HINTS:
            if loc in t:
                self.slots["location_hint"] = loc.upper() if loc in ("dha",) else loc.title()
                break

        m = re.search(r"(\d+)\s*(bed|bedroom)", t)
        if m:
            self.slots["bedrooms"] = int(m.group(1))

        if any(w in t for w in ["rent", "kiraya", "rental"]):
            self.slots["purpose"] = "For Rent"
        elif any(w in t for w in ["buy", "sale", "khareed", "purchase"]):
            self.slots["purpose"] = "For Sale"

        if any(w in t for w in ["invest", "roi", "appreciation"]):
            self.slots["intent"] = "investment"
        elif any(w in t for w in ["commercial", "shop", "office", "plaza"]):
            self.slots["intent"] = "commercial"
        elif self.slots["purpose"] == "For Rent":
            self.slots["intent"] = "rental"
        elif not self.slots["intent"]:
            self.slots["intent"] = "buyer"

        # "us se sasti" / "cheaper than that" -> lower the budget relative to
        # what's already known, regardless of whether results were shown yet
        # in this standalone test (voice_pipeline.py sets last_result_ids in
        # the real flow, giving an even sharper anchor)
        if any(p in t for p in ["sasti", "cheaper", "kam mein", "less than that"]):
            if self.slots["budget"]:
                self.slots["budget"] = int(self.slots["budget"] * 0.85)

    def last_budget_seen(self):
        return self.slots["budget"]

    def log_turn(self, role, text):
        self.history.append({"role": role, "text": text, "ts": time.time()})

    def summary(self):
        s = self.slots
        parts = []
        if s["city"]:
            parts.append(s["city"])
        if s["location_hint"]:
            parts.append(s["location_hint"])
        if s["budget"]:
            parts.append(f"budget PKR {s['budget']:,}")
        if s["bedrooms"]:
            parts.append(f"{s['bedrooms']} bed")
        parts.append(s["purpose"])
        return ", ".join(parts) if parts else "(no slots filled yet)"


if __name__ == "__main__":
    cs = ConversationState()
    for u in ["Budget 3 crore hai.", "DHA mein kya options hain?", "Us se sasti koi option?"]:
        cs.update_from_utterance(u)
        print(f"> {u}\n  slots: {cs.summary()}\n")
