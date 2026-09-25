"""
Day 3 - Task 4: Objection Handling.
Detects the objection category in a caller's turn and returns the matching
empathy-first rebuttal template from the Day 1 persona, optionally grounded
with a real fact pulled from the Day 2 retrieval layer (developer name,
comparable price) when one is available.
"""
import re

OBJECTION_KEYWORDS = {
    "price": ["expensive", "mehnga", "zyada", "budget se zyada", "too much", "costly"],
    "trust": ["trust", "fraud", "dhoka", "reliable", "scam", "fake"],
    "location": ["door", "far", "dor", "location theek nahi", "developing area", "undeveloped"],
    "investment": ["appreciation", "resale", "roi", "future value", "investment safe"],
    "builder": ["builder", "developer", "construction quality", "handover"],
    "maintenance": ["maintenance", "upkeep", "society fee", "monthly charges"],
}

REBUTTALS = {
    "price": ("Main samajh sakti hoon budget important hai — is se milta julta ek "
              "option hai jo thora kam mein aa jayega, ya hum installment plan bhi dekh sakte hain."),
    "trust": ("Bilkul samajh sakti hoon yeh concern — {agency} humara verified listing "
              "partner hai aur humne pehle bhi is society mein successful bookings ki hain. "
              "Main aap ko unki detail bhi bhej sakti hoon."),
    "location": ("Ye area thora developing hai lekin infrastructure tezi se ban raha hai — "
                 "agle 2 saal mein achi appreciation expected hai, aur schools/hospitals "
                 "nearby already available hain."),
    "investment": ("Historical trend dekhein toh is society mein steady appreciation raha hai. "
                   "Main aap ko comparable resale prices bhi share kar sakti hoon taake aap "
                   "khud decide kar sakein."),
    "builder": ("{developer} ka track record achha hai — pehle bhi projects on-time handover "
                "hue hain. Main aap ko unki past projects ki list bhej deti hoon."),
    "maintenance": ("Society-level maintenance ek monthly fee se cover hoti hai jo bohot "
                     "reasonable hai — main aap ko exact figure confirm kar deti hoon."),
}


def detect_objection(user_text):
    t = user_text.lower()
    for category, keywords in OBJECTION_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return category
    return None


def handle_objection(user_text, property_context=None):
    """property_context: optional dict with 'agency', 'developer' pulled
    from the SQL/payment_plans tables (Day 2) to make the rebuttal specific
    instead of generic."""
    category = detect_objection(user_text)
    if not category:
        return None, None

    template = REBUTTALS[category]
    ctx = property_context or {}
    text = template.format(
        agency=ctx.get("agency", "RealEstate Hub"),
        developer=ctx.get("developer", "the developer"),
    )
    return category, text


if __name__ == "__main__":
    tests = [
        "Yeh property thori expensive lag rahi hai humare budget se.",
        "Mujhe is developer par trust nahi hai, pehle fraud suna hai.",
        "Yeh area thora door hai city se, kya yahan future mein value badhegi?",
        "Maintenance kitni hogi is society mein?",
        "Theek hai, visit book kar dein.",  # no objection
    ]
    for t in tests:
        cat, resp = handle_objection(t, {"agency": "Lions Home Design & Build", "developer": "Bahria Town Pvt. Ltd."})
        print(f"Caller: {t}")
        print(f"  Detected: {cat}")
        if resp:
            print(f"  Rebuttal: {resp}")
        print()
