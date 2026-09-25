"""
Day 2 - Task 2: Answer generation (final RAG step).
Pluggable LLM call: reads PROVIDER + API key from environment so the exact
same retrieval pipeline works with OpenAI, Anthropic, Gemini, or Groq once
a key is available. Without a key, use --dry-run to see the exact assembled
prompt/context the LLM would receive (useful for grading/demo without
paying for API calls).
"""
import os
import argparse
import sys
from dotenv import load_dotenv
from retriever import structured_search, semantic_search

# Loads a .env file (if present) in the current working directory, or any
# parent directory -- so `GROQ_API_KEY=...` / `LLM_PROVIDER=groq` in
# day2/.env is picked up automatically without manual `export` each time.
load_dotenv()

SYSTEM_PROMPT = """You are Ayesha, a real estate voice assistant for RealEstate Hub in Pakistan.
Answer ONLY using the CONTEXT provided below. If the context doesn't contain
the answer, say you don't have that information rather than guessing. Keep
answers short (2-4 sentences), natural, and factual. Cite property_id when
recommending a specific listing.

IMPORTANT: The STRUCTURED MATCHES section already comes pre-filtered from our
database using the caller's budget, city, location, and bedroom requirements.
If any properties are listed there, they ARE valid matches -- recommend 2-3
of them with their property_id, location, price, and bedrooms. Do NOT say
"no listings" unless the STRUCTURED MATCHES section is literally empty.

Speak naturally in UrduLish (Urdu + English mixed), the way a Pakistani real
estate agent talks on the phone. Example: "Ji sir, DHA mein 3 bedroom ka
option hai, price 1.5 crore, 5 Marla, possession ready. Aap visit karna
chahenge?"
"""


def build_context(user_query, city=None, purpose=None, max_price=None,
                  min_bedrooms=None, location=None):
    struct_df = structured_search(city=city, purpose=purpose, max_price=max_price,
                                   min_bedrooms=min_bedrooms,
                                   location_contains=location, limit=5)
    struct_txt = "\n".join(
        f"- property_id={r.property_id}: {r.property_type} in {r.location}, {r.city}, "
        f"PKR {r.price:,}, {r.bedrooms} bed / {r.baths} bath, {r.area_raw}, agent {r.agent}"
        for r in struct_df.itertuples()
    ) or "(no properties matched the structured filters)"

    sem_hits = semantic_search(user_query, collection="property_docs", n_results=3)
    sem_txt = "\n".join(f"- {h['text']}" for h in sem_hits) or "(no semantic matches)"

    faq_hits = semantic_search(user_query, collection="faqs", n_results=2)
    faq_txt = "\n".join(f"- {h['text']}" for h in faq_hits) or "(no relevant FAQ)"

    return f"""STRUCTURED MATCHES (from SQL, exact facts):
{struct_txt}

SEMANTIC MATCHES (from vector search, descriptive):
{sem_txt}

RELEVANT FAQ:
{faq_txt}"""


def call_llm(user_query, context):
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}\n\nCALLER QUESTION: {user_query}\n\nAyesha's answer:"

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=300,
                                       messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(model="gpt-4o-mini",
                                               messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text
    elif provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    else:
        raise RuntimeError(
            "No LLM_PROVIDER set (expected 'openai', 'anthropic', 'gemini', or 'groq') "
            "or key missing. Use --dry-run to see the assembled prompt without calling an API."
        )


def answer(user_query, city=None, purpose=None, max_price=None,
           min_bedrooms=None, location=None, dry_run=False):
    context = build_context(user_query, city, purpose, max_price,
                            min_bedrooms, location)
    if dry_run:
        return f"[DRY RUN -- no LLM called]\n\n--- CONTEXT SENT TO LLM ---\n{context}\n\n--- QUESTION ---\n{user_query}"
    return call_llm(user_query, context)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--city")
    ap.add_argument("--purpose", default="For Sale")
    ap.add_argument("--max_price", type=int)
    ap.add_argument("--min_bedrooms", type=int)
    ap.add_argument("--location", help="e.g. DHA, Bahria, Johar Town")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(answer(args.query, city=args.city, purpose=args.purpose,
                 max_price=args.max_price, min_bedrooms=args.min_bedrooms,
                 location=args.location,
                 dry_run=args.dry_run or not os.environ.get("LLM_PROVIDER")))