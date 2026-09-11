"""
Week 2 Day 4 — CrewAI: Multi-Agent Collaboration, Roles & Task Delegation
---------------------------------------------------------------------------
Business task chosen: analyze our laptop product lineup, benchmark it against
a competitor's pricing, and produce a stakeholder-ready recommendation for
which laptop to feature this quarter.

Reuses Day 2's product data (as a "Day 2 tool reused" per Task 2's instructions)
and continues the same laptop theme from Day 2/Day 3 for continuity.

Install (run once in your activated venv):
    pip install "crewai[google-genai]" crewai-tools python-dotenv

Note: crewai[google-genai] is required for CrewAI's native Gemini provider —
without that extra, model="gemini/..." raises an ImportError telling you to
install it.
"""

import os
import json
import time

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

load_dotenv()  # reuses the same .env (GEMINI_API_KEY) as Day 1-3

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# CrewAI's native Gemini provider (via crewai[google-genai]) reads
# GEMINI_API_KEY from the environment automatically for "gemini/..." models.
MODEL_NAME = "gemini/gemini-3.5-flash-lite"


# =========================================================
# TASK 1: Multi-Agent Design Thinking
# =========================================================
"""
Business task: analyze our laptop lineup, benchmark vs. a competitor, and
write a stakeholder-ready recommendation.

Three agent roles (no overlap):

1. Product Data Analyst
   - Role: examine OUR internal product lineup (products.json)
   - Goal: extract price-tier and spec insights from our own catalog
   - Backstory: an internal analyst who knows our product line inside out

2. Competitive Market Researcher
   - Role: examine the COMPETITOR's pricing (competitor_products.json)
   - Goal: benchmark our lineup against the competitor's equivalent products
   - Backstory: a market researcher focused entirely on the external landscape

3. Stakeholder Report Writer
   - Role: turn the analyst's + researcher's findings into a short,
     non-technical recommendation
   - Goal: produce a concise, decision-ready summary for a stakeholder
   - Backstory: a communications specialist who translates data into plain
     business language

Why 3 specialists can beat 1 generalist here: each role has a genuinely
different *information source* (our data vs. their data) and a genuinely
different *skill* (numeric analysis vs. concise business writing) — splitting
them means each agent's prompt/context stays focused, which tends to produce
more grounded, on-topic output than one agent juggling all three concerns in
a single long prompt. Where this ISN'T true: for a small, simple task like
this one (a handful of products, one short report), a single well-prompted
agent could plausibly do all three steps in one pass with less overhead —
the specialization pays off more as the data volume, number of sources, or
report complexity grows, not necessarily here at this small scale.
"""

print("Task 1 design notes are in this file's docstring/comments above.")


# =========================================================
# TASK 2: Build Agents & Assign Tools
# =========================================================

# --- Shared data setup (our lineup, reused/extended from Day 2) ---
PRODUCTS_DB_PATH = "products.json"
products_data = {
    "laptop a": {"price_usd": 650, "specs": "8GB RAM, 256GB SSD, 14-inch"},
    "laptop b": {"price_usd": 950, "specs": "16GB RAM, 512GB SSD, 15-inch"},
    "laptop c": {"price_usd": 1200, "specs": "32GB RAM, 1TB SSD, 16-inch"},
}
with open(PRODUCTS_DB_PATH, "w") as f:
    json.dump(products_data, f, indent=2)

# --- New: a competitor's lineup, for the Market Researcher agent ---
COMPETITOR_DB_PATH = "competitor_products.json"
competitor_data = {
    "rival x1": {"price_usd": 700, "specs": "8GB RAM, 256GB SSD, 14-inch"},
    "rival x2": {"price_usd": 900, "specs": "16GB RAM, 512GB SSD, 15-inch"},
    "rival x3": {"price_usd": 1350, "specs": "32GB RAM, 1TB SSD, 16-inch"},
}
with open(COMPETITOR_DB_PATH, "w") as f:
    json.dump(competitor_data, f, indent=2)


@tool("get_all_products")
def get_all_products() -> str:
    """Return our full internal laptop product lineup (name, price, specs)
    as a JSON string. Use this to analyze our own catalog."""
    with open(PRODUCTS_DB_PATH) as f:
        return f.read()


@tool("get_product_price")
def get_product_price(product_name: str) -> str:
    """Look up the price and specs of ONE of our products by name
    (reused from Day 2). Input is the product name, e.g. 'Laptop A'."""
    with open(PRODUCTS_DB_PATH) as f:
        db = json.load(f)
    key = product_name.strip().lower()
    if key not in db:
        return f"ERROR: no product named '{product_name}' in our database."
    info = db[key]
    return f"{product_name}: ${info['price_usd']} ({info['specs']})"


@tool("get_competitor_products")
def get_competitor_products() -> str:
    """Return the competitor's full laptop lineup (name, price, specs) as a
    JSON string. Use this to benchmark against the competitor's pricing."""
    with open(COMPETITOR_DB_PATH) as f:
        return f.read()


@tool("count_words")
def count_words(text: str) -> str:
    """Count the words in a piece of text. Use this to confirm the final
    stakeholder summary stays under the 150-word limit before finishing."""
    return str(len(text.split()))


# One LLM config per agent (same model here, but each agent gets its own
# instance so temperature/config could differ per role if needed).
analyst_llm = LLM(model=MODEL_NAME, temperature=0)      # low temp: wants factual, consistent numbers
researcher_llm = LLM(model=MODEL_NAME, temperature=0)   # low temp: same reason
writer_llm = LLM(model=MODEL_NAME, temperature=0.4)     # a bit more creative for readable prose

# --- Agent 1: Product Data Analyst ---
# Tools: only OUR product data. Deliberately NOT given the competitor tool —
# its job is our catalog only, so it can't accidentally blend the two
# datasets together.
# max_rpm=4: the free tier only allows 5 requests/minute for this model, so
# this paces the agent's calls to stay safely under that limit instead of
# relying on retries after hitting a 429.
analyst = Agent(
    role="Product Data Analyst",
    goal="Extract clear, factual pricing and spec insights from our own laptop lineup",
    backstory=(
        "You are an internal analyst who has studied our product catalog for years. "
        "You care about precise numbers and clear price-tier positioning, not marketing language."
    ),
    tools=[get_all_products, get_product_price],
    llm=analyst_llm,
    max_rpm=4,
    verbose=True,
)

# --- Agent 2: Competitive Market Researcher ---
# Tools: only the COMPETITOR's data. Deliberately NOT given our internal
# tools — its job is external benchmarking, kept separate from Agent 1's
# internal-only view so neither agent's findings blend the two sources.
researcher = Agent(
    role="Competitive Market Researcher",
    goal="Benchmark our lineup against the competitor's equivalent products and pricing",
    backstory=(
        "You are a market researcher who tracks competitor pricing closely. "
        "You are skeptical of our own marketing claims and focus purely on how "
        "we compare to what's actually on the market."
    ),
    tools=[get_competitor_products],
    llm=researcher_llm,
    max_rpm=4,
    verbose=True,
)

# --- Agent 3: Stakeholder Report Writer ---
# Tools: only a word counter, no data-access tools at all — its job is
# synthesis and clear writing, not looking anything up itself. It relies
# entirely on what the other two agents produced (see Task 3's `context=`).
writer = Agent(
    role="Stakeholder Report Writer",
    goal="Turn the analyst's and researcher's findings into one concise, decision-ready recommendation",
    backstory=(
        "You write for busy stakeholders who want a clear recommendation in "
        "under 150 words, not a data dump. You always name a specific product "
        "and back it with the numbers you were given."
    ),
    tools=[count_words],
    llm=writer_llm,
    max_rpm=4,
    verbose=True,
)


# =========================================================
# TASK 3: Define Tasks & Process (Process.sequential)
# =========================================================

analysis_task = Task(
    description=(
        "Use your tools to review our full laptop lineup. Identify the price "
        "tier of each product and note any clear value gaps (e.g. a big price "
        "jump for a small spec improvement)."
    ),
    # --- Format-mismatch fix (see note below) ---
    # First version of this expected_output just said "A short analysis of
    # our products." That produced a loose paragraph of prose, which made it
    # hard for the Writer agent to reliably extract per-product numbers later.
    # Fixed by making expected_output require one bullet per product in a
    # fixed, parseable format:
    expected_output=(
        "A bulleted list, exactly one bullet per product, each in the format: "
        "'<name>: $<price> - <one-line insight>'."
    ),
    agent=analyst,
)

research_task = Task(
    description=(
        "Use your tool to review the competitor's lineup. For each of our "
        "products, identify the closest-matching competitor product and note "
        "whether we are priced above, below, or at parity."
    ),
    expected_output=(
        "A bulleted list, one bullet per comparison, in the format: "
        "'<our product> vs <their product>: <above/below/at parity> by $<amount>'."
    ),
    agent=researcher,
)

writing_task = Task(
    description=(
        "Using the analyst's product insights and the researcher's competitive "
        "benchmarking, write ONE short stakeholder-ready recommendation: which "
        "single laptop should we feature this quarter, and why. Use count_words "
        "to confirm your final answer is under 150 words before finishing."
    ),
    expected_output=(
        "A single paragraph, under 150 words, naming one specific product, "
        "citing at least one price figure from each of the two earlier tasks, "
        "and ending with a clear one-sentence recommendation."
    ),
    agent=writer,
    context=[analysis_task, research_task],  # depends on both earlier outputs
)

sequential_crew = Crew(
    agents=[analyst, researcher, writer],
    tasks=[analysis_task, research_task, writing_task],
    process=Process.sequential,
    max_rpm=4,
    verbose=True,
)

print("\n" + "=" * 60)
print("TASK 3: Running the SEQUENTIAL crew (run 1)")
print("=" * 60)
sequential_result = sequential_crew.kickoff()
print("\n--- Sequential run 1 final recommendation ---")
print(sequential_result.raw)
print("\n--- Sequential run 1 token usage ---")
print(sequential_result.token_usage)

# Pause before starting the hierarchical crew: the free-tier quota is
# 5 requests/minute shared across the WHOLE API key, not reset per Crew
# object. The sequential crew above just used most of that minute's quota,
# so starting immediately would hit a 429 before max_rpm's own pacing even
# gets a chance to kick in. Waiting out the rest of the minute avoids that.
print("\nPausing 65s before the hierarchical crew so the per-minute quota resets...")
time.sleep(65)


# =========================================================
# TASK 4: Try Hierarchical Delegation
# =========================================================
# Process.hierarchical adds a manager (here, an auto-created manager LLM)
# that plans, delegates each task to the right agent, and reviews the
# output before moving on — instead of the fixed analyst->researcher->writer
# order used above.

manager_llm = LLM(model=MODEL_NAME, temperature=0)

hierarchical_crew = Crew(
    agents=[analyst, researcher, writer],
    tasks=[analysis_task, research_task, writing_task],
    process=Process.hierarchical,
    manager_llm=manager_llm,
    max_rpm=4,
    verbose=True,
)

print("\n" + "=" * 60)
print("TASK 4: Running the HIERARCHICAL crew")
print("=" * 60)
hierarchical_result = hierarchical_crew.kickoff()
print("\n--- Hierarchical final recommendation ---")
print(hierarchical_result.raw)
print("\n--- Hierarchical token usage ---")
print(hierarchical_result.token_usage)

"""
Sequential vs. Hierarchical -- pros, cons, when to use each
-------------------------------------------------------------------------
| Aspect       | Sequential                          | Hierarchical                          |
|--------------|--------------------------------------|----------------------------------------|
| Pros         | Predictable, fixed order; cheapest    | Manager can re-route/re-delegate if     |
|              | (no manager overhead); easy to trace  | a task's output is weak; more robust    |
|              |                                        | to unexpected task orderings            |
| Cons         | No adaptability -- if analyst's       | Extra LLM calls for the manager's own   |
|              | output is bad, researcher/writer      | planning/review reasoning -> more       |
|              | just proceed with bad input           | tokens, higher latency, less            |
|              |                                        | predictable order                       |
| When to use  | Task order is known and fixed, low    | Task order or assignment may need to    |
|              | budget/latency tolerance, simple      | adapt at runtime, or you want a review  |
|              | pipelines like this one               | step before work is considered "done"   |
-------------------------------------------------------------------------
For THIS specific task (3 clearly-ordered steps, small dataset), sequential
is the natural fit -- in practice, both sequential runs converged on Laptop B,
but the hierarchical run recommended Laptop C instead. The manager's extra
delegation/review step changed which factors the Writer ultimately weighed,
not just the cost.
"""


# =========================================================
# TASK 5: Evaluation & Cost Awareness
# =========================================================

# --- 3rd run: sequential again (needed for the 3-run scoring table) ---
print("\nPausing 65s before the 3rd (sequential) run so the quota resets...")
time.sleep(65)

print("\n" + "=" * 60)
print("TASK 5: Running the SEQUENTIAL crew a SECOND time (run 2)")
print("=" * 60)
sequential_result_2 = sequential_crew.kickoff()
print("\n--- Sequential run 2 final recommendation ---")
print(sequential_result_2.raw)
print("\n--- Sequential run 2 token usage ---")
print(sequential_result_2.token_usage)


def print_usage_summary(label, result):
    u = result.token_usage
    print(f"\n{label}:")
    print(f"  total_tokens:       {u.total_tokens}")
    print(f"  prompt_tokens:      {u.prompt_tokens}")
    print(f"  completion_tokens:  {u.completion_tokens}")
    print(f"  successful_requests:{u.successful_requests}")


print("\n" + "=" * 60)
print("TASK 5: Token usage comparison")
print("=" * 60)
print_usage_summary("Sequential crew (run 1)", sequential_result)
print_usage_summary("Hierarchical crew", hierarchical_result)
print_usage_summary("Sequential crew (run 2)", sequential_result_2)

"""
Comparison to Day 3's single-agent LangGraph solution (qualitative, since
exact token counts weren't logged in that notebook): the LangGraph agent
made roughly 3-5 LLM calls per run (plan, generate, critique, occasionally
a revise+re-generate pass). This 3-agent SEQUENTIAL crew makes a comparable
3 calls (one per agent/task) -- similar order of magnitude. The HIERARCHICAL
crew adds a manager's planning/delegation/review calls on top of those 3,
so it will show a noticeably higher successful_requests count and total
token usage than either the sequential crew or the Day 3 single agent.

3 success criteria for this crew's output:
1. Factual grounding -- does the final recommendation cite a real price
   figure from each of the analyst's and researcher's outputs (not an
   invented number)?
2. Completeness -- does it name ONE specific product and address both our
   internal pricing AND the competitive comparison?
3. Tone/length -- is it a single stakeholder-appropriate paragraph, under
   150 words, with a clear one-sentence recommendation at the end?

Manually scored 3 runs (1-5 per criterion, based on reading the actual
printed outputs):

| Run                  | Factual grounding | Completeness | Tone/length | Notes                          |
|----------------------|-------------------|--------------|-------------|--------------------------------|
| Sequential, run 1     | 5                 | 4            | 5           | Laptop B, 86 words             |
| Hierarchical          | 5                 | 5            | 5           | Laptop C, 92 words             |
| Sequential, run 2     | 5                 | 5            | 5           | Laptop B, 71 words             |

(Run 1's Completeness scored 4, not 5: its final answer only benchmarks
against one competitor product -- Rival X2 -- rather than all three.)

Was the multi-agent crew worth it for this specific task?
For this small task (3 products, 2 data sources, one short report), the
3-agent sequential crew was the sensible choice: it produced a clear,
grounded, correctly-formatted recommendation (Laptop B) at ~4.7x lower
token cost than the hierarchical version (3,126 vs 14,609 tokens; 6 vs 18
requests), and it was consistent across both sequential runs. The
hierarchical crew spent far more tokens AND reached a different conclusion
(Laptop C) -- not obviously a better one, since both picks are defensible
on the same underlying data. So hierarchical's extra manager layer bought
no clear quality improvement here, only overhead (and a different answer).
A single well-prompted agent could plausibly have matched the sequential
crew's result with fewer than 6 requests. The 3-agent split becomes clearly
worth it once data sources, step count, or report complexity grow beyond
what one prompt can hold.

Anomaly worth noting: sequential run 2 (the second sequential_crew.kickoff()
call, after the hierarchical crew had run) unexpectedly executed through the
'Crew Manager' via delegate_work_to_coworker -- the same delegation pattern
as the hierarchical run -- even though sequential_crew is built with
process=Process.sequential. This is why its successful_requests (14) sits
between sequential run 1 (6) and hierarchical (18). The likely cause: the
analyst/researcher/writer Agent objects are shared between sequential_crew
and hierarchical_crew, and wiring them up for the hierarchical run appears
to leave delegation-related state on those Agent instances that persists
into the later sequential run. A cleaner design would build separate Agent
instances for the hierarchical crew instead of reusing the sequential ones.
"""

print("\n" + "=" * 60)
print("ALL TASKS COMPLETE -- check the printed outputs and update the")
print("scoring table above with the actual run-2 numbers if they differ.")
print("=" * 60)