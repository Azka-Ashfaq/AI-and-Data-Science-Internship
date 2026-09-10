# Week 2, Day 4 — CrewAI: Multi-Agent Collaboration, Roles & Task Delegation

## Business task

Analyze our laptop product lineup, benchmark it against a competitor's pricing, and produce a stakeholder-ready recommendation for which laptop to feature this quarter.

A 3-agent CrewAI crew handles this end to end:

| Agent | Role | Data source | Tools |
|---|---|---|---|
| **Product Data Analyst** | Extract price-tier and spec insights from our own catalog | `products.json` (internal) | `get_all_products`, `get_product_price` |
| **Competitive Market Researcher** | Benchmark our lineup against the competitor's pricing | `competitor_products.json` (external) | `get_competitor_products` |
| **Stakeholder Report Writer** | Synthesize both agents' findings into one concise, decision-ready recommendation | Context from the two tasks above (no raw data access) | `count_words` |

Each agent is deliberately scoped to only the tool(s) its role needs, so no single agent blends internal and competitor data.

## What's in this repo

- **`crewAI_multi_agent.py`** — the full script: agent/tool definitions, sequential crew (Task 3), hierarchical crew with a manager agent (Task 4), and token-usage/cost evaluation across 3 runs (Task 5).
- **`Week2_Day4_CrewAI_Writeup.docx`** — write-up covering all 5 tasks: design rationale, tool justification, the sequential-vs-hierarchical comparison, and cost/quality evaluation, backed by actual execution logs.

`products.json` and `competitor_products.json` are **not** committed — the script generates both fresh on every run (see lines ~84–101), so they don't need to be tracked.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install "crewai[google-genai]" crewai-tools python-dotenv
```

Create a `.env` file in the project root (never commit this file):

```
GEMINI_API_KEY=your_key_here
```

> `crewai[google-genai]` is required for CrewAI's native Gemini provider — without it, `model="gemini/..."` raises an `ImportError`.

## Running

```bash
python crewAI_multi_agent.py
```

The script runs, in order:
1. **Sequential crew** (`Process.sequential`) — analyst → researcher → writer, run 1.
2. **Hierarchical crew** (`Process.hierarchical`) — a manager agent delegates each task and reviews the output.
3. **Sequential crew, run 2** — for a 3-run cost/quality comparison.

Each stage prints full CrewAI execution logs plus a token-usage summary. There's a ~65s pause between stages to stay under the Gemini free tier's 5 requests/minute limit (`max_rpm=4` on every agent).

**Runtime note:** a full run takes several minutes due to the built-in rate-limit pauses.

## Key findings (from actual execution logs)

| Metric | Sequential (run 1) | Hierarchical | Sequential (run 2) |
|---|---|---|---|
| Total tokens | 3,126 | 14,609 | 8,037 |
| Successful requests | 6 | 18 | 14 |
| Final recommendation | Laptop B | **Laptop C** | Laptop B |

- Hierarchical delegation genuinely changed the outcome (not just the cost) — the manager's re-framing led the Writer to a different pick than either sequential run.
- Hierarchical cost ~4.7× more tokens and 3× more requests than a single sequential run, without a clearly better result.
- **Anomaly**: sequential run 2 unexpectedly executed through the Crew Manager (`delegate_work_to_coworker`) despite being configured with `Process.sequential` — likely because the Analyst/Researcher/Writer `Agent` objects are shared between the sequential and hierarchical `Crew` instances, and hierarchical wiring leaves delegation state on those shared agents. A cleaner design would use separate `Agent` instances per crew.

Full task-by-task write-up, the pros/cons table for sequential vs. hierarchical, and the scored evaluation criteria are in `Week2_Day4_CrewAI_Writeup.docx`.

## Deliverables checklist

- [x] Task 1 — Multi-agent design rationale (3 non-overlapping roles)
- [x] Task 2 — Agents built with role-appropriate tools and individual LLM configs
- [x] Task 3 — Sequential crew with `Task` context dependencies; format-mismatch fix documented
- [x] Task 4 — Hierarchical crew with manager delegation; sequential vs. hierarchical comparison table
- [x] Task 5 — Token usage/cost logged across 3 runs; 3 success criteria defined and manually scored

Author:

Azka-Ashfaq
AI-Data Science Intern
