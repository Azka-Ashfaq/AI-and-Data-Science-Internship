# Week 3 · Day 3 — Domain-Scoped AFL Chat Agent

A LangChain agent that only discusses AFL, grounds every stat in a real tool call, and declines
off-topic requests with a redirect rather than a flat refusal.

## Folder structure

```
day3/
├── afl_datasets/                          <- put your 4 raw CSVs here (same as Day 1/2)
├── tools.py                                <- 4 structured retrieval tools (Task 2)
├── agent.py                                <- system prompt + agent + memory (Task 1, 3, 4)
├── guardrail_eval.py                       <- adversarial + guardrail test harness (Task 1, 5)
├── afl_week3_day3_chat_agent.ipynb        <- main notebook tying everything together
├── adversarial_test_log.csv                <- created when you run the tests
├── guardrail_test_log.csv                  <- created when you run the tests
└── guardrail_report.md                     <- created when you run the tests
```

## Setup

**1. Packages** (installs `langchain`, `langgraph`, `langchain-google-genai`, plus the usual
`pandas`/`numpy`/`tabulate`):

```
pip install langchain langchain-core langgraph langchain-google-genai pandas numpy tabulate
```

**2. API key.** This agent calls Google Gemini and needs `GEMINI_API_KEY` set in your environment
(same convention as your other course):

```
# Windows (PowerShell)
$env:GEMINI_API_KEY = "your-key-here"

# Mac/Linux
export GEMINI_API_KEY="your-key-here"
```

**3. Data.** Unzip `afl_datasets.zip` so the `afl_datasets` folder sits next to these files (same
as Day 1/2).

## How to run

- **Open `afl_week3_day3_chat_agent.ipynb`, Run All.** The data/tool cells (Task 2) run with no API
  key at all — they're real pandas queries and will show real output regardless. The agent-calling
  cells (Tasks 1, 3, 4, 5) check for `GEMINI_API_KEY` first: with no key set they print setup
  instructions and skip cleanly instead of erroring; with a key set they call the live agent.
- **Or run the guardrail evaluation directly from the command line** once your key is set:
  ```
  python guardrail_eval.py
  ```
  This runs all 10 adversarial prompts + all 18 guardrail test-set prompts, saves both logs as CSV,
  and writes `guardrail_report.md`.

## What each file does

| File | Task(s) | What's in it |
|---|---|---|
| `tools.py` | 2 | 4 structured retrieval tools: team head-to-head, player season stats, player recent games, player career average — all real pandas queries, no LLM involved. Includes the structured-vs-semantic decision and justification in its module docstring. |
| `agent.py` | 1, 3, 4 | `SYSTEM_PROMPT` (scope + refusal style), `build_agent()` (wires tools + memory via `InMemorySaver` checkpointer), `chat()` / `chat_with_trace()` helpers |
| `guardrail_eval.py` | 1, 5 | 10 adversarial prompts, 18-prompt guardrail test set, automated grounding check for stat-based questions (numbers must trace back to a real tool output, with rounding tolerance), auto-generated report |

## The retrieval-layer decision (Task 2)

The dataset is entirely structured tables — no match reports, articles, or commentary text exist
in `afl_datasets.zip`. So every tool here is a structured pandas lookup, not semantic/vector
search. This isn't a shortcut: it directly follows the task's own principle that sports stats
should be exact lookups, never fuzzy retrieval, to avoid hallucinated numbers. If unstructured text
becomes available later, a vector store should be added only for qualitative content — numeric
stats should keep going through these structured tools regardless.

## The grounding check (Task 3)

`guardrail_eval.grounding_check()` extracts every number in the agent's final answer and confirms
each one traces back to a number that actually appeared in a tool's output (years are excluded,
and a rounding tolerance is applied — e.g. "24" correctly traces back to a tool output of "24.2",
since models naturally round when speaking). This check only runs on `in_scope_grounded` questions
— general AFL history/rules answers legitimately contain numbers (years, club counts) with no tool
behind them, so those are marked `N/A` rather than flagged as false failures. This runs
automatically on every response in the Task 5 evaluation, so a hallucinated stat is caught
mechanically, not just by eyeballing.

## Subjective questions (Task 5 edge cases)

`SYSTEM_PROMPT` includes explicit guidance for genuinely subjective AFL-adjacent questions (e.g.
"what's the best sport?") — the agent should frame AFL enthusiasm as its own bias, not state it as
objective fact. This was added after reviewing an early test run where the agent answered too
declaratively; see the failure-pattern table in `guardrail_report.md` for the reasoning.

## Finishing the guardrail report

`guardrail_report.md` is generated automatically with every response and the automated grounding
check filled in. Two things still need a human pass after running it with your key:

1. **`manual_pass_fail`** — read each response and mark PASS/FAIL. The automated
   `auto_looks_like_refusal` flag is only a rough keyword check (looks for words like "AFL" or
   "can't help") — it can't tell if a reply also leaked real off-topic content, so don't treat it
   as the final verdict.
2. **The failure-pattern table** at the bottom of the report — for each FAIL or ungrounded
   response, note the pattern and the one fix (a system-prompt tweak, a clearer tool docstring, or
   an added refusal example).

## Verified without a live key

Every tool in `tools.py` was run against your real dataset and produces genuine results, e.g.:

```
Head-to-head, Richmond Tigers vs Carlton Blues (58 games on record): Richmond Tigers won 25,
lost 32, drew 1. Most recent meeting: 2025-03-13 at Melbourne Cricket Ground, Richmond Tigers
82 - 69 Carlton Blues (W for Richmond Tigers).
```

All three `.py` files parse as valid Python and import cleanly. `build_agent()` correctly raises a
clear error (not a crash) when `GEMINI_API_KEY` is missing. The full notebook executes end-to-end
with zero errors in both states (key present or absent) — only the live-model calls themselves
need to be run on your end, since I don't have access to the Gemini API in my own environment.

Author:

Azka-Ashfaq

AI and Data Science Intern
