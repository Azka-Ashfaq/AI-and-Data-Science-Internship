# AFL Assistant — Day 5 Deliverables

Drop this folder's files next to your existing `afl_datasets/` and
`model_store/` directories from Day 1-4 (same layout as before) and
everything below works as-is — nothing in Day 1-4's data layout changed.

## What changed from Day 4 (hardening)

- **Bugfix:** `nodes_router.py` had a dead branch referencing an undefined
  `FUTURE_MARKERS` name that crashed on some "will..." queries. Removed.
- **`resilience.py`** (new): every tool call now goes through a shared
  timeout + error-handling wrapper (retrieval: 6s, prediction: 8s).
- **Standardized prediction disclaimer** in `tools_prediction.py`.
- **Prompt-injection resistance**: `nodes_router.py` now has a
  `JAILBREAK_PATTERNS` list checked before anything else; 4 injection-style
  prompts were tested live (see `eval_results.md`, category
  `scope_guardrail`, cases G5-G8) plus empty-input and oversized-payload
  abuse guards (G9, G10) — all 10 scope cases passed.
- **Abuse guards**: empty-input and oversized-payload rejection, and an
  `off_topic_streak` counter that escalates the refusal wording after 3+
  consecutive off-scope turns.

## Evaluation headline (full run, data files present)

`eval_results.md` / `eval_results.json`: **29 of 31 cases PASS, 2 FAIL.**
All failures are in the `multi_turn` category (M1, M3) — see below.

| Category | Pass | Fail | Pass rate |
|---|---|---|---|
| factual | 5 | 0 | 100% |
| scope_guardrail | 10 | 0 | 100% |
| retrieval | 6 | 0 | 100% |
| prediction_sanity | 6 | 0 | 100% |
| multi_turn | 2 | 2 | 50% |

## Known gap, called out honestly

`conversation_history` is accepted into the graph's state but no node in
Day 4's graph actually reads it — every turn was classified independently.
`api.py` / `streamlit_app.py` add a narrow session-layer fix (entity
carryover for follow-ups that *omit an entity*, e.g. "what about his career
average"). That fix works for retrieval follow-ups (M2 passes) but **does
not** cover follow-ups with no entity at all ("why do you think that" after
a prediction, M1) or follow-ups whose entity is a *pair* rather than a
single name ("who won the most recent one" after a head-to-head, M3). Both
remain open; see "Recommended Next Steps" in the executive report.

## Running things

```bash
# 1. Regression + eval suite (needs afl_datasets/ + model_store/ present)
python router_test.py
python eval_suite.py            # writes eval_results.md / eval_results.json

# 2. Naive-baseline comparison (needs your Day 1/2 holdout CSV + model)
python baseline_comparison.py --holdout afl_datasets/match_holdout.csv \
                               --model model_store/match_winner_model.joblib

# 3. API
pip install fastapi uvicorn
uvicorn api:api_app --reload --port 8000
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"message": "who will win the Pies vs Cats this week"}'

# 4. Demo UI
pip install streamlit
streamlit run streamlit_app.py

# 5. Rebuild the executive report PDF after any report-content change
pip install reportlab
python build_report.py
## Files

| **File**                                                                                                                                                                                                     | **Task** | **What it is**                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------: | --------------------------------------------------------- |
| `resilience.py`                                                                                                                                                                                              |        1 | Timeout + uniform error handling for every tool call      |
| `nodes_router.py`, `nodes_prediction.py`, `nodes_retrieval.py`, `nodes_validation.py`, `nodes_formatter.py`, `state.py`, `graph_app.py`, `entity_resolution.py`, `tools_prediction.py`, `tools_retrieval.py` |        1 | Hardened Day 4 graph (drop-in replacements)               |
| `eval_suite.py`, `eval_results.md`, `eval_results.json`                                                                                                                                                      |        2 | 31-case combined eval suite + full-run results            |
| `baseline_comparison.py`                                                                                                                                                                                     |        2 | Model vs. naive ladder-position baseline                  |
| `api.py`                                                                                                                                                                                                     |        3 | FastAPI `/chat` wrapper, sessions, logging, rate limiting |
| `streamlit_app.py`                                                                                                                                                                                           |        3 | Minimal demo chat UI                                      |
| `monitoring_checklist.md`                                                                                                                                                                                    |        4 | One-page monitoring + weekly retrain loop                 |
| `build_report.py`, `AFL_Assistant_Executive_Report.pdf` (top-level)                                                                                                                                          |        5 | 2-page executive report generator + output                |
| `demo_script.md`                                                                                                                                                                                             |        5 | 5-7 minute demo script / slide outline                    |


