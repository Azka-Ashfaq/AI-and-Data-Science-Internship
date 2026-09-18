# AFL Assistant — Monitoring & Maintenance Checklist

## What to track (from api.py's structured logs)

| Metric | Source field | Alert threshold | Cadence |
|---|---|---|---|
| Response latency (p50 / p95) | `latency_ms` | p95 > 3000ms (retrieval) / > 6000ms (prediction) | Real-time dashboard; review daily |
| Tool error rate | `validation_status == tool_error` ÷ total turns | > 2% over any rolling 1-hour window | Real-time alert |
| Tool timeout rate | `resilience.py` `error_type == timeout` | > 1% over any rolling 1-hour window | Real-time alert |
| Off-topic leak rate | Turns where an off-topic ask got anything other than the scope-refusal message | Any occurrence (target: 0%) | Real-time alert — this is a scope-integrity failure, treat as a bug, not noise |
| Off-topic *volume* (not leak) | Count of `intent == off_topic` per session | `off_topic_streak >= 3` sessions, as a share of all sessions | Daily |
| Router confidence distribution | `intent_confidence` | Share of turns with confidence < 0.6 rising week over week | Weekly |
| Clarification rate | `validation_status == needs_clarification` ÷ total turns | Sustained rise > 5 points week over week | Weekly |
| Prediction accuracy drift | Predicted winner vs. actual result, as each round completes | Rolling 5-round accuracy drops > 8 points below the holdout benchmark (see baseline_comparison.py) | After each round's results are final |
| Rate-limit hits | 429 responses ÷ total requests | > 5% sustained for one conversation_id → investigate as abuse, not just throttle | Daily |

## Weekly retraining / refresh loop

1. **After each round's matches are final** (typically Monday), append the new match results and player box scores to the raw tables (`matches_home_away`, `round_by_round`).
2. **Regenerate the feature snapshots** (`team_latest_snapshot.csv`, `player_latest_snapshot.csv`) the prediction tools read from — these are the *only* files `tools_prediction.py` needs refreshed; the model files themselves don't need to change for a snapshot refresh.
3. **Recompute the rolling accuracy metric**: run `baseline_comparison.py` against the newly-completed round(s) folded into the holdout window, compare to both the original Day 2 holdout numbers and the naive ladder baseline.
4. **Retrain the models** (`predict_match_winner`, `predict_top_player`) on a monthly cadence, or immediately if the rolling accuracy drift alert above fires — whichever comes first. Retraining more often than monthly on AFL's ~23-round season adds noise without enough new data to justify it.
5. **Re-run `eval_suite.py`** after every retrain before promoting the new model files into `model_store/` — a model swap that passes the sanity/scope categories but regresses prediction accuracy should not ship.
6. **Re-run `router_test.py`** after any change to `nodes_router.py` or its pattern lists — it's cheap (no data dependency) and catches routing regressions immediately, as it did in Day 5's hardening (20/20 maintained after the fix).

## Known limitations to keep an eye on

- **No real multi-turn memory in the graph itself** — `conversation_history` is accepted but unused by any node; `api.py`'s entity-carryover retry is a narrow session-layer patch, not general coreference resolution. Watch the clarification rate on genuine follow-up questions; if it climbs, that's the signal to build real context-aware entity resolution into the graph.
- **Naive-baseline comparison must be re-run locally** — `baseline_comparison.py` needs the Day 1/2 holdout file and trained model artifacts, which aren't part of this deliverable; treat the accuracy-drift alert above as provisional until that script has been run once to set the actual threshold numbers.
