
```markdown
# AFL Assistant — LangGraph Integration (Chat, Retrieval & Prediction)

A LangGraph-orchestrated AFL assistant that routes user queries between chat,
statistical retrieval, ML-based predictions, refusals, and clarification
loops — with guaranteed disclaimers on every prediction and safe fallbacks
for ambiguous or unsupported requests.

---

## Features

- **Intent routing** — a rule-based classifier sends each query to the correct
  node: `prediction`, `retrieval`, `factual`, or `off_topic`.
- **Statistical retrieval** — H2H records, player season/career/recent stats
  via Day-3 structured query tools.
- **ML predictions** — match-winner and top-fantasy-player predictions with
  probability, confidence level, and top feature drivers.
- **Guaranteed disclaimers** — every prediction passes through a formatter
  node that hard-codes the probabilistic disclaimer.
- **Self-correction** — ambiguous entities (e.g., "Smith") trigger a
  clarification loop instead of guessing.
- **Safe fallbacks** — unsupported stat types (e.g., "predict tackles") are
  refused with a clear explanation instead of hallucinating.

---

## Architecture

```
                    ┌──────────────────────┐
   USER QUERY ────► │   router_node        │  (intent classifier)
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┬────────────────┐
              ▼                ▼                ▼                ▼
       prediction_node   retrieval_node   direct_answer    refusal_node
              │                │                │                │
              └────────┬───────┘                │                │
                       ▼                        ▼                ▼
                 validate_node                END              END
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    formatter_node  clarify_node  fallback_node
          │            │            │
          ▼            ▼            ▼
         END          END          END
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure the model assets are present

```
model_store/
  ├── match_winner_model.joblib
  ├── top_player_model.joblib
  ├── team_latest_snapshot.csv
  ├── player_latest_snapshot.csv
  ├── known_players.csv
  └── known_teams.csv

afl_datasets/
  ├── afl_players_info_raw.csv
  ├── afl_round_by_round_raw.csv
  └── afl_matches_home_away_raw.csv
```

### 3. Run the router accuracy test

```bash
python router_test.py
```

Expected output: `Accuracy: 20/20 = 100.0%`

### 4. Run the full graph on sample queries

```bash
python graph_app.py
```

### 5. Run the 11-conversation end-to-end test

```bash
python e2e_tests.py
```

Writes `e2e_traces.json` with the full state trace for each conversation.

---

## Example Interactions

### Match-winner prediction

```
USER: who will win the Pies vs Cats this week

Predicted winner: Collingwood Magpies
(win probability 64.9%, confidence Moderate).

Top drivers:
  - Home team accumulated ladder points/standing (impact 1.475)
  - Away team recent win rate (last 5 games) (impact 0.454)
  - Home H2H Games Played Prior (impact 0.306)

Predictions are probabilistic statistical estimates based on historical
machine learning models and team snapshots. Actual match outcomes are
subject to real-time form, injuries, and in-game variance.
```

### Top-player prediction

```
USER: who will top-score for Carlton

Predicted top player for Carlton Blues: Nic Newman
(projected fantasy score 102.8).

Grounding:
  - High recent 5-game fantasy baseline (126.4 pts avg)
  - Consistent ball-winning midfield/rebound role (~28.2 disposals/game)
  - High on-ground presence and primary role in club's setup

Player rankings represent statistical expected value simulations under
historical conditions. Player selection, tactical matchups, and ground
time impact actual fantasy outcomes.
```

### Retrieval

```
USER: what were Dustin Martin's 2017 stats

Dustin Martin in 2017 (25 games): 709 disposals (avg 29.5),
37 goals (avg 1.5), 2784 fantasy points (avg 111.4).
```

### Clarification (ambiguous entity)

```
USER: will Smith play well this week

I need the player's full name and team to predict their performance.
Could you clarify which player you mean?
```

### Off-topic refusal

```
USER: what's the weather in Sydney

I can only answer AFL-related questions — stats, head-to-heads, and
match/player predictions. Please ask me something about the AFL.
```

---

## Routing Accuracy

Tested on 20 labeled queries (5 per intent class):

| Intent     | Correct | Total | Accuracy |
|------------|---------|-------|----------|
| prediction | 6       | 6     | 100%     |
| retrieval  | 6       | 6     | 100%     |
| factual    | 4       | 4     | 100%     |
| off_topic  | 4       | 4     | 100%     |
| **Total**  | **20**  | **20** | **100%** |

Full table: see [`routing_accuracy.md`](./routing_accuracy.md).

---

## Project Structure

```
day4/
├── state.py                  # LangGraph state schema
├── entity_resolution.py      # Team / player / date resolution
├── tools_prediction.py       # ML prediction tools (match winner, top player)
├── tools_retrieval.py        # Structured retrieval tools (H2H, season, career)
│
├── nodes_router.py           # Intent classifier + routing edge
├── nodes_prediction.py       # Prediction node (entity resolution + tool call)
├── nodes_retrieval.py        # Retrieval node (tool dispatch)
├── nodes_validation.py       # Validation + clarify / fallback / refusal / direct
├── nodes_formatter.py        # Response formatter with guaranteed disclaimers
│
├── graph_app.py              # Compiled LangGraph application
├── router_test.py            # 20-query routing accuracy test
├── e2e_tests.py              # 11-conversation end-to-end test
│
├── e2e_traces.json           # Auto-generated state traces (from e2e_tests.py)
├── routing_accuracy.md       # Standalone routing accuracy table
├── README_day4.md            # Full deliverable write-up (design + justification)
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## Design Notes

### Why explicit routing instead of a single free-form agent?

1. **Guaranteed disclaimers.** Every prediction response passes through
   `formatter_node`, which hard-codes the probabilistic disclaimer. A free
   agent can forget it; a graph edge cannot skip it.

2. **Deterministic validation.** After any tool call, `validate_node`
   inspects the tool's status (`success` / `error` / `ambiguous` / `not_found`
   / `unsupported`) and routes to the correct handler. No silent failure.

3. **Forced clarification on ambiguity.** Unresolved entities trigger
   `clarify_node` — the system asks the user rather than guessing.

4. **Debuggability.** Each node is a pure function of state. Any failure is
   traceable to one specific node.

See [`README_day4.md`](./README_day4.md) for the full write-up including
annotated state traces for 3 representative conversations.

---

## Requirements

- Python 3.11+
- See `requirements.txt` for package versions.

---



Once `README.md` is saved, you're ready to push. GitHub will automatically render it on the repo homepage with the architecture diagram, example interactions, accuracy table, and design rationale — everything a grader needs to evaluate the project at a glance.
