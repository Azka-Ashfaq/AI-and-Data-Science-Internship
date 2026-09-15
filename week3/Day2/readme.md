# Week 3 · Day 2 — Prediction Models

Match-winner and top-player prediction models, trained on Day 1's cleaned data and feature
definitions. Both are packaged as plain Python functions with a clean, documented interface.

## Folder structure

```
day2/
├── afl_datasets/                              <- put your 4 raw CSVs here (same as Day 1)
├── afl_week3_day2_prediction_models.ipynb     <- main notebook (Tasks 1-5)
├── predict.py                                  <- callable prediction functions
└── model_store/                                <- created automatically when you run the notebook
    ├── match_winner_model.joblib
    ├── top_player_model.joblib
    ├── team_latest_snapshot.csv
    ├── player_latest_snapshot.csv
    ├── known_teams.csv
    └── known_players.csv
```

`predict.py` reads everything it needs from `model_store/` at import time — keep that folder sitting
right next to it.

## Setup

Requires Python 3 with `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `joblib`. If any are
missing:

```
pip install pandas numpy matplotlib scikit-learn joblib
```

## How to run

1. Unzip `afl_datasets.zip` so the `afl_datasets` folder sits **next to** the notebook (same as Day 1).
2. Open `afl_week3_day2_prediction_models.ipynb` in VS Code / Jupyter.
3. Select a kernel that has the packages above installed.
4. **Run All.**

The notebook rebuilds Day 1's cleaning + feature engineering from the raw CSVs first, so it runs
standalone. Running it produces `model_store/` and `predict.py` fresh.

## What the notebook does

| Task | Output |
|---|---|
| 1. Baselines | Match winner: majority-class vs ladder-based. Top player: "last game repeats" vs "rolling 10-game average" |
| 2. Match Winner Model | Logistic Regression + HistGradientBoosting, evaluated on accuracy/F1/ROC AUC/Brier score + calibration curve; final model chosen and justified |
| 3. Top Player Model | Regression on `fantasy_points` (framing justified over learning-to-rank); MAE/RMSE + top-5 hit rate, compared to Task 1 baselines |
| 4. Feature Importance & Sanity Checks | Coefficients/permutation importance for both models, leakage check, manual sniff test on 3 holdout matches |
| 5. Packaging | Both models saved to `model_store/`; `predict.py` written with input-validated callable functions |

## Results summary

- **Match winner**: Logistic Regression chosen as final model — near-identical accuracy/ROC AUC to
  HistGBM, but better-calibrated probabilities and full interpretability. Both models beat the
  ladder baseline (66% accuracy) and majority-class baseline.
- **Top player**: HistGBM regressor beats both baselines on MAE (17.5 vs 17.8/19.1), and beats the
  weaker baseline on top-5 hit rate — but **ties** the rolling-average baseline on top-5 hit rate
  (0.759 both). Documented honestly in the notebook rather than overstated.
- No leakage red flags found — all rolling/expanding features are `shift(1)`'d before any
  window, and no feature derived from the match being predicted appears in either model.

## How to call the models

```python
from predict import predict_match_winner, predict_top_player

predict_match_winner("Richmond Tigers", "Carlton Blues", "2026-06-01")
# -> {'home_team': ..., 'away_team': ..., 'date': ..., 'winner': ..., 'probability': ...}

predict_top_player("Richmond Tigers", "2026-06-01", top_k=5)
# -> [{'player_id': ..., 'predicted_score': ...}, ...]
```

Both raise a clear `ValueError` (not a crash) on an unknown team name, an unknown/unsupported
`stat_type`, or a date before the data's known range (1983-01-01) — this was smoke-tested in the
notebook's last cell.

## Limitation

`predict_match_winner` / `predict_top_player` use each team's/player's **most recent snapshot in
the historical dataset** as "current form" — there is no live data feed, so predictions reflect
form as of wherever the training data ends.

Author:

Azka_Ashfaq

AI and Data-science Intern
