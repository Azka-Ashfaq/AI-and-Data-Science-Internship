"""
baseline_comparison.py — Model vs. naive baseline (Day 5, Task 2).

Contextualizes "how good is good enough": compares the trained
match-winner model's accuracy/Brier score on the same holdout split used
in Day 2 against a naive ladder-position baseline that predicts the team
with the higher pre-match ladder points always wins (prob = 1.0), a common
sanity-check benchmark for match-outcome models.

Run locally (needs the Day 1/2 outputs next to this file):
    python baseline_comparison.py \
        --holdout afl_datasets/match_holdout.csv \
        --model model_store/match_winner_model.joblib

If your Day 2 holdout file has a different name/columns, pass --home-ladder-col
/ --away-ladder-col / --label-col to match it — defaults match the column
names used elsewhere in this project (home_ladder_pts_cum_pre,
away_ladder_pts_cum_pre, a binary home-win label).
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def naive_ladder_baseline(df: pd.DataFrame, home_col: str, away_col: str) -> np.ndarray:
    """Predicts P(home win) = 1.0 if home has more ladder points pre-match,
    0.0 if away does, 0.5 on an exact tie. This is the simplest 'public'
    benchmark available without any modeling — if the trained model can't
    beat this, the model isn't adding value over reading the ladder."""
    diff = df[home_col] - df[away_col]
    probs = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="afl_datasets/match_holdout.csv",
                     help="Time-based holdout CSV produced in Day 1/2 (must include the same "
                          "feature columns the model was trained on, plus the true-label column).")
    ap.add_argument("--model", default="model_store/match_winner_model.joblib")
    ap.add_argument("--home-ladder-col", default="home_ladder_pts_cum_pre")
    ap.add_argument("--away-ladder-col", default="away_ladder_pts_cum_pre")
    ap.add_argument("--label-col", default="home_win",
                     help="Binary column: 1 if the home team won, 0 otherwise.")
    args = ap.parse_args()

    try:
        df = pd.read_csv(args.holdout)
        model = joblib.load(args.model)
    except FileNotFoundError as e:
        print(f"Could not find a required file: {e}")
        print("Run this from your project folder with afl_datasets/ and model_store/ present, "
              "or pass --holdout / --model explicitly.")
        sys.exit(1)

    if args.label_col not in df.columns:
        print(f"Label column '{args.label_col}' not found in holdout file. "
              f"Available columns: {list(df.columns)}")
        sys.exit(1)

    y_true = df[args.label_col].astype(int).values

    # --- Naive baseline ---
    naive_probs = naive_ladder_baseline(df, args.home_ladder_col, args.away_ladder_col)
    naive_preds = (naive_probs >= 0.5).astype(int)
    naive_acc = accuracy_score(y_true, naive_preds)
    # clip away exact 0/1 for a defined log-loss/Brier score
    naive_probs_clipped = np.clip(naive_probs, 1e-6, 1 - 1e-6)
    naive_brier = brier_score_loss(y_true, naive_probs_clipped)

    # --- Trained model ---
    feature_cols = [c for c in df.columns if c != args.label_col]
    model_probs = model.predict_proba(df[feature_cols])[:, 1]
    model_preds = (model_probs >= 0.5).astype(int)
    model_acc = accuracy_score(y_true, model_preds)
    model_brier = brier_score_loss(y_true, model_probs)

    print("=" * 60)
    print(f"Holdout size: {len(df)} matches")
    print("=" * 60)
    print(f"{'Method':<28} {'Accuracy':>10} {'Brier score':>14}")
    print(f"{'Naive ladder-position':<28} {naive_acc:>10.3f} {naive_brier:>14.3f}")
    print(f"{'Trained model':<28} {model_acc:>10.3f} {model_brier:>14.3f}")
    print("=" * 60)
    lift = (model_acc - naive_acc) * 100
    print(f"Accuracy lift over naive baseline: {lift:+.1f} points")
    if model_brier < naive_brier:
        print("Model's probabilities are better calibrated than the naive baseline (lower Brier).")
    else:
        print("WARNING: model's Brier score is not better than the naive baseline — "
              "re-check calibration before shipping.")


if __name__ == "__main__":
    main()
