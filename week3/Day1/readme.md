# Week 3 · Day 1 — AFL Data Foundations

EDA, feature engineering, and prediction-target definitions for the AFL match-winner and
top-player prediction system (and the domain-locked chat assistant planned for later this week).

## Folder structure

```
day1/
├── afl_datasets/                          <- put your 4 raw CSVs here (unzip afl_datasets.zip)
├── afl_data_foundations.ipynb             <- main notebook (Tasks 1-5)
├── AFL_Data_Dictionary_Week3_Day1.docx    <- 1-page data dictionary + target definitions
├── feature_store/                          <- created automatically when you run the notebook
│   ├── match_feature_table_v<date>.csv
│   └── player_feature_table_v<date>.csv
└── README.md
```

## Setup

Requires Python 3 with `pandas`, `numpy`, and `matplotlib`. If any are missing:

```
pip install pandas numpy matplotlib
```

No other packages are needed.

## How to run

1. Unzip `afl_datasets.zip` so the `afl_datasets` folder sits **next to** the notebook (see
   folder structure above).
2. Open `afl_data_foundations.ipynb` in VS Code / Jupyter.
3. Select a kernel that has the packages above installed.
4. **Run All.**

The loader finds the 4 CSVs by keyword match, so it doesn't matter that the raw filenames are
long or inconsistent — just make sure all 4 original files are somewhere inside `afl_datasets/`.

## What the notebook does

| Task | Output |
|---|---|
| 1. Data Inventory & Understanding | Table grain, join keys, date/season/team/player coverage, structural changes over time, data-quality checks (confirmed against the real data, not assumed) |
| 2. Prediction Targets | `match_result` (Home Win/Away Win/Draw), `match_margin`, `top_disposals`, `top_goalkicker`, `fantasy_score` — exact formulas + a one-row-per-match table |
| 3. EDA | Win rates over time, home-ground advantage, player distributions, 5 prediction-relevant relationships (form, venue, rest days, margin spread, player form vs. output) |
| 4. Feature Engineering | Leakage-safe rolling/form, ladder, head-to-head, and venue features; saved as a versioned CSV feature table |
| 5. Train/Hold-Out Split | Reusable `time_based_split()` function; explanation of why a random split leaks future information; discussion of a realistic accuracy ceiling |

## Data-quality issues found (and fixed in the notebook)

- **Western Bulldogs** is stored as `"W. Bulldogs"` when it's the subject team but
  `"Western Bulldogs"` as an opponent — this silently breaks joins/pairing unless mapped to one
  canonical name first (done in the notebook).
- The `id` column in the team-matches table is a row number, **not** a shared match ID — the real
  match grain is reconstructed from `(date, venue, sorted team pair)`.
- `seasonal.player_id` has a stray `"ID_"` text prefix on a handful of rows.
- ~0.8% of player-match rows have no matching player-bio record (left as unmatched, not dropped).
- Several "advanced" stats (contested possessions, hit-outs, etc.) are `NaN` in early seasons —
  this reflects stats not being tracked yet, not true zeros, so they are **not** imputed.
- No `position` column exists in any table — a documented statistical-profile proxy (ruck /
  forward / defender / midfielder) is used instead, flagged as a known gap for later days.

Author:

Azka-Ashfaq

Ai-Data Science Intern
