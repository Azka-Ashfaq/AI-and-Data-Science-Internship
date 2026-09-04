# Adult Income Prediction — Day 1 README

## 1. Objective
Establish the problem, explore the Adult Census dataset, create reproducible
train/dev/test splits, and build simple **baseline** predictors before any
real machine learning model is trained. These baselines set the floor that
Day 2+ models must beat.

## 2. Dataset
- Source: `fetch_openml('adult', version=2, as_frame=True)`
- Target: `class`, converted to binary (`1` = `>50K`, `0` = `<=50K`)
- Missing values (`'?'` / `' ?'`) converted to `NaN`
- Class distribution: ~24% positive (`>50K`), 76% negative — imbalanced

## 3. Business Objective & Success Metric
Identify individuals likely to earn `>$50K/year` for targeted marketing
outreach, so campaigns contact high-potential candidates efficiently.

**Primary metric: F1-score** — balances Precision (avoid wasting outreach on
unlikely candidates) against Recall (don't miss genuine high-income
prospects). A missed high-income candidate (false negative) is a lost
opportunity; contacting a low-income candidate (false positive) is wasted
spend.

## 4. Data Splits
Stratified splits with `random_state=42`, reused unchanged in every later
day of the project:
- Train: 72%
- Dev: 8%
- Test: 20% (held out and untouched until final evaluation)

## 5. Baseline Models & Results
Three baselines were built and evaluated on the test set — see
`baseline_results.csv` for the full table:

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Majority Class | 0.7607 | 0.0000 | 0.0000 | 0.0000 | — | — |
| Education Rule (education-num ≥ 13) | 0.7530 | 0.4844 | 0.4970 | 0.4906 | 0.6653 | 0.3611 |
| **Advanced Rule** (education, capital-gain, or overtime + high-paying occupation) | 0.7439 | 0.4745 | 0.6527 | **0.5495** | 0.7126 | 0.3928 |

**Best baseline: Advanced Rule, F1 = 0.5495.** It outperforms the simpler
education-only rule because it combines multiple meaningful signals
(education, investment income, and overtime hours in high-paying
occupations). The Majority Class predictor scores F1 = 0 because it never
predicts the positive class at all, despite having the highest raw accuracy
— a reminder that accuracy alone is misleading on an imbalanced dataset.

**Target for the week: F1 > 0.60**, roughly a 10% improvement over this best
baseline.

## 6. Error Analysis
`false_positives_sample.csv` and `false_negatives_sample.csv` each contain
15 sampled rows from the Advanced Rule's errors on the test set, for
manual inspection:

- **False Positives** (predicted `>50K`, actually `<=50K`): tend to be
  older workers in steady but not high-paying roles (e.g. Craft-repair,
  Other-service) who work full hours without matching income, or highly
  educated individuals (e.g. a Doctorate holder) whose real income didn't
  cross the threshold that year.
- **False Negatives** (predicted `<=50K`, actually `>50K`): tend to be
  workers in occupations the rule doesn't flag as "high-paying"
  (e.g. Protective-serv, Transport-moving) despite full-time or overtime
  hours — the rule's occupation list misses real high earners outside its
  three named categories.

## 7. Issues to Fix in Later Days
1. Missing values need proper imputation (median/most-frequent), not just cleaning
2. `capital-gain` is highly skewed — needs a log transform
3. Categorical features need proper (one-hot) encoding, not hand-written rules
4. Class imbalance (~24% positive) needs class weighting
5. Feature engineering: age buckets, education categories, interaction terms
6. `hours-per-week` has extreme outlier values to handle

## 8. Files in This Folder
| File | Description |
|---|---|
| `adult_income_prediction.py` | Day 1 script: problem definition, EDA, splits, 3 baselines, error analysis |
| `baseline_results.csv` | Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC for all 3 baselines |
| `false_positives_sample.csv` | 15 sampled false-positive rows from the Advanced Rule baseline |
| `false_negatives_sample.csv` | 15 sampled false-negative rows from the Advanced Rule baseline |
| `figs/eda_visualizations.png` | 6-panel EDA chart (income split, age, education, hours, capital gain, occupation) |
| `figs/baseline_confusion_matrices.png` | Confusion matrices for all 3 baselines |
| `figs/baseline_curves.png` | ROC and Precision-Recall curves for all 3 baselines |

## 9. How to Reproduce
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python adult_income_prediction.py
```
Uses `random_state=42` throughout, so results and the train/dev/test split
are identical on every run.

## 👤 Author

*Azka Ashfaq*

- GitHub: [@Azka-Ashfaq](https://github.com/Azka-Ashfaq)
- LinkedIn: [Azka Ashfaq](https://linkedin.com/in/azka-ashfaq)

 📅 Date
August 31, 2026
