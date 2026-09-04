# Adult Income Prediction — Day 1 README

## 1. Objective
Establish the problem, explore the Adult Census dataset, create reproducible
train/dev/test splits, and build simple **baseline** predictors before any
real machine learning model is trained. These baselines set the floor that
Day 2+ models must beat.

## 2. Business Objective & Success Metric
Identify individuals likely to earn **>$50K/year** for targeted marketing
outreach, so campaigns contact high-potential candidates efficiently.

**Primary metric: F1-score** — balances Precision (avoid wasting outreach on
unlikely candidates) against Recall (don't miss genuine high-income
prospects). A missed high-income candidate (false negative) is a lost
opportunity; contacting a low-income candidate (false positive) is wasted
spend.

**Stakeholder summary:** "We predict who earns over $50K to target marketing
efficiently. Using F1-Score balances finding the right people while
minimizing wasted outreach. We aim to improve from baseline F1 of 0.00 to
above 0.60."

## 3. Dataset
- Source: `fetch_openml('adult', version=2, as_frame=True)`
- **Shape:** 48,842 rows × 14 features
- **Target:** `class`, converted to binary (`1` = `>50K`, `0` = `<=50K`)
- **Class distribution:** `<=50K`: 37,155 (76.1%) · `>50K`: 11,687 (23.9%) —
  base rate 0.239, i.e. moderately imbalanced

**Missing values** (encoded as `'?'` in the raw data, converted to `NaN`):

| Column | Missing count |
|---|---|
| `workclass` | 2,799 |
| `occupation` | 2,809 |
| `native-country` | 857 |

**Key numeric feature summaries:**

| Feature | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|
| age | 38.6 | 13.7 | 17 | 37 | 90 |
| education-num | 10.1 | 2.6 | 1 | 10 | 16 |
| capital-gain | 1,079.1 | 7,452.0 | 0 | 0 | 99,999 |
| capital-loss | 87.5 | 403.0 | 0 | 0 | 4,356 |
| hours-per-week | 40.4 | 12.4 | 1 | 40 | 99 |

Capital-gain's huge std relative to its mean (and 75th percentile of 0)
confirms it's extremely right-skewed — flagged below as something to fix.

**Top categories** (by count): `workclass` is dominated by Private (33,906);
`education` by HS-grad (15,784) and Some-college (10,878); `marital-status`
by Married-civ-spouse (22,379) and Never-married (16,117).

## 4. Data Splits
Stratified splits with `random_state=42`, reused unchanged in every later
day of the project:

| Split | Size | Positive rate |
|---|---|---|
| Train | 39,073 (72%) | 0.239 |
| Dev | 3,908 (8%) | 0.239 |
| Test | 9,769 (20%) | 0.239 |

Matching positive rates across all three splits confirm the stratification
worked correctly. The test set is held out and untouched until final
evaluation.

**Why a holdout test set is necessary:** it simulates genuinely new data the
model hasn't seen, gives an unbiased estimate of real-world performance, and
prevents overfitting to the test set during tuning.

## 5. Baseline Models & Results
Three baselines were built and evaluated on the test set (see
`baseline_results.csv`):

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

## 6. Error Analysis (Advanced Rule, on the test set)
- **Total predictions:** 9,769
- **False Positives:** 1,690 (predicted `>50K`, actually `<=50K`)
- **False Negatives:** 812 (predicted `<=50K`, actually `>50K`)

**False Positive profile** (median across sampled rows, in
`false_positives_sample.csv`): age 35, education-num 13, hours-per-week 40.
These tend to be workers with solid education and full-time hours in roles
like Exec-managerial or Prof-specialty whose actual income still fell at or
below $50K that year — the rule over-trusts education/occupation alone.

**False Negative profile** (median, in `false_negatives_sample.csv`): age
50, education-num 10, hours-per-week 40. These tend to be older workers in
occupations the rule doesn't recognize as "high-paying" (e.g.
Protective-serv, Transport-moving, Handlers-cleaners) despite full-time
hours — real high earners the rule's fixed occupation list misses.

## 7. Issues to Fix in Later Days
1. Missing values (`workclass`, `occupation`, `native-country`) need proper
   imputation, not just cleaning
2. `capital-gain` is highly skewed (mean 1,079 vs. 75th-percentile of 0) —
   needs a log transform
3. Categorical features need proper (one-hot) encoding, not hand-written rules
4. Class imbalance (23.9% positive) needs class weighting
5. Feature engineering: age buckets, education categories, interaction terms
6. `hours-per-week` has extreme outlier values (up to 99) to handle

## 8. Files in This Folder
| File | Description |
|---|---|
| `adult_income_prediction.py` | Day 1 script: problem definition, EDA, splits, 3 baselines, error analysis |
| `baseline_results.csv` | Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC for all 3 baselines |
| `false_positives_sample.csv` | Sampled false-positive rows from the Advanced Rule baseline |
| `false_negatives_sample.csv` | Sampled false-negative rows from the Advanced Rule baseline |
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

## 10. Next Steps
- **Day 2:** Build ML models (Logistic Regression, Decision Tree) with a
  proper preprocessing pipeline — target F1 ~0.60
- **Day 3:** Feature engineering + cross-validated model comparison — target F1 ~0.65
- **Day 4:** Hyperparameter tuning, calibration, threshold selection — target F1 > 0.70
- **Day 5:** Final validation, interpretation, and production inference

Author
Azka Ashfaq - AI and Data Science Intern
