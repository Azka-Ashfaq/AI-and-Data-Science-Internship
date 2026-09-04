# Adult Income Prediction — Day 2 README

## 1. Objective
Move beyond Day 1's hand-written rule baselines and train real machine
learning models — Logistic Regression and a Decision Tree — with a proper,
reusable preprocessing pipeline, then compare them against the Day 1
baselines on the same held-out test set.

## 2. Dataset
- Source: `fetch_openml('adult', version=2, as_frame=True)`
- Shape: 48,842 rows × 15 columns (14 features + `income` target)
- Target: `income`, converted to binary (`1` = `>50K`, `0` = `<=50K`)
- Class distribution: `<=50K` 37,155 (76.1%) · `>50K` 11,687 (23.9%)
- Missing values: `workclass` 2,799 · `occupation` 2,809 · `native-country` 857
- Same stratified 80/20 split as Day 1, `random_state=42`: Train 39,073 rows /
  Test 9,769 rows, both with a 0.239 positive rate — identical test set
  reused across every day of the project

## 3. Preprocessing Pipeline
A single `ColumnTransformer` handles all preprocessing, wrapped inside each
model's own `Pipeline` so preprocessing is refit correctly per model and
never leaks test-set information into training:

- **Numeric features** (`age`, `fnlwgt`, `education-num`, `capital-gain`,
  `capital-loss`, `hours-per-week`): median imputation → `StandardScaler`
- **Categorical features** (`workclass`, `education`, `marital-status`,
  `occupation`, `relationship`, `race`, `sex`, `native-country`):
  most-frequent imputation → `OneHotEncoder(handle_unknown='ignore')`

## 4. Models Trained
| Model | Key settings |
|---|---|
| Logistic Regression | `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'` (compensates for the 76%/24% class imbalance) |
| Decision Tree | `max_depth=10`, `min_samples_split=100`, `min_samples_leaf=50` (all limit tree complexity to reduce overfitting) |

## 5. Results (test set)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline 1: Majority Class | 0.7607 | 0.0000 | 0.0000 | 0.0000 | — | — |
| Baseline 2: Education Rule | 0.7530 | 0.4844 | 0.4970 | 0.4906 | — | — |
| Baseline 3: Advanced Rule | 0.7439 | 0.4745 | 0.6527 | 0.5495 | — | — |
| **Logistic Regression** | 0.8072 | 0.5651 | 0.8443 | **0.6771** | 0.9044 | 0.7619 |
| Decision Tree | 0.8606 | 0.7742 | 0.5894 | 0.6693 | 0.9079 | 0.7826 |

**Best model by F1-score: Logistic Regression (F1 = 0.6771).**

✅ **Goal achieved** — the Day 1 target of F1 > 0.60 was reached by both ML
models, and both comfortably beat the best Day 1 baseline (Advanced Rule,
F1 = 0.5495).

**Reading the trade-off between the two models:** Logistic Regression has
much higher recall (0.8443 vs. 0.5894) — it catches far more actual high
earners — while the Decision Tree has much higher precision (0.7742 vs.
0.5651) — when it predicts `>50K` it's right more often, but it misses more
genuine high earners. Since the project's stated priority is not missing
high-income prospects (a false negative is a lost opportunity), Logistic
Regression's higher recall is why it wins on F1 despite the Decision Tree
having a slightly higher ROC-AUC (0.9079 vs. 0.9044) and a much higher raw
accuracy (0.8606 vs. 0.8072) — accuracy alone favors the tree because it's
better at correctly predicting the majority `<=50K` class.

## 6. Model Analysis

**Logistic Regression — top features:**

| Direction | Top features |
|---|---|
| Toward `>50K` | `capital-gain` (coef 2.28), `marital-status: Married-civ-spouse` (1.75), `marital-status: Married-AF-spouse` (1.73), `native-country: England` (1.09), `native-country: Ireland` (0.88) |
| Toward `<=50K` | `occupation: Priv-house-serv` (−1.63), `native-country: Columbia` (−1.54), `marital-status: Never-married` (−1.25), `native-country: Dominican-Republic` (−1.18), `occupation: Farming-fishing` (−0.88) |

Capital-gain and marital status dominate the top signals — consistent with
Day 1's error analysis, which found the hand-written rules were missing
real high earners the education/occupation-only logic didn't cover.

**Decision Tree:**
- Depth: 10 (capped by `max_depth=10`)
- Leaves: 184
- Train accuracy: 0.8603 · Test accuracy: 0.8606
- Overfitting gap: **−0.0003** — test accuracy is essentially identical to
  train accuracy, so the depth/leaf-size limits successfully prevented
  overfitting; the model generalizes well.

## 7. Files in This Folder
| File | Description |
|---|---|
| `AIP_Supervised_model.py` | Day 2 script: preprocessing pipeline, baseline re-evaluation, Logistic Regression + Decision Tree training/evaluation, saving |
| `models/preprocessor.joblib` | The fitted `ColumnTransformer` alone |
| `models/log_reg_pipeline.joblib` | Full Logistic Regression pipeline (preprocessing + classifier) |
| `models/tree_pipeline.joblib` | Full Decision Tree pipeline (preprocessing + classifier) |
| `models/model_comparison.csv` | Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC for all 5 models |
| `figs/roc_curves.png` | ROC curves, Logistic Regression vs. Decision Tree |
| `figs/pr_curves.png` | Precision-Recall curves, Logistic Regression vs. Decision Tree |
| `figs/confusion_matrices.png` | Side-by-side confusion matrices for both models |
| `figs/feature_importance.png` | Top 10 Logistic Regression coefficients pushing toward each class |
| `figs/decision_tree.png` | Decision Tree structure, first 3 levels |
| `figs/model_comparison.png` | Bar chart comparing accuracy/precision/recall/F1 across models |

## 8. How to Reproduce
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python AIP_Supervised_model.py
```
`random_state=42` is used throughout, so the split, Logistic Regression fit,
and Decision Tree fit are all identical on every run.

Author
Azka Ashfaq - AI and Data Science Intern
