# Adult Income Prediction — Day 2 README

## 1. Objective
Move beyond Day 1's hand-written rule baselines and train real machine
learning models — Logistic Regression and a Decision Tree — with a proper,
reusable preprocessing pipeline, then compare them against the Day 1
baselines on the same held-out test set.

## 2. Dataset
- Source: `fetch_openml('adult', version=2, as_frame=True)`
- Target: `income` (from the dataset's `class` column), converted to binary
  (`1` = `>50K`, `0` = `<=50K`)
- Missing values (`'?'` / `' ?'`) converted to `NaN`
- Same stratified 80/20 split as Day 1, with `random_state=42` — the test
  set stays identical across every day of the project

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
| Logistic Regression | `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'` (compensates for the ~24%/76% class imbalance) |
| Decision Tree | `max_depth=10`, `min_samples_split=100`, `min_samples_leaf=50` (all limit tree complexity to reduce overfitting) |

Both baselines from Day 1 (Majority Class, Education Rule, Advanced Rule)
are re-evaluated here for a single side-by-side comparison table.

## 5. Results
Full numbers are written to `models/model_comparison.csv` when the script
runs — paste your actual values into the table below after running it:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline 1: Majority Class | | | | | | |
| Baseline 2: Education Rule | | | | | | |
| Baseline 3: Advanced Rule | | | | | | |
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |

The script prints the best model by F1-score and states whether the Day 1
goal of **F1 > 0.60** was reached.

## 6. Model Analysis
- **Logistic Regression:** coefficients ranked to show the top 5 features
  pushing predictions toward `>50K` and the top 5 pushing toward `<=50K`
  (see `figs/feature_importance.png`).
- **Decision Tree:** depth, number of leaves, and a train-vs-test accuracy
  gap are printed to check for overfitting; a gap over 0.10 triggers a
  warning in the script output.

## 7. Files in This Folder
| File | Description |
|---|---|
| `AIP_Supervised_model.py` | Day 2 script: preprocessing pipeline, baseline re-evaluation, Logistic Regression + Decision Tree training/evaluation, saving |
| `models/preprocessor.joblib` | The fitted `ColumnTransformer` alone |
| `models/log_reg_pipeline.joblib` | Full Logistic Regression pipeline (preprocessing + classifier) |
| `models/tree_pipeline.joblib` | Full Decision Tree pipeline (preprocessing + classifier) |
| `models/model_comparison.csv` | Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC for all 5 models (3 baselines + 2 ML models) |
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
Uses `random_state=42` throughout, so the split and results are identical
on every run (model training itself is otherwise deterministic for both
Logistic Regression and this Decision Tree configuration).

## 9. Next Steps
- **Day 3:** Feature engineering (new derived features) + 5-fold
  cross-validated comparison of Logistic Regression, Random Forest, and
  HistGradientBoosting
- **Day 4:** Hyperparameter tuning, overfitting/underfitting diagnostics,
  probability calibration, and threshold selection
- **Day 5:** Final validation, error analysis, interpretation, and a
  production-ready inference function

Author
Azka Ashfaq - AI and Data Science Intern
