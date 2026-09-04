# Adult Income Prediction — Final Project README

## 1. Project Objective
Predict whether an individual earns **more than $50K/year** using the UCI/OpenML
Adult Census dataset, so the result can be used to target marketing outreach at
higher-income prospects efficiently (maximize useful contacts, minimize wasted
outreach on low-probability leads).

## 2. Dataset Description
- Source: `fetch_openml('adult', version=2, as_frame=True)`
- ~48,842 rows, 14 raw input features (demographic and employment attributes:
  age, workclass, education, marital-status, occupation, relationship, race,
  sex, capital-gain, capital-loss, hours-per-week, native-country, fnlwgt).
- Missing values are encoded as `'?'` / `' ?'` in the raw data and were
  converted to `NaN` before modelling.

## 3. Target Variable
- `class` (renamed conceptually to "income"), converted to a binary label:
  `1` = `>50K`, `0` = `<=50K`.
- Class distribution is imbalanced: roughly 24% positive class.

## 4. Feature Engineering
Six engineered features are created inside the pipeline (`engineer_features`),
so they are produced identically at training and inference time:
1. `age_bucket` — age binned into 6 ranges (<25, 25-34, 35-44, 45-54, 55-64, 65+)
2. `hours_bin` — part-time / full-time / overtime buckets from hours-per-week
3. `has_capital_gain` — binary flag for any investment income
4. `log_capital_gain` — log1p transform of capital-gain to fix skew
5. `higher_ed` — flag for education-num >= 13 (some college and above)
6. `edu_hours_interaction` — education-num × hours-per-week

## 5. Preprocessing Steps
All preprocessing lives inside a single `ColumnTransformer` (`build_preprocessor`)
so it is applied identically in training and inference:
- **Numeric features:** median imputation → `StandardScaler`
- **Categorical features:** most-frequent imputation → `OneHotEncoder(handle_unknown='ignore')`

The full pipeline order is: `feature engineering → preprocessing → classifier`.

## 6. Models Tested
- Baselines (Day 1): majority-class, single-rule (education), multi-feature rule
- Day 2: Logistic Regression, Decision Tree
- Day 3: Logistic Regression, Random Forest, HistGradientBoosting (5-fold
  cross-validated comparison + feature-selection check)
- Day 4: the same three models, hyperparameter-tuned

## 7. Hyperparameter Tuning Approach
`RandomizedSearchCV` (20 iterations per model) with 5-fold `StratifiedKFold`,
optimizing **ROC-AUC**, run on a 15,000-row training subsample for speed:
- Logistic Regression: `penalty` (l1/l2), `C` (log-uniform 1e-3–1e2)
- Random Forest: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
- HistGradientBoosting: `learning_rate`, `max_iter`, `max_depth`, `l2_regularization`

The tuned models were then refit and cross-validated on the **full** training
set; the model with the best mean CV ROC-AUC was selected as final.

## 8. Best Parameters
See `artifacts/best_params.csv` (written by Day 4) for the exact winning
hyperparameters found by the search for each of the three candidate models.

## 9. Selected Final Model
See `artifacts/cv_results.csv` for the CV ROC-AUC of every tuned candidate —
the row with the highest `roc_auc_mean` is the model saved as
`artifacts/final_pipeline.joblib`. If probability calibration (isotonic
regression) improved the Brier score on out-of-fold predictions, the saved
pipeline is that model wrapped in `CalibratedClassifierCV`; otherwise it is
the plain tuned pipeline.

## 10. Classification Threshold
Threshold was chosen by maximizing F1-score on the training set's
cross-validated probabilities. See `artifacts/threshold_info.csv` for the
exact value (default 0.50 is also reported there for comparison).

## 11. Final Test Performance
See `day5_outputs/tables/final_comparison_table.csv`, produced by
`day5_final_project.py`, for the final held-out test-set metrics: accuracy,
precision, recall, F1, ROC-AUC, PR-AUC, and Brier score.

## 12. Important Features
See `day5_outputs/tables/feature_importance.csv` and
`day5_outputs/figs/feature_importance.png` for the top permutation-importance
features on the held-out test set (computed on the raw input columns, so it
is valid for whichever model was selected).

## 13. Known Limitations
- The dataset is from 1994 U.S. Census data — income patterns and dollar
  thresholds do not reflect the present day.
- `fnlwgt` is a census sampling weight, not a real personal attribute, and
  can add noise if over-weighted by the model.
- `marital-status` / `relationship` carry meaningful predictive weight but
  are demographic proxies, not skill or effort signals — the model should
  not be used for individual high-stakes decisions without a fairness
  review, even though it's appropriate for the aggregate marketing use case
  it was built for.
- Some demographic subgroups have small sample sizes in this dataset, so
  their subgroup metrics (see `day5_outputs/tables/subgroup_performance.csv`)
  are less reliable.

## 14. How to Reproduce Training
Run the four day scripts in order, each of which reads/creates files the
next script needs:
```bash
python adult_income_prediction.py   # Day 1 - baselines
python AIP_Supervised_model.py      # Day 2 - Logistic Regression / Decision Tree
python FE_CV_MC.py                  # Day 3 - feature engineering + CV comparison
python model_tuning.py              # Day 4 - tuning, calibration, saves artifacts/
```
All scripts use `RANDOM_STATE = 42` and the same `train_test_split` call, so
the test set is identical and untouched across every stage.

## 15. How to Run Inference
```bash
python day5_final_project.py
```
Or import the inference function directly in your own code:
```python
from day5_final_project import predict_income
import pandas as pd

new_rows = pd.DataFrame([...])   # same raw columns as the Adult dataset
result = predict_income(new_rows)
print(result)   # probability_gt_50k, prediction
```
The inference function loads `artifacts/final_pipeline.joblib` and applies
all preprocessing automatically — no manual feature engineering or encoding
is required from the caller.

## 16. Environment / Library Versions
Printed at the end of every script's run, and captured in `requirements.txt`.
See that file for the pinned versions used for this project.

Author
Azka Ashfaq
AI and Data Science Intern
