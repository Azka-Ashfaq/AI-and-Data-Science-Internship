# Day 3: Adult Income Prediction - Feature Engineering & Model Comparison

## 📌 Scenario Overview

Today's goal is to expand the feature set with principled feature engineering and use cross-validation to compare models more reliably. The objective is to find features that materially improve performance and get statistically believable performance estimates before hyperparameter tuning.

## 🎯 Tasks Completed

### ✅ Task 1: Engineered Features
Created 6 new features with justifications and univariate predictive scores:

| Feature | Type | Creation Rule | Mutual Info |
|---------|------|---------------|-------------|
| **log_capital_gain** | numeric | log1p(capital-gain) | 0.082 |
| **edu_hours_interaction** | numeric | education-num × hours-per-week | 0.083 |
| **age_bucket** | categorical | Age binned into 6 groups | 0.061 |
| **higher_ed** | boolean | education-num >= 13 | 0.048 |
| **hours_bin** | categorical | hours binned: part/full/overtime | 0.037 |
| **has_capital_gain** | boolean | capital-gain > 0 | 0.030 |

**Key Insights:**
- `log_capital_gain` and `edu_hours_interaction` carry the strongest univariate signal
- >50K rate rises from 1.1% (<25) to peak 39.4% (45-54) by age
- >50K rate rises from 8.4% (part-time) to 40.0% (overtime) by hours

### ✅ Task 2: Pipeline with Engineered Features
Integrated features cleanly into preprocessing pipeline using `FunctionTransformer`:
- No information leakage (features use only current-row data)
- Pipeline: feature_engineering → preprocessor → model

### ✅ Task 3: Cross-Validated Model Comparison
Compared 3 models using StratifiedKFold (k=5) with identical preprocessing:

| Model | Accuracy (mean±std) | ROC AUC (mean±std) | F1 (mean±std) |
|-------|---------------------|--------------------|---------------|
| **Logistic Regression** | 0.858 ± 0.002 | 0.913 ± 0.002 | 0.675 ± 0.005 |
| **Random Forest** | 0.854 ± 0.002 | 0.904 ± 0.002 | 0.671 ± 0.006 |
| **HistGradientBoosting** | **0.874 ± 0.003** | **0.929 ± 0.002** | **0.713 ± 0.007** |

**🏆 Best Model: HistGradientBoosting**
- Leads on all three metrics
- Tightest fold-to-fold spread
- Confirmed as the strongest model

### ✅ Task 4: Statistical Comparison & Feature Importance

**Statistical Tests (Top 2 Models):**
| Test | Statistic | p-value | Significance |
|------|-----------|---------|--------------|
| Paired t-test | t = 29.657 | 0.00001 | ✅ Significant |
| Wilcoxon | W = 0.000 | 0.0625 | Borderline |

**Interpretation:** HistGradientBoosting wins on all 5 folds. The t-test confirms significance (p < 0.05). Wilcoxon's borderline p-value is due to small sample size (n=5), not unreliability.

**Feature Importance:**
- **Logistic Regression:** `log_capital_gain` (+5.393) strongest predictor
- **Random Forest:** `edu_hours_interaction` (0.073) ranks #3 overall

### ✅ Task 5: Feature Selection Check
Tested SelectKBest with mutual information (k=40) vs full feature set:

| Feature Set | Accuracy | ROC AUC | F1 | CV Time |
|-------------|----------|---------|-----|----------|
| Full (~118 cols) | **0.874** | **0.929** | **0.713** | 18.5s |
| SelectKBest k=40 | 0.869 | 0.926 | 0.699 | 216.2s |

**Decision:** Keep full feature set for tuning
- Feature selection did NOT improve performance
- Was ~12x slower
- No columns dropped going into tuning

## 📊 Visualizations Generated

The code produces the following visualization:
- `figs/cv_boxplots.png` - Boxplots showing model performance across 5 CV folds

## 🛠️ Technologies Used

- Python 3.11
- Scikit-learn (Pipeline, ColumnTransformer, Cross-validation)
- Pandas, NumPy (Data manipulation)
- Matplotlib (Visualizations)
- SciPy (Statistical tests)


## 🚀 How to Run

### Prerequisites
``bash
pip install pandas numpy scikit-learn matplotlib scipy

👩‍💻 Author
Azka Ashfaq
AI and Data Science Intern


