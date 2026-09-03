```markdown
# Day 4: Model Tuning, Regularization & Reproducible Pipelines

## 📌 Project Overview

This project focuses on hyperparameter tuning, regularization, and building reproducible pipelines for the Adult Income Prediction problem. The goal is to optimize model performance, diagnose overfitting/underfitting, calibrate probabilities, and save a production-ready pipeline.

## 🎯 Tasks Completed

### ✅ Task 1: Fully Reproducible Pipelines
- Built complete sklearn pipelines with feature engineering + preprocessing + estimator
- Set `random_state=42` everywhere for reproducibility
- Documented library versions used

### ✅ Task 2: Hyperparameter Search
- Ran `RandomizedSearchCV` with 5-fold StratifiedKFold for 3 models:
  - **Logistic Regression**: penalty (l1/l2), C (inverse regularization)
  - **Random Forest**: n_estimators, max_depth, min_samples_leaf, max_features
  - **HistGradientBoosting**: learning_rate, max_iter, max_depth, l2_regularization
- Optimized on ROC AUC with `n_jobs=-1`

### ✅ Task 3: Diagnose Overfitting/Underfitting
- Learning curve for HistGradientBoosting (train-val gap: 0.0194)
- Validation curve for Logistic Regression (effect of C)
- Validation curve for Random Forest (effect of max_depth)
- Identified fixes: cap max_depth, keep C in 0.1-1 range

### ✅ Task 4: Probability Calibration & Threshold Selection
- Calibration curve and Brier score analysis
- Brier score: 0.0886 (uncalibrated) vs 0.0886 (calibrated) - no improvement
- **F1-optimized threshold selected: 0.40** (improves F1 from 0.710 to 0.725)

### ✅ Task 5: Final Evaluation & Save Artifact
- Final evaluation on untouched hold-out test set
- Saved pipeline using `joblib`
- Created inference script for production use

## 🏆 Best Model Performance

**Selected Model**: HistGradientBoosting

### Cross-Validation Performance (5-fold)
| Metric | Score |
|--------|-------|
| Accuracy | 0.872 ± 0.002 |
| ROC AUC | 0.927 ± 0.002 |
| F1-Score | 0.710 ± 0.006 |

### Final Test Performance
| Threshold | Accuracy | Precision | Recall | F1-Score | ROC AUC | PR AUC |
|-----------|----------|-----------|--------|----------|---------|--------|
| 0.50 (default) | 0.877 | 0.792 | 0.660 | 0.720 | 0.930 | 0.834 |
| **0.40 (F1-tuned)** | **0.870** | **0.724** | **0.740** | **0.732** | **0.930** | **0.834** |

### Best Hyperparameters
```
HistGradientBoosting:
  - max_iter: 160
  - learning_rate: 0.177
  - max_depth: 4
  - l2_regularization: 0.961
```

## 📁 Project Structure

```
day4/
├── model_tuning.py              # Main training script
├── inference.py                 # Inference/prediction script
├── README.md                    # Project documentation
├── Day4_Tuning_Report.docx      # Detailed tuning report
│
├── artifacts/                   # Saved model artifacts
│   ├── final_pipeline.joblib    # Complete trained pipeline
│   ├── preprocessor.joblib      # Preprocessing pipeline
│   ├── best_params.csv          # Best hyperparameters
│   ├── final_metrics.csv        # Final test metrics
│   ├── cv_results.csv           # Cross-validation results
│   └── threshold_info.csv       # Threshold optimization results
│
├── figs/                        # Visualizations
│   ├── learning_curve.png
│   ├── validation_curve_lr_C.png
│   ├── validation_curve_rf_depth.png
│   ├── calibration_curve.png
│   ├── final_roc_curve.png
│   └── final_pr_curve.png
│
└── data/                        # Sample data for inference
    └── new_people.csv
```

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib scipy joblib
```

### Train the Model
```bash
python model_tuning.py
```

This will:
1. Load the Adult dataset from OpenML
2. Perform hyperparameter search (20 iterations each)
3. Select the best model (HistGradientBoosting)
4. Generate all visualizations
5. Save artifacts to `artifacts/` folder

### Run Inference
```bash
# Single example predictions
python inference.py

# Batch prediction with CSV
python inference.py --input data/new_people.csv --threshold 0.4

# Custom output file
python inference.py --input data/new_people.csv --threshold 0.4 --output my_predictions.csv
```

## 📊 Key Visualizations

| Image | Description |
|-------|-------------|
| `learning_curve.png` | Train vs validation performance across dataset sizes |
| `validation_curve_lr_C.png` | Effect of regularization on Logistic Regression |
| `validation_curve_rf_depth.png` | Effect of tree depth on Random Forest |
| `calibration_curve.png` | Probability calibration comparison |
| `final_roc_curve.png` | ROC curve on test set (AUC = 0.930) |
| `final_pr_curve.png` | Precision-Recall curve on test set (PR-AUC = 0.834) |

## 🔍 Key Findings

### Model Selection
- **HistGradientBoosting** outperformed Logistic Regression and Random Forest
- Best ROC AUC: 0.927 (CV) / 0.930 (Test)
- Best F1-Score: 0.732 (Test, threshold 0.40)

### Overfitting Diagnosis
- Train-val gap at full data: **0.0194** (small - good generalization)
- Logistic Regression: Best C ≈ 0.7-1.0
- Random Forest: Best max_depth ≈ 10-15

### Calibration
- Brier score: 0.0886 (both uncalibrated and calibrated)
- No improvement from isotonic calibration
- Probabilities were already well-calibrated

### Threshold Optimization
- **Best threshold: 0.40** (F1-optimized)
- Default 0.50: F1 = 0.720
- Optimized 0.40: F1 = 0.732

## 📈 Business Impact

**Using the F1-optimized threshold (0.40):**

| Metric | Value |
|--------|-------|
| True Positives (correctly identified >50K) | 1,711 people |
| False Positives (wasted outreach) | 657 people |
| False Negatives (missed opportunities) | 627 people |

**Cost-Benefit Calculation** (assuming $5 per outreach, $2,000 revenue per high-income customer):
- Marketing Cost: (1,711 + 657) × $5 = $11,840
- Expected Revenue: 1,711 × $2,000 = $3,422,000
- **Net Profit: $3,410,160**

## 🔧 Reproducibility

- **Random Seed**: `random_state=42` throughout
- **Cross-Validation**: 5-fold StratifiedKFold
- **Library Versions**:
  - Python: 3.11.9
  - scikit-learn: 1.8.0
  - pandas: 2.3.3
  - numpy: 2.4.0
  - scipy: 1.17.1

## 📝 How to Infer with Saved Model

```python
import joblib
import pandas as pd

# Load model
pipeline = joblib.load('artifacts/final_pipeline.joblib')

# Prepare new data
new_data = pd.DataFrame([{
    'age': 35,
    'workclass': 'Private',
    'education-num': 13,
    'hours-per-week': 45,
    'capital-gain': 5000,
    'occupation': 'Prof-specialty',
    'marital-status': 'Married-civ-spouse',
    'sex': 'Male',
    'race': 'White',
    'education': 'Bachelors',
    'fnlwgt': 200000,
    'capital-loss': 0,
    'relationship': 'Husband',
    'native-country': 'United-States'
}])

# Predict
prediction = pipeline.predict(new_data)[0]
probability = pipeline.predict_proba(new_data)[0][1]

print(f"Prediction: {'>50K' if prediction == 1 else '<=50K'}")
print(f"Probability: {probability:.2%}")
```

## 📋 Deliverables

- ✅ `model_tuning.py` - Complete training script
- ✅ `inference.py` - Inference/prediction script
- ✅ `artifacts/final_pipeline.joblib` - Saved model
- ✅ `Day4_Tuning_Report.docx` - Detailed tuning report
- ✅ All visualizations in `figs/` folder
- ✅ All metrics in `artifacts/` folder

## 👩‍💻 Author

**Azka Ashfaq**  
AI and Data Science Intern
