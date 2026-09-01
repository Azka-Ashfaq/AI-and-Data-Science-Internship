# Adult Income Prediction

Predict whether an individual earns >$50K/year using the Adult Census Income dataset.

## Models Implemented

### Baselines
- **Majority Class**: Always predicts <=50K
- **Simple Rule**: Predicts >50K if education >= 13
- **Advanced Rule**: Predicts >50K if (education >= 13) OR (capital-gain > 0) OR (hours > 40 AND high-paying occupation)

### ML Models
- **Logistic Regression** (Best F1: 0.590)
- **Decision Tree** (F1: 0.560)

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Majority Class | 0.760 | 0.000 | 0.000 | 0.000 | - |
| Education Rule | 0.742 | 0.443 | 0.529 | 0.482 | - |
| Advanced Rule | 0.724 | 0.426 | 0.608 | 0.501 | - |
| Logistic Regression | 0.800 | 0.650 | 0.540 | 0.590 | 0.850 |
| Decision Tree | 0.785 | 0.620 | 0.510 | 0.560 | 0.790 |

**Best**: Logistic Regression with F1-Score 0.590

## Features

14 features including: age, education, occupation, marital-status, capital-gain, hours-per-week, and more.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run the script
python AIP_Supervised_model.py
