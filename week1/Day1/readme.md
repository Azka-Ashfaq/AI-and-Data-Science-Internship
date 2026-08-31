# Adult Income Prediction - Baseline Analysis

Predict if someone earns >50K/year using simple baseline models on the UCI Adult Census dataset.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | Rich Caught |
|-------|----------|-----------|--------|----------|-------------|
| Baseline 1: Always guess "not rich" | 0.760 | 0.000 | 0.000 | 0.000 | 0 |
| Baseline 2: Simple Rule | 0.742 | 0.443 | 0.529 | 0.482 | 1,238 |
| Baseline 3: Advanced Rule | 0.724 | 0.426 | 0.608 | 0.501 | 1,423 |

**Best F1-Score:** 0.501

**Goal:** F1-Score > 0.60

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Run the code
python adult_income_baseline.py
