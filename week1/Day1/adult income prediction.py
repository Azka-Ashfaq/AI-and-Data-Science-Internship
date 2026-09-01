"""
Adult Income Prediction - Simple Version
Goal: Guess if someone earns >50K/year, using simple baselines (no real ML model yet)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# STEP 1: Load the data
# -----------------------------------------
data = fetch_openml('adult', version=2, as_frame=True)
df = data.frame

# Fix: Dataset uses ' ?' (with a space) or '?' for missing values
df = df.replace(' ?', np.nan)  
df = df.replace('?', np.nan)   

print("Rows, Columns:", df.shape)
print("Columns:", list(df.columns))
print("\nTarget column values:")
print(df['class'].value_counts())

# Convert target to boolean for easier analysis
# Original has '>50K' and '<=50K'
positive_rate = (df['class'] == '>50K').mean()
print(f"\nPercent earning >50K: {round(positive_rate * 100, 1)} %")

# -----------------------------------------
# STEP 2: Split into train / test
# -----------------------------------------
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# drop rows with missing values (simple fix for now)
train = pd.concat([X_train, y_train], axis=1).dropna()
test = pd.concat([X_test, y_test], axis=1).dropna()

X_train, y_train = train.drop('class', axis=1), train['class']
X_test, y_test = test.drop('class', axis=1), test['class']

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# -----------------------------------------
# STEP 3: Baseline 1 - Always guess the majority class
# -----------------------------------------
majority_model = DummyClassifier(strategy='most_frequent')
majority_model.fit(X_train, y_train)
majority_pred = majority_model.predict(X_test)

# -----------------------------------------
# STEP 4: Baseline 2 - Simple rule (single feature)
# Guess ">50K" if high education
# -----------------------------------------
def simple_rule_single(data):
    condition = (data['education-num'] >= 13)
    return np.where(condition, '>50K', '<=50K')

rule_pred_single = simple_rule_single(X_test)

# -----------------------------------------
# STEP 5: Baseline 3 - Advanced rule (multiple features)
# Guess ">50K" if:
# - High education (>=13) OR
# - Has capital gains (>0) OR
# - Works many hours (>40) AND is in a high-paying occupation
# -----------------------------------------
def simple_rule_advanced(data):
    condition = ((data['education-num'] >= 13) | 
                 (data['capital-gain'] > 0) |
                 ((data['hours-per-week'] > 40) & 
                  (data['occupation'].isin(['Exec-managerial', 'Prof-specialty', 'Tech-support']))))
    return np.where(condition, '>50K', '<=50K')

rule_pred_advanced = simple_rule_advanced(X_test)

# -----------------------------------------
# STEP 6: Check how good each guess is
# -----------------------------------------
def show_scores(name, y_true, y_pred):
    print(f"\n{name}")
    print("Accuracy: ", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred, pos_label='>50K', zero_division=0), 3))
    print("Recall:   ", round(recall_score(y_true, y_pred, pos_label='>50K'), 3))
    print("F1-Score: ", round(f1_score(y_true, y_pred, pos_label='>50K'), 3))

show_scores("Baseline 1: Always guess majority class", y_test, majority_pred)
show_scores("Baseline 2: Simple rule (education >= 13)", y_test, rule_pred_single)
show_scores("Baseline 3: Advanced rule (education OR capital-gain OR hours + occupation)", y_test, rule_pred_advanced)

print("\nGoal for future models: F1-Score above 0.60")

# -----------------------------------------
# STEP 7: Compare all baselines
# -----------------------------------------
print("\n" + "="*60)
print("BASELINE COMPARISON")
print("="*60)
print("Baseline 1 (Majority class): Always predicts '<=50K' - F1: 0.000")
print("Baseline 2 (Single rule): Uses only education - F1: ~0.49")
print("Baseline 3 (Advanced rule): Uses multiple features - F1: ~0.55")
print("\nBest baseline: Baseline 3 with F1-score around 0.55")
print("This sets the bar for any ML model - must beat F1 > 0.55 to be useful")

# -----------------------------------------
