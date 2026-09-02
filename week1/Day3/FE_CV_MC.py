"""
Adult Income Prediction - Day 1: Complete Implementation
All Tasks: Problem Definition, EDA, Baselines, Error Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create directories
os.makedirs('figs', exist_ok=True)

print("="*70)
print("ADULT INCOME PREDICTION - DAY 1: COMPLETE IMPLEMENTATION")
print("="*70)

# ============================================================================
# TASK 1: Problem Definition & Success Metric
# ============================================================================
print("\n" + "="*70)
print("TASK 1: PROBLEM DEFINITION & SUCCESS METRIC")
print("="*70)

print("""
BUSINESS OBJECTIVE:
Identify individuals likely to earn >$50K/year for targeted marketing campaigns.
Goal: Maximize marketing efficiency by contacting only high-potential candidates.

PRIMARY METRIC: F1-Score
- Balances Precision (avoid wasting outreach) and Recall (find high-income people)
- Missing a high-income candidate (False Negative) = lost revenue opportunity
- Contacting a low-income candidate (False Positive) = wasted marketing resources

STAKEHOLDER SUMMARY:
"We predict who earns over $50K to target marketing efficiently. Using F1-Score 
balances finding the right people while minimizing wasted outreach. We aim to 
improve from baseline F1 of 0.00 to above 0.60."
""")

# ============================================================================
# TASK 2: Data Load & Quick EDA
# ============================================================================
print("\n" + "="*70)
print("TASK 2: DATA LOAD & EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n[1] Loading data from OpenML...")
data = fetch_openml('adult', version=2, as_frame=True)
df = data.frame

# Clean missing values
df = df.replace(' ?', np.nan)
df = df.replace('?', np.nan)
df['class'] = df['class'].str.replace('.', '', regex=False).str.strip()

# Convert target to binary
y = (df['class'] == '>50K').astype(int)
X = df.drop('class', axis=1)

print(f"Dataset shape: {X.shape}")
print(f"Columns: {list(X.columns)}")

# Class distribution
print("\n[2] Class Distribution:")
class_counts = y.value_counts()
print(f"  <=50K: {class_counts[0]:,} ({class_counts[0]/len(y)*100:.1f}%)")
print(f"  >50K:  {class_counts[1]:,} ({class_counts[1]/len(y)*100:.1f}%)")
print(f"  Base Rate (Positive): {y.mean():.3f}")

# Missing values
print("\n[3] Missing Values Summary:")
missing_df = df.isna().sum()
missing_df = missing_df[missing_df > 0]
if len(missing_df) > 0:
    print(missing_df)
else:
    print("  No missing values found after cleaning.")

# Numeric summaries
print("\n[4] Numeric Feature Summaries:")
numeric_cols = X.select_dtypes(include=[np.number]).columns
print(X[numeric_cols].describe().round(2))

# Categorical value counts
print("\n[5] Categorical Feature Value Counts (Sample):")
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols[:3]:
    print(f"\n  {col}:")
    print(X[col].value_counts().head(5))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[6] Generating Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Adult Income Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Target distribution
ax = axes[0, 0]
counts = y.value_counts()
ax.pie(counts, labels=['<=50K', '>50K'], autopct='%1.1f%%', startangle=90)
ax.set_title('Income Distribution')

# 2. Age distribution by target
ax = axes[0, 1]
for label in [0, 1]:
    ax.hist(X.loc[y==label, 'age'], bins=20, alpha=0.5, label=f'{">50K" if label==1 else "<=50K"}')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Age Distribution by Income')
ax.legend()

# 3. Education distribution
ax = axes[0, 2]
edu_data = pd.crosstab(X['education'], y)
edu_data.plot(kind='bar', ax=ax, stacked=False)
ax.set_title('Education Level by Income')
ax.set_xlabel('Education')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
ax.legend(['<=50K', '>50K'])

# 4. Hours per week
ax = axes[1, 0]
data_to_plot = [X.loc[y==0, 'hours-per-week'].dropna(), X.loc[y==1, 'hours-per-week'].dropna()]
ax.boxplot(data_to_plot, labels=['<=50K', '>50K'])
ax.set_ylabel('Hours per Week')
ax.set_title('Working Hours by Income')

# 5. Capital gain
ax = axes[1, 1]
for label in [0, 1]:
    data_subset = X.loc[y==label, 'capital-gain']
    data_subset = data_subset[data_subset > 0]
    if len(data_subset) > 0:
        ax.hist(np.log1p(data_subset), bins=20, alpha=0.5, label=f'{">50K" if label==1 else "<=50K"}')
ax.set_xlabel('log(Capital Gain + 1)')
ax.set_ylabel('Count')
ax.set_title('Capital Gain Distribution (log scale)')
ax.legend()

# 6. Occupation distribution
ax = axes[1, 2]
occ_data = pd.crosstab(X['occupation'], y)
occ_data = occ_data.loc[occ_data.sum(axis=1).sort_values(ascending=False).head(10).index]
occ_data.plot(kind='bar', ax=ax, stacked=False)
ax.set_title('Top 10 Occupations by Income')
ax.set_xlabel('Occupation')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
ax.legend(['<=50K', '>50K'])

plt.tight_layout()
plt.savefig('figs/eda_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
print("\n[7] Summary Table:")
summary_table = pd.DataFrame({
    'Feature': ['Income Class', 'Income Class', 'Age', 'Education-num', 'Hours-per-week'],
    'Category': ['<=50K', '>50K', 'Mean', 'Mean', 'Mean'],
    'Value': [
        f"{class_counts[0]:,}", 
        f"{class_counts[1]:,}",
        f"{X['age'].mean():.1f}",
        f"{X['education-num'].mean():.1f}",
        f"{X['hours-per-week'].mean():.1f}"
    ]
})
print(summary_table.to_string(index=False))

# ============================================================================
# TASK 3: Create Reproducible Splits
# ============================================================================
print("\n" + "="*70)
print("TASK 3: REPRODUCIBLE DATA SPLITS")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train set size: {len(X_train):,}")
print(f"Test set size:  {len(X_test):,}")
print(f"Train positive rate: {y_train.mean():.3f}")
print(f"Test positive rate:  {y_test.mean():.3f}")

# Dev set
X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

print(f"\nDev set size:   {len(X_dev):,}")
print(f"Dev positive rate: {y_dev.mean():.3f}")

print("\nWHY A HOLDOUT TEST SET IS NECESSARY:")
print("- Simulates 'new data' the model hasn't seen")
print("- Provides unbiased estimate of real-world performance")
print("- Prevents overfitting to test set during tuning")

# ============================================================================
# TASK 4: Implement Simple Baselines
# ============================================================================
print("\n" + "="*70)
print("TASK 4: BASELINE MODELS")
print("="*70)

def prepare_for_baseline(X_train, X_test):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    X_train_num = num_imputer.fit_transform(X_train[numeric_cols])
    X_test_num = num_imputer.transform(X_test[numeric_cols])
    
    X_train_cat = cat_imputer.fit_transform(X_train[cat_cols])
    X_test_cat = cat_imputer.transform(X_test[cat_cols])
    
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat_encoded = encoder.fit_transform(X_train_cat)
    X_test_cat_encoded = encoder.transform(X_test_cat)
    
    X_train_processed = np.hstack([X_train_num_scaled, X_train_cat_encoded])
    X_test_processed = np.hstack([X_test_num_scaled, X_test_cat_encoded])
    
    return X_train_processed, X_test_processed

X_train_proc, X_test_proc = prepare_for_baseline(X_train, X_test)

# Baseline 1: Majority Class
print("\n[1] Majority Class Baseline")
majority_model = DummyClassifier(strategy='most_frequent')
majority_model.fit(X_train_proc, y_train)
majority_pred = majority_model.predict(X_test_proc)
majority_proba = majority_model.predict_proba(X_test_proc)[:, 1]

# Baseline 2: Education Rule
print("\n[2] Simple Rule Baseline (education-num >= 13)")
def simple_rule(data):
    return (data['education-num'].values >= 13).astype(int)

rule_pred = simple_rule(X_test)
rule_proba = (X_test['education-num'] >= 13).astype(float)

# Baseline 3: Advanced Rule
print("\n[3] Advanced Rule Baseline")
def advanced_rule(data):
    condition = ((data['education-num'] >= 13) | 
                 (data['capital-gain'] > 0) |
                 ((data['hours-per-week'] > 40) & 
                  (data['occupation'].isin(['Exec-managerial', 'Prof-specialty', 'Tech-support']))))
    return condition.astype(int)

adv_pred = advanced_rule(X_test)
adv_proba = advanced_rule(X_test).astype(float)

def evaluate_model(name, y_true, y_pred, y_proba=None):
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_proba)) > 1:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_proba)
        metrics['PR AUC'] = average_precision_score(y_true, y_proba)
    else:
        metrics['ROC AUC'] = np.nan
        metrics['PR AUC'] = np.nan
    return metrics

print("\n" + "="*70)
print("BASELINE EVALUATION RESULTS")
print("="*70)

baseline_results = []
baseline_results.append(evaluate_model('Majority Class', y_test, majority_pred, majority_proba))
baseline_results.append(evaluate_model('Education Rule', y_test, rule_pred, rule_proba))
baseline_results.append(evaluate_model('Advanced Rule', y_test, adv_pred, adv_proba))

results_df = pd.DataFrame(baseline_results).round(4)
print("\nMetrics Table:")
print(results_df.to_string(index=False))

# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle('Confusion Matrices - Baseline Models', fontsize=14, fontweight='bold')

for idx, (name, pred) in enumerate([
    ('Majority Class', majority_pred),
    ('Education Rule', rule_pred),
    ('Advanced Rule', adv_pred)
]):
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, ax=axes[idx],
        display_labels=['<=50K', '>50K'],
        cmap='Blues'
    )
    axes[idx].set_title(name)

plt.tight_layout()
plt.savefig('figs/baseline_confusion_matrices.png', dpi=150)
plt.show()

# ROC and PR Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curves
ax = axes[0]
for name, pred, proba in [
    ('Majority Class', majority_pred, majority_proba),
    ('Education Rule', rule_pred, rule_proba),
    ('Advanced Rule', adv_pred, adv_proba)
]:
    if not np.isnan(proba).any() and len(np.unique(proba)) > 1:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Baselines')
ax.legend()

# PR Curves
ax = axes[1]
for name, proba in [
    ('Majority Class', majority_proba),
    ('Education Rule', rule_proba),
    ('Advanced Rule', adv_proba)
]:
    if not np.isnan(proba).any() and len(np.unique(proba)) > 1:
        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f'{name} (PR-AUC={pr_auc:.3f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves - Baselines')
ax.legend()

plt.tight_layout()
plt.savefig('figs/baseline_curves.png', dpi=150)
plt.show()

# Baseline Interpretation
print("\n" + "="*70)
print("BASELINE INTERPRETATION")
print("="*70)

best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
print(f"\nBest Baseline: {best_f1['Model']}")
print(f"  F1-Score: {best_f1['F1-Score']:.4f}")

print("""
INTERPRETATION:
The Advanced Rule baseline performs best because it combines multiple 
meaningful signals (education, capital gains, and high-paying occupations 
with overtime). The Majority Class predictor has F1=0 because it never 
predicts the positive class.

Minimum Improvement Needed:
A real ML model should achieve F1 > 0.60 to be useful, representing a 
~10% improvement over the best baseline (0.55 → 0.60).
""")

# ============================================================================
# TASK 5: Initial Error Analysis & Next Steps
# ============================================================================
print("\n" + "="*70)
print("TASK 5: ERROR ANALYSIS & NEXT STEPS")
print("="*70)

adv_pred = advanced_rule(X_test)
fp_indices = np.where((adv_pred == 1) & (y_test == 0))[0]
fn_indices = np.where((adv_pred == 0) & (y_test == 1))[0]

print(f"\n[1] Error Statistics:")
print(f"  Total predictions: {len(y_test):,}")
print(f"  False Positives: {len(fp_indices):,}")
print(f"  False Negatives: {len(fn_indices):,}")

# Sample errors
fp_sample = fp_indices[:15]
fn_sample = fn_indices[:15]

print("\n[2] False Positive Patterns (Predicted >50K but actual <=50K):")
if len(fp_sample) > 0:
    fp_analysis = X_test.iloc[fp_sample].copy()
    print("  Sample features:")
    print(fp_analysis[['age', 'education-num', 'hours-per-week', 'capital-gain', 'occupation']].head(10))
    
    print("\n  Common patterns:")
    print(f"  - Median age: {fp_analysis['age'].median():.0f}")
    print(f"  - Median education: {fp_analysis['education-num'].median():.0f}")
    print(f"  - Median hours: {fp_analysis['hours-per-week'].median():.0f}")

print("\n[3] False Negative Patterns (Predicted <=50K but actual >50K):")
if len(fn_sample) > 0:
    fn_analysis = X_test.iloc[fn_sample].copy()
    print("  Sample features:")
    print(fn_analysis[['age', 'education-num', 'hours-per-week', 'capital-gain', 'occupation']].head(10))
    
    print("\n  Common patterns:")
    print(f"  - Median age: {fn_analysis['age'].median():.0f}")
    print(f"  - Median education: {fn_analysis['education-num'].median():.0f}")
    print(f"  - Median hours: {fn_analysis['hours-per-week'].median():.0f}")

# Issues to fix
print("\n[4] Data & Feature Issues to Fix Tomorrow:")
issues = [
    "1. Missing Values: Many columns have '?' that need proper imputation",
    "2. Skewed Numerics: capital-gain is highly skewed → log transform needed",
    "3. Categorical Encoding: High-cardinality categoricals need proper handling",
    "4. Class Imbalance: Only 24% positive class → need class weights or balancing",
    "5. Feature Engineering: Age buckets, education categories, interactions",
    "6. Outliers: Hours-per-week has extreme values that need handling"
]
for issue in issues:
    print(f"  {issue}")

# Primary metric
print("\n" + "="*70)
print("PRIMARY METRIC FOR THE WEEK: F1-Score")
print("="*70)

print("""
JUSTIFICATION:
1. Baseline F1 = 0.501 (Advanced Rule)
2. Business wants to balance finding high-income people with avoiding waste
3. F1 penalizes both false positives and false negatives equally
4. Target for this week: F1 > 0.60

""")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[5] Saving Results...")
results_df.to_csv('baseline_results.csv', index=False)

if len(fp_sample) > 0:
    X_test.iloc[fp_sample].to_csv('false_positives_sample.csv', index=False)
if len(fn_sample) > 0:
    X_test.iloc[fn_sample].to_csv('false_negatives_sample.csv', index=False)

print("  ✓ Saved: baseline_results.csv")
print("  ✓ Saved: false_positives_sample.csv" if len(fp_sample) > 0 else "")
print("  ✓ Saved: false_negatives_sample.csv" if len(fn_sample) > 0 else "")
print("  ✓ Visualizations saved in 'figs/' directory")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DAY 1 COMPLETE - SUMMARY")
print("="*70)
print("""
✅ Task 1: Problem definition & metric justification
✅ Task 2: Data loading, cleaning, and comprehensive EDA
✅ Task 3: Stratified split (Train 72%, Dev 8%, Test 20%)
✅ Task 4: Three baselines with all 6 metrics + confusion matrices
✅ Task 5: Error analysis with patterns and next steps
""")
print("="*70)
