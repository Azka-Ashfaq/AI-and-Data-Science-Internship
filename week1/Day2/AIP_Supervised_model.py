"""
Adult Income Prediction - Complete Pipeline
This script trains Logistic Regression and Decision Tree models to predict income >50K
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
pd.set_option('display.max_columns', None)

# Create directories for saving outputs
os.makedirs('figs', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*60)
print("ADULT INCOME PREDICTION - COMPLETE PIPELINE")
print("="*60)

# -----------------------------------------
# STEP 1: Load and prepare the data
# -----------------------------------------
print("\n[1] Loading data from OpenML...")
data = fetch_openml('adult', version=2, as_frame=True)
df = data.frame

# Define column names
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
df.columns = column_names

# Handle missing values (dataset uses '?' or ' ?')
df = df.replace(' ?', np.nan)
df = df.replace('?', np.nan)

# Clean the target variable
df['income'] = df['income'].str.replace('.', '', regex=False).str.strip()

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Show class distribution
print("\nClass distribution:")
print(df['income'].value_counts())
print(f"\nPercentage earning >50K: {(df['income'] == '>50K').mean() * 100:.1f}%")

# Show missing values
missing = df.isna().sum()
if missing.sum() > 0:
    print("\nMissing values per column:")
    print(missing[missing > 0])

# -----------------------------------------
# STEP 2: Split data into train/test sets
# -----------------------------------------
print("\n[2] Splitting data into train/test sets...")
y = (df['income'] == '>50K').astype(int)  # 1 = >50K, 0 = <=50K
X = df.drop(columns=['income'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")
print(f"Positive rate - Train: {y_train.mean():.3f}")
print(f"Positive rate - Test: {y_test.mean():.3f}")

# -----------------------------------------
# STEP 3: Define preprocessing pipelines
# -----------------------------------------
print("\n[3] Setting up preprocessing pipelines...")

# Define feature types
numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                    'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 
                        'occupation', 'relationship', 'race', 'sex', 
                        'native-country']

# Numeric pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# -----------------------------------------
# STEP 4: Create and evaluate baselines
# -----------------------------------------
print("\n[4] Evaluating baseline models...")

# Baseline 1: Always predict majority class
majority_model = DummyClassifier(strategy='most_frequent')
majority_model.fit(X_train, y_train)
majority_pred = majority_model.predict(X_test)

# Baseline 2: Simple rule (education >= 13)
def simple_rule_single(data):
    condition = (data['education-num'] >= 13)
    return condition.astype(int)  # Returns 1 for >50K, 0 for <=50K

rule_pred_single = simple_rule_single(X_test)

# Baseline 3: Advanced rule with multiple features
def simple_rule_advanced(data):
    condition = ((data['education-num'] >= 13) | 
                 (data['capital-gain'] > 0) |
                 ((data['hours-per-week'] > 40) & 
                  (data['occupation'].isin(['Exec-managerial', 'Prof-specialty', 
                                            'Tech-support']))))
    return condition.astype(int)

rule_pred_advanced = simple_rule_advanced(X_test)

# Evaluate baselines
def evaluate_baseline(name, y_true, y_pred, y_proba=None):
    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    return metrics

baseline_results = [
    evaluate_baseline('Baseline 1: Majority Class', y_test, majority_pred),
    evaluate_baseline('Baseline 2: Education Rule', y_test, rule_pred_single),
    evaluate_baseline('Baseline 3: Advanced Rule', y_test, rule_pred_advanced),
]

print("\nBaseline Performance:")
baseline_df = pd.DataFrame(baseline_results).set_index('model').round(4)
print(baseline_df)

# -----------------------------------------
# STEP 5: Train machine learning models
# -----------------------------------------
print("\n[5] Training machine learning models...")

# Logistic Regression
log_reg = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, 
                                random_state=RANDOM_STATE, 
                                class_weight='balanced'))  # Handle class imbalance
])

# Decision Tree
tree = Pipeline([
    ('prep', preprocessor),
    ('clf', DecisionTreeClassifier(max_depth=10,  # Limit depth to prevent overfitting
                                   min_samples_split=100,
                                   min_samples_leaf=50,
                                   random_state=RANDOM_STATE))
])

# Train models
log_reg.fit(X_train, y_train)
tree.fit(X_train, y_train)
print("✓ Models trained successfully!")

# -----------------------------------------
# STEP 6: Evaluate models
# -----------------------------------------
print("\n[6] Evaluating models on test set...")

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
    }

# Evaluate models
model_results = baseline_results + [
    evaluate_model('Logistic Regression', log_reg, X_test, y_test),
    evaluate_model('Decision Tree', tree, X_test, y_test),
]

# Create comparison table
comparison_df = pd.DataFrame(model_results).set_index('model').round(4)
print("\n" + "="*60)
print("MODEL COMPARISON (Test Set)")
print("="*60)
print(comparison_df)

# Identify best model
best_f1 = comparison_df['f1'].max()
best_model = comparison_df['f1'].idxmax()
print(f"\n✓ Best model by F1-score: {best_model} (F1 = {best_f1:.4f})")

# -----------------------------------------
# STEP 7: Visualizations
# -----------------------------------------
print("\n[7] Creating visualizations...")

# 1. ROC Curves
fig, ax = plt.subplots(figsize=(7, 6))
for model in [log_reg, tree]:
    name = 'Logistic Regression' if model == log_reg else 'Decision Tree'
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (AUC=0.5)')
ax.set_title('ROC Curves - Test Set', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/roc_curves.png', dpi=150)
plt.show()

# 2. Precision-Recall Curves
fig, ax = plt.subplots(figsize=(7, 6))
for model in [log_reg, tree]:
    name = 'Logistic Regression' if model == log_reg else 'Decision Tree'
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.set_title('Precision-Recall Curves - Test Set', fontsize=14, fontweight='bold')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/pr_curves.png', dpi=150)
plt.show()

# 3. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (model, name) in enumerate([(log_reg, 'Logistic Regression'), 
                                      (tree, 'Decision Tree')]):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, 
        ax=axes[idx],
        display_labels=['<=50K', '>50K'], 
        colorbar=False,
        cmap='Blues'
    )
    axes[idx].set_title(name, fontsize=12, fontweight='bold')
plt.suptitle('Confusion Matrices - Test Set', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figs/confusion_matrices.png', dpi=150)
plt.show()

# 4. Feature Importance (Logistic Regression)
feature_names = log_reg.named_steps['prep'].get_feature_names_out()
coefs = log_reg.named_steps['clf'].coef_[0]

coef_df = pd.DataFrame({
    'feature': feature_names, 
    'coefficient': coefs
}).sort_values('coefficient', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top positive coefficients
top_positive = coef_df.head(10)
axes[0].barh(top_positive['feature'], top_positive['coefficient'], color='green')
axes[0].set_title('Top 10 Features - Higher Income (>50K)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Coefficient')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3)

# Top negative coefficients
top_negative = coef_df.tail(10)
axes[1].barh(top_negative['feature'], top_negative['coefficient'], color='red')
axes[1].set_title('Top 10 Features - Lower Income (<=50K)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Coefficient')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Logistic Regression Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figs/feature_importance.png', dpi=150)
plt.show()

# 5. Decision Tree Visualization (simplified)
plt.figure(figsize=(20, 10))
plot_tree(tree.named_steps['clf'], 
          feature_names=feature_names,
          class_names=['<=50K', '>50K'],
          filled=True, 
          rounded=True,
          max_depth=3,  # Show only top 3 levels for clarity
          fontsize=8)
plt.title('Decision Tree Structure (First 3 Levels)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figs/decision_tree.png', dpi=150)
plt.show()

# 6. Model Performance Comparison Bar Chart
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
models_to_plot = ['Logistic Regression', 'Decision Tree', 'Baseline 3: Advanced Rule']

plot_data = comparison_df.loc[models_to_plot, metrics_to_plot]

fig, ax = plt.subplots(figsize=(12, 6))
plot_data.plot(kind='bar', ax=ax, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figs/model_comparison.png', dpi=150)
plt.show()

# -----------------------------------------
# STEP 8: Model Analysis
# -----------------------------------------
print("\n[8] Model Analysis...")

# Logistic Regression analysis
print("\n" + "="*60)
print("LOGISTIC REGRESSION ANALYSIS")
print("="*60)
print(f"Top 5 features for >50K:")
print(coef_df.head(5).to_string(index=False))
print(f"\nTop 5 features for <=50K:")
print(coef_df.tail(5).to_string(index=False))

# Decision Tree analysis
tree_clf = tree.named_steps['clf']
train_score = tree.score(X_train, y_train)
test_score = tree.score(X_test, y_test)

print("\n" + "="*60)
print("DECISION TREE ANALYSIS")
print("="*60)
print(f"Tree depth: {tree_clf.get_depth()}")
print(f"Number of leaves: {tree_clf.get_n_leaves()}")
print(f"Train accuracy: {train_score:.4f}")
print(f"Test accuracy: {test_score:.4f}")
print(f"Overfitting gap: {train_score - test_score:.4f}")

if train_score - test_score > 0.1:
    print("⚠️  Warning: Significant overfitting detected!")
else:
    print("✓ Model shows good generalization")

# -----------------------------------------
# STEP 9: Save models
# -----------------------------------------
print("\n[9] Saving models...")
joblib.dump(preprocessor, 'models/preprocessor.joblib')
joblib.dump(log_reg, 'models/log_reg_pipeline.joblib')
joblib.dump(tree, 'models/tree_pipeline.joblib')
print("✓ Models saved to 'models/' directory")

# Save comparison table
comparison_df.to_csv('models/model_comparison.csv')
print("✓ Comparison table saved to 'models/model_comparison.csv'")

# -----------------------------------------
# STEP 10: Summary
# -----------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Best performing model: {best_model}")
print(f"✓ Best F1-score: {best_f1:.4f}")
print(f"✓ Best ROC-AUC: {comparison_df['roc_auc'].max():.4f}")
print(f"✓ Models saved in 'models/' directory")
print(f"✓ Visualizations saved in 'figs/' directory")
print("\nGoal achieved!" if best_f1 > 0.60 else "\nGoal not yet achieved. Consider improving models.")
print("="*60)