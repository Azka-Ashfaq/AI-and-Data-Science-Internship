"""
Day 4: Model Tuning, Regularization & Reproducible Pipelines
Adult Income Prediction - Hyperparameter Tuning & Model Selection

TASKS:
1. Build Fully Reproducible Pipelines
2. Hyperparameter Search (3 models)
3. Diagnose Overfitting/Underfitting
4. Probability Calibration & Threshold Selection
5. Final Evaluation & Save Artifacts
"""

import os
import time
import warnings
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import sklearn
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate,
    cross_val_predict, learning_curve, validation_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay
)
from scipy.stats import loguniform, randint, uniform

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42          # For reproducibility
N_ITER = 20                # RandomizedSearchCV iterations (reduced for speed)
SEARCH_SUBSAMPLE = 15000   # Use subset for faster search (None = full data)
CV_FOLDS = 5               # Number of cross-validation folds

# Create directories for outputs
os.makedirs('figs', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

print("="*70)
print("DAY 4: MODEL TUNING, REGULARIZATION & REPRODUCIBLE PIPELINES")
print("="*70)

print("\nLibrary versions:")
print(f"  python:     {platform.python_version()}")
print(f"  numpy:      {np.__version__}")
print(f"  pandas:     {pd.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  scipy:      {scipy.__version__}")

# ============================================================================
# TASK 1: Build Fully Reproducible Pipelines
# ============================================================================
print("\n" + "="*70)
print("TASK 1: BUILD REPRODUCIBLE PIPELINES")
print("="*70)
print("Building pipelines with feature engineering + preprocessing + classifier")
print("All components use random_state=42 for reproducibility")

def load_data():
    """
    Load Adult dataset from OpenML
    Handles missing values and converts target to binary
    """
    print("\nLoading data from OpenML...")
    data = fetch_openml('adult', version=2, as_frame=True)
    df = data.frame
    
    # Handle missing values (dataset uses '?' and ' ?')
    df = df.replace(' ?', np.nan)
    df = df.replace('?', np.nan)
    
    # Clean target column (remove trailing dots)
    df['class'] = df['class'].str.replace('.', '', regex=False).str.strip()
    
    # Convert target to binary (1 = >50K, 0 = <=50K)
    y = (df['class'] == '>50K').astype(int)
    X = df.drop(columns=['class'])
    
    print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.3f} ({y.sum():,} people earn >50K)")
    
    return X, y

def engineer_features(X_in):
    """
    Feature engineering - creates 6 new features from existing ones
    All features use only current-row data (no leakage)
    """
    X = X_in.copy()
    
    # 1. Age buckets - captures non-linear age effects
    X['age_bucket'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 65, 120],
                              labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'], 
                              right=False)
    
    # 2. Hours buckets - part-time vs full-time vs overtime
    X['hours_bin'] = pd.cut(X['hours-per-week'], bins=[0, 35, 40, 200],
                             labels=['part_time', 'full_time', 'overtime'], 
                             include_lowest=True)
    
    # 3. Capital gain flag - indicates investment income
    X['has_capital_gain'] = (X['capital-gain'] > 0).astype(int)
    
    # 4. Log capital gain - handles skewness
    X['log_capital_gain'] = np.log1p(X['capital-gain'])
    
    # 5. Higher education flag - education level indicator
    X['higher_ed'] = (X['education-num'] >= 13).astype(int)
    
    # 6. Interaction feature - education × hours (captures "educated and works hard")
    X['edu_hours_interaction'] = X['education-num'] * X['hours-per-week']
    
    return X

# Define feature lists for preprocessing
# NOTE: These are the features that will be available AFTER feature engineering
NUMERIC_FEATURES = [
    'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 
    'hours-per-week', 'has_capital_gain', 'log_capital_gain', 
    'higher_ed', 'edu_hours_interaction'
]

CATEGORICAL_FEATURES = [
    'workclass', 'education', 'marital-status', 'occupation', 
    'relationship', 'race', 'sex', 'native-country', 
    'age_bucket', 'hours_bin'
]

def build_preprocessor():
    """
    Build preprocessing pipeline:
    - Numeric: median imputation + StandardScaler
    - Categorical: most_frequent imputation + OneHotEncoder
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ])

def build_pipeline(estimator):
    """
    Build complete pipeline: feature_engineering -> preprocessor -> classifier
    This ensures all models use the exact same preprocessing
    """
    return Pipeline([
        ('feat_eng', FunctionTransformer(engineer_features)),
        ('prep', build_preprocessor()),
        ('clf', estimator),
    ])

# Load data and create reproducible splits
X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nData splits (stratified on target):")
print(f"  Train set: {X_train.shape[0]:,} samples")
print(f"  Test set:  {X_test.shape[0]:,} samples (UNTAPPED until final evaluation)")
print(f"  Train positive rate: {y_train.mean():.3f}")
print(f"  Test positive rate:  {y_test.mean():.3f}")

# Cross-validation strategy for all tasks
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
print(f"\nCross-validation: {CV_FOLDS}-fold StratifiedKFold with random_state={RANDOM_STATE}")

# ============================================================================
# TASK 2: Hyperparameter Search
# ============================================================================
print("\n" + "="*70)
print("TASK 2: HYPERPARAMETER SEARCH")
print("="*70)
print(f"Searching {N_ITER} random combinations for each model")
print("Using ROC AUC as optimization metric (handles class imbalance)")

# Use subsample for faster search (if configured)
if SEARCH_SUBSAMPLE:
    X_search, _, y_search, _ = train_test_split(
        X_train, y_train, train_size=SEARCH_SUBSAMPLE, 
        random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"\nUsing {X_search.shape[0]:,} samples for search (speed optimization)")
else:
    X_search, y_search = X_train, y_train
    print(f"\nUsing full training set ({X_search.shape[0]:,} samples) for search")

# Store all search results for comparison
search_results = {}

# --- Model 1: Logistic Regression ---
print("\n[1] Tuning Logistic Regression...")
print("  Parameters: penalty (L1/L2), C (regularization strength)")

lr_pipe = build_pipeline(LogisticRegression(
    max_iter=1000, 
    random_state=RANDOM_STATE, 
    solver='saga',      # Supports both L1 and L2, faster than liblinear
    n_jobs=-1           # Use all CPU cores
))

lr_param_dist = {
    'clf__penalty': ['l1', 'l2'],
    'clf__C': loguniform(1e-3, 1e2),  # Log-uniform sampling (better for regularization)
}

lr_search = RandomizedSearchCV(
    lr_pipe, lr_param_dist, 
    n_iter=N_ITER, 
    cv=skf, 
    scoring='roc_auc', 
    random_state=RANDOM_STATE, 
    n_jobs=-1,          # Parallelize CV folds
    refit=True,
    verbose=0
)

t0 = time.time()
lr_search.fit(X_search, y_search)
lr_time = time.time() - t0

print(f"  ✅ Completed in {lr_time:.1f}s")
print(f"  Best params: {lr_search.best_params_}")
print(f"  Best CV ROC AUC: {lr_search.best_score_:.4f}")
search_results['LogisticRegression'] = lr_search

# --- Model 2: Random Forest ---
print("\n[2] Tuning Random Forest...")
print("  Parameters: n_estimators, max_depth, min_samples_leaf, max_features")

rf_pipe = build_pipeline(RandomForestClassifier(
    random_state=RANDOM_STATE, 
    n_jobs=-1
))

rf_param_dist = {
    'clf__n_estimators': randint(100, 300),      # Number of trees
    'clf__max_depth': randint(4, 15),            # Tree depth
    'clf__min_samples_leaf': randint(1, 10),     # Minimum samples per leaf
    'clf__max_features': ['sqrt', 'log2'],       # Features per split
}

rf_search = RandomizedSearchCV(
    rf_pipe, rf_param_dist, 
    n_iter=N_ITER, 
    cv=skf,
    scoring='roc_auc', 
    random_state=RANDOM_STATE, 
    n_jobs=-1, 
    refit=True,
    verbose=0
)

t0 = time.time()
rf_search.fit(X_search, y_search)
rf_time = time.time() - t0

print(f"  ✅ Completed in {rf_time:.1f}s")
print(f"  Best params: {rf_search.best_params_}")
print(f"  Best CV ROC AUC: {rf_search.best_score_:.4f}")
search_results['RandomForest'] = rf_search

# --- Model 3: HistGradientBoosting ---
print("\n[3] Tuning HistGradientBoosting...")
print("  Parameters: learning_rate, max_iter, max_depth, l2_regularization")

hgb_pipe = build_pipeline(HistGradientBoostingClassifier(
    random_state=RANDOM_STATE
))

hgb_param_dist = {
    'clf__learning_rate': loguniform(0.01, 0.3),  # Step size
    'clf__max_iter': randint(50, 200),            # Number of boosting rounds
    'clf__max_depth': randint(3, 10),             # Tree depth
    'clf__l2_regularization': uniform(0, 1),      # Regularization strength
}

hgb_search = RandomizedSearchCV(
    hgb_pipe, hgb_param_dist, 
    n_iter=N_ITER, 
    cv=skf,
    scoring='roc_auc', 
    random_state=RANDOM_STATE, 
    n_jobs=-1, 
    refit=True,
    verbose=0
)

t0 = time.time()
hgb_search.fit(X_search, y_search)
hgb_time = time.time() - t0

print(f"  ✅ Completed in {hgb_time:.1f}s")
print(f"  Best params: {hgb_search.best_params_}")
print(f"  Best CV ROC AUC: {hgb_search.best_score_:.4f}")
search_results['HistGradientBoosting'] = hgb_search

# Summary of best parameters
best_params = {
    name: {k.replace('clf__', ''): v for k, v in s.best_params_.items()} 
    for name, s in search_results.items()
}

print("\n" + "-"*70)
print("BEST PARAMETERS SUMMARY:")
for model, params in best_params.items():
    print(f"  {model}:")
    for param, value in params.items():
        print(f"    {param}: {value}")

# ============================================================================
# REFIT TUNED MODELS ON FULL TRAINING SET
# ============================================================================
print("\n" + "="*70)
print("REFITTING TUNED MODELS ON FULL TRAINING SET")
print("="*70)
print("Cross-validating tuned models on full training data")

# Create tuned estimators with best parameters
tuned_estimators = {
    'LogisticRegression': LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE,
        solver='saga', n_jobs=-1,
        **best_params['LogisticRegression']
    ),
    'RandomForest': RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1,
        **best_params['RandomForest']
    ),
    'HistGradientBoosting': HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        **best_params['HistGradientBoosting']
    ),
}

# Cross-validate tuned models
tuned_cv_results = {}
for name, est in tuned_estimators.items():
    print(f"\n{name}:")
    pipe = build_pipeline(est)
    res = cross_validate(pipe, X_train, y_train, cv=skf, 
                         scoring=['accuracy', 'roc_auc', 'f1'])
    tuned_cv_results[name] = res
    
    print(f"  Accuracy: {res['test_accuracy'].mean():.4f} ± {res['test_accuracy'].std():.4f}")
    print(f"  ROC AUC:  {res['test_roc_auc'].mean():.4f} ± {res['test_roc_auc'].std():.4f}")
    print(f"  F1:       {res['test_f1'].mean():.4f} ± {res['test_f1'].std():.4f}")

# Select final model based on best ROC AUC
FINAL_MODEL_NAME = max(tuned_cv_results, key=lambda n: tuned_cv_results[n]['test_roc_auc'].mean())
print(f"\n🏆 FINAL MODEL SELECTED: {FINAL_MODEL_NAME}")
print(f"  ROC AUC: {tuned_cv_results[FINAL_MODEL_NAME]['test_roc_auc'].mean():.4f}")

# ============================================================================
# TASK 3: Diagnose Overfitting / Underfitting
# ============================================================================
print("\n" + "="*70)
print("TASK 3: DIAGNOSE OVERFITTING / UNDERFITTING")
print("="*70)
print("Using learning curves and validation curves to diagnose model behavior")

# --- Learning Curve ---
print("\n[1] Learning Curve - Training Size vs Performance")
print("  Shows if model benefits from more data or is overfitting")

final_pipe_for_curve = build_pipeline(tuned_estimators[FINAL_MODEL_NAME])

train_sizes, train_scores, val_scores = learning_curve(
    final_pipe_for_curve, X_train, y_train, 
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc', 
    train_sizes=np.linspace(0.1, 1.0, 6),  # 6 points from 10% to 100%
    n_jobs=-1
)

# Plot learning curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train ROC AUC', color='blue')
ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation ROC AUC', color='orange')
ax.fill_between(train_sizes, 
                train_scores.mean(1) - train_scores.std(1), 
                train_scores.mean(1) + train_scores.std(1), 
                alpha=0.15, color='blue')
ax.fill_between(train_sizes, 
                val_scores.mean(1) - val_scores.std(1), 
                val_scores.mean(1) + val_scores.std(1), 
                alpha=0.15, color='orange')
ax.set_xlabel('Training set size')
ax.set_ylabel('ROC AUC')
ax.set_title(f'Learning Curve - {FINAL_MODEL_NAME}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/learning_curve.png', dpi=150)
print("  ✓ Saved: figs/learning_curve.png")
plt.show()

# Calculate and interpret gap
gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
print(f"\n  Train-Val gap at full data: {gap:.4f}")
if gap < 0.03:
    print("  ✅ Small gap - low variance, model generalizes well")
elif gap < 0.08:
    print("  ⚠ Moderate gap - some overfitting, consider regularization")
else:
    print("  ❌ Large gap - significant overfitting, need more regularization or data")

# --- Validation Curve: Logistic Regression C ---
print("\n[2] Validation Curve - Logistic Regression (effect of C)")
print("  Shows how regularization strength affects performance")

# Use subset for speed
X_vc, _, y_vc, _ = train_test_split(
    X_train, y_train, train_size=min(15000, len(X_train)),
    random_state=RANDOM_STATE, stratify=y_train
)

lr_vc_pipe = build_pipeline(LogisticRegression(
    max_iter=500, random_state=RANDOM_STATE, solver='lbfgs', penalty='l2'
))

C_range = np.logspace(-3, 2, 8)  # 0.001 to 100
train_lr, val_lr = validation_curve(
    lr_vc_pipe, X_vc, y_vc, 
    param_name='clf__C', param_range=C_range,
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc', 
    n_jobs=-1
)

# Plot validation curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(C_range, train_lr.mean(1), 'o-', label='Train ROC AUC', color='blue')
ax.plot(C_range, val_lr.mean(1), 'o-', label='Validation ROC AUC', color='orange')
ax.set_xscale('log')
ax.set_xlabel('C (inverse regularization strength)')
ax.set_ylabel('ROC AUC')
ax.set_title('Validation Curve - Logistic Regression: effect of C')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/validation_curve_lr_C.png', dpi=150)
print("  ✓ Saved: figs/validation_curve_lr_C.png")
plt.show()

best_c_idx = np.argmax(val_lr.mean(1))
print(f"\n  Best C: {C_range[best_c_idx]:.4f}")
print(f"  Validation ROC AUC at best C: {val_lr.mean(1)[best_c_idx]:.4f}")
print("  Interpretation: Very small C underfits, very large C overfits")

# --- Validation Curve: Random Forest max_depth ---
print("\n[3] Validation Curve - Random Forest (effect of max_depth)")
print("  Shows how tree depth affects performance")

rf_vc_pipe = build_pipeline(RandomForestClassifier(
    n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1
))

depth_range = [2, 4, 6, 8, 10, 15, 20, None]
depth_labels = [str(d) for d in depth_range]
train_rf, val_rf = validation_curve(
    rf_vc_pipe, X_vc, y_vc, 
    param_name='clf__max_depth', param_range=depth_range,
    cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
    scoring='roc_auc', 
    n_jobs=-1
)

# Plot validation curve
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = range(len(depth_range))
ax.plot(x_pos, train_rf.mean(1), 'o-', label='Train ROC AUC', color='blue')
ax.plot(x_pos, val_rf.mean(1), 'o-', label='Validation ROC AUC', color='orange')
ax.set_xticks(list(x_pos))
ax.set_xticklabels(depth_labels)
ax.set_xlabel('max_depth')
ax.set_ylabel('ROC AUC')
ax.set_title('Validation Curve - Random Forest: effect of max_depth')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/validation_curve_rf_depth.png', dpi=150)
print("  ✓ Saved: figs/validation_curve_rf_depth.png")
plt.show()

best_depth_idx = np.argmax(val_rf.mean(1))
print(f"\n  Best max_depth: {depth_labels[best_depth_idx]}")
print(f"  Validation ROC AUC: {val_rf.mean(1)[best_depth_idx]:.4f}")
print("  Interpretation: Train score keeps climbing, validation peaks then drops - classic overfitting")

# ============================================================================
# TASK 4: Probability Calibration & Threshold Selection
# ============================================================================
print("\n" + "="*70)
print("TASK 4: PROBABILITY CALIBRATION & THRESHOLD SELECTION")
print("="*70)
print("Using cross-validated probabilities to calibrate and find optimal threshold")

# Get out-of-fold predictions (honest, no leakage)
print("\n[1] Generating out-of-fold probabilities...")
final_pipe_calib = build_pipeline(tuned_estimators[FINAL_MODEL_NAME])

proba_raw = cross_val_predict(
    final_pipe_calib, X_train, y_train, cv=skf, 
    method='predict_proba', n_jobs=-1
)[:, 1]

brier_raw = brier_score_loss(y_train, proba_raw)
print(f"  Brier score (uncalibrated): {brier_raw:.4f}")

# Calibrate using isotonic regression
print("\n[2] Calibrating probabilities with isotonic regression...")
calibrated_pipe = CalibratedClassifierCV(
    build_pipeline(tuned_estimators[FINAL_MODEL_NAME]), 
    method='isotonic', 
    cv=5
)

proba_cal = cross_val_predict(
    calibrated_pipe, X_train, y_train, cv=skf, 
    method='predict_proba', n_jobs=-1
)[:, 1]

brier_cal = brier_score_loss(y_train, proba_cal)
print(f"  Brier score (calibrated): {brier_cal:.4f}")

# Decide whether to use calibration (only if it improves Brier score)
USE_CALIBRATION = brier_cal < brier_raw - 0.001
print(f"\n  Using calibration: {USE_CALIBRATION}")
if USE_CALIBRATION:
    print("  ✅ Calibration improves Brier score")
else:
    print("  ℹ Calibration does not improve Brier score meaningfully")

# Calibration curve
print("\n[3] Plotting calibration curve...")
frac_pos_raw, mean_pred_raw = calibration_curve(
    y_train, proba_raw, n_bins=10, strategy='quantile'
)
frac_pos_cal, mean_pred_cal = calibration_curve(
    y_train, proba_cal, n_bins=10, strategy='quantile'
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
ax.plot(mean_pred_raw, frac_pos_raw, 'o-', 
        label=f'{FINAL_MODEL_NAME} (Brier={brier_raw:.4f})')
ax.plot(mean_pred_cal, frac_pos_cal, 's-', 
        label=f'+ isotonic (Brier={brier_cal:.4f})')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve (5-fold cross-validated predictions)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/calibration_curve.png', dpi=150)
print("  ✓ Saved: figs/calibration_curve.png")
plt.show()

# Threshold optimization
print("\n[4] Optimizing threshold for F1-score...")
proba_for_threshold = proba_cal if USE_CALIBRATION else proba_raw

thresholds = np.arange(0.10, 0.95, 0.05)
threshold_rows = []

for t in thresholds:
    pred = (proba_for_threshold >= t).astype(int)
    threshold_rows.append({
        'threshold': round(t, 2),
        'precision': precision_score(y_train, pred, zero_division=0),
        'recall': recall_score(y_train, pred, zero_division=0),
        'f1': f1_score(y_train, pred, zero_division=0),
    })

threshold_table = pd.DataFrame(threshold_rows)
print("\nThreshold analysis (on training set):")
print(threshold_table.round(4).to_string(index=False))

BEST_THRESHOLD = float(threshold_table.loc[threshold_table['f1'].idxmax(), 'threshold'])
print(f"\n🏆 Best threshold for F1: {BEST_THRESHOLD:.2f}")

# Compare thresholds
print("\n[5] Comparing default vs optimized threshold:")
for t, label in [(0.5, 'default'), (BEST_THRESHOLD, 'F1-optimized')]:
    pred = (proba_for_threshold >= t).astype(int)
    cm = confusion_matrix(y_train, pred)
    f1 = f1_score(y_train, pred)
    print(f"\nThreshold = {t:.2f} ({label}):")
    print(f"  F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

# ============================================================================
# TASK 5: Final Evaluation & Save Artifacts
# ============================================================================
print("\n" + "="*70)
print("TASK 5: FINAL EVALUATION ON HOLD-OUT TEST SET")
print("="*70)
print("Evaluating final model ONCE on untouched test set")

# Build final pipeline (with or without calibration)
if USE_CALIBRATION:
    final_pipeline = CalibratedClassifierCV(
        build_pipeline(tuned_estimators[FINAL_MODEL_NAME]), 
        method='isotonic', 
        cv=5
    )
else:
    final_pipeline = build_pipeline(tuned_estimators[FINAL_MODEL_NAME])

# Train on full training set
print("\n[1] Training final pipeline on full training set...")
t0 = time.time()
final_pipeline.fit(X_train, y_train)
train_time = time.time() - t0
print(f"  Training complete in {train_time:.1f}s")

# Evaluate on test set
print("\n[2] Evaluating on hold-out test set...")
proba_test = final_pipeline.predict_proba(X_test)[:, 1]
pred_default = (proba_test >= 0.5).astype(int)
pred_tuned = (proba_test >= BEST_THRESHOLD).astype(int)

def report_metrics(y_true, pred, proba):
    """Calculate all relevant metrics"""
    return {
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall': recall_score(y_true, pred, zero_division=0),
        'f1': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, proba),
        'pr_auc': average_precision_score(y_true, proba),
    }

metrics_default = report_metrics(y_test, pred_default, proba_test)
metrics_tuned = report_metrics(y_test, pred_tuned, proba_test)

final_metrics_table = pd.DataFrame([
    {'threshold': '0.50 (default)', **metrics_default},
    {'threshold': f'{BEST_THRESHOLD:.2f} (F1-optimized)', **metrics_tuned},
]).set_index('threshold').round(4)

print("\nFinal hold-out test metrics:")
print(final_metrics_table)

# Confusion matrices
print("\n[3] Confusion matrices:")
print("\nDefault threshold (0.5):")
cm_default = confusion_matrix(y_test, pred_default)
print(cm_default)
print(f"  True Negatives: {cm_default[0,0]:,}")
print(f"  False Positives: {cm_default[0,1]:,}")
print(f"  False Negatives: {cm_default[1,0]:,}")
print(f"  True Positives: {cm_default[1,1]:,}")

print("\nOptimized threshold (F1-optimal):")
cm_tuned = confusion_matrix(y_test, pred_tuned)
print(cm_tuned)
print(f"  True Negatives: {cm_tuned[0,0]:,}")
print(f"  False Positives: {cm_tuned[0,1]:,}")
print(f"  False Negatives: {cm_tuned[1,0]:,}")
print(f"  True Positives: {cm_tuned[1,1]:,}")

# ROC Curve
print("\n[4] Generating final ROC curve...")
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, proba_test, name=FINAL_MODEL_NAME, ax=ax)
ax.plot([0, 1], [0, 1], '--', color='gray', label='Random (AUC=0.5)')
ax.set_title(f'ROC Curve - {FINAL_MODEL_NAME} on Hold-out Test')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/final_roc_curve.png', dpi=150)
print("  ✓ Saved: figs/final_roc_curve.png")
plt.show()

# Precision-Recall Curve
print("\n[5] Generating final Precision-Recall curve...")
fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_predictions(y_test, proba_test, name=FINAL_MODEL_NAME, ax=ax)
ax.set_title(f'Precision-Recall Curve - {FINAL_MODEL_NAME} on Hold-out Test')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figs/final_pr_curve.png', dpi=150)
print("  ✓ Saved: figs/final_pr_curve.png")
plt.show()

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================
print("\n" + "="*70)
print("SAVING ARTIFACTS")
print("="*70)

# Save the final pipeline
joblib.dump(final_pipeline, 'artifacts/final_pipeline.joblib')
print("✓ Saved: artifacts/final_pipeline.joblib")

# ============================================================================
# FIX: Save preprocessor correctly - apply feature engineering first
# ============================================================================
print("\nSaving preprocessor...")
# Apply feature engineering first, then fit preprocessor
X_train_engineered = engineer_features(X_train)
preprocessor = build_preprocessor()
preprocessor.fit(X_train_engineered)
joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
print("✓ Saved: artifacts/preprocessor.joblib")

# Save best parameters
best_params_df = pd.DataFrame([
    {'model': name, 'params': str(params)} 
    for name, params in best_params.items()
])
best_params_df.to_csv('artifacts/best_params.csv', index=False)
print("✓ Saved: artifacts/best_params.csv")

# Save final metrics
final_metrics_table.to_csv('artifacts/final_metrics.csv')
print("✓ Saved: artifacts/final_metrics.csv")

# Save threshold info
threshold_info = pd.DataFrame({
    'threshold': [0.5, BEST_THRESHOLD],
    'type': ['default', 'F1-optimized'],
    'f1_on_train': [
        f1_score(y_train, (proba_for_threshold >= 0.5).astype(int)),
        f1_score(y_train, (proba_for_threshold >= BEST_THRESHOLD).astype(int))
    ]
})
threshold_info.to_csv('artifacts/threshold_info.csv', index=False)
print("✓ Saved: artifacts/threshold_info.csv")

# Save CV results
cv_results_df = pd.DataFrame([
    {
        'model': name,
        'accuracy_mean': res['test_accuracy'].mean(),
        'accuracy_std': res['test_accuracy'].std(),
        'roc_auc_mean': res['test_roc_auc'].mean(),
        'roc_auc_std': res['test_roc_auc'].std(),
        'f1_mean': res['test_f1'].mean(),
        'f1_std': res['test_f1'].std(),
    }
    for name, res in tuned_cv_results.items()
])
cv_results_df.to_csv('artifacts/cv_results.csv', index=False)
print("✓ Saved: artifacts/cv_results.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DAY 4 COMPLETE - SUMMARY")
print("="*70)

print(f"""
✅ TASK 1: Reproducible Pipelines
   - Feature engineering: 6 new features
   - Preprocessing: median impute, scale, one-hot encode
   - All components with random_state=42

✅ TASK 2: Hyperparameter Search
   - 3 models: LogisticRegression, RandomForest, HistGradientBoosting
   - {N_ITER} random combinations each
   - 5-fold StratifiedKFold CV
   - Optimized on ROC AUC

✅ TASK 3: Overfitting/Underfitting Diagnosis
   - Learning curve: train-val gap = {gap:.4f}
   - Validation curve: Logistic Regression (C)
   - Validation curve: Random Forest (max_depth)

✅ TASK 4: Probability Calibration & Threshold
   - Calibration: {'Used isotonic' if USE_CALIBRATION else 'Not used'}
   - Best threshold: {BEST_THRESHOLD:.2f} (F1-optimized)
   - Calibration curve saved

✅ TASK 5: Final Evaluation
   - Test set evaluated ONCE (no leakage)
   - ROC and PR curves generated
   - All artifacts saved to 'artifacts/'

🏆 BEST MODEL: {FINAL_MODEL_NAME}

FINAL TEST PERFORMANCE (F1-optimized threshold = {BEST_THRESHOLD:.2f}):
  - Accuracy:  {metrics_tuned['accuracy']:.4f}
  - Precision: {metrics_tuned['precision']:.4f}
  - Recall:    {metrics_tuned['recall']:.4f}
  - F1-Score:  {metrics_tuned['f1']:.4f}
  - ROC AUC:   {metrics_tuned['roc_auc']:.4f}
  - PR AUC:    {metrics_tuned['pr_auc']:.4f}

📁 FILES SAVED:
  figs/
    - learning_curve.png
    - validation_curve_lr_C.png
    - validation_curve_rf_depth.png
    - calibration_curve.png
    - final_roc_curve.png
    - final_pr_curve.png
  
  artifacts/
    - final_pipeline.joblib
    - preprocessor.joblib
    - best_params.csv
    - final_metrics.csv
    - threshold_info.csv
    - cv_results.csv

📋 NEXT STEPS (Day 5):
  - Final interpretation and business impact analysis
  - Deployment recommendations
  - Create final presentation
""")

print("="*70)
print("END OF DAY 4 IMPLEMENTATION")
print("="*70)