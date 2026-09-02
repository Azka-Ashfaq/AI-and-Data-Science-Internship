import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
pd.set_option('display.max_columns', None)

# Create figures directory if it doesn't exist
os.makedirs('figs', exist_ok=True)

print("="*60)
print("ADULT INCOME PREDICTION - TASK 3: FEATURE ENGINEERING & MODEL COMPARISON")
print("="*60)

# -----------------------------------------
# STEP 1: Load the data using fetch_openml
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

# Handle missing values
df = df.replace(' ?', np.nan)
df = df.replace('?', np.nan)

# Clean the target variable
df['income'] = df['income'].str.replace('.', '', regex=False).str.strip()

# Prepare features and target
y = (df['income'] == '>50K').astype(int)  # 1 = >50K, 0 = <=50K
X = df.drop(columns=['income'])

print(f"Dataset shape: {X.shape}")
print(f"Positive rate (>50K): {y.mean():.3f}")

# -----------------------------------------
# STEP 2: Feature Engineering Function
# -----------------------------------------
print("\n[2] Creating engineered features...")

def engineer_features(X_in):
    X = X_in.copy()
    # Age buckets
    X['age_bucket'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 65, 120],
                              labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'], right=False)
    # Hours buckets
    X['hours_bin'] = pd.cut(X['hours-per-week'], bins=[0, 35, 40, 200],
                             labels=['part_time', 'full_time', 'overtime'], include_lowest=True)
    # Capital gain features
    X['has_capital_gain'] = (X['capital-gain'] > 0).astype(int)
    X['log_capital_gain'] = np.log1p(X['capital-gain'])
    # Education features
    X['higher_ed'] = (X['education-num'] >= 13).astype(int)
    # Interaction feature
    X['edu_hours_interaction'] = X['education-num'] * X['hours-per-week']
    return X

X_eng = engineer_features(X)

# Show engineered features
eng_feats = ['age_bucket', 'hours_bin', 'has_capital_gain', 'log_capital_gain', 
             'higher_ed', 'edu_hours_interaction']
print("\nEngineered features sample:")
print(X_eng[eng_feats].head())

# -----------------------------------------
# STEP 3: Mutual Information Analysis
# -----------------------------------------
print("\n[3] Computing Mutual Information scores...")

# Prepare data for mutual information
mi_input = X_eng[eng_feats].copy()
mi_input['age_bucket'] = LabelEncoder().fit_transform(mi_input['age_bucket'].astype(str))
mi_input['hours_bin'] = LabelEncoder().fit_transform(mi_input['hours_bin'].astype(str))

discrete_mask = [f in ('age_bucket', 'hours_bin', 'has_capital_gain', 'higher_ed') for f in eng_feats]

mi_scores = mutual_info_classif(mi_input, y, discrete_features=discrete_mask, random_state=RANDOM_STATE)

feature_dictionary = pd.DataFrame({
    'feature': eng_feats,
    'type': ['categorical', 'categorical', 'boolean', 'numeric', 'boolean', 'numeric'],
    'mutual_info': mi_scores,
}).sort_values('mutual_info', ascending=False).reset_index(drop=True)

print("\nMutual Information scores:")
print(feature_dictionary)

# -----------------------------------------
# STEP 4: Analyze target rate by engineered features
# -----------------------------------------
print("\n[4] Analyzing target rates by feature categories...")

print("\nAge Bucket Analysis:")
print(X_eng.groupby('age_bucket', observed=True).apply(lambda g: y.loc[g.index].mean()).rename('>50K rate'))

print("\nHours Bin Analysis:")
print(X_eng.groupby('hours_bin', observed=True).apply(lambda g: y.loc[g.index].mean()).rename('>50K rate'))

# -----------------------------------------
# STEP 5: Define preprocessing pipelines
# -----------------------------------------
print("\n[5] Setting up preprocessing pipelines...")

numeric_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'has_capital_gain', 'log_capital_gain', 'higher_ed', 'edu_hours_interaction']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'native-country', 'age_bucket', 'hours_bin']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

feature_engineering = FunctionTransformer(engineer_features)

print("Pipeline stages: feature_engineering -> preprocessor -> [model]")

# -----------------------------------------
# STEP 6: Cross-validation of models
# -----------------------------------------
print("\n[6] Performing cross-validation...")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=RANDOM_STATE),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ['accuracy', 'roc_auc', 'f1']

cv_results = {}
for name, clf in models.items():
    pipe = Pipeline([('feat_eng', feature_engineering), ('prep', preprocessor), ('clf', clf)])
    t0 = time.time()
    cv_results[name] = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_train_score=False)
    print(f"{name}: fit+score for 5 folds took {time.time()-t0:.1f}s")

# Summary of results
summary_rows = []
for name, res in cv_results.items():
    summary_rows.append({
        'model': name,
        'accuracy_mean': res['test_accuracy'].mean(), 'accuracy_std': res['test_accuracy'].std(),
        'roc_auc_mean': res['test_roc_auc'].mean(), 'roc_auc_std': res['test_roc_auc'].std(),
        'f1_mean': res['test_f1'].mean(), 'f1_std': res['test_f1'].std(),
    })
cv_summary = pd.DataFrame(summary_rows).set_index('model').round(4)

print("\nCross-validation Results:")
print(cv_summary)

# -----------------------------------------
# STEP 7: Visualization - Box plots
# -----------------------------------------
print("\n[7] Creating visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, metric in zip(axes, ['test_accuracy', 'test_roc_auc', 'test_f1']):
    data = [cv_results[name][metric] for name in models]
    ax.boxplot(data, tick_labels=list(models.keys()))
    ax.set_title(metric.replace('test_', '').upper())
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3)
plt.suptitle('Model Performance Comparison (5-fold CV)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figs/cv_boxplots.png', dpi=150)
plt.show()

# -----------------------------------------
# STEP 8: Statistical Significance Tests
# -----------------------------------------
print("\n[8] Statistical significance testing...")

means = {name: res['test_accuracy'].mean() for name, res in cv_results.items()}
ranked = sorted(means, key=means.get, reverse=True)
top2 = ranked[:2]
a, b = cv_results[top2[0]]['test_accuracy'], cv_results[top2[1]]['test_accuracy']

t_stat, p_val_t = stats.ttest_rel(a, b)
w_stat, p_val_w = stats.wilcoxon(a, b)

print(f"Top 2 models by mean accuracy: {top2[0]} ({a.mean():.4f}) vs {top2[1]} ({b.mean():.4f})")
print(f"Paired t-test:  t={t_stat:.3f}, p={p_val_t:.5f}")
print(f"Wilcoxon test:  W={w_stat:.3f}, p={p_val_w:.4f}")

if p_val_t < 0.05:
    print(f"✓ Significant difference between {top2[0]} and {top2[1]} (p < 0.05)")
else:
    print(f"✗ No significant difference between {top2[0]} and {top2[1]} (p >= 0.05)")

# -----------------------------------------
# STEP 9: Feature Importance Analysis
# -----------------------------------------
print("\n[9] Analyzing feature importance...")

# Split data for model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# Logistic Regression coefficients
print("\n--- Logistic Regression Coefficients ---")
lr_pipe = Pipeline([('feat_eng', feature_engineering), ('prep', preprocessor),
                     ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]).fit(X_train, y_train)

feat_names = lr_pipe.named_steps['prep'].get_feature_names_out()
lr_coefs = lr_pipe.named_steps['clf'].coef_[0]

engineered_markers = ['has_capital_gain', 'log_capital_gain', 'higher_ed', 'edu_hours_interaction', 'age_bucket', 'hours_bin']
eng_mask = np.array([any(m in f for m in engineered_markers) for f in feat_names])

lr_eng = pd.DataFrame({'feature': feat_names[eng_mask], 'coefficient': lr_coefs[eng_mask]})
lr_eng = lr_eng.sort_values('coefficient', key=abs, ascending=False).reset_index(drop=True)
print("Top engineered features from Logistic Regression:")
print(lr_eng.head(10))

# RandomForest feature importances
print("\n--- RandomForest Feature Importances ---")
rf_pipe = Pipeline([('feat_eng', feature_engineering), ('prep', preprocessor),
                     ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1))]).fit(X_train, y_train)

rf_feat_names = rf_pipe.named_steps['prep'].get_feature_names_out()
rf_importances = rf_pipe.named_steps['clf'].feature_importances_
rf_eng_mask = np.array([any(m in f for m in engineered_markers) for f in rf_feat_names])

rf_all = pd.DataFrame({'feature': rf_feat_names, 'importance': rf_importances}).sort_values('importance', ascending=False).reset_index(drop=True)
rf_all['rank'] = rf_all.index + 1

rf_eng = rf_all[rf_all['feature'].isin(rf_feat_names[rf_eng_mask])].reset_index(drop=True)
print(f"(RandomForest has {len(rf_feat_names)} total post-encoding features)")
print("Top engineered features from RandomForest:")
print(rf_eng.head(10))

# -----------------------------------------
# STEP 10: Feature Selection Comparison
# -----------------------------------------
print("\n[10] Feature selection comparison...")

# Full feature set
pipe_full = Pipeline([('feat_eng', feature_engineering), ('prep', preprocessor),
                       ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))])
t0 = time.time()
res_full = cross_validate(pipe_full, X, y, cv=skf, scoring=scoring)
time_full = time.time() - t0

# Selected features (k=40)
pipe_selected = Pipeline([('feat_eng', feature_engineering), ('prep', preprocessor),
                           ('select', SelectKBest(mutual_info_classif, k=40)),
                           ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))])
t0 = time.time()
res_selected = cross_validate(pipe_selected, X, y, cv=skf, scoring=scoring)
time_selected = time.time() - t0

selection_comparison = pd.DataFrame({
    'accuracy': [res_full['test_accuracy'].mean(), res_selected['test_accuracy'].mean()],
    'roc_auc': [res_full['test_roc_auc'].mean(), res_selected['test_roc_auc'].mean()],
    'f1': [res_full['test_f1'].mean(), res_selected['test_f1'].mean()],
    'total_cv_time_sec': [round(time_full, 1), round(time_selected, 1)],
}, index=['Full feature set (~118 cols)', 'SelectKBest k=40'])

print("\nFeature Selection Comparison:")
print(selection_comparison.round(4))

# -----------------------------------------
# STEP 11: Summary
# -----------------------------------------
print("\n" + "="*60)
print("SUMMARY - TASK 3 COMPLETE")
print("="*60)
print(f"✓ Best performing model: {ranked[0]} (Accuracy: {means[ranked[0]]:.4f})")
print(f"✓ Best F1-score: {cv_summary['f1_mean'].max():.4f}")
print(f"✓ Best ROC-AUC: {cv_summary['roc_auc_mean'].max():.4f}")
print(f"✓ Feature engineering added {len(eng_feats)} new features")
print(f"✓ Visualizations saved in 'figs/' directory")
print("="*60)