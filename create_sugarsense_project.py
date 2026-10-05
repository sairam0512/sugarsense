import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV, cross_val_predict, learning_curve
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
                             r2_score, mean_absolute_error)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

print("Starting SugarSense complete execution script...")

# ---------------------------------------------------------
# 1. DATA AUDIT & INITIAL SETUP
# ---------------------------------------------------------
data_path = 'C:/Users/psair/.gemini/antigravity/scratch/SugarSense/diabetes_screening_data.csv'
df_raw = pd.read_csv(data_path)

print(f"Data Shape: {df_raw.shape}")
print(f"Duplicates: {df_raw.duplicated().sum()}")
print(f"Class Balance:\n{df_raw['Outcome'].value_counts(normalize=True)}")

# Impossible zero features
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
missing_stats = {}
for col in zero_cols:
    zeros_count = (df_raw[col] == 0).sum()
    zeros_pct = (zeros_count / len(df_raw)) * 100
    missing_stats[col] = zeros_pct
    print(f"Column '{col}': {zeros_count} zeros ({zeros_pct:.2f}%)")

# Save missing percentage bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(missing_stats.keys(), missing_stats.values(), color='#e74c3c', edgecolor='black', alpha=0.85)
plt.title('Percentage of Impossible Zero Values (Missing Data) per Feature', fontsize=12, fontweight='bold')
plt.xlabel('Clinical Feature', fontsize=11)
plt.ylabel('Missing Percentage (%)', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/missing_values.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
df_clean = df_raw.copy()
# Count missing values before replacing with NaN
df_clean['Missing_Count'] = (df_clean[zero_cols] == 0).sum(axis=1)

# Replace impossible zeros with NaN in feature columns
for col in zero_cols:
    df_clean[col] = df_clean[col].replace(0, np.nan)

# Create domain-driven features
df_clean['Is_Obese'] = (df_clean['BMI'] >= 30.0).astype(float)
df_clean['Glucose_Age_Product'] = df_clean['Glucose'] * df_clean['Age']
df_clean['Risk_Pedigree_BMI'] = df_clean['DiabetesPedigreeFunction'] * df_clean['BMI']

print("Engineered columns created successfully.")

# Save Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df_clean.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar=True)
plt.title('Feature Correlation Matrix (Including Engineered Features)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/correlation_heatmap.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. TRAIN-TEST SPLIT & PIPELINES
# ---------------------------------------------------------
X = df_clean.drop(columns=['Outcome'])
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Define baseline model pipelines
pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

models = {
    'Logistic Regression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=7))
    ]),
    'Decision Tree': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42))
    ]),
    'Support Vector Machine': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', SVC(class_weight='balanced', probability=True, random_state=42))
    ])
}

# ---------------------------------------------------------
# 4. MODEL COMPARISON (5-FOLD CV)
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

comparison_results = []

for name, pipeline in models.items():
    cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    
    # Fit on full training set to get overall train score
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict_proba(X_train)[:, 1] if hasattr(pipeline, 'predict_proba') else pipeline.predict(X_train)
    train_auc = roc_auc_score(y_train, train_pred)
    
    val_auc_mean = cv_res['test_roc_auc'].mean()
    val_auc_std = cv_res['test_roc_auc'].std()
    
    comparison_results.append({
        'Model': name,
        'Accuracy': cv_res['test_accuracy'].mean(),
        'Precision': cv_res['test_precision'].mean(),
        'Recall': cv_res['test_recall'].mean(),
        'F1 Score': cv_res['test_f1'].mean(),
        'ROC-AUC (Mean)': val_auc_mean,
        'ROC-AUC (Std)': val_auc_std,
        'Train ROC-AUC': train_auc,
        'Overfit Gap': train_auc - val_auc_mean
    })

df_comp = pd.DataFrame(comparison_results).sort_values(by='ROC-AUC (Mean)', ascending=False)
print("\n--- MODEL COMPARISON TABLE (TRAINING 5-FOLD CV) ---")
print(df_comp.to_string(index=False))

# Plot Model Comparison Bar Chart
plt.figure(figsize=(12, 6))
x_axis = np.arange(len(df_comp))
width = 0.25

plt.bar(x_axis - width, df_comp['ROC-AUC (Mean)'], width, label='Val ROC-AUC', color='#2ecc71')
plt.bar(x_axis, df_comp['Recall'], width, label='Val Recall', color='#3498db')
plt.bar(x_axis + width, df_comp['Overfit Gap'], width, label='Overfit Gap', color='#e74c3c')

plt.xticks(x_axis, df_comp['Model'], rotation=15, ha='right', fontsize=10)
plt.ylabel('Score', fontsize=11)
plt.title('Baseline Model Performance & Overfit Gap Comparison (5-Fold CV)', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/model_comparison.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 5. HYPERPARAMETER TUNING
# ---------------------------------------------------------
print("\n--- TUNING HYPERPARAMETERS WITH RANDOMIZEDSEARCHCV ---")

# 1. Random Forest Tuning
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

rf_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 8, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2', 0.5]
}

rf_search = RandomizedSearchCV(rf_pipeline, rf_param_grid, n_iter=15, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)

print(f"Best RF Params: {rf_search.best_params_}")
print(f"Best RF CV ROC-AUC: {rf_search.best_score_:.4f}")

# 2. XGBoost Tuning
xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42))
])

xgb_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [2, 3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_param_grid, n_iter=15, cv=cv, scoring='roc_auc', random_state=42, n_jobs=-1)
xgb_search.fit(X_train, y_train)

print(f"Best XGB Params: {xgb_search.best_params_}")
print(f"Best XGB CV ROC-AUC: {xgb_search.best_score_:.4f}")

# Select the ultimate best tuned pipeline
if rf_search.best_score_ >= xgb_search.best_score_:
    best_pipeline = rf_search.best_estimator_
    best_model_name = "Tuned Random Forest"
else:
    best_pipeline = xgb_search.best_estimator_
    best_model_name = "Tuned XGBoost"

print(f"\nFinal Selected Best Model: {best_model_name}")

# Save tuned model to disk
joblib.dump(best_pipeline, 'C:/Users/psair/.gemini/antigravity/scratch/SugarSense/sugarsense_model.joblib')
print("Model pipeline saved to sugarsense_model.joblib")

# ---------------------------------------------------------
# 6. FINAL TEST EVALUATION
# ---------------------------------------------------------
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
y_test_pred_default = (y_test_proba >= 0.5).astype(int)

test_acc = accuracy_score(y_test, y_test_pred_default)
test_prec = precision_score(y_test, y_test_pred_default)
test_rec = recall_score(y_test, y_test_pred_default)
test_f1 = f1_score(y_test, y_test_pred_default)
test_auc = roc_auc_score(y_test, y_test_proba)

cm = confusion_matrix(y_test, y_test_pred_default)
fn = cm[1, 0]
fp = cm[0, 1]

print("\n--- FINAL TEST SET PERFORMANCE (Default 0.5 Threshold) ---")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1 Score:  {test_f1:.4f}")
print(f"ROC-AUC:   {test_auc:.4f}")
print(f"False Negatives (Missed Diabetic Patients): {fn}")
print(f"False Positives (Healthy Wrongly Flagged):  {fp}")

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetic', 'Diabetic'])
disp.plot(cmap='Blues', ax=ax)
plt.title(f'Test Confusion Matrix: {best_model_name} (Threshold 0.5)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/confusion_matrix_default.png', dpi=300)
plt.close()

# Plot ROC and Precision-Recall Curves
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_test_proba)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC Curve
axes[0].plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'ROC Curve (AUC = {test_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=11)
axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)

# Precision-Recall Curve
axes[1].plot(rec_vals, prec_vals, color='#8e44ad', lw=2.5, label='Precision-Recall Curve')
axes[1].set_xlabel('Recall (Sensitivity)', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/roc_pr_curves.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 7. THRESHOLD TUNING ("THINK LIKE A DOCTOR")
# ---------------------------------------------------------
print("\n--- THRESHOLD TUNING FOR CLINICAL RECALL TARGET >= 85% ---")
oof_proba = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

thresholds = np.linspace(0.01, 0.99, 200)
best_thresh = 0.5
best_prec_at_target_rec = 0.0

for t in thresholds:
    t_pred = (oof_proba >= t).astype(int)
    rec = recall_score(y_train, t_pred, zero_division=0)
    prec = precision_score(y_train, t_pred, zero_division=0)
    
    if rec >= 0.85:
        if prec > best_prec_at_target_rec:
            best_prec_at_target_rec = prec
            best_thresh = t

print(f"Optimal Medical Threshold: {best_thresh:.3f}")
print(f"Train OOF Recall at Optimal Threshold: {recall_score(y_train, (oof_proba >= best_thresh).astype(int)):.4f}")
print(f"Train OOF Precision at Optimal Threshold: {precision_score(y_train, (oof_proba >= best_thresh).astype(int)):.4f}")

# Apply Tuned Threshold on Test Set
y_test_pred_tuned = (y_test_proba >= best_thresh).astype(int)
cm_tuned = confusion_matrix(y_test, y_test_pred_tuned)

fn_tuned = cm_tuned[1, 0]
fp_tuned = cm_tuned[0, 1]

print("\n--- TEST SET PERFORMANCE WITH TUNED MEDICAL THRESHOLD ---")
print(f"Tuned Threshold: {best_thresh:.3f}")
print(f"Recall:    {recall_score(y_test, y_test_pred_tuned):.4f} (was {test_rec:.4f})")
print(f"Precision: {precision_score(y_test, y_test_pred_tuned):.4f} (was {test_prec:.4f})")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred_tuned):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_test_pred_tuned):.4f}")
print(f"False Negatives (Missed Diabetic Patients): {fn_tuned} (reduced from {fn})")
print(f"False Positives (Healthy Wrongly Flagged):  {fp_tuned} (increased from {fp})")

# Plot Tuned Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp_t = ConfusionMatrixDisplay(confusion_matrix=cm_tuned, display_labels=['Non-Diabetic', 'Diabetic'])
disp_t.plot(cmap='Oranges', ax=ax)
plt.title(f'Tuned Test Confusion Matrix (Threshold {best_thresh:.3f})', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/confusion_matrix_tuned.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 8. FEATURE IMPORTANCE & EXPLANATION
# ---------------------------------------------------------
perm_imp = permutation_importance(best_pipeline, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42)
sorted_idx = perm_imp.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], perm_imp.importances_mean[sorted_idx], color='#34495e', edgecolor='black')
plt.xlabel('Permutation Importance (Decrease in Test ROC-AUC)', fontsize=11)
plt.title('Feature Importance Analysis (Permutation Importance on Test Set)', fontsize=13, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/feature_importance.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# 9. BONUS CHALLENGES
# ---------------------------------------------------------
# Bonus 1: Linear Regression predicting Glucose from cheap measurements
cheap_features = ['Pregnancies', 'BloodPressure', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
df_reg = df_clean.dropna(subset=['Glucose'])

X_reg = df_reg[cheap_features]
y_reg = df_reg['Glucose']

X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

reg_pipe.fit(X_reg_tr, y_reg_tr)
y_reg_pred = reg_pipe.predict(X_reg_te)

r2 = r2_score(y_reg_te, y_reg_pred)
mae = mean_absolute_error(y_reg_te, y_reg_pred)

print(f"\n--- BONUS 1: GLUCOSE REGRESSION MODEL ---")
print(f"Cheap Features: {cheap_features}")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.2f} mg/dL")

# Bonus 2: Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    best_pipeline, X_train, y_train, cv=cv, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='#e74c3c', label='Training Score')
plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color='#2ecc71', label='Validation Score')
plt.title('Learning Curve: ROC-AUC vs Training Set Size', fontsize=12, fontweight='bold')
plt.xlabel('Training Set Size (Patients)', fontsize=11)
plt.ylabel('ROC-AUC Score', fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('C:/Users/psair/.gemini/antigravity/scratch/SugarSense/learning_curve.png', dpi=300)
plt.close()

# Bonus 3: Imputer Strategy Comparison
imputer_results = {}
for imp_name, imp_obj in [('Median', SimpleImputer(strategy='median')),
                          ('KNN', KNNImputer(n_neighbors=5)),
                          ('Drop Rows', None)]:
    if imp_name == 'Drop Rows':
        df_dropped = df_clean.dropna()
        X_dr = df_dropped.drop(columns=['Outcome'])
        y_dr = df_dropped['Outcome']
        rf_dr = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        cv_dr = cross_validate(rf_dr, X_dr, y_dr, cv=5, scoring='roc_auc')
        imputer_results[imp_name] = cv_dr['test_score'].mean()
    else:
        pipe_imp = Pipeline([
            ('imputer', imp_obj),
            ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])
        cv_imp = cross_validate(pipe_imp, X_train, y_train, cv=5, scoring='roc_auc')
        imputer_results[imp_name] = cv_imp['test_score'].mean()

print("\n--- BONUS 3: IMPUTER COMPARISON ---")
for k, v in imputer_results.items():
    print(f"Imputer '{k}': Mean Validation ROC-AUC = {v:.4f}")

print("\nAll execution steps completed successfully!")
