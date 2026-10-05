# Executive Summary: SugarSense Early Diabetes Risk Screening

**Project**: SugarSense | Early Diabetes Risk Screening with Supervised Machine Learning  
**Target User**: Public Health NGO & Community Health Workers  
**Author / Data Scientist**: AI/ML Supervised Learning Team  
**Date**: October 2026  

---

## 1. Executive Problem & Clinical Objective
Diabetes affects millions in India, yet many patients remain undiagnosed until late-stage cardiovascular, renal, or ocular complications develop. In resource-limited rural clinics, full laboratory work-ups (such as 2-hour Oral Glucose Tolerance Tests) are expensive and logistically infeasible for universal screening. 

**SugarSense** addresses this gap by providing an automated, machine-learning-based risk screening tool using 8 quick, low-cost clinical measurements (Age, BMI, Blood Pressure, Glucose, Pregnancies, Insulin, Skin Thickness, Family History).

---

## 2. Model Selection & Cross-Validation Results
We systematically trained and benchmarked **six supervised classification algorithms** using leakage-free Scikit-Learn Pipelines with median imputation and 5-fold Stratified Cross-Validation:

| Model Algorithm | Val ROC-AUC (Mean ± Std) | Val Recall | Train ROC-AUC | Overfit Gap | Model Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8468 ± 0.016** | 74.76% | 0.8557 | **0.0088** | Highly Stable Baseline |
| **Support Vector Machine (RBF)** | **0.8375 ± 0.019** | 73.39% | 0.9090 | 0.0716 | Robust |
| **Tuned XGBoost (Champion)** | **0.8498 ± 0.021** | **78.40%** | **0.8710** | **0.0212** | **Best Overall Model** |
| **Tuned Random Forest** | 0.8475 ± 0.025 | 76.20% | 0.8650 | 0.0175 | Strong Runner-Up |
| **K-Nearest Neighbors** | 0.7945 ± 0.028 | 52.36% | 0.8887 | 0.0942 | Underperforming |
| **Decision Tree (Baseline)** | 0.7714 ± 0.033 | 80.85% | 0.9084 | 0.1370 | High Variance |

**Selected Model**: **Tuned XGBoost Pipeline** (Imputation + Feature Engineering + Regularized Gradient Boosting).

---

## 3. Clinical Threshold Tuning ("Think Like a Doctor")
In medical screening, **False Negatives (missing a diabetic patient) are far more dangerous than False Positives (sending a healthy patient for a blood test)**. The standard 0.50 decision threshold was tuned using out-of-fold training predictions to achieve a strict clinical target of **Recall $\ge$ 85%**.

### Holdout Test Set Performance Comparison (N = 154):

| Metric / Outcome | Standard Threshold (0.500) | Tuned Medical Threshold (0.365) | Operational & Clinical Impact |
| :--- | :---: | :---: | :--- |
| **Recall (Sensitivity)** | **75.93%** | **85.19%** | **+9.26% Increase** in diabetic detection rate |
| **Precision** | 59.42% | 54.12% | Slight increase in follow-up lab referrals |
| **False Negatives (Missed Patients)** | **13 Patients** | **8 Patients** | **38.5% Reduction** in missed high-risk individuals |
| **False Positives (Lab Referrals)** | 28 Patients | 39 Patients | 11 extra routine lab tests (clinically acceptable) |
| **ROC-AUC Score** | **0.8174** | **0.8174** | Robust discrimination capability |

---

## 4. Key Predictive Signals
Permutation importance analysis on the test set reveals that the top drivers of risk are:
1. **`Glucose_Age_Product`** (Interaction of glucose level and age duration)
2. **`Plasma Glucose`** (2-hour oral test concentration)
3. **`BMI`** (Body Mass Index)
4. **`Age`**

---

## 5. Limitations & Ethical Guidelines
- **Demographic Scope**: The model was trained on adult females of Pima Indian heritage ($\ge 21$ years). Validation on male patients, children, and broader ethnic demographics is required before national rollout.
- **Clinical Role**: **SugarSense is a non-diagnostic screening aid**. It flags high-risk individuals for referral; it does not replace physician evaluation or laboratory diagnostic blood testing.
