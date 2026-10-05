# 🩺 SugarSense: Early Diabetes Risk Screening

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)

**SugarSense** is a Supervised Machine Learning early diabetes screening system designed for community health workers and NGO mobile clinics in resource-constrained environments. By leveraging 8 low-cost clinical measurements, SugarSense identifies high-risk individuals and recommends follow-up laboratory testing, prioritizing high **Recall** to minimize dangerous missed diagnoses.

---

## 📌 Problem Statement
Diabetes affects over 100 million people in India, with a significant proportion remaining undiagnosed due to the cost and time required for laboratory blood tests. 

Missed detection leads to irreversible macro- and micro-vascular complications (heart disease, kidney failure, neuropathy, and retinopathy). **SugarSense** predicts whether a patient is likely diabetic (`Outcome = 1`) or non-diabetic (`Outcome = 0`), optimizing the decision threshold to achieve **Recall $\ge$ 85%** on clinical screenings.

---

## 📊 Dataset Summary
- **Source**: Pima Indians Diabetes Database (768 patients, females aged $\ge 21$)
- **File**: `diabetes_screening_data.csv`
- **Features (8 Input Features)**:
  1. `Pregnancies`: Number of pregnancies
  2. `Glucose`: Plasma glucose concentration (2-hour oral glucose tolerance test)
  3. `BloodPressure`: Diastolic blood pressure (mm Hg)
  4. `SkinThickness`: Triceps skin fold thickness (mm)
  5. `Insulin`: 2-hour serum insulin ($\mu\text{U/ml}$)
  6. `BMI`: Body Mass Index ($\text{kg/m}^2$)
  7. `DiabetesPedigreeFunction`: Genetic risk pedigree score
  8. `Age`: Patient age (years)
- **Target**: `Outcome` (1 = Diabetic [34.9%], 0 = Non-Diabetic [65.1%])
- **Data Audit Finding**: Hidden missing values encoded as `0` in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`. Missing values are imputed inside leakage-free pipelines.

---

## 🛠️ Project Structure & Architecture
```
SugarSense/
│
├── diabetes_screening_data.csv   # Dataset file
├── SugarSense.ipynb              # Complete executed Jupyter Notebook (Phases 1-10)
├── screen_patient.py             # Python screening module & CLI demo tool
├── sugarsense_model.joblib       # Saved champion pipeline model artifact
├── SUMMARY_REPORT.md             # 1-Page Executive Summary Report
├── README.md                     # Repository documentation & Git commit history
│
├── missing_values.png            # Missing data audit chart
├── correlation_heatmap.png       # Correlation matrix visualization
├── model_comparison.png          # 5-fold CV baseline model comparison chart
├── roc_pr_curves.png             # Test set ROC & Precision-Recall curves
├── confusion_matrix_default.png  # Confusion matrix at 0.5 threshold
├── confusion_matrix_tuned.png    # Confusion matrix at tuned medical threshold (0.365)
├── feature_importance.png        # Permutation feature importance plot
└── learning_curve.png            # Model learning curve plot
```

---

## 🚀 How to Run the Project

### 1. Environment Setup
Clone the repository and install required dependencies:
```bash
git clone https://github.com/your-org/SugarSense.git
cd SugarSense
pip install -r requirements.txt
```
*Required packages*: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`.

### 2. Run the Jupyter Notebook
Launch Jupyter Notebook to view full step-by-step code, plots, and markdown analyses:
```bash
jupyter notebook SugarSense.ipynb
```

### 3. Test Patient Screening Module
Run the standalone screening tool to evaluate individual patient records:
```bash
python screen_patient.py
```

---

## 📈 Key Results & Model Comparison

### 5-Fold Stratified Cross-Validation Results:
| Model Algorithm | Val ROC-AUC | Val Recall | Train ROC-AUC | Overfit Gap | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.8468 | 74.76% | 0.8557 | 0.0088 | Stable Baseline |
| **Support Vector Machine** | 0.8375 | 73.39% | 0.9090 | 0.0716 | Robust |
| **Tuned XGBoost (Champion)** | **0.8498** | **78.40%** | **0.8710** | **0.0212** | **Best Model** |
| **Tuned Random Forest** | 0.8475 | 76.20% | 0.8650 | 0.0175 | Strong |
| **K-Nearest Neighbors** | 0.7945 | 52.36% | 0.8887 | 0.0942 | Moderate |
| **Decision Tree** | 0.7714 | 80.85% | 0.9084 | 0.1370 | High Variance |

---

## 🩺 Threshold Tuning ("Think Like a Doctor")

By lowering the decision threshold from **0.500 to 0.365** based on out-of-fold cross-validation predictions:
- **Test Recall**: Increased from **75.93% to 85.19%** (meeting the target clinical threshold).
- **False Negatives**: Reduced from **13 to 8 missed patients** (38.5% reduction in missed diabetes cases).
- **False Positives**: Increased from 28 to 39 (acceptable referral trade-off for confirmatory lab blood test).

---

## 📜 Git Commit History Breakdown (6 Meaningful Commits)

1. `commit 1`: `feat(data): initial commit of dataset and data audit phase (Phase 1)`
2. `commit 2`: `feat(eda): add exploratory data analysis distribution plots and correlation heatmap (Phase 2)`
3. `commit 3`: `feat(engineering): implement domain features and leakage-free pipeline setup (Phases 3-4)`
4. `commit 4`: `feat(modeling): train and compare 6 classification algorithms with 5-fold CV (Phase 5)`
5. `commit 5`: `feat(tuning): hyperparameter tuning and medical threshold optimization for Recall >= 85% (Phases 6-8)`
6. `commit 6`: `docs(deliverables): add feature importance, bonus challenges, executive summary, and joblib screening app`

---

## ⚠️ Limitations & Ethical Disclaimer
1. **Demographic Constraint**: Model was trained on adult females of Pima Indian heritage. Generalizability to males, children, and other ethnic populations requires further validation.
2. **Clinical Role**: SugarSense is a **screening recommendation tool**, not a diagnostic system. All high-risk referrals must be confirmed by standard lab testing (e.g. 2-hour OGTT or HbA1c).
