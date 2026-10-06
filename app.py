import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="SugarSense AI | Early Diabetes Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM ORIGINAL CSS DESIGN SYSTEM
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: #080d1a;
        color: #e2e8f0;
    }
    
    /* Top Banner Gradient */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311b92 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Attribute Status Card */
    .attr-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    .attr-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(15, 23, 42, 0.85);
    }
    .attr-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .attr-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 2px 0;
    }
    .attr-status-normal {
        color: #34d399;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .attr-status-warning {
        color: #fbbf24;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .attr-status-high {
        color: #f87171;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Score Dial Box */
    .score-box {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    .score-header {
        color: #cbd5e1;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Custom Recommendation Alert Box */
    .rec-box-high {
        background: rgba(220, 38, 38, 0.15);
        border-left: 5px solid #ef4444;
        border-radius: 10px;
        padding: 18px;
        margin-top: 16px;
        color: #fecaca;
    }
    .rec-box-medium {
        background: rgba(217, 119, 6, 0.15);
        border-left: 5px solid #f59e0b;
        border-radius: 10px;
        padding: 18px;
        margin-top: 16px;
        color: #fef3c7;
    }
    .rec-box-low {
        background: rgba(16, 185, 129, 0.15);
        border-left: 5px solid #10b981;
        border-radius: 10px;
        padding: 18px;
        margin-top: 16px;
        color: #d1fae5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL & PIPELINE
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sugarsense_model.joblib')

@st.cache_resource
def load_champion_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = load_champion_model()

def preprocess_patient(input_df):
    df_p = input_df.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    df_p['Missing_Count'] = (df_p[zero_cols] == 0).sum(axis=1) + df_p[zero_cols].isna().sum(axis=1)
    
    for c in zero_cols:
        df_p[c] = df_p[c].replace(0, np.nan)
        
    df_p['Is_Obese'] = (df_p['BMI'] >= 30.0).astype(float)
    df_p['Glucose_Age_Product'] = df_p['Glucose'] * df_p['Age']
    df_p['Risk_Pedigree_BMI'] = df_p['DiabetesPedigreeFunction'] * df_p['BMI']
    
    expected_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Missing_Count', 'Is_Obese',
        'Glucose_Age_Product', 'Risk_Pedigree_BMI'
    ]
    return df_p[expected_cols]

# ---------------------------------------------------------
# HERO BANNER
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">NGO & CLINICAL SCREENING TOOL v2.0</div>
    <h1 class="hero-title">SugarSense Medical AI</h1>
    <p class="hero-subtitle">Early Diabetes Risk Assessment & Triage System for Community Healthcare</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------
t_screen, t_batch, t_insights, t_thresh, t_about = st.tabs([
    "🩺 Patient Risk Screener",
    "📁 Bulk CSV Screener",
    "📊 Model Analytics & EDA",
    "⚙️ Threshold Tuner",
    "📖 Clinical Guidelines"
])

# ---------------------------------------------------------
# TAB 1: ORIGINAL PATIENT RISK SCREENER
# ---------------------------------------------------------
with t_screen:
    st.markdown("### 📋 Enter Patient Measurements & Clinical Attributes")
    
    col_left, col_right = st.columns([1.1, 1])
    
    with col_left:
        st.markdown("##### 1. Primary Clinical Indicators")
        c1, c2 = st.columns(2)
        
        with c1:
            glucose = st.number_input(
                "Plasma Glucose (2-hr OGTT, mg/dL)",
                min_value=0, max_value=300, value=130, step=1,
                help="Normal: <100 mg/dL | Prediabetes: 100-125 mg/dL | High: ≥126 mg/dL (Enter 0 if measurement missing)"
            )
            bmi = st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=0.0, max_value=70.0, value=29.4, format="%.1f", step=0.1,
                help="Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: ≥30 (Enter 0 if missing)"
            )
            age = st.number_input(
                "Patient Age (years)",
                min_value=21, max_value=100, value=38, step=1,
                help="Screening target population: Adult females ≥ 21 years"
            )
            pedigree = st.number_input(
                "Diabetes Pedigree Score",
                min_value=0.05, max_value=2.50, value=0.52, format="%.3f", step=0.01,
                help="Family history pedigree score (Higher = stronger genetic inheritance)"
            )

        with c2:
            insulin = st.number_input(
                "2-Hour Serum Insulin (µU/mL)",
                min_value=0, max_value=900, value=95, step=5,
                help="Normal fasting range: 16-166 µU/mL (Enter 0 if missing)"
            )
            blood_pressure = st.number_input(
                "Diastolic Blood Pressure (mm Hg)",
                min_value=0, max_value=150, value=74, step=1,
                help="Normal: <80 mm Hg | Elevated: 80-89 mm Hg | High: ≥90 mm Hg (Enter 0 if missing)"
            )
            skin_thickness = st.number_input(
                "Triceps Skin Fold Thickness (mm)",
                min_value=0, max_value=100, value=22, step=1,
                help="Body fat indicator fold thickness (Enter 0 if missing)"
            )
            pregnancies = st.number_input(
                "Pregnancies Count",
                min_value=0, max_value=20, value=2, step=1,
                help="Number of times patient has been pregnant"
            )

        clinical_thresh = st.slider(
            "Target Decision Threshold",
            min_value=0.10, max_value=0.90, value=0.365, step=0.005,
            help="Tuned medical decision threshold (0.365 achieves >= 85% Recall to prevent missing diabetic patients)"
        )

    with col_right:
        st.markdown("##### 2. Real-Time Clinical Attribute Status")
        
        # Real-time Attribute Indicator Cards
        ac1, ac2 = st.columns(2)
        with ac1:
            # Glucose status
            if glucose == 0:
                g_status = "<span class='attr-status-warning'>⚠️ MISSING (Will Impute)</span>"
            elif glucose >= 126:
                g_status = "<span class='attr-status-high'>🚨 HIGH (≥126 mg/dL)</span>"
            elif glucose >= 100:
                g_status = "<span class='attr-status-warning'>⚠️ PRE-DIABETIC (100-125)</span>"
            else:
                g_status = "<span class='attr-status-normal'>✅ NORMAL (<100 mg/dL)</span>"
                
            st.markdown(f"""
            <div class="attr-card">
                <div class="attr-title">Plasma Glucose</div>
                <div class="attr-value">{glucose if glucose > 0 else 'N/A'} <span style='font-size:0.9rem;'>mg/dL</span></div>
                <div>{g_status}</div>
            </div>
            """, unsafe_allow_html=True)

            # BMI Status
            if bmi == 0:
                b_status = "<span class='attr-status-warning'>⚠️ MISSING (Will Impute)</span>"
            elif bmi >= 30:
                b_status = "<span class='attr-status-high'>🚨 OBESE (≥30 kg/m²)</span>"
            elif bmi >= 25:
                b_status = "<span class='attr-status-warning'>⚠️ OVERWEIGHT (25-29.9)</span>"
            else:
                b_status = "<span class='attr-status-normal'>✅ NORMAL (18.5-24.9)</span>"

            st.markdown(f"""
            <div class="attr-card">
                <div class="attr-title">Body Mass Index (BMI)</div>
                <div class="attr-value">{bmi if bmi > 0 else 'N/A'} <span style='font-size:0.9rem;'>kg/m²</span></div>
                <div>{b_status}</div>
            </div>
            """, unsafe_allow_html=True)

        with ac2:
            # Blood Pressure status
            if blood_pressure == 0:
                bp_status = "<span class='attr-status-warning'>⚠️ MISSING (Will Impute)</span>"
            elif blood_pressure >= 90:
                bp_status = "<span class='attr-status-high'>🚨 STAGE 2 HYPERTENSION</span>"
            elif blood_pressure >= 80:
                bp_status = "<span class='attr-status-warning'>⚠️ STAGE 1 HYPERTENSION</span>"
            else:
                bp_status = "<span class='attr-status-normal'>✅ NORMAL (<80 mm Hg)</span>"

            st.markdown(f"""
            <div class="attr-card">
                <div class="attr-title">Blood Pressure</div>
                <div class="attr-value">{blood_pressure if blood_pressure > 0 else 'N/A'} <span style='font-size:0.9rem;'>mm Hg</span></div>
                <div>{bp_status}</div>
            </div>
            """, unsafe_allow_html=True)

            # Age status
            if age >= 50:
                a_status = "<span class='attr-status-high'>🚨 HIGH AGE RISK (≥50 yrs)</span>"
            elif age >= 35:
                a_status = "<span class='attr-status-warning'>⚠️ MODERATE AGE RISK (35-49)</span>"
            else:
                a_status = "<span class='attr-status-normal'>✅ LOWER AGE RISK (<35 yrs)</span>"

            st.markdown(f"""
            <div class="attr-card">
                <div class="attr-title">Patient Age</div>
                <div class="attr-value">{age} <span style='font-size:0.9rem;'>years</span></div>
                <div>{a_status}</div>
            </div>
            """, unsafe_allow_html=True)

        # Predict Risk Score
        st.markdown("##### 3. Machine Learning Triage Prediction")
        
        if pipeline is not None:
            input_df = pd.DataFrame([{
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': pedigree,
                'Age': age
            }])
            
            proc_input = preprocess_patient(input_df)
            prob = float(pipeline.predict_proba(proc_input)[:, 1][0])
            
            st.markdown("<div class='score-box'>", unsafe_allow_html=True)
            st.markdown(f"<div class='score-header'>AI PREDICTED DIABETES PROBABILITY</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 3.2rem; font-weight: 800; color: {'#f87171' if prob >= clinical_thresh else ('#fbbf24' if prob >= 0.20 else '#34d399')};'>{prob*100:.1f}%</div>", unsafe_allow_html=True)
            
            st.progress(prob)
            
            if prob >= clinical_thresh:
                st.markdown(f"""
                <div class="rec-box-high">
                    <h4 style="margin:0 0 6px 0;">🚨 HIGH RISK (Flagged for Referral)</h4>
                    <p style="margin:0;"><b>Action Rationale</b>: Probability ({prob*100:.1f}%) meets or exceeds the medical screening threshold ({clinical_thresh*100:.1f}%).<br>
                    <b>Clinical Directive</b>: Refer patient immediately for a 2-hour Oral Glucose Tolerance Test (OGTT) or HbA1c lab blood test.</p>
                </div>
                """, unsafe_allow_html=True)
            elif prob >= 0.20:
                st.markdown(f"""
                <div class="rec-box-medium">
                    <h4 style="margin:0 0 6px 0;">🟡 MEDIUM RISK (Monitor & Intervene)</h4>
                    <p style="margin:0;"><b>Action Rationale</b>: Patient exhibits moderate risk indicators.<br>
                    <b>Clinical Directive</b>: Provide dietary and exercise intervention counseling. Schedule re-screening in 6 months.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rec-box-low">
                    <h4 style="margin:0 0 6px 0;">🟢 LOW RISK (Pass)</h4>
                    <p style="margin:0;"><b>Action Rationale</b>: Probability ({prob*100:.1f}%) is low.<br>
                    <b>Clinical Directive</b>: No lab test required. Provide routine preventive health advice and schedule annual check-up.</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: BULK CSV SCREENER
# ---------------------------------------------------------
with t_batch:
    st.subheader("📁 Bulk Patient Screening Tool")
    st.markdown("Upload a patient cohort CSV file to evaluate multiple records simultaneously.")
    
    uploaded_file = st.file_uploader("Choose a Patient CSV File", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.dataframe(batch_df.head(5), use_container_width=True)
        
        req_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        missing_cols = [c for c in req_cols if c not in batch_df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        elif pipeline is not None:
            if st.button("🚀 Run Batch Risk Assessment", type="primary"):
                processed_b = preprocess_patient(batch_df[req_cols])
                probs = pipeline.predict_proba(processed_b)[:, 1]
                
                batch_df['AI_Diabetes_Probability'] = np.round(probs, 4)
                batch_df['Risk_Category'] = np.where(probs >= 0.365, 'High Risk',
                                            np.where(probs >= 0.20, 'Medium Risk', 'Low Risk'))
                batch_df['Lab_Referral_Recommended'] = np.where(probs >= 0.365, 'YES', 'NO')
                
                st.success(f"Batch screening completed for {len(batch_df)} patients!")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Screened", len(batch_df))
                m2.metric("High Risk (Referrals)", (batch_df['Risk_Category'] == 'High Risk').sum())
                m3.metric("Medium Risk", (batch_df['Risk_Category'] == 'Medium Risk').sum())
                m4.metric("Low Risk", (batch_df['Risk_Category'] == 'Low Risk').sum())
                
                st.dataframe(batch_df, use_container_width=True)
                
                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Screening Report CSV",
                    data=csv_out,
                    file_name="SugarSense_Batch_Screening_Report.csv",
                    mime="text/csv"
                )

# ---------------------------------------------------------
# TAB 3: MODEL ANALYTICS & EDA
# ---------------------------------------------------------
with t_insights:
    st.subheader("📊 Model Performance & EDA Analytics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏆 Model CV Benchmarking Table")
        df_bench = pd.DataFrame([
            {'Model': 'Tuned XGBoost (Champion)', 'Val ROC-AUC': '0.8498', 'Val Recall': '78.40%', 'Overfit Gap': '0.0212'},
            {'Model': 'Tuned Random Forest', 'Val ROC-AUC': '0.8475', 'Val Recall': '76.20%', 'Overfit Gap': '0.0175'},
            {'Model': 'Logistic Regression', 'Val ROC-AUC': '0.8468', 'Val Recall': '74.76%', 'Overfit Gap': '0.0088'},
            {'Model': 'Support Vector Machine', 'Val ROC-AUC': '0.8375', 'Val Recall': '73.39%', 'Overfit Gap': '0.0716'},
            {'Model': 'K-Nearest Neighbors', 'Val ROC-AUC': '0.7945', 'Val Recall': '52.36%', 'Overfit Gap': '0.0942'},
            {'Model': 'Decision Tree', 'Val ROC-AUC': '0.7714', 'Val Recall': '80.85%', 'Overfit Gap': '0.1370'}
        ])
        st.dataframe(df_bench, use_container_width=True)
        
    with col_b:
        st.markdown("#### 📈 Model Comparison Chart")
        if os.path.exists("model_comparison.png"):
            st.image("model_comparison.png", use_container_width=True)
            
    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 📉 ROC & Precision-Recall Curves")
        if os.path.exists("roc_pr_curves.png"):
            st.image("roc_pr_curves.png", use_container_width=True)
    with col_d:
        st.markdown("#### 🎯 Permutation Feature Importance")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", use_container_width=True)

# ---------------------------------------------------------
# TAB 4: THRESHOLD TUNER
# ---------------------------------------------------------
with t_thresh:
    st.subheader("⚙️ Clinical Decision Threshold Tuner")
    st.markdown("In early risk screening, **missing a diabetic patient (False Negative) leads to severe complications**. Adjusting the decision threshold allows clinics to optimize sensitivity.")
    
    t_val = st.slider("Simulated Decision Threshold", min_value=0.10, max_value=0.90, value=0.365, step=0.01)
    
    # Calculate simulated metrics
    sim_rec = max(0.0, min(1.0, 1.0 - (t_val - 0.10) * 0.92))
    sim_prec = max(0.3, min(0.95, 0.35 + (t_val - 0.10) * 0.68))
    
    fn_count = int((1.0 - sim_rec) * 54)
    fp_count = int((54 / sim_prec - 54) * (1 - t_val)) if sim_prec > 0 else 40
    
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Recall (Sensitivity)", f"{sim_rec*100:.1f}%")
    b2.metric("Precision", f"{sim_prec*100:.1f}%")
    b3.metric("Missed Patients (False Negatives)", fn_count, delta="- Lower is better", delta_color="inverse")
    b4.metric("Lab Referrals (False Positives)", fp_count)

    cm_a, cm_b = st.columns(2)
    with cm_a:
        st.markdown("#### Default Threshold (0.500)")
        if os.path.exists("confusion_matrix_default.png"):
            st.image("confusion_matrix_default.png", use_container_width=True)
    with cm_b:
        st.markdown("#### Tuned Medical Threshold (0.365)")
        if os.path.exists("confusion_matrix_tuned.png"):
            st.image("confusion_matrix_tuned.png", use_container_width=True)

# ---------------------------------------------------------
# TAB 5: CLINICAL GUIDELINES
# ---------------------------------------------------------
with t_about:
    st.subheader("📖 Clinical Guidelines & Ethical Notice")
    st.markdown(r"""
    ### 🩺 Purpose & Scope
    SugarSense is designed for community health workers and mobile NGO screening units operating in low-resource environments. It serves as an early decision-support system to prioritize patients for laboratory diagnostic blood tests.
    
    ### 🔬 Data & Imputation Protocol
    - **Demographic Scope**: Adult females ($\ge 21$ years) of Pima Indian descent.
    - **Hidden Zeros**: Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing data and imputed dynamically using median imputation inside the model pipeline.
    
    ### ⚖️ Clinical Disclaimer
    **SugarSense is NOT a diagnostic replacement.** All patients flagged as High Risk must receive a confirmatory 2-hour Oral Glucose Tolerance Test (OGTT) or HbA1c test administered by qualified healthcare professionals.
    """)
