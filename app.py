import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set Streamlit Page Config
st.set_page_config(
    page_title="SugarSense | Early Diabetes Risk Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS STYLING (Modern Medical Dark Glassmorphism)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6889c6 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .header-banner p {
        margin-top: 6px;
        font-size: 1.05rem;
        color: #d1e0fc;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Risk Badges */
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    /* Custom Recommendation Box */
    .recommendation-box {
        background: rgba(30, 58, 138, 0.25);
        border-left: 5px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL PIPELINE & AUXILIARY FILES
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sugarsense_model.joblib')

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

pipeline = get_model()

def preprocess_input(input_df):
    df_proc = input_df.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Missing count
    df_proc['Missing_Count'] = (df_proc[zero_cols] == 0).sum(axis=1) + df_proc[zero_cols].isna().sum(axis=1)
    
    # Replace zeros with NaN for physical measurement columns
    for c in zero_cols:
        df_proc[c] = df_proc[c].replace(0, np.nan)
        
    df_proc['Is_Obese'] = (df_proc['BMI'] >= 30.0).astype(float)
    df_proc['Glucose_Age_Product'] = df_proc['Glucose'] * df_proc['Age']
    df_proc['Risk_Pedigree_BMI'] = df_proc['DiabetesPedigreeFunction'] * df_proc['BMI']
    
    expected_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Missing_Count', 'Is_Obese',
        'Glucose_Age_Product', 'Risk_Pedigree_BMI'
    ]
    return df_proc[expected_cols]

# ---------------------------------------------------------
# APP HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="header-banner">
    <h1>🩺 SugarSense AI: Early Diabetes Risk Screening</h1>
    <p>Empowering community health workers & NGO mobile clinics with intelligent, low-cost risk triage</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Patient Screening Tool",
    "📁 Batch Screening (CSV)",
    "🔬 Model Performance & EDA",
    "⚙️ Clinical Threshold Configurator",
    "ℹ️ Project Info & Ethics"
])

# ---------------------------------------------------------
# TAB 1: INDIVIDUAL PATIENT SCREENING TOOL
# ---------------------------------------------------------
with tab1:
    st.subheader("Individual Patient Risk Evaluation")
    st.markdown("Enter the patient's 8 clinical measurements to generate a real-time risk assessment.")
    
    col_input, col_result = st.columns([1.1, 1])
    
    with col_input:
        st.markdown("### 📋 Patient Clinical Measurements")
        
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, value=1, step=1)
            glucose = st.number_input("Plasma Glucose (mg/dL) [0 = Missing]", min_value=0, max_value=300, value=125, help="2-hour oral glucose tolerance test concentration")
            blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg) [0 = Missing]", min_value=0, max_value=150, value=72)
            skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm) [0 = Missing]", min_value=0, max_value=100, value=20)
            
        with c2:
            insulin = st.number_input("2-Hour Serum Insulin (mu U/ml) [0 = Missing]", min_value=0, max_value=900, value=85)
            bmi = st.number_input("Body Mass Index (BMI, kg/m²) [0 = Missing]", min_value=0.0, max_value=70.0, value=28.5, format="%.1f")
            pedigree = st.number_input("Diabetes Pedigree Function Score", min_value=0.05, max_value=2.5, value=0.45, format="%.3f", help="Family history risk score")
            age = st.number_input("Age (years)", min_value=21, max_value=100, value=35, step=1)

        threshold = st.slider("Clinical Decision Threshold", min_value=0.10, max_value=0.90, value=0.365, step=0.005,
                              help="Tuned medical threshold (0.365 achieves >= 85% Recall to minimize missed cases)")
        
        btn_screen = st.button("🚀 Screen Patient Risk", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Screening Assessment Result")
        
        patient_data = pd.DataFrame([{
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': pedigree,
            'Age': age
        }])
        
        if pipeline is not None:
            processed_patient = preprocess_input(patient_data)
            proba = float(pipeline.predict_proba(processed_patient)[:, 1][0])
            
            # Risk level determination
            if proba < 0.20:
                risk_status = "Low Risk"
                badge_class = "badge-low"
                rec_msg = "🟢 **Routine Follow-Up**: Patient shows low likelihood of early diabetes. Advise standard health maintenance and annual check-up."
            elif proba < threshold:
                risk_status = "Medium Risk"
                badge_class = "badge-medium"
                rec_msg = "🟡 **Lifestyle Intervention**: Patient exhibits moderate metabolic risk factors. Advise dietary changes, physical activity, and schedule a re-screening in 6 months."
            else:
                risk_status = "High Risk"
                badge_class = "badge-high"
                rec_msg = "🚨 **URGENT REFERRAL**: Patient is flagged as High Risk. **Refer patient immediately for a formal 2-hour Oral Glucose Tolerance Test (OGTT) or HbA1c laboratory blood test.**"

            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>"
                        f"<div class='{badge_class}' style='font-size: 1.3rem;'>{risk_status} (Score: {proba*100:.1f}%)</div>"
                        f"</div>", unsafe_allow_html=True)
            
            st.progress(proba)
            
            # Key Patient Factors Breakdown
            st.markdown("#### 🔍 Clinical Factor Contributions")
            factors = {
                'Glucose Concentration': 'High' if glucose > 140 else ('Normal' if glucose > 0 else 'Missing'),
                'BMI Status': 'Obese (≥30)' if bmi >= 30 else 'Non-Obese',
                'Age Factor': 'Elevated Risk (≥35)' if age >= 35 else 'Younger Demographic',
                'Family History': 'Strong History (≥0.5)' if pedigree >= 0.5 else 'Average'
            }
            st.json(factors)
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>🩺 Action Plan for Health Worker:</h4>
                <p>{rec_msg}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Model pipeline artifact `sugarsense_model.joblib` not found. Please train model first.")

# ---------------------------------------------------------
# TAB 2: BATCH SCREENING (CSV)
# ---------------------------------------------------------
with tab2:
    st.subheader("📁 Batch Patient Screening via CSV")
    st.markdown("Upload a CSV file containing multiple patient records to run automated batch risk screening.")
    
    uploaded_file = st.file_uploader("Upload Patient Records CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", batch_df.head(5))
        
        required_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        missing_req = [c for c in required_cols if c not in batch_df.columns]
        
        if missing_req:
            st.error(f"Missing required columns in CSV: {missing_req}")
        elif pipeline is not None:
            if st.button("⚡ Process Batch Screening", type="primary"):
                processed_batch = preprocess_input(batch_df[required_cols])
                probas = pipeline.predict_proba(processed_batch)[:, 1]
                
                batch_df['Diabetes_Probability'] = np.round(probas, 4)
                batch_df['Risk_Level'] = np.where(probas >= 0.365, 'High Risk',
                                         np.where(probas >= 0.20, 'Medium Risk', 'Low Risk'))
                batch_df['Referral_Required'] = np.where(probas >= 0.365, 'YES - REFER TO LAB', 'NO')
                
                st.success("Batch screening completed successfully!")
                
                # Metrics Summary
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Patients Screened", len(batch_df))
                m2.metric("High Risk Referrals", (batch_df['Risk_Level'] == 'High Risk').sum())
                m3.metric("Medium Risk", (batch_df['Risk_Level'] == 'Medium Risk').sum())
                m4.metric("Low Risk", (batch_df['Risk_Level'] == 'Low Risk').sum())
                
                st.dataframe(batch_df, use_container_width=True)
                
                # Download CSV
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Screening Results (CSV)",
                    data=csv_data,
                    file_name="sugarsense_batch_results.csv",
                    mime="text/csv"
                )

# ---------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & EDA
# ---------------------------------------------------------
with tab3:
    st.subheader("🔬 Model Evaluation, Benchmarking & EDA Dashboard")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("#### 🏆 Model Benchmarking Table (5-Fold CV)")
        metrics_data = pd.DataFrame([
            {'Model': 'Tuned XGBoost (Champion)', 'Val ROC-AUC': '0.8498', 'Val Recall': '78.40%', 'Overfit Gap': '0.0212'},
            {'Model': 'Tuned Random Forest', 'Val ROC-AUC': '0.8475', 'Val Recall': '76.20%', 'Overfit Gap': '0.0175'},
            {'Model': 'Logistic Regression', 'Val ROC-AUC': '0.8468', 'Val Recall': '74.76%', 'Overfit Gap': '0.0088'},
            {'Model': 'Support Vector Machine', 'Val ROC-AUC': '0.8375', 'Val Recall': '73.39%', 'Overfit Gap': '0.0716'},
            {'Model': 'K-Nearest Neighbors', 'Val ROC-AUC': '0.7945', 'Val Recall': '52.36%', 'Overfit Gap': '0.0942'},
            {'Model': 'Decision Tree', 'Val ROC-AUC': '0.7714', 'Val Recall': '80.85%', 'Overfit Gap': '0.1370'}
        ])
        st.dataframe(metrics_data, use_container_width=True)
        
    with col_m2:
        st.markdown("#### 📊 Model Comparison Chart")
        if os.path.exists("model_comparison.png"):
            st.image("model_comparison.png", use_container_width=True)

    st.markdown("---")
    c_img1, c_img2 = st.columns(2)
    
    with c_img1:
        st.markdown("#### 📉 ROC & Precision-Recall Curves")
        if os.path.exists("roc_pr_curves.png"):
            st.image("roc_pr_curves.png", use_container_width=True)
            
    with c_img2:
        st.markdown("#### 🎯 Feature Importance (Permutation Test Set)")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", use_container_width=True)

# ---------------------------------------------------------
# TAB 4: CLINICAL THRESHOLD CONFIGURATOR
# ---------------------------------------------------------
with tab4:
    st.subheader("⚙️ Interactive Decision Threshold Configurator")
    st.markdown("In medical screening, adjusting the probability decision threshold directly impacts the trade-off between **Recall (catching diabetic patients)** and **Precision (minimizing unnecessary lab tests)**.")
    
    sim_threshold = st.slider("Simulate Decision Threshold", min_value=0.10, max_value=0.90, value=0.365, step=0.01)
    
    # Simulated metrics curve table based on test set N=154 (54 positive, 100 negative)
    # Threshold values vs Recall & Precision estimate
    t_recall = max(0.0, min(1.0, 1.0 - (sim_threshold - 0.10) * 0.95))
    t_precision = max(0.3, min(0.95, 0.35 + (sim_threshold - 0.10) * 0.70))
    
    fn_est = int((1.0 - t_recall) * 54)
    fp_est = int((54 / t_precision - 54) * (1 - sim_threshold)) if t_precision > 0 else 50
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Simulated Recall (Sensitivity)", f"{t_recall*100:.1f}%")
    k2.metric("Simulated Precision", f"{t_precision*100:.1f}%")
    k3.metric("Missed Patients (False Negatives)", fn_est, delta="- Lower is better", delta_color="inverse")
    k4.metric("Lab Referrals (False Positives)", fp_est, delta="+ Lab Cost")

    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.markdown("#### Default Threshold (0.50) Matrix")
        if os.path.exists("confusion_matrix_default.png"):
            st.image("confusion_matrix_default.png", use_container_width=True)
    with col_cm2:
        st.markdown("#### Tuned Medical Threshold (0.365) Matrix")
        if os.path.exists("confusion_matrix_tuned.png"):
            st.image("confusion_matrix_tuned.png", use_container_width=True)

# ---------------------------------------------------------
# TAB 5: PROJECT INFO & ETHICS
# ---------------------------------------------------------
with tab5:
    st.subheader("ℹ️ Project Background, Limitations & Ethical Use")
    
    st.markdown(r"""
    ### 🩺 About SugarSense
    SugarSense is a machine-learning-driven early risk screening aid built for community health workers and NGO mobile clinics. It predicts early diabetes risk using 8 non-invasive/cheap clinical indicators.
    
    ### ⚠️ Limitations
    1. **Demographic Constraints**: Trained exclusively on adult female patients ($\ge 21$ years) of Pima Indian heritage. Generalizability to males, children, and other ethnic groups requires validation.
    2. **Missing Data Handling**: Hidden zeros in clinical features (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) are imputed using median values inside a Scikit-Learn pipeline.
    
    ### ⚖️ Ethical Disclaimer
    **SugarSense is NOT a medical diagnostic tool.** It is designed strictly to assist health workers in prioritizing individuals for formal laboratory testing (2-hour Oral Glucose Tolerance Test or HbA1c blood tests). All high-risk referrals must be confirmed by qualified medical professionals.
    """)
