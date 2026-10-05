import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sugarsense_model.joblib')

def load_screening_model():
    """Loads the trained SugarSense pipeline model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run the training pipeline first.")
    return joblib.load(MODEL_PATH)

def preprocess_patient_input(patient_data):
    """
    Preprocesses patient input dictionary or pandas DataFrame.
    Calculates engineered features and converts impossible 0 values to NaN.
    """
    if isinstance(patient_data, dict):
        df_patient = pd.DataFrame([patient_data])
    elif isinstance(patient_data, pd.DataFrame):
        df_patient = patient_data.copy()
    else:
        raise ValueError("patient_data must be a dictionary or pandas DataFrame")

    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Calculate Missing_Count before replacing 0s
    for col in zero_cols:
        if col not in df_patient.columns:
            df_patient[col] = np.nan
            
    df_patient['Missing_Count'] = (df_patient[zero_cols] == 0).sum(axis=1) + df_patient[zero_cols].isna().sum(axis=1)
    
    # Replace zeros with NaN for physical measurement columns
    for col in zero_cols:
        df_patient[col] = df_patient[col].replace(0, np.nan)
        
    # Engineered features
    df_patient['Is_Obese'] = (df_patient['BMI'] >= 30.0).astype(float)
    df_patient['Glucose_Age_Product'] = df_patient['Glucose'] * df_patient['Age']
    df_patient['Risk_Pedigree_BMI'] = df_patient['DiabetesPedigreeFunction'] * df_patient['BMI']
    
    expected_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Missing_Count', 'Is_Obese',
        'Glucose_Age_Product', 'Risk_Pedigree_BMI'
    ]
    
    return df_patient[expected_cols]

def screen_patient(patient_data, threshold=0.365):
    """
    Evaluates diabetes risk for a single patient or batch.
    
    Parameters:
        patient_data (dict or pd.DataFrame): Clinical measurements of patient(s).
        threshold (float): Clinical decision threshold (default 0.365 tuned for >=85% Recall).
        
    Returns:
        dict: Patient screening assessment result.
    """
    pipeline = load_screening_model()
    X_processed = preprocess_patient_input(patient_data)
    
    proba = float(pipeline.predict_proba(X_processed)[:, 1][0])
    
    # Categorize Risk
    if proba < 0.20:
        risk_level = "Low Risk"
        action = "Standard routine follow-up. No immediate lab test required."
    elif proba < threshold:
        risk_level = "Medium Risk"
        action = "Monitor patient. Advise lifestyle intervention and schedule follow-up in 6 months."
    else:
        risk_level = "High Risk"
        action = "URGENT REFERRAL: Send patient for full 2-hour Oral Glucose Tolerance Test (OGTT)."

    return {
        'Diabetes_Probability': round(proba, 4),
        'Risk_Level': risk_level,
        'Screening_Flag': "HIGH_RISK_REFER" if proba >= threshold else "LOW_RISK_PASS",
        'Tuned_Threshold_Used': threshold,
        'Recommended_Action': action
    }

if __name__ == '__main__':
    print("--- SugarSense Patient Screening Tool Demo ---")
    sample_patient_high_risk = {
        'Pregnancies': 5,
        'Glucose': 160,
        'BloodPressure': 74,
        'SkinThickness': 0,  # Missing measurement
        'Insulin': 0,        # Missing measurement
        'BMI': 34.2,
        'DiabetesPedigreeFunction': 0.65,
        'Age': 45
    }
    
    sample_patient_low_risk = {
        'Pregnancies': 1,
        'Glucose': 85,
        'BloodPressure': 66,
        'SkinThickness': 29,
        'Insulin': 94,
        'BMI': 22.5,
        'DiabetesPedigreeFunction': 0.20,
        'Age': 24
    }
    
    print("\nHigh Risk Test Patient Result:")
    print(screen_patient(sample_patient_high_risk))
    
    print("\nLow Risk Test Patient Result:")
    print(screen_patient(sample_patient_low_risk))
