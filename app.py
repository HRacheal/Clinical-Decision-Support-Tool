import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. LOAD THE BRAIN DIRECTLY (No API needed)
@st.cache_resource
def load_clinical_models():
    # Make sure these files are uploaded to the same folder on Hugging Face
    with open('clinic_brain.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('text_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('sex_encoder.pkl', 'rb') as f:
        le_sex = pickle.load(f)
    return model, tfidf, le_sex

# Attempt to load models
try:
    model, tfidf, le_sex = load_clinical_models()
except FileNotFoundError:
    st.error("Error: Brain files (.pkl) not found. Please upload clinic_brain.pkl, text_vectorizer.pkl, and sex_encoder.pkl.")
    st.stop()

# 2. THE USER INTERFACE (What doctors see)
st.title("👨‍⚕️ Clinical Decision Support System")
st.markdown("Enter patient vitals and symptoms for an instant AI assessment.")

with st.form("patient_data"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 25)
        sex = st.selectbox("Sex", ["M", "F"])
        temp = st.number_input("Temperature (°C)", 34.0, 42.0, 37.0)
    with col2:
        hr = st.number_input("Heart Rate", 40, 200, 80)
        sys = st.number_input("Systolic BP", 80, 200, 120)
        dia = st.number_input("Diastolic BP", 50, 120, 80)
    
    symptoms_text = st.text_area("Patient Symptoms", "High fever and chills...")
    submit = st.form_submit_button("Analyze Patient")

# 3. THE LOGIC (Runs when button is clicked)
if submit:
    # Process inputs
    sex_encoded = le_sex.transform([sex])[0]
    # We use 20 as a default for Respiratory Rate if not asked
    vitals = np.array([[age, sex_encoded, temp, hr, 20, sys, dia]])
    
    # Process text
    text_vec = tfidf.transform([symptoms_text]).toarray()
    
    # Combine (Hybrid Feature)
    combined_input = np.hstack((vitals, text_vec))
    
    # Predict
    target_cols = ['Malaria', 'Typhoid', 'Pneumonia', 'TB', 'Dengue', 'Cholera', 'Hypertension', 'Diabetes']
    prediction_probs = model.predict_proba(combined_input)
    
    st.subheader("Diagnostic Risk Analysis")
    
    # Show results in a clean grid
    cols = st.columns(2)
    referral_urgent = False
    
    for i, disease in enumerate(target_cols):
        # Get risk percentage
        risk_score = prediction_probs[i][0][1] * 100
        
        with cols[i % 2]:
            if risk_score > 70:
                st.error(f"**{disease}**: {risk_score:.1f}% (High Risk)")
                referral_urgent = True
            elif risk_score > 30:
                st.warning(f"**{disease}**: {risk_score:.1f}% (Moderate)")
            else:
                st.success(f"**{disease}**: {risk_score:.1f}% (Low)")

    if referral_urgent:
        st.divider()
        st.error("🚨 **ALERT:** High-risk indicators detected. Automated referral recommended.")
