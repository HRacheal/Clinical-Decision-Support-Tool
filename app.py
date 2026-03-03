import streamlit as st
import requests

st.title("👨‍⚕️ Digital Clinic: AI Diagnostic Support")

with st.form("patient_form"):
    age = st.number_input("Age", 1, 100)
    sex = st.selectbox("Sex", ["M", "F"])
    temp = st.number_input("Temperature (°C)", 35.0, 42.0)
    symptoms = st.text_area("Describe Symptoms (e.g., high fever, chills)")
    submit = st.form_submit_button("Run AI Diagnosis")

if submit:
    # This sends the data to your API from Step 1
    payload = {
        "age": age, "sex": sex, "temp": temp, 
        "heart_rate": 80, "resp_rate": 20, "bp_sys": 120, "bp_dia": 80,
        "symptoms_text": symptoms
    }
    response = requests.post("YOUR_API_URL_HERE/diagnose", json=payload)
    st.write(response.json())