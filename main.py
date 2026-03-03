from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="AI Clinical Support System")

# Load your 100% accurate models
with open('clinic_brain.pkl', 'rb') as f:
    model = pickle.load(f)
with open('text_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('sex_encoder.pkl', 'rb') as f:
    le_sex = pickle.load(f)

# Define what the doctor needs to send
class PatientData(BaseModel):
    age: int
    sex: str
    temp: float
    heart_rate: int
    resp_rate: int
    bp_sys: int
    bp_dia: int
    symptoms_text: str

@app.post("/diagnose")
def diagnose(data: PatientData):
    # 1. Process Input
    sex_val = le_sex.transform([data.sex])[0]
    vitals = np.array([[data.age, sex_val, data.temp, data.heart_rate, 
                        data.resp_rate, data.bp_sys, data.bp_dia]])
    text_vec = tfidf.transform([data.symptoms_text]).toarray()
    combined = np.hstack((vitals, text_vec))

    # 2. Predict
    target_cols = ['malaria', 'typhoid', 'pneumonia', 'tb', 'dengue', 'cholera', 'hypertension', 'diabetes']
    probs = model.predict_proba(combined)
    
    results = {}
    for i, disease in enumerate(target_cols):
        results[disease] = f"{probs[i][0][1] * 100:.1f}%"

    return {"diagnosis_results": results}