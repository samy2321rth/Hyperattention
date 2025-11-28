# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="HealthAI Guardian – Chronic Disease Management")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Auto-find model file (works everywhere)
model_path = os.path.join(os.path.dirname(__file__), "models", "hypertension_model_final.pkl")
model = joblib.load(model_path)

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message": "HealthAI Guardian – Live for Chronic Disease Prediction"}

@app.post("/predict")
def predict(data: PatientData):
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    probability = float(model.predict_proba(features)[0][1])
    risk = "High Risk" if probability > 0.5 else "Low Risk"

    return {
        "risk_level": risk,
        "risk_probability": round(probability, 3),
        "recommendation": 
            "High stress levels detected – recommend daily meditation. Reduce salt intake, walk 30 mins daily and monitor BP regularly." 
            if probability > 0.4 else 
            "Excellent lifestyle! Continue healthy habits.",
        "disclaimer": "This AI does not replace professional medical advice."
    }
