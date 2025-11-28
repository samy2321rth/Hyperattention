# main.py - HealthAI Guardian (Official Hackathon Submission)
# AI-Driven Predictive Health Monitoring for Chronic Disease Management
# Built for Shri Vaishnav Vidyapeeth Vishwavidyalaya Hackathon 2025

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(
    title="HealthAI Guardian",
    description="AI-powered early detection & management of chronic diseases (Hypertension, Cardiovascular Risk, Stress) for Indian students",
    version="1.0.0"
)

# Allow frontend (React/Flutter/mobile) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model (automatically finds the .pkl file)
model_path = os.path.join(os.path.dirname(__file__), "models", "hypertension_model_final.pkl")
model = joblib.load(model_path)

# Input format from user (symptoms, vitals, lifestyle
class HealthInput(BaseModel):
    age: float
    sex: float          # 0 = female, 1 = male
    cp: float           # chest pain type (0-3)
    trestbps: float     # resting blood pressure (mmHg)
    chol: float       # cholesterol level
    fbs: float          # fasting blood sugar >120 mg/dl (1=true, 0=false)
    restecg: float      # resting ECG result
    thalach: float      # maximum heart rate achieved
    exang: float        # exercise induced angina (1=yes, 0=no)
    oldpeak: float      # ST depression induced by exercise
    slope: float        # slope of peak exercise ST segment
    ca: float           # number of major vessels colored by fluoroscopy
    thal: float         # 3 = normal, 6 = fixed defect, 7 = reversible defect

# Home route
@app.get("/")
def home():
    return {
        "project": "HealthAI Guardian",
        "message": "AI-Driven Predictive Health Monitoring System for Chronic Disease Management",
        "status": "Live & Active",
        "hackathon": "Shri Vaishnav Vidyapeeth Vishwavidyalaya"
    }

# Main prediction endpoint - matches ALL core features
@app.post("/predict")
def predict_risk(data: HealthInput):
    # Convert input to array for model
    features = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    # Get prediction probability
    probability = float(model.predict_proba(features)[0][1])
    risk_level = "High Risk" if probability > 0.5 else "Low Risk"

    # Personalized recommendation based on risk
    if probability > 0.7:
        recommendation = "URGENT: High cardiovascular risk detected! Reduce salt intake immediately, avoid stress, walk 30 mins daily, and consult a doctor within 48 hours."
    elif probability > 0.4:
        recommendation = "High stress levels detected – recommend daily meditation. Reduce salt, walk 30 mins daily, monitor BP regularly."
    else:
        recommendation = "Excellent lifestyle! Continue healthy habits. Keep monitoring BP and glucose monthly."

    return {
        "condition": "Cardiovascular Disease Risk (Hypertension + Heart Disease)",
        "risk_level": risk_level,
        "risk_probability": round(probability, 3),
        "confidence": round(max(model.predict_proba(features)[0]), 3),
        "recommendation": recommendation,
        "mock_wearable_data": {
            "heart_rate_bpm": int(data.thalach),
            "steps_today": 6800,
            "sleep_hours_last_night": 6.8,
            "stress_indicator": "Elevated" if probability > 0.5 else "Normal"
        },
        "disclaimer": "This AI does not replace professional medical advice. Please consult a qualified doctor for diagnosis and treatment."
    }

# Mock wearable data endpoint (for future Flutter integration)
@app.get("/wearables/mock")
def mock_wearables():
    return {
        "device": "HealthAI Band (Mock",
        "heart_rate": 88,
        "steps": 7200,
        "sleep_hours": 7.1,
        "stress_score": 0.68,
        "timestamp": "2025-11-28T20:30:00Z"
    }
