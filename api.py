from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load('readmission_model.joblib')

class PatientData(BaseModel):
    age: int
    num_comorbidities: int
    lab_trend: float
    discharge_home: int

@app.post("/predict")
def predict_readmission(data: PatientData):
    features = np.array([[data.age, data.num_comorbidities, data.lab_trend, data.discharge_home]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]
    return {"readmission_risk": int(prediction), "probability": float(probability)}
