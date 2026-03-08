import joblib
import pandas as pd
import numpy as np

model = joblib.load("artifacts/disease_prediction_model.pkl")

def confidence_level(prob):
    if prob < 0.50:
        return "Low"
    elif prob < 0.75:
        return "Medium"
    return "High"

def predict_disease(input_dict: dict):

    print("INPUT RECEIVED:", input_dict)

    df = pd.DataFrame([input_dict])

    binary_cols = [
        "appetite_loss","vomiting","diarrhea","lethargy",
        "coughing","nasal_discharge","weight_loss",
        "excessive_salivation","seizures"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"yes":1, "no":0})

    pred = model.predict(df)[0]
    prob = np.max(model.predict_proba(df)[0])

    return {
        "prediction": pred,
        "probability": float(prob),
        "confidence": confidence_level(prob)
    }