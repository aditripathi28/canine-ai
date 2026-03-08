from fastapi import FastAPI
from inference import predict_disease

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Disease Prediction API is running"}

@app.post("/predict")
def predict(data: dict):

    result = predict_disease(data)

    return result