from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="Smart Email Classifier API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "email_classifier_v1.joblib"

model = joblib.load(MODEL_PATH)


class MessageRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Smart Email Classifier API is running"}


@app.post("/predict")
def predict_message(request: MessageRequest):
    prediction = model.predict([request.text])[0]
    probabilities = model.predict_proba([request.text])[0]
    confidence = float(max(probabilities))

    return {
        "text": request.text,
        "predicted_label": prediction,
        "confidence": round(confidence, 4)
    }