import joblib
from pathlib import Path

#Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'email_classifier_v1.joblib'

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_text(model, text:str):
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = max(probabilities)
    return prediction, confidence

if __name__ == "__main__":
    model = load_model()
    while True:
        text = input("Enter an email message (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break

        label, confidence = predict_text(model, text)
        print(f"Predicted Label: {label} (Confidence: {confidence:.2f})\n")
        