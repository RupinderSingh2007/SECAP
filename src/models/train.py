import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix

#Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / 'data' / 'processed' / 'cleaned_messages.csv'
MODEL_PATH = BASE_DIR / 'models' / 'email_classifier_v1.joblib'

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def train_model(df):
    x = df["cleaned_text"]
    y = df["label"]

    #Split data (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #Pipeline = vectorizer +model
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    #Train model
    pipeline.fit(x_train, y_train)

    #Predict
    y_pred = pipeline.predict(x_test)

    #Evaluate
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return pipeline

def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    save_model(model)

