import pandas as pd
import re
from pathlib import Path

def clean_text(text: str) -> str:
    """
    Clean a single message/email text.
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower().strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Cleaning to dataset.
    """
    df = df.copy()

    # Remove rows with missing values in text or label
    df = df.dropna(subset=['text', 'label'])
    # Clean the text column
    df["cleaned_text"] = df["text"].apply(clean_text)
    # Remove rows where cleaned text is empty
    df = df[df["cleaned_text"] != ""]
    # Standardize labels
    df["label"] = df["label"].str.lower().str.strip()
    return df

def save_processed_data(df:pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned dataset to CSV.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_path = "data/raw/messages.csv"
    output_path = "data/processed/cleaned_messages.csv"

    df = load_data(input_path)
    processed_df = preprocess_data(df)
    save_processed_data(processed_df, output_path)

    print("Preprocessing complete.")
    print(f"Original Rows: {len(df)}")
    print(f"Cleaned Rows: {len(processed_df)}")
    print(f"Cleaned data saved to: {output_path}")