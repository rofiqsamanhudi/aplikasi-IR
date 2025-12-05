import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path

def build_index(input_file="data/quotes_clean.csv", model_file="data/tfidf_index.pkl"):
    Path("data").mkdir(exist_ok=True)

    print("[INFO] Loading cleaned data...")
    df = pd.read_csv(input_file)

    print("[INFO] Building TF-IDF index...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])

    print("[INFO] Saving index...")
    with open(model_file, "wb") as f:
        pickle.dump((vectorizer, X, df), f)

    print(f"[DONE] Index saved to {model_file}")


if __name__ == "__main__":
    build_index()
