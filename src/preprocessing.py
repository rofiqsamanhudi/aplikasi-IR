import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# download stopwords jika belum ada
nltk.download("stopwords")

def clean_text(text):
    # lowercase
    text = text.lower()

    # hilangkan karakter non-huruf
    text = re.sub(r"[^a-zA-Z ]", " ", text)

    # tokenizing
    tokens = text.split()

    # hapus stopwords
    stop = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop]

    # gabungkan kembali
    return " ".join(tokens)

def preprocess(input_file="data/quotes_raw.csv", output_file="data/quotes_clean.csv"):
    print("[INFO] Loading data...")
    df = pd.read_csv(input_file)

    print("[INFO] Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    print("[INFO] Saving cleaned data...")
    df.to_csv(output_file, index=False)

    print(f"[DONE] Cleaned dataset saved to {output_file}")


if __name__ == "__main__":
    preprocess()
