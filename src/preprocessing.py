import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

def clean_text(text):
    if isinstance(text, float) or text is None:
        return ""
    # Lowercase
    text = text.lower()
    # Hapus karakter selain huruf & spasi
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    # Tokenization
    tokens = text.split()
    # Stopwords
    stop = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop]
    # Gabungkan kembali
    return " ".join(tokens)

def preprocess(
    input_file="data/tmdb_10000_film.csv", 
    output_file="data/tmdb_10000_film_clean.csv"
):
    print("[INFO] Loading dataset TMDb...")
    df = pd.read_csv(input_file)

    print("[INFO] Cleaning text (judul + sinopsis)...")
    df["judul_clean"] = df["judul_display"].astype(str).apply(clean_text)
    df["sinopsis_clean"] = df["sinopsis_asli"].astype(str).apply(clean_text)

    print("[INFO] Saving cleaned data...")
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"[DONE] Cleaned dataset saved to {output_file}")
    print(f"[INFO] Total data: {len(df)}")

if __name__ == "__main__":
    preprocess()
