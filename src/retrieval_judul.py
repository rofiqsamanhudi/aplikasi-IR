import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path="data/tmdb_10000_film_clean.csv"):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(path)
    df["text"] = df["judul_clean"].astype(str)
    print(f"[INFO] Data loaded: {len(df)}")
    return df

def build_tfidf(text):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    tfidf = vectorizer.fit_transform(text)
    return vectorizer, tfidf

def search(query, df, vectorizer, tfidf, top_k=10):
    query_clean = re.sub(r"[^a-zA-Z ]", " ", query.lower())
    q_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    
    idx = sims.argsort()[::-1][:top_k]
    result = df.iloc[idx].copy()
    result["similarity"] = sims[idx]
    return result

if __name__ == "__main__":
    df = load_data()
    vectorizer, tfidf = build_tfidf(df["text"])

    print("\n=== PENCARIAN BERDASARKAN JUDUL ===")
    print("ketik 'exit' untuk keluar.\n")

    while True:
        q = input("Masukkan kata kunci (judul): ")
        if q.lower() == "exit":
            break
        r = search(q, df, vectorizer, tfidf)
        print("\n--- HASIL ---")
        for i, row in r.iterrows():
            print(f"{row['judul_display']} | Score: {row['similarity']:.4f}")
        print()
