import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def search(query, model_file="data/tfidf_index.pkl", top_k=5):
    print("[INFO] Loading index...")
    with open(model_file, "rb") as f:
        vectorizer, X, df = pickle.load(f)

    print("[INFO] Computing similarity...")
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, X)[0]

    # Ranking: highest → lowest
    top_idx = np.argsort(scores)[::-1][:top_k]

    print("\n===== SEARCH RESULT =====")
    for rank, i in enumerate(top_idx, start=1):
        print(f"\nRank {rank} | Score: {scores[i]:.4f}")
        print(f"Text   : {df.iloc[i]['text']}")
        print(f"Author : {df.iloc[i]['author']}")
        print(f"Tags   : {df.iloc[i]['tags']}")
    print("========================\n")


if __name__ == "__main__":
    query = input("Masukkan query: ")
    search(query)
