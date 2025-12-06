import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# Streamlit Config
# ----------------------------------------------------------
st.set_page_config(
    page_title="TMDb Movie Retrieval",
    layout="wide"
)

# ==========================================================
#                      Utilities
# ==========================================================
def safe_split_genres(cell):
    """Safely split genre strings into a list."""
    if pd.isna(cell) or not cell:
        return []
    if isinstance(cell, list):
        return cell
    return [p.strip() for p in str(cell).split(",") if p.strip()]


def build_df_display(df_subset, scores, cols):
    """
    Create a DataFrame for display:
    - Insert 'retrieval_score' as the first column.
    - Only include selected metadata columns.
    """
    out = df_subset.copy().reset_index(drop=True)
    out["retrieval_score"] = scores

    # Ensure all required columns exist
    for c in cols:
        if c not in out.columns:
            out[c] = ""

    return out[["retrieval_score"] + cols]


# ==========================================================
#                     Load / Prepare Data
# ==========================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Ensure all required columns exist
    expected_cols = [
        "id", "judul_asli", "judul_display", "sinopsis_asli",
        "genre", "tanggal_rilis", "rating", "popularitas",
        "bahasa_asli", "judul_clean", "sinopsis_clean"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""

    # Fill missing values
    fill_cols = [
        "sinopsis_asli", "sinopsis_clean", "judul_display",
        "judul_asli", "genre", "tanggal_rilis", "rating",
        "popularitas", "bahasa_asli", "judul_clean"
    ]
    for c in fill_cols:
        df[c] = df[c].fillna("")

    df["genres_list"] = df["genre"].apply(safe_split_genres)
    return df


# ==========================================================
#                Sidebar — Dataset Settings
# ==========================================================
st.sidebar.title("Settings")

data_path = st.sidebar.text_input(
    "Path CSV dataset",
    value="data/tmdb_10000_film_clean.csv"
)

top_k = st.sidebar.number_input(
    "Jumlah hasil (top K)",
    value=10, min_value=1, max_value=200, step=1
)

reload_button = st.sidebar.button("Load dataset (refresh cache)")

# Load data
try:
    if reload_button:
        load_data.clear()  # type: ignore
    df = load_data(data_path)

except FileNotFoundError:
    st.sidebar.error(f"File tidak ditemukan: {data_path}")
    st.stop()

except pd.errors.EmptyDataError:
    st.sidebar.error("File kosong atau tidak valid.")
    st.stop()


# ==========================================================
#                     Vectorizer Setup
# ==========================================================
@st.cache_resource
def build_vectorizers(title_corpus, synopsis_corpus, genre_corpus):

    title_vec = TfidfVectorizer(stop_words="english", max_features=10000)
    synopsis_vec = TfidfVectorizer(stop_words="english", max_features=20000)
    genre_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

    title_matrix = title_vec.fit_transform(title_corpus)
    synopsis_matrix = synopsis_vec.fit_transform(synopsis_corpus)
    genre_matrix = genre_vec.fit_transform(genre_corpus)

    return (
        title_vec, title_matrix,
        synopsis_vec, synopsis_matrix,
        genre_vec, genre_matrix
    )


(
    title_vec, title_matrix,
    synopsis_vec, synopsis_matrix,
    genre_vec, genre_matrix
) = build_vectorizers(
    df["judul_clean"].astype(str),
    df["sinopsis_clean"].astype(str),
    df["genre"].astype(str)
)


# ==========================================================
#                    Search Helper Functions
# ==========================================================
def topk_by_similarity(vec, matrix, topk=10):
    sim = cosine_similarity(vec, matrix).flatten()
    idx = sim.argsort()[::-1][:topk]
    return idx, sim[idx]


def search_title_scores(query, topk=10):
    vec = title_vec.transform([query])
    sim = cosine_similarity(vec, title_matrix).flatten()
    idx = np.argsort(sim)[::-1][:topk]
    return idx, sim[idx]


def search_synopsis_scores(query, topk=10):
    vec = synopsis_vec.transform([query])
    sim = cosine_similarity(vec, synopsis_matrix).flatten()
    idx = np.argsort(sim)[::-1][:topk]
    return idx, sim[idx]


def search_genre_match_counts(selected_genres, topk=None):
    if not selected_genres:
        return np.array([], int), np.array([], float)

    def match_count(row):
        return len(set(row["genres_list"]) & set(selected_genres))

    counts = df.apply(match_count, axis=1).to_numpy(int)
    pos = np.where(counts > 0)[0]

    if pos.size == 0:
        return np.array([], int), np.array([], float)

    pop = pd.to_numeric(df["popularitas"], errors="coerce").fillna(0).to_numpy(float)
    sort_idx = np.lexsort((-pop[pos], -counts[pos]))
    pos_sorted = pos[sort_idx]

    if topk:
        pos_sorted = pos_sorted[:topk]

    return pos_sorted, counts[pos_sorted].astype(float)


# ==========================================================
#                           UI
# ==========================================================
st.title("TMDb Movie Retrieval App")
st.write("Aplikasi search engine TMDb menggunakan Streamlit.")

mode = st.radio(
    "Mode pencarian:",
    ["Judul", "Sinopsis", "Genre", "Hybrid"],
    horizontal=True
)

meta_cols = [
    "id", "judul_display", "genre", "tanggal_rilis",
    "rating", "popularitas", "bahasa_asli",
    "judul_clean", "sinopsis_clean"
]

# ----------------------------------------------------------
# Mode: Judul
# ----------------------------------------------------------
if mode == "Judul":
    st.header("Pencarian — Judul")

    q = st.text_input("Masukkan judul:")
    restrict_substring = st.checkbox(
        "Filter judul yang mengandung query (substring)",
        value=False
    )

    if st.button("Cari Judul"):

        if not q.strip():
            st.warning("Masukkan judul terlebih dahulu.")
            st.stop()

        idx, sims = search_title_scores(q, topk=len(df))

        if restrict_substring:
            mask = df["judul_display"].str.contains(q, case=False, na=False)
            filtered = [i for i in idx if mask.iloc[i]]

            if not filtered:
                st.info("Tidak ada hasil sesuai substring.")
                st.stop()

            chosen = filtered[:top_k]
            chosen_sims = sims[[list(idx).index(i) for i in chosen]]

        else:
            chosen = idx[:top_k]
            chosen_sims = sims[:top_k]

        out = build_df_display(
            df.iloc[chosen].reset_index(drop=True),
            chosen_sims,
            meta_cols
        )

        st.write("Hasil pencarian (mode Judul):")
        st.dataframe(out)


# ----------------------------------------------------------
# Mode: Sinopsis
# ----------------------------------------------------------
elif mode == "Sinopsis":
    st.header("Pencarian — Sinopsis")

    q = st.text_input("Masukkan kata kunci sinopsis:")

    if st.button("Cari Sinopsis"):

        if not q.strip():
            st.warning("Masukkan kata kunci sinopsis.")
            st.stop()

        idx, sims = search_synopsis_scores(q, topk=len(df))

        out = build_df_display(
            df.iloc[idx[:top_k]].reset_index(drop=True),
            sims[:top_k],
            meta_cols
        )

        st.write("Hasil pencarian (mode Sinopsis):")
        st.dataframe(out)


# ----------------------------------------------------------
# Mode: Genre
# ----------------------------------------------------------
elif mode == "Genre":
    st.header("Pencarian & Rekomendasi — Genre")

    all_genres = sorted({g for sub in df["genres_list"] for g in sub})
    selected = st.multiselect("Pilih genre:", all_genres)

    if st.button("Cari Genre"):

        if not selected:
            st.warning("Pilih minimal satu genre.")
            st.stop()

        pos, counts = search_genre_match_counts(selected, topk=top_k)

        if pos.size == 0:
            st.info("Tidak ada film sesuai genre.")
            st.stop()

        out = build_df_display(
            df.iloc[pos].reset_index(drop=True),
            counts,
            meta_cols
        )

        st.write("Hasil pencarian (mode Genre):")
        st.dataframe(out)


# ----------------------------------------------------------
# Mode: Hybrid
# ----------------------------------------------------------
else:
    st.header("Hybrid Query — Judul + Sinopsis + Genre")
    st.write("Isi minimal dua dari tiga input.")

    q_title = st.text_input("Judul (opsional):")
    q_syn = st.text_input("Sinopsis (opsional):")
    q_gen = st.multiselect("Genre (opsional):", sorted({g for sub in df["genres_list"] for g in sub}))

    if st.button("Cari Hybrid"):

        filled = sum([
            bool(q_title.strip()),
            bool(q_syn.strip()),
            len(q_gen) > 0
        ])

        if filled < 2:
            st.error("Isi minimal dua input untuk Hybrid Search.")
            st.stop()

        n = len(df)
        title_sim = np.zeros(n)
        syn_sim = np.zeros(n)
        genre_norm = np.zeros(n)

        if q_title.strip():
            vec_t = title_vec.transform([q_title])
            title_sim = cosine_similarity(vec_t, title_matrix).flatten()

        if q_syn.strip():
            vec_s = synopsis_vec.transform([q_syn])
            syn_sim = cosine_similarity(vec_s, synopsis_matrix).flatten()

        if q_gen:
            match = df["genres_list"].apply(lambda g: len(set(g) & set(q_gen))).to_numpy(float)
            genre_norm = match / match.max() if match.max() > 0 else np.zeros_like(match)

        final_score = (
            0.4 * title_sim +
            0.4 * syn_sim +
            0.2 * genre_norm
        )

        idx = np.argsort(final_score)[::-1][:top_k]

        out = build_df_display(
            df.iloc[idx].reset_index(drop=True),
            final_score[idx],
            meta_cols
        )

        st.write("Hasil (mode Hybrid):")
        st.dataframe(out)


# ==========================================================
#                          Footer
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.write("Gunakan field 'Path CSV dataset' untuk mengganti lokasi file CSV.")
