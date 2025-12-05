import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# =====================================================================
# CONFIG
# =====================================================================
st.set_page_config(page_title="Quotes Retrieval System", layout="wide")

# =====================================================================
# LOAD DATA
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/quotes_clean.csv")
    df["tags"] = df["tags"].fillna("")
    df["tags_list"] = df["tags"].apply(lambda x: x.split(", ") if x else [])
    return df

df = load_data()

# =====================================================================
# TF-IDF MODEL
# =====================================================================
@st.cache_resource
def build_tfidf():
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["clean_text"])
    return vectorizer, matrix

vectorizer, tfidf_matrix = build_tfidf()


# =====================================================================
# HIGHLIGHT FUNCTION
# =====================================================================
def highlight_text(text, query):
    q = re.escape(query)
    style = "background-color:#fff3a3; padding:2px 4px; border-radius:4px;"
    return re.sub(fr"(?i)({q})", rf"<span style='{style}'>\1</span>", text)


# =====================================================================
# GLOBAL CSS
# =====================================================================
st.markdown("""
<style>

body {
    background-color: #f5f6fa;
}

/* Card grid masonry style */
.grid-container {
    column-count: 2;
    column-gap: 20px;
}

.card {
    display: inline-block;
    background: #ffffff;
    padding: 18px;
    margin-bottom: 20px;
    width: 100%;
    border-radius: 12px;
    border: 1px solid #e4e4e4;
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    transition: 0.2s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}

.quote-text {
    font-size: 18px;
    font-style: italic;
    line-height: 1.5;
    color: #333;
}

.author {
    margin-top: 6px;
    font-weight: 600;
    color: #555;
}

.tag {
    display: inline-block;
    padding: 4px 9px;
    background-color: #1a73e8;
    color: white;
    border-radius: 10px;
    font-size: 11px;
    margin-right: 5px;
    margin-top: 6px;
}

.score-box {
    margin-top: 10px;
    padding: 4px 8px;
    background-color: #eef5ff;
    color: #1a73e8;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)


# =====================================================================
# TITLE & SIDEBAR
# =====================================================================
st.sidebar.title("Navigation")
st.sidebar.info("Quotes Retrieval & Recommendation System")

mode = st.sidebar.radio("Pilih Mode:", ["Pencarian TF-IDF", "Rekomendasi Berdasarkan Tags"])

st.title("Quotes Retrieval & Recommendation System")
st.write("Sistem pencarian dan rekomendasi menggunakan TF-IDF dan tag relevance scoring.")


# =====================================================================
# MODE 1 – TF-IDF SEARCH
# =====================================================================
if mode == "Pencarian TF-IDF":

    st.subheader("Pencarian Berdasarkan Query Kata Kunci")

    query = st.text_input("Masukkan query pencarian:")

    if query:
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        df["score"] = scores
        results = df.sort_values(by="score", ascending=False).head(10)

        st.write("Hasil Pencarian Teratas")

        st.markdown("<div class='grid-container'>", unsafe_allow_html=True)

        for _, row in results.iterrows():

            highlighted_text = highlight_text(row["clean_text"], query)
            tags_html = "".join([f"<span class='tag'>{t}</span>" for t in row['tags_list']])

            st.markdown(f"""
                <div class="card">
                    <div class="quote-text">"{highlighted_text}"</div>
                    <div class="author">— {row['author']}</div>
                    {tags_html}
                    <div class="score-box">Score: {row['score']:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# MODE 2 — TAG RECOMMENDATION
# =====================================================================
else:
    st.subheader("Rekomendasi Berdasarkan Tags")

    all_tags = sorted({t for sub in df["tags_list"] for t in sub})
    selected = st.multiselect("Pilih Tags:", all_tags)

    if selected:
        def match_score(row):
            return len(set(row["tags_list"]) & set(selected))

        df["tag_score"] = df.apply(match_score, axis=1)

        results = df[df["tag_score"] > 0].sort_values(by="tag_score", ascending=False)

        st.write("Hasil Rekomendasi")

        st.markdown("<div class='grid-container'>", unsafe_allow_html=True)

        for _, row in results.iterrows():
            tags_html = "".join([f"<span class='tag'>{t}</span>" for t in row['tags_list']])

            st.markdown(f"""
                <div class="card">
                    <div class="quote-text">"{row['text']}"</div>
                    <div class="author">— {row['author']}</div>
                    {tags_html}
                    <div class="score-box">Match: {row['tag_score']}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if results.empty:
            st.warning("Tidak ada hasil yang cocok.")
