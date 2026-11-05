import re
from pathlib import Path

import arxiv
import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
DATA_DIR, ART_DIR = ROOT / "data", ROOT / "artifacts"
K = 6
TITLE_COL, SUMMARY_COL = "title", "summary"

CLUSTER_NAMES = {}


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
wnl = WordNetLemmatizer()


def preprocessing(text):
    text_clean = re.sub("[^a-zA-Z]", " ", text)
    text_clean = text_clean.lower()
    text_clean = text_clean.split()

    sw = set(stopwords.words("english"))

    text_clean = [wnl.lemmatize(word) for word in text_clean if not word in sw]
    text_final = " ".join(text_clean)
    return text_final


def build_text(title: str, summary: str) -> str:
    """Raw text before preprocessing. Must match how df['text'] was built."""
    return f"{title} {summary}"


@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")


@st.cache_resource
def load_artifacts():
    reducer = joblib.load(ART_DIR / "umap10.joblib")
    X_norm = normalize(np.load(DATA_DIR / "arxiv_embeddings_v2.npy"))
    papers = pd.read_csv(DATA_DIR / f"data_clustered_k{K}.csv")
    centroids = (
        pd.read_csv(DATA_DIR / f"centroids_k{K}.csv").drop(columns="cluster").to_numpy()
    )
    reducer.transform(X_norm[:1])
    return reducer, X_norm, papers, centroids


@st.cache_data(show_spinner=False)
def fetch_summary(title: str):
    """Look up a paper by title on the arXiv API; returns (title, summary) or None."""
    search = arxiv.Search(query=f'ti:"{title}"', max_results=1)
    result = next(arxiv.Client().results(search), None)
    if result is None:
        return None
    return " ".join(result.title.split()), " ".join(result.summary.split())


def recommend(title: str, summary: str, n: int = 3):
    reducer, X_norm, papers, centroids = load_artifacts()

    text = preprocessing(build_text(title, summary))
    vec = normalize(load_model().encode([text]))

    vec_10d = reducer.transform(vec)
    dists = np.linalg.norm(centroids - vec_10d, axis=1)
    cluster = int(dists.argmin())

    idx = np.where(papers["cluster"].to_numpy() == cluster)[0]
    sims = X_norm[idx] @ vec[0]
    order = sims.argsort()[::-1]

    same = papers[TITLE_COL].str.lower().str.strip().to_numpy()[idx] == title.lower().strip()
    order = [i for i in order if not same[i]][:n]
    return cluster, papers.iloc[idx[order]].assign(similarity=sims[order])



st.title("arXiv paper recommender")

missing = {TITLE_COL, SUMMARY_COL} - set(load_artifacts()[2].columns)
if missing:
    st.error(f"Columns {missing} not found. Available: {list(load_artifacts()[2].columns)}")
    st.stop()

title = st.text_input("Research paper title")

if title:
    with st.spinner("Searching arXiv..."):
        found = fetch_summary(title)

    if found:
        found_title, summary = found
        st.caption(f"Found on arXiv: {found_title}")
        with st.expander("Summary of the searched paper"):
            st.write(summary)
    else:
        st.warning("Paper not found on arXiv. Paste its abstract instead.")
        found_title, summary = title, st.text_area("Abstract")

    if summary:
        with st.spinner("Finding similar papers..."):
            cluster, recs = recommend(found_title, summary)
        st.subheader(CLUSTER_NAMES.get(cluster, f"Cluster {cluster}"))
        for _, row in recs.iterrows():
            st.markdown(f"**{row[TITLE_COL]}** (similarity {row['similarity']:.2f})")
            st.write(row[SUMMARY_COL])