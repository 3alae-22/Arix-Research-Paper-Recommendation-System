import re
from pathlib import Path

import arxiv
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

BASE_MODEL = "BAAI/bge-base-en-v1.5"
ROOT = Path(__file__).resolve().parent
DATA_DIR, ART_DIR, MODEL_DIR = ROOT / "data", ROOT / "artifacts", ROOT / "models"
K = 9
TITLE_COL, SUMMARY_COL = "title", "summary"

CLUSTER_NAMES = {
    0: "Control, Robotics & Power Systems",
    1: "Computer Vision & Imaging",
    2: "Plasma Physics & Fluid Dynamics",
    3: "LLMs, Agents & Machine Learning",
    4: "Neuroscience & Brain Dynamics",
    5: "Molecular & Computational Biology",
    6: "Speech & Audio Processing",
    7: "Wireless Communications & Sensing",
    8: "Optics, Photonics & Quantum Physics",
}

st.set_page_config(
    page_title="arXiv paper recommender",
    page_icon="📄",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

    :root { --accent: #0f766e; }

    .block-container { max-width: 800px; padding-top: 2.5rem; }

    h1, h2, h3 { font-family: 'Newsreader', Georgia, serif; letter-spacing: -0.01em; }
    h1 { font-size: 2.6rem !important; margin-bottom: 0.2rem; }

    .subtitle { font-size: 1.05rem; opacity: 0.72; margin: 0 0 1.8rem 0; max-width: 60ch; }

    .topic {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        border: 1px solid var(--accent);
        color: var(--accent);
        font-weight: 600;
        font-size: 0.95rem;
        margin: 0.2rem 0 1.2rem 0;
    }

    .paper-title { font-weight: 600; font-size: 1.05rem; line-height: 1.4; margin-bottom: 0.4rem; }

    .stProgress > div > div > div > div { background-color: var(--accent); }

    [data-testid="stSidebar"] h3 { font-size: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def preprocessing(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\$([^$]*)\$", r"\1", text)
    return text.strip()


def build_text(title: str, summary: str) -> str:
    """Raw text before preprocessing. Must match how df['text'] was built."""
    return f"{title} {summary}"


@st.cache_resource
def load_model():
    tuned_model = SentenceTransformer(
        BASE_MODEL,
        device="cpu"
    )

    tuned_model.load_adapter(
        str(MODEL_DIR / "bge-arxiv-lora")
    )

    tuned_model.eval()

    return tuned_model


@st.cache_resource
def load_artifacts():
    reducer = joblib.load(ART_DIR / "umap10.joblib")
    X_norm = normalize(np.load(DATA_DIR / "arxiv_embeddings_v2.npy"))
    papers = pd.read_csv(DATA_DIR / f"data_clustered_k{K}_v2.csv")
    centroids = (
        pd.read_csv(DATA_DIR / f"centroids_k{K}_v2.csv").drop(columns="cluster").to_numpy()
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


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. Enter the title of a paper.\n"
        "2. Its abstract is fetched from arXiv.\n"
        "3. The paper is placed in one of "
        f"{K} research topics.\n"
        "4. The closest papers in that topic are returned."
    )
    st.divider()
    st.caption(f"Embeddings: `{BASE_MODEL}` fine-tuned with LoRA")
    st.caption(f"Topics: {K} clusters on a 10D UMAP projection")

st.title("arXiv paper recommender")
st.markdown(
    '<p class="subtitle">Enter a paper title and find similar work in the same research topic.</p>',
    unsafe_allow_html=True,
)

missing = {TITLE_COL, SUMMARY_COL} - set(load_artifacts()[2].columns)
if missing:
    st.error(f"Columns {missing} not found. Available: {list(load_artifacts()[2].columns)}")
    st.stop()

title = st.text_input(
    "Research paper title",
    placeholder="e.g. Attention Is All You Need",
)

if not title:
    st.info("Type a title above to get recommendations.", icon="🔎")

if title:
    with st.spinner("Searching arXiv..."):
        found = fetch_summary(title)

    if found:
        found_title, summary = found
        with st.container(border=True):
            st.caption("Found on arXiv")
            st.markdown(f'<div class="paper-title">{found_title}</div>', unsafe_allow_html=True)
            with st.expander("Abstract"):
                st.write(summary)
    else:
        st.warning("Paper not found on arXiv. Paste its abstract instead.")
        found_title, summary = title, st.text_area("Abstract", height=180)

    if summary:
        with st.spinner("Finding similar papers..."):
            cluster, recs = recommend(found_title, summary)

        st.markdown("### Research topic")
        st.markdown(
            f'<span class="topic">{CLUSTER_NAMES.get(cluster, f"Cluster {cluster}")}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("### Similar papers")
        for _, row in recs.iterrows():
            with st.container(border=True):
                st.markdown(f'<div class="paper-title">{row[TITLE_COL]}</div>', unsafe_allow_html=True)
                sim = float(row["similarity"])
                bar, value = st.columns([5, 1])
                bar.progress(max(0.0, min(1.0, sim)))
                value.caption(f"{sim:.2f}")
                with st.expander("Abstract"):
                    st.write(row[SUMMARY_COL])