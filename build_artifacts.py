from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import normalize

DATA_DIR, ART_DIR = Path("data"), Path("artifacts")
ART_DIR.mkdir(exist_ok=True)
K = 6

X_norm = normalize(np.load(DATA_DIR / "arxiv_embeddings_v2.npy"))
papers = pd.read_csv(DATA_DIR / f"data_clustered_k{K}.csv")
centroids = (
    pd.read_csv(DATA_DIR / f"centroids_k{K}.csv").drop(columns="cluster").to_numpy()
)
assert len(papers) == len(X_norm), "CSV rows and embeddings are not aligned"


reducer = umap.UMAP(
    n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42
)
X_umap = reducer.fit_transform(X_norm)


labels = papers["cluster"].to_numpy()
rebuilt = np.vstack([X_umap[labels == c].mean(axis=0) for c in range(K)])
print("max centroid gap:", np.abs(rebuilt - centroids).max())

joblib.dump(reducer, ART_DIR / "umap10.joblib")