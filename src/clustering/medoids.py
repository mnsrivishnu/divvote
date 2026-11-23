"""k-means / medoid selection.

Purpose
-------
Diversity clustering (PLAN.md section 4.2, STEP 3): run k-means on the prompt
embeddings {e_i}, then select the medoid m_j of each cluster (the prompt
closest to its cluster centroid) to form the final decorrelated prompt set
{m_1 .. m_k}. k-means is the default; hierarchical and HDBSCAN are ablation
alternatives (section 7.4).

Maps to: Week 3, Day 2 (PLAN.md section 10) — "cluster with k-means; select
k=5 medoids."

Implements the skeleton given in PLAN.md section 8.5. No logic beyond it.
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

encoder = SentenceTransformer("BAAI/bge-m3")


def select_medoids(prompts, k=5):
    E = encoder.encode(prompts)
    km = KMeans(n_clusters=k, random_state=42).fit(E)
    medoids = []
    for c in range(k):
        idxs = np.where(km.labels_ == c)[0]
        centroid = km.cluster_centers_[c]
        best = min(idxs, key=lambda i: np.linalg.norm(E[i] - centroid))
        medoids.append(prompts[best])
    return medoids
