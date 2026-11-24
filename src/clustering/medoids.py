"""k-means / medoid selection.

Purpose
-------
Diversity clustering (PLAN.md section 4.2, STEPS 2-3): embed the prompt pool
with the sentence encoder (bge-m3), run k-means with k clusters, then select the
medoid m_j of each cluster (the prompt whose embedding is closest to the cluster
centroid) to form the final decorrelated prompt set {m_1 .. m_k}. k-means is the
default; hierarchical and HDBSCAN are ablation alternatives (section 7.4).

Implements the selection in PLAN.md section 8.5 (KMeans random_state=42; medoid =
min distance to centroid), using the project's Encoder wrapper for embeddings so
the model is loaded lazily rather than at import time.

Maps to: Week 3, Day 2 (PLAN.md section 10) — "Embed pool with bge-m3; cluster
with k-means; select k=5 medoids."

Run from the repo root:
    python -m src.clustering.medoids           # uses data/prompts/pool.json
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.embedding.encoder import Encoder

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
POOL_PATH = PROMPTS_DIR / "pool.json"
MEDOIDS_PATH = PROMPTS_DIR / "medoids.json"

DEFAULT_K = 5
RANDOM_STATE = 42


def select_medoids(prompts, k=DEFAULT_K, encoder=None, random_state=RANDOM_STATE):
    """Cluster prompt embeddings with k-means and return each cluster's medoid.

    Follows PLAN.md section 8.5. Returns (medoids, labels, medoid_indices) where
    `medoids` is the list of selected prompt strings (one per cluster), `labels`
    is the per-prompt cluster assignment, and `medoid_indices` indexes into
    `prompts`.
    """
    encoder = encoder or Encoder()
    E = encoder.encode(prompts)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(E)

    medoids, medoid_indices = [], []
    for c in range(k):
        idxs = np.where(km.labels_ == c)[0]
        centroid = km.cluster_centers_[c]
        best = min(idxs, key=lambda i: np.linalg.norm(E[i] - centroid))
        medoids.append(prompts[best])
        medoid_indices.append(int(best))
    return medoids, km.labels_, medoid_indices


def _load_pool(pool_path):
    data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    prompts = data["prompts"]
    cat_of = {it["text"]: it["category"]
              for it in data.get("prompts_with_category", [])}
    return prompts, cat_of, data.get("metadata", {})


def build_medoids(pool_path=POOL_PATH, k=DEFAULT_K, encoder_name=None,
                  out_path=MEDOIDS_PATH, verbose=True):
    """Load the pool, select k medoids, save them to medoids.json, and print."""
    prompts, cat_of, pool_meta = _load_pool(pool_path)
    encoder = Encoder(encoder_name) if encoder_name else Encoder()

    medoids, labels, medoid_indices = select_medoids(prompts, k=k, encoder=encoder)

    records = []
    for cluster_id, (idx, text) in enumerate(zip(medoid_indices, medoids)):
        members = [prompts[i] for i in np.where(labels == cluster_id)[0]]
        records.append({"cluster": cluster_id, "text": text,
                        "category": cat_of.get(text), "pool_index": idx,
                        "cluster_size": len(members), "members": members})

    meta = {"date": str(date.today()), "encoder": encoder.model_name, "k": k,
            "random_state": RANDOM_STATE, "pool_size": len(prompts),
            "source_pool": str(Path(pool_path).name),
            "source_pool_encoder": pool_meta.get("encoder")}
    payload = {"metadata": meta,
               "medoids": medoids,
               "medoids_detail": records}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if verbose:
        print(f"Selected {k} medoids from {len(prompts)} prompts "
              f"(encoder={encoder.model_name}, random_state={RANDOM_STATE})\n")
        for r in records:
            cat = f"[{r['category']}]" if r["category"] else "[?]"
            print(f"  cluster {r['cluster']} (size {r['cluster_size']}) "
                  f"{cat}  {r['text']}")
        print(f"\nSaved to {out_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Select k prompt medoids.")
    parser.add_argument("--pool", default=str(POOL_PATH))
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--encoder", default=None,
                        help="override encoder model (default: bge-m3)")
    parser.add_argument("--out", default=str(MEDOIDS_PATH))
    args = parser.parse_args()
    build_medoids(pool_path=args.pool, k=args.k, encoder_name=args.encoder,
                  out_path=args.out)


if __name__ == "__main__":
    main()
