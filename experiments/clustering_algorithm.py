"""Clustering-algorithm ablation (PLAN.md section 7.4): k-means vs hierarchical
vs HDBSCAN on the same bge-m3 k=5 prompt pool.

Re-clusters the identical pool embeddings three ways, selects each cluster's
medoid (member closest to the cluster centroid = mean of member embeddings),
saves the medoid sets, runs uniform DivVote on all five benchmarks (reference
style), and reports SHD per benchmark for the three algorithms. k-means is the
canonical k=5 set (data/prompts/medoids.json); hierarchical uses n_clusters=5 to
match; HDBSCAN chooses its own cluster count (reported, with any noise points).

New medoids mean fresh LLM votes; medoids shared with the k-means set reuse the
on-disk cache, so only genuinely new medoid prompts cost API calls.

Run from the repo root:
    python -m experiments.clustering_algorithm
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans  # noqa: E402

from src.embedding.encoder import Encoder            # noqa: E402
from src.metrics import metrics                      # noqa: E402
from src.voting import baseline, divvote             # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(REPO_ROOT, "data", "prompts")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
POOL_PATH = os.path.join(PROMPTS_DIR, "pool.json")
K = 5
RANDOM_STATE = 42
STYLE = "reference"
BENCHMARKS = ["asia", "cancer", "medical_diagnostic", "covid19", "endorisk"]


def medoids_from_labels(prompts, E, labels, cat_of, algo, encoder_name):
    """Select one medoid per (non-noise) cluster; return a medoids.json payload.

    Medoid = member whose embedding is closest to the cluster centroid (mean of
    member embeddings). Label -1 (HDBSCAN noise) is skipped. Clusters are
    renumbered 0..m-1 in ascending original-label order.
    """
    clusters = sorted(set(int(c) for c in labels) - {-1})
    medoids, records = [], []
    for new_id, c in enumerate(clusters):
        idxs = np.where(labels == c)[0]
        centroid = E[idxs].mean(axis=0)
        best = int(min(idxs, key=lambda i: np.linalg.norm(E[i] - centroid)))
        members = [prompts[i] for i in idxs]
        medoids.append(prompts[best])
        records.append({"cluster": new_id, "text": prompts[best],
                        "category": cat_of.get(prompts[best]),
                        "pool_index": best, "cluster_size": len(members),
                        "members": members})
    n_noise = int(np.sum(np.asarray(labels) == -1))
    payload = {"metadata": {"date": str(date.today()), "encoder": encoder_name,
                            "algorithm": algo, "k": len(clusters),
                            "random_state": RANDOM_STATE,
                            "pool_size": len(prompts), "n_noise": n_noise},
               "medoids": medoids, "medoids_detail": records}
    return payload, n_noise


def category_summary(records, cat_of):
    """Per-cluster category composition string, for the merge/split readout."""
    out = []
    for rec in records:
        cats = defaultdict(int)
        for m in rec["members"]:
            cats[cat_of.get(m, "?")] += 1
        tag = "MERGED" if len(cats) > 1 else "pure"
        out.append(f"    cluster {rec['cluster']} (size {rec['cluster_size']:>2}) "
                   f"[{tag:>6}]  "
                   + ", ".join(f"{c}:{n}" for c, n in sorted(cats.items())))
    return out


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pool = json.loads(open(POOL_PATH, encoding="utf-8").read())
    prompts = pool["prompts"]
    cat_of = {it["text"]: it["category"]
              for it in pool.get("prompts_with_category", [])}

    encoder = Encoder()
    E = encoder.encode(prompts)                      # L2-normalized
    name_enc = encoder.model_name

    # Labels per algorithm on the SAME embeddings.
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(E)
    hier = AgglomerativeClustering(n_clusters=K).fit(E)
    hdb = HDBSCAN(min_cluster_size=2, min_samples=1).fit(E)
    algo_labels = {"kmeans": km.labels_, "hierarchical": hier.labels_,
                   "hdbscan": hdb.labels_}

    # Build + save medoid sets. k-means reuses the canonical medoids.json.
    paths, payloads = {}, {}
    for algo, labels in algo_labels.items():
        payload, n_noise = medoids_from_labels(prompts, E, labels, cat_of, algo,
                                               name_enc)
        if algo == "kmeans":
            path = os.path.join(PROMPTS_DIR, "medoids.json")   # canonical k=5
        else:
            path = os.path.join(PROMPTS_DIR, f"medoids_{algo}.json")
            open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
        paths[algo] = path
        payloads[algo] = payload
        print(f"{algo:<13} clusters={payload['metadata']['k']}  "
              f"noise={n_noise}  -> {os.path.relpath(path, REPO_ROOT)}")

    print("\nCluster composition per algorithm (seed-category merges)")
    for algo in algo_labels:
        print(f"\n--- {algo} (k={payloads[algo]['metadata']['k']}, "
              f"noise={payloads[algo]['metadata']['n_noise']}) ---")
        for line in category_summary(payloads[algo]["medoids_detail"], cat_of):
            print(line)

    # DivVote (uniform) per algorithm per benchmark; baseline once.
    base = {n: baseline.run_baseline(n, prompt_style=STYLE, verbose=False)["SHD"]
            for n in BENCHMARKS}
    shd = defaultdict(dict)
    for algo in algo_labels:
        print(f"\nDivVote (uniform) with {algo} medoids "
              f"(k={payloads[algo]['metadata']['k']})")
        for n in BENCHMARKS:
            res = divvote.run_divvote(n, prompt_style=STYLE,
                                      medoids_path=paths[algo], verbose=False)
            shd[algo][n] = res["SHD"]
            print(f"  {n:<20} SHD={res['SHD']:>3}")

    print("\n" + "=" * 62)
    print("SUMMARY  SHD by clustering algorithm vs 7-verb baseline (Raw)")
    print("=" * 62)
    print(f"{'benchmark':<20}{'Base':>6}{'k-means':>10}{'hier':>10}{'hdbscan':>10}")
    print("-" * 56)
    for n in BENCHMARKS:
        print(f"{n:<20}{base[n]:>6}{shd['kmeans'][n]:>10}"
              f"{shd['hierarchical'][n]:>10}{shd['hdbscan'][n]:>10}")
    print(f"\n(k-means/hier at k={K}; hdbscan k="
          f"{payloads['hdbscan']['metadata']['k']}, "
          f"min_cluster_size=2. SHD lower is better.)")

    csv_path = os.path.join(RESULTS_DIR, "clustering_algorithm.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "baseline_SHD", "SHD_kmeans",
                    "SHD_hierarchical", "SHD_hdbscan", "k_kmeans",
                    "k_hierarchical", "k_hdbscan"])
        for n in BENCHMARKS:
            w.writerow([n, base[n], shd["kmeans"][n], shd["hierarchical"][n],
                        shd["hdbscan"][n], K, K,
                        payloads["hdbscan"]["metadata"]["k"]])
    print(f"\nSaved {os.path.relpath(csv_path, REPO_ROOT)}")
    return base, shd, payloads


if __name__ == "__main__":
    run()
