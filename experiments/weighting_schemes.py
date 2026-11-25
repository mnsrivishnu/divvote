"""Four weighting schemes on the cached k=5 votes (PLAN.md sections 4.4 / 7.4).

Re-aggregates the SAME cached DivVote vote tensors (uniform collection pass) four
ways and reports SHD per benchmark side by side:
  - uniform            w_j = 1/k
  - cluster-size       w_j = |C_j| / sum|C_i|
  - inverse-variance   w_j ~ 1 / sigma_j^2   (sigma_j^2 = cluster embedding spread)
  - learned-reliability w_j = r_j^gamma / sum r_i^gamma  (r learned on Asia, gamma=1)

No new API calls: every scheme aggregates the identical cached (k, n, n) tensor,
so only the weight vector changes. The learned-reliability column is IN-SAMPLE on
Asia (weights are learned there) and flagged as such.

Run from the repo root:
    python -m experiments.weighting_schemes
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402

from src.metrics import metrics                      # noqa: E402
from src.voting import baseline, divvote             # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
VALIDATION = "asia"
GAMMA = 1.0
SCHEMES = ["uniform", "cluster-size", "inverse-variance", "learned-reliability"]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    medoids = divvote.load_medoids()
    k = len(medoids)

    # Build the weight vectors once (all derived from the k=5 medoids/pool; the
    # reliability weights need the cached Asia votes).
    w_uniform = np.ones(k) / k
    w_size = divvote.cluster_size_weights()
    w_invvar = divvote.inverse_variance_weights()
    w_rel, r = divvote.reliability_weights(validation=VALIDATION, gamma=GAMMA)
    variances = divvote.cluster_embedding_variances()
    weights = {"uniform": w_uniform, "cluster-size": w_size,
               "inverse-variance": w_invvar, "learned-reliability": w_rel}

    print(f"k = {k} medoids; weights per medoid (cluster order)")
    print(f"{'medoid':<7}{'|C_j|':>7}{'sigma^2':>10}{'r_j(asia)':>11}"
          f"{'w_unif':>9}{'w_size':>9}{'w_ivar':>9}{'w_rel':>9}   prompt")
    print("-" * 100)
    detail = divvote._load_medoids_detail()
    for j in range(k):
        print(f"m{j:<6}{detail[j]['cluster_size']:>7}{variances[j]:>10.4f}"
              f"{r[j]:>11.3f}{w_uniform[j]:>9.3f}{w_size[j]:>9.3f}"
              f"{w_invvar[j]:>9.3f}{w_rel[j]:>9.3f}   {medoids[j][:40]}")

    rows = []
    for name, style in divvote.ALL_SPECS:
        true_adj, nodes = baseline.load_ground_truth(name)
        states = divvote._resolve_states(name, style, nodes, None)
        V = divvote.collect_votes(nodes, medoids, prompt_style=style,
                                  states=states, use_cache=True)
        shd = {s: metrics.shd(divvote.aggregate_votes(V, weights=weights[s])[0],
                              true_adj) for s in SCHEMES}
        base = baseline.run_baseline(name, prompt_style=style, verbose=False)
        rows.append((name, base["SHD"], shd))

    print("\nSHD by weighting scheme (k=5, cached votes re-aggregated) vs "
          "7-verb baseline (Raw)")
    print(f"{'benchmark':<20}{'Base':>6}{'uniform':>9}{'size':>9}"
          f"{'inv-var':>9}{'learned':>9}")
    print("-" * 62)
    for name, b, shd in rows:
        line = f"{name:<20}{b:>6}"
        for s in SCHEMES:
            line += f"{shd[s]:>9}"
        print(line)
    print("\n(SHD, lower is better. 'learned' is IN-SAMPLE on Asia -- weights "
          "are learned there; treat Asia's learned cell as not held-out.)")

    csv_path = os.path.join(RESULTS_DIR, "weighting_schemes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "baseline_SHD"] + [f"SHD_{s}" for s in SCHEMES])
        for name, b, shd in rows:
            w.writerow([name, b] + [shd[s] for s in SCHEMES])
    print(f"\nSaved {os.path.relpath(csv_path, REPO_ROOT)}")
    return rows


if __name__ == "__main__":
    run()
