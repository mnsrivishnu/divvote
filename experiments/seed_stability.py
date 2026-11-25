"""Seed-stability / statistical-rigor pass (PLAN.md section 7.2).

Runs DivVote at k=5 (bge-m3, reference) across many k-means random seeds. For
each seed we re-cluster the SAME pool embeddings, re-select medoids, and
re-aggregate -- for both uniform and cluster-size weighting -- then compare to
the fixed 7-verb Raw baseline.

Cost is capped regardless of seed count: we vote every pool prompt x ordered
pair ONCE per benchmark (cached), giving a full (P, n, n) tensor. Each seed's
medoids are a subset of pool prompts, so its aggregation just slices and
re-weights that cached tensor -- zero extra API calls per seed.

Reports, per benchmark and weighting scheme:
  - mean +/- std SHD over seeds, plus min/max;
  - how often DivVote beats the baseline across seeds;
  - whether cluster-size's advantage over uniform is stable or within noise
    (paired per-seed delta);
  - a significance statement (is the baseline outside the seed SHD
    distribution; z = (baseline - mean) / std).

Run from the repo root:
    python -m experiments.seed_stability
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                   # noqa: E402
from sklearn.cluster import KMeans                   # noqa: E402

from src.embedding.encoder import Encoder            # noqa: E402
from src.metrics import metrics                      # noqa: E402
from src.voting import baseline, divvote             # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(REPO_ROOT, "data", "prompts")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
POOL_PATH = os.path.join(PROMPTS_DIR, "pool.json")
K = 5
N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
STYLE = "reference"
BENCHMARKS = ["asia", "cancer", "medical_diagnostic", "covid19", "endorisk"]


def medoids_for_seed(E, seed):
    """k-means at random_state=seed; return (medoid_indices, cluster_sizes)."""
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(E)
    idx, sizes = [], []
    for c in range(K):
        members = np.where(km.labels_ == c)[0]
        centroid = km.cluster_centers_[c]
        best = int(min(members, key=lambda i: np.linalg.norm(E[i] - centroid)))
        idx.append(best)
        sizes.append(len(members))
    return idx, np.array(sizes, dtype=float)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pool = json.loads(open(POOL_PATH, encoding="utf-8").read())
    prompts = pool["prompts"]
    encoder = Encoder()
    E = encoder.encode(prompts)                      # bge-m3, L2-normalized

    # Pre-vote every pool prompt x pair once per benchmark (bounded cost).
    print(f"Pre-voting full pool ({len(prompts)} prompts) per benchmark "
          f"(cached; cost is capped regardless of the {N_SEEDS} seeds)")
    Vfull, base = {}, {}
    for name in BENCHMARKS:
        true_adj, nodes = baseline.load_ground_truth(name)
        states = divvote._resolve_states(name, STYLE, nodes, None)
        Vfull[name] = (divvote.collect_votes(nodes, prompts, prompt_style=STYLE,
                                             states=states, use_cache=True),
                       true_adj)
        base[name] = baseline.run_baseline(name, prompt_style=STYLE,
                                           verbose=False)["SHD"]
        print(f"  {name:<20} pool votes cached; baseline SHD={base[name]}")

    # Per seed: slice medoids, aggregate uniform + cluster-size, score SHD.
    # shd[name][scheme] = list over seeds.
    shd = {n: {"uniform": [], "cluster-size": []} for n in BENCHMARKS}
    for seed in SEEDS:
        idx, sizes = medoids_for_seed(E, seed)
        w_uniform = np.ones(K) / K
        w_size = sizes / sizes.sum()
        for name in BENCHMARKS:
            V, true_adj = Vfull[name]
            Vseed = V[idx]                           # (K, n, n) from cache
            adj_u = divvote.aggregate_votes(Vseed, weights=w_uniform)[0]
            adj_s = divvote.aggregate_votes(Vseed, weights=w_size)[0]
            shd[name]["uniform"].append(metrics.shd(adj_u, true_adj))
            shd[name]["cluster-size"].append(metrics.shd(adj_s, true_adj))

    # Report.
    print(f"\n{'='*86}")
    print(f"SEED STABILITY  k=5, bge-m3, reference, {N_SEEDS} k-means seeds "
          f"(random_state 0..{N_SEEDS-1})")
    print(f"{'='*86}")
    print(f"{'benchmark':<18}{'Base':>5}{'scheme':>14}{'mean':>7}{'std':>7}"
          f"{'min':>5}{'max':>5}{'beat base':>11}{'z=(B-mu)/sd':>13}")
    print("-" * 86)
    summary_rows = []
    for name in BENCHMARKS:
        b = base[name]
        for scheme in ("uniform", "cluster-size"):
            arr = np.array(shd[name][scheme], dtype=float)
            mu, sd = float(arr.mean()), float(arr.std(ddof=0))
            n_beat = int(np.sum(arr < b))
            n_tie = int(np.sum(arr == b))
            z = (b - mu) / sd if sd > 0 else float("inf")
            print(f"{name:<18}{b:>5}{scheme:>14}{mu:>7.2f}{sd:>7.2f}"
                  f"{int(arr.min()):>5}{int(arr.max()):>5}"
                  f"{n_beat:>4}/{N_SEEDS} ({n_tie} tie){z:>+9.2f}")
            summary_rows.append((name, scheme, b, mu, sd, int(arr.min()),
                                 int(arr.max()), n_beat, n_tie, z))

    # Cluster-size vs uniform: paired per-seed delta.
    print(f"\n{'='*70}")
    print("CLUSTER-SIZE vs UNIFORM  (paired per-seed delta = size_SHD - "
          "uniform_SHD)")
    print(f"{'='*70}")
    print(f"{'benchmark':<20}{'mean d':>8}{'std d':>7}{'size<unif':>11}"
          f"{'tie':>5}{'size>unif':>11}")
    print("-" * 62)
    paired_rows = []
    for name in BENCHMARKS:
        u = np.array(shd[name]["uniform"], dtype=float)
        s = np.array(shd[name]["cluster-size"], dtype=float)
        d = s - u
        better = int(np.sum(d < 0))
        tie = int(np.sum(d == 0))
        worse = int(np.sum(d > 0))
        print(f"{name:<20}{d.mean():>+8.2f}{d.std(ddof=0):>7.2f}"
              f"{better:>11}{tie:>5}{worse:>11}")
        paired_rows.append((name, float(d.mean()), float(d.std(ddof=0)),
                            better, tie, worse))

    # Significance statements.
    print(f"\n{'='*70}")
    print("SIGNIFICANCE (is the baseline outside DivVote's seed distribution?)")
    print(f"{'='*70}")
    for name in BENCHMARKS:
        b = base[name]
        for scheme in ("uniform", "cluster-size"):
            arr = np.array(shd[name][scheme], dtype=float)
            allbeat = bool(np.all(arr < b))
            worst = int(arr.max())
            verdict = ("baseline ABOVE every seed (strictly beaten by all "
                       f"{N_SEEDS})" if allbeat else
                       f"baseline within/below seed range (worst seed SHD "
                       f"{worst} vs base {b})")
            print(f"  {name:<18} {scheme:<13} {verdict}")

    # Persist.
    with open(os.path.join(RESULTS_DIR, "seed_stability.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "scheme", "baseline_SHD", "mean_SHD",
                    "std_SHD", "min_SHD", "max_SHD", "n_beat_base",
                    "n_tie_base", "n_seeds", "z_base_minus_mean_over_std"])
        for (name, scheme, b, mu, sd, mn, mx, nb, nt, z) in summary_rows:
            w.writerow([name, scheme, b, f"{mu:.4f}", f"{sd:.4f}", mn, mx,
                        nb, nt, N_SEEDS, f"{z:.4f}"])
    with open(os.path.join(RESULTS_DIR, "seed_stability_paired.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "mean_delta_size_minus_uniform", "std_delta",
                    "n_size_better", "n_tie", "n_size_worse", "n_seeds"])
        for (name, md, sdd, bt, ti, wo) in paired_rows:
            w.writerow([name, f"{md:.4f}", f"{sdd:.4f}", bt, ti, wo, N_SEEDS])
    print("\nSaved results/seed_stability.csv and "
          "results/seed_stability_paired.csv")
    return summary_rows, paired_rows


if __name__ == "__main__":
    run()
