"""Cluster-count (k) ablation for DivVote medoid selection (PLAN.md section 7.4).

Re-selects medoids from the SAME bge-m3 prompt pool at k = 3, 7, 10 (k=5 is the
canonical set already in data/prompts/medoids.json), identical encoder and
random_state=42, and saves each new set to data/prompts/medoids_k{k}.json. For
every k it runs DivVote with UNIFORM weighting on all five benchmarks in
reference style and scores SHD/TENE/TERE against the 7-verb Raw baseline (whose
numbers do not depend on k, so they are computed once).

Two outputs:
  1. One summary table -- per benchmark, baseline SHD vs DivVote SHD at
     k = 3, 5, 7, 10, with delta and rel% in each cell.
  2. A seed-category merge/split report per k: which of the six pool categories
     (direct, risk, counterfactual, interventional, mechanistic, temporal) share
     a cluster (merged) and which are spread across clusters (split). We already
     saw direct+mechanistic merge at k=5; this shows whether k>=6 splits them.

Medoids are actual pool prompts, so a medoid shared across k reuses the on-disk
LLM cache -- only genuinely new medoid prompts cost fresh API calls.

Run from the repo root:
    python -m experiments.cluster_count_ablation
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import medoids as medoids_mod   # noqa: E402
from src.metrics import metrics                      # noqa: E402
from src.voting import baseline, divvote             # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(REPO_ROOT, "data", "prompts")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

K_NEW = [3, 7, 10]            # k=5 already selected (medoids.json)
K_ALL = [3, 5, 7, 10]
BENCHMARKS = ["asia", "cancer", "medical_diagnostic", "covid19", "endorisk"]
STYLE = "reference"

SEED_CATEGORIES = ["direct", "risk", "counterfactual", "interventional",
                   "mechanistic", "temporal"]


def medoids_path_for(k):
    if k == 5:
        return os.path.join(PROMPTS_DIR, "medoids.json")
    return os.path.join(PROMPTS_DIR, f"medoids_k{k}.json")


def ensure_medoids(k):
    """Build medoids_k{k}.json for the new k values; reuse medoids.json for k=5."""
    path = medoids_path_for(k)
    if k == 5:
        print(f"k=5: reusing canonical {os.path.relpath(path, REPO_ROOT)}")
        return path
    print(f"k={k}: selecting medoids (bge-m3, random_state=42) -> "
          f"{os.path.relpath(path, REPO_ROOT)}")
    medoids_mod.build_medoids(k=k, out_path=path, verbose=False)
    return path


def category_report(k):
    """Return (clusters, split_map) describing category merges/splits at k.

    clusters: list of {cluster, size, medoid_category, categories:{cat:count}}.
    split_map: {category: [cluster_ids it appears in]}.
    """
    payload = json.loads(open(medoids_path_for(k), encoding="utf-8").read())
    # Map each pool prompt -> its seed category (medoids_detail lacks per-member
    # categories, so read them from the pool once).
    pool = json.loads(open(os.path.join(PROMPTS_DIR, "pool.json"),
                            encoding="utf-8").read())
    cat_of = {it["text"]: it["category"]
              for it in pool.get("prompts_with_category", [])}

    clusters = []
    cat_to_clusters = defaultdict(set)
    for rec in payload["medoids_detail"]:
        cats = defaultdict(int)
        for member in rec["members"]:
            c = cat_of.get(member, "?")
            cats[c] += 1
            cat_to_clusters[c].add(rec["cluster"])
        clusters.append({
            "cluster": rec["cluster"],
            "size": rec["cluster_size"],
            "medoid_category": rec.get("category"),
            "categories": dict(cats),
        })
    split_map = {c: sorted(cat_to_clusters[c]) for c in SEED_CATEGORIES}
    return clusters, split_map


def print_category_report(k, clusters, split_map):
    print(f"\n--- k={k}: cluster composition (seed-category merges) ---")
    for cl in clusters:
        parts = ", ".join(f"{cat}:{n}" for cat, n in sorted(cl["categories"].items()))
        tag = "MERGED" if len(cl["categories"]) > 1 else "pure"
        print(f"  cluster {cl['cluster']} (size {cl['size']:>2}) [{tag:>6}]  "
              f"{parts}")
    merged = [cl for cl in clusters if len(cl["categories"]) > 1]
    split = {c: cls for c, cls in split_map.items() if len(cls) > 1}
    print(f"  merges: {len(merged)} cluster(s) mix categories; "
          f"splits: {sorted(split) if split else 'none'}")
    dm = split_map["direct"]
    mm = split_map["mechanistic"]
    together = bool(set(dm) & set(mm))
    print(f"  direct clusters={dm}  mechanistic clusters={mm}  "
          f"-> direct+mechanistic {'SHARE a cluster' if together else 'SEPARATED'}")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Select medoids for the new k values (clustering only; no API calls).
    print("=" * 66)
    print("STEP 1  Medoid selection per k (same pool, encoder, random_state=42)")
    print("=" * 66)
    for k in K_ALL:
        ensure_medoids(k)

    # 2. Baseline (7-verb Raw) once per benchmark -- independent of k.
    print("\n" + "=" * 66)
    print("STEP 2  7-verb baseline (Raw, reference) -- computed once")
    print("=" * 66)
    base = {}
    for name in BENCHMARKS:
        b = baseline.run_baseline(name, prompt_style=STYLE, verbose=False)
        base[name] = {"SHD": b["SHD"], "TENE": b["TENE"], "TERE": b["TERE"]}
        print(f"  {name:<20} SHD={b['SHD']:>3}  TENE={b['TENE']:>3}  "
              f"TERE={b['TERE']:>3}")

    # 3. DivVote (uniform) per k per benchmark.
    print("\n" + "=" * 66)
    print("STEP 3  DivVote (uniform) per k per benchmark -- SHD/TENE/TERE")
    print("=" * 66)
    # dv[name][k] = {"SHD":..,"TENE":..,"TERE":..}
    dv = defaultdict(dict)
    for k in K_ALL:
        mpath = medoids_path_for(k)
        print(f"\nk={k}  (medoids: {os.path.relpath(mpath, REPO_ROOT)})")
        for name in BENCHMARKS:
            res = divvote.run_divvote(name, prompt_style=STYLE,
                                      medoids_path=mpath, verbose=False)
            dv[name][k] = {"SHD": res["SHD"], "TENE": res["TENE"],
                           "TERE": res["TERE"]}
            print(f"  {name:<20} SHD={res['SHD']:>3}  TENE={res['TENE']:>3}  "
                  f"TERE={res['TERE']:>3}")

    # 4. Summary table: baseline SHD vs DivVote SHD at each k, delta + rel%.
    print("\n" + "=" * 86)
    print("SUMMARY  Baseline SHD vs DivVote SHD by k  (cell = SHD  dSHD  rel%; "
          "rel% target >= 20%)")
    print("=" * 86)
    header = f"{'benchmark':<20}{'Base':>6}"
    for k in K_ALL:
        header += f"{'k=' + str(k):>18}"
    print(header)
    print("-" * 86)
    for name in BENCHMARKS:
        b = base[name]["SHD"]
        line = f"{name:<20}{b:>6}"
        for k in K_ALL:
            s = dv[name][k]["SHD"]
            d = s - b
            rel = (b - s) / b * 100 if b else 0.0
            line += f"{s:>7}{d:>+5}{rel:>5.0f}%"
        print(line)
    print("\n(dSHD = DivVote - baseline, negative = improvement; "
          "rel% = SHD reduction vs baseline.)")

    # 5. Seed-category merge/split per k.
    print("\n" + "=" * 66)
    print("STEP 5  Seed-category merge/split per k")
    print("=" * 66)
    cat_reports = {}
    for k in K_ALL:
        clusters, split_map = category_report(k)
        cat_reports[k] = (clusters, split_map)
        print_category_report(k, clusters, split_map)

    # 6. Persist.
    csv_path = os.path.join(RESULTS_DIR, "cluster_count_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "k", "baseline_SHD", "divvote_SHD", "dSHD",
                    "rel_pct", "divvote_TENE", "divvote_TERE",
                    "baseline_TENE", "baseline_TERE"])
        for name in BENCHMARKS:
            b = base[name]
            for k in K_ALL:
                d = dv[name][k]
                rel = (b["SHD"] - d["SHD"]) / b["SHD"] if b["SHD"] else 0.0
                w.writerow([name, k, b["SHD"], d["SHD"], d["SHD"] - b["SHD"],
                            f"{rel:.4f}", d["TENE"], d["TERE"],
                            b["TENE"], b["TERE"]])
    print(f"\nSaved {os.path.relpath(csv_path, REPO_ROOT)}")
    return base, dv, cat_reports


if __name__ == "__main__":
    run()
