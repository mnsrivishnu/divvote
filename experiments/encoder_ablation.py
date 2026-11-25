"""Encoder ablation (PLAN.md section 7.4): bge-m3 vs e5-large-v2 vs
all-MiniLM-L6-v2 for medoid selection.

Regenerates the k=5 medoid set with each sentence encoder (same pool,
random_state=42, same k-means medoid rule), then runs uniform DivVote on all
five benchmarks (reference style) and reports SHD per benchmark for the three
encoders. Tests whether bge-m3 was a good choice and how sensitive medoid
selection is to the encoder.

bge-m3 is the canonical set (data/prompts/medoids.json); the other two are saved
to data/prompts/medoids_enc_{tag}.json. New medoids mean fresh LLM votes; medoids
shared with the bge-m3 set reuse the on-disk cache.

Caveat: e5-large-v2 is trained with "query:"/"passage:" prefixes. We embed the
prompts plain (no prefix), identically to bge-m3, so the ablation isolates the
selection pipeline rather than each model's best-case retrieval setup.

Run from the repo root:
    python -m experiments.encoder_ablation
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clustering import medoids as medoids_mod    # noqa: E402
from src.voting import baseline, divvote              # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(REPO_ROOT, "data", "prompts")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
POOL_PATH = os.path.join(PROMPTS_DIR, "pool.json")
K = 5
STYLE = "reference"
BENCHMARKS = ["asia", "cancer", "medical_diagnostic", "covid19", "endorisk"]

# (tag, HuggingFace model name). bge-m3 is canonical (medoids.json).
ENCODERS = [
    ("bge-m3", "BAAI/bge-m3"),
    ("e5-large-v2", "intfloat/e5-large-v2"),
    ("minilm-l6", "sentence-transformers/all-MiniLM-L6-v2"),
]


def medoids_path_for(tag):
    if tag == "bge-m3":
        return os.path.join(PROMPTS_DIR, "medoids.json")
    return os.path.join(PROMPTS_DIR, f"medoids_enc_{tag}.json")


def category_lines(detail, cat_of):
    out = []
    for rec in detail:
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
    cat_of = {it["text"]: it["category"]
              for it in pool.get("prompts_with_category", [])}

    # 1. Build/locate the medoid set for each encoder.
    paths, payloads = {}, {}
    for tag, model in ENCODERS:
        path = medoids_path_for(tag)
        if tag == "bge-m3":
            print(f"{tag:<13} reusing canonical {os.path.relpath(path, REPO_ROOT)}")
        else:
            print(f"{tag:<13} selecting k={K} medoids with {model} ...")
            medoids_mod.build_medoids(k=K, encoder_name=model, out_path=path,
                                      verbose=False)
            print(f"{'':13} -> {os.path.relpath(path, REPO_ROOT)}")
        paths[tag] = path
        payloads[tag] = json.loads(open(path, encoding="utf-8").read())

    # 2. Medoid prompts + cluster composition per encoder.
    print("\nSelected medoids + cluster composition per encoder")
    for tag, _ in ENCODERS:
        print(f"\n--- {tag} ---")
        for line in category_lines(payloads[tag]["medoids_detail"], cat_of):
            print(line)

    # Overlap with the canonical bge-m3 medoid set (how much the choice moves).
    bge_set = set(payloads["bge-m3"]["medoids"])
    print("\nMedoid overlap with bge-m3 (shared medoids reuse the vote cache)")
    for tag, _ in ENCODERS:
        shared = len(set(payloads[tag]["medoids"]) & bge_set)
        print(f"  {tag:<13} {shared}/{K} medoids shared with bge-m3")

    # 3. Baseline once; DivVote uniform per encoder.
    base = {n: baseline.run_baseline(n, prompt_style=STYLE, verbose=False)["SHD"]
            for n in BENCHMARKS}
    shd = defaultdict(dict)
    for tag, _ in ENCODERS:
        print(f"\nDivVote (uniform) with {tag} medoids")
        for n in BENCHMARKS:
            res = divvote.run_divvote(n, prompt_style=STYLE,
                                      medoids_path=paths[tag], verbose=False)
            shd[tag][n] = res["SHD"]
            print(f"  {n:<20} SHD={res['SHD']:>3}")

    # 4. Summary.
    print("\n" + "=" * 64)
    print("SUMMARY  SHD by encoder (uniform DivVote, k=5) vs baseline (Raw)")
    print("=" * 64)
    print(f"{'benchmark':<20}{'Base':>6}{'bge-m3':>10}{'e5-large':>10}"
          f"{'minilm':>10}")
    print("-" * 56)
    for n in BENCHMARKS:
        print(f"{n:<20}{base[n]:>6}{shd['bge-m3'][n]:>10}"
              f"{shd['e5-large-v2'][n]:>10}{shd['minilm-l6'][n]:>10}")
    print("\n(SHD lower is better; bge-m3 is the canonical choice.)")

    csv_path = os.path.join(RESULTS_DIR, "encoder_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "baseline_SHD", "SHD_bge_m3",
                    "SHD_e5_large_v2", "SHD_minilm_l6"])
        for n in BENCHMARKS:
            w.writerow([n, base[n], shd["bge-m3"][n], shd["e5-large-v2"][n],
                        shd["minilm-l6"][n]])
    print(f"\nSaved {os.path.relpath(csv_path, REPO_ROOT)}")
    return base, shd, payloads


if __name__ == "__main__":
    run()
