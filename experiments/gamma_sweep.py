"""gamma-sensitivity ablation for DivVote's learned reliability weighting.

DESCRIPTIVE ONLY -- this is NOT gamma selection. Reporting-benchmark selection
would be test-set leakage, and tuning gamma on the in-sample Asia validation set
overfits an 8-node graph. gamma=1 is the pre-specified headline weighting; this
sweep only characterises how sensitive SHD is to the reliability sharpness.

Learns per-medoid reliability r_j on Asia (held out of the table), then for each
gamma re-weights the SAME cached medoid-vote tensors (w_j = r_j^gamma / sum r),
so the whole sweep makes zero new API calls. Prints SHD per benchmark per gamma
for the four held-out networks (Cancer, Medical Diagnostic, Covid-19, Endorisk).

Maps to: Week 3, Days 5-6 (PLAN.md section 10) -- reliability-weighting ablation.

Run from the repo root:
    python -m experiments.gamma_sweep
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voting import divvote, baseline   # noqa: E402
from src.metrics import metrics             # noqa: E402

GAMMAS = [1, 2, 3, 5, 10]
SPECS = [("cancer", "reference"), ("medical_diagnostic", "reference"),
         ("covid19", "reference"), ("endorisk", "reference")]


def run(validation="asia"):
    # Learn per-medoid reliability once on the validation benchmark (cached).
    r, medoids = divvote.medoid_reliability(validation)      # r_j in [0, 1]
    weights_by_gamma = {g: (r ** g) / (r ** g).sum() for g in GAMMAS}

    rows = []
    for name, style in SPECS:
        true_adj, nodes = baseline.load_ground_truth(name)
        states = divvote._resolve_states(name, style, nodes, None)
        V = divvote.collect_votes(nodes, medoids, prompt_style=style,
                                  states=states)              # cached -> no API
        base = baseline.run_baseline(name, prompt_style=style,
                                     verbose=False)["SHD"]
        unif = metrics.shd(divvote.aggregate_votes(V, None)[0], true_adj)
        per_g = {g: metrics.shd(
            divvote.aggregate_votes(V, weights_by_gamma[g])[0], true_adj)
            for g in GAMMAS}
        rows.append((name, base, unif, per_g))

    print("gamma-sensitivity ablation (SHD) - reliability weights learned on "
          f"'{validation}'")
    print("DESCRIPTIVE ONLY. No gamma was selected from these results.\n")
    print("medoid reliabilities r_j: "
          f"{', '.join(f'm{i}={rj:.3f}' for i, rj in enumerate(r))}\n")
    hdr = f"{'benchmark':<20}{'Base':>6}{'Unif':>6}" + "".join(
        f"{'g=' + str(g):>6}" for g in GAMMAS)
    print(hdr)
    print("-" * len(hdr))
    for name, base, unif, per_g in rows:
        print(f"{name:<20}{base:>6}{unif:>6}"
              + "".join(f"{per_g[g]:>6}" for g in GAMMAS))
    print("\nUnif = uniform (w_j=1/k). g=1 is the pre-specified headline "
          "weighting.")
    print(f"All SHD; lower is better. '{validation}' is the validation set and "
          "is held out.")
    return rows


if __name__ == "__main__":
    run()
