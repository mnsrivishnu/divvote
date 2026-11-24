"""Per-benchmark ECE + reliability-diagram data for DivVote confidence (C3).

Runs uniform DivVote (the headline weighting) on all five benchmarks in
reference style, reusing the cached medoid votes (no new API calls), and for
each benchmark reports Expected Calibration Error over the predicted edges plus
the reliability-diagram table. Writes results/ece.csv (one row per benchmark)
and results/ece_reliability.csv (one row per confidence bin).

Run from the repo root:
    python -m experiments.ece_eval
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voting import divvote, baseline   # noqa: E402
from src.metrics import calibration         # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
TARGET = 0.1


def run(specs=None):
    specs = specs or divvote.ALL_SPECS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary, rel_rows = [], []

    for name, style in specs:
        true_adj, _ = baseline.load_ground_truth(name)
        res = divvote.run_divvote(name, prompt_style=style, verbose=False)
        e, table = calibration.ece_for_prediction(
            res["pred_adj"], res["confidence"], true_adj)
        n_pred = int(res["pred_adj"].sum())
        summary.append((name, n_pred, e))

        print(f"\n{name} (style={style}) - ECE = {e:.4f}  "
              f"[{'PASS' if e < TARGET else 'FAIL'} target < {TARGET}]  "
              f"predicted edges: {n_pred}")
        print(f"{'conf bin':<14}{'count':>7}{'mean conf':>11}{'accuracy':>10}"
              f"{'gap':>8}")
        print("-" * 50)
        for r in table:
            label = f"[{r['bin_lo']:.1f},{r['bin_hi']:.1f}" \
                    f"{']' if r['bin_hi'] == 1.0 else ')'}"
            if r["count"]:
                print(f"{label:<14}{r['count']:>7}{r['mean_confidence']:>11.3f}"
                      f"{r['accuracy']:>10.3f}{r['gap']:>8.3f}")
            else:
                print(f"{label:<14}{0:>7}{'-':>11}{'-':>10}{'-':>8}")
            rel_rows.append({"benchmark": name, "bin_lo": r["bin_lo"],
                             "bin_hi": r["bin_hi"], "count": r["count"],
                             "mean_confidence": r["mean_confidence"],
                             "accuracy": r["accuracy"], "gap": r["gap"]})

    print("\n" + "=" * 46)
    print(f"{'benchmark':<20}{'pred edges':>11}{'ECE':>9}{'':>6}")
    print("-" * 46)
    for name, n_pred, e in summary:
        flag = "PASS" if e < TARGET else "FAIL"
        print(f"{name:<20}{n_pred:>11}{e:>9.4f}{flag:>6}")
    print(f"\n(ECE over predicted edges, uniform weighting; target < {TARGET}.)")

    with open(os.path.join(RESULTS_DIR, "ece.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "predicted_edges", "ece", "target", "pass"])
        for name, n_pred, e in summary:
            w.writerow([name, n_pred, f"{e:.6f}", TARGET, int(e < TARGET)])
    with open(os.path.join(RESULTS_DIR, "ece_reliability.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "bin_lo", "bin_hi",
                                          "count", "mean_confidence",
                                          "accuracy", "gap"])
        w.writeheader()
        w.writerows(rel_rows)
    print(f"\nSaved results/ece.csv and results/ece_reliability.csv")
    return summary


if __name__ == "__main__":
    run()
