"""Post-hoc temperature scaling of DivVote confidence: pre vs post ECE (C3).

Fits a SINGLE temperature T on the Asia validation set only (held-out
discipline, same as the reliability weighting), by minimising NLL over Asia's
predicted-edge vote margins, then applies that fixed T to the four held-out
benchmarks and reports pre- vs post-calibration ECE. T is NOT selected on the
reported benchmarks. Reuses cached votes (uniform DivVote); no new API calls.

Calibrated confidence = sigmoid(margin / T), margin = confidence * k. Writes
results/ece_calibrated.csv.

Run from the repo root:
    python -m experiments.ece_calibrate
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
VALIDATION = "asia"
HELD_OUT = [("cancer", "reference"), ("medical_diagnostic", "reference"),
            ("covid19", "reference"), ("endorisk", "reference")]


def _edges(name, style):
    """(confidence, correctness, k) over predicted edges, from cached votes."""
    true_adj, _ = baseline.load_ground_truth(name)
    res = divvote.run_divvote(name, prompt_style=style, verbose=False)
    conf, correct = calibration.confidence_and_correctness(
        res["pred_adj"], res["confidence"], true_adj)
    return conf, correct, res["k"]


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Fit T on the validation benchmark only.
    v_conf, v_correct, k = _edges(VALIDATION, "reference")
    T = calibration.fit_temperature(
        calibration.margin_from_confidence(v_conf, k), v_correct)
    pre_v = calibration.ece(v_conf, v_correct)
    post_v = calibration.ece(
        calibration.apply_temperature(
            calibration.margin_from_confidence(v_conf, k), T), v_correct)
    print(f"Temperature fit on '{VALIDATION}' (validation) by NLL: T = {T:.3f}")
    print(f"  (T > 1 softens overconfidence; k = {k})")
    print(f"  {VALIDATION} in-sample ECE: pre {pre_v:.4f} -> post {post_v:.4f} "
          f"  [in-sample, shown for transparency; not a held-out result]\n")

    rows = []
    for name, style in HELD_OUT:
        conf, correct, kk = _edges(name, style)
        margin = calibration.margin_from_confidence(conf, kk)
        pre = calibration.ece(conf, correct)
        post = calibration.ece(calibration.apply_temperature(margin, T), correct)
        rows.append((name, int(conf.size), pre, post))

    print(f"Pre- vs post-calibration ECE (T={T:.3f} fit on {VALIDATION} only)")
    print(f"{'benchmark':<20}{'edges':>6}{'pre ECE':>10}{'post ECE':>10}"
          f"{'delta':>9}{'post<0.1':>10}")
    print("-" * 65)
    for name, n, pre, post in rows:
        flag = "PASS" if post < TARGET else "FAIL"
        print(f"{name:<20}{n:>6}{pre:>10.4f}{post:>10.4f}{post - pre:>+9.4f}"
              f"{flag:>10}")
    print(f"\n(ECE over predicted edges, uniform weighting. T selected on "
          f"'{VALIDATION}' by NLL only -- NOT tuned on the four above.)")

    with open(os.path.join(RESULTS_DIR, "ece_calibrated.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "predicted_edges", "ece_pre", "ece_post",
                    "temperature", "validation", "post_pass"])
        for name, n, pre, post in rows:
            w.writerow([name, n, f"{pre:.6f}", f"{post:.6f}", f"{T:.6f}",
                        VALIDATION, int(post < TARGET)])
    print("\nSaved results/ece_calibrated.csv")
    return T, rows


if __name__ == "__main__":
    run()
