"""Expected Calibration Error (ECE) for DivVote's per-edge confidence.

Purpose
-------
Contribution C3 (PLAN.md sections 6, 7.2): DivVote outputs a per-edge confidence
= |2 * edge_score - 1| in [0, 1]. This module measures how well that confidence
tracks empirical correctness, via the standard binned Expected Calibration
Error, and produces the underlying reliability-diagram table.

Scope: calibration is measured over the edges DivVote *predicts* (adj == 1). For
each predicted directed edge i -> j:
    confidence = conf[i, j]
    correct    = 1 if the edge is in the ground truth (true_adj[i, j] == 1) else 0
Predicted edges are binned by confidence; per bin we compare mean confidence to
empirical accuracy. With B equal-width bins over [0, 1]:
    ECE = sum_b (n_b / N) * | acc_b - conf_b |
summed over non-empty bins (N = number of predicted edges). Target: ECE < 0.1.

Maps to: Week 3+ / Feb 2027 (PLAN.md section 10) -- per-edge confidence + ECE.
"""

import numpy as np

DEFAULT_BINS = 10


def confidence_and_correctness(pred_adj, confidence, true_adj):
    """Return (conf, correct) 1-D arrays over the predicted directed edges.

    conf[e] is the confidence of predicted edge e; correct[e] is 1.0 if that
    edge exists in the ground truth, else 0.0. Order follows row-major traversal
    of pred_adj. Empty arrays if nothing is predicted.
    """
    pred_adj = np.asarray(pred_adj)
    confidence = np.asarray(confidence, dtype=float)
    true_adj = np.asarray(true_adj)
    mask = pred_adj == 1
    conf = confidence[mask]
    correct = (true_adj[mask] == 1).astype(float)
    return conf, correct


def reliability_table(conf, correct, n_bins=DEFAULT_BINS):
    """Reliability-diagram rows: one dict per equal-width confidence bin.

    Each row has bin_lo, bin_hi, count, mean_confidence, accuracy, gap. Empty
    bins report count 0 and NaN for the three statistics. The last bin is closed
    on the right so confidence == 1.0 is included.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            in_bin = (conf >= lo) & (conf <= hi)   # closed right: include 1.0
        else:
            in_bin = (conf >= lo) & (conf < hi)
        n = int(in_bin.sum())
        if n:
            mc = float(conf[in_bin].mean())
            acc = float(correct[in_bin].mean())
            gap = abs(acc - mc)
        else:
            mc = acc = gap = float("nan")
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": n,
                     "mean_confidence": mc, "accuracy": acc, "gap": gap})
    return rows


def ece(conf, correct, n_bins=DEFAULT_BINS):
    """Expected Calibration Error over the given (confidence, correctness) pairs.

    Returns 0.0 when there are no samples (nothing predicted -> nothing to
    miscalibrate).
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    if n == 0:
        return 0.0
    total = 0.0
    for row in reliability_table(conf, correct, n_bins):
        if row["count"]:
            total += (row["count"] / n) * row["gap"]
    return total


def ece_for_prediction(pred_adj, confidence, true_adj, n_bins=DEFAULT_BINS):
    """Convenience wrapper: (ECE, reliability_table) for a predicted graph."""
    conf, correct = confidence_and_correctness(pred_adj, confidence, true_adj)
    return ece(conf, correct, n_bins), reliability_table(conf, correct, n_bins)


# --- Post-hoc temperature scaling (Guo et al. 2017) --------------------------
# DivVote's confidence is a discrete vote margin, so we scale the *signed* vote
# margin m = (2*score - 1) * k (for a predicted edge, m = confidence * k) and
# read calibrated confidence as sigmoid(m / T). Unlike logit(confidence), the
# margin is finite at unanimous agreement, so a single temperature T can pull
# overconfident unanimous edges toward 0.5. T is fit by minimising negative
# log-likelihood on a validation set; T > 1 softens, T < 1 sharpens.

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def margin_from_confidence(confidence_edges, k):
    """Signed vote margin m = confidence * k for already-predicted edges."""
    return np.asarray(confidence_edges, dtype=float) * k


def apply_temperature(margin, T):
    """Calibrated confidence sigmoid(margin / T) for a vote margin array."""
    return _sigmoid(np.asarray(margin, dtype=float) / float(T))


def fit_temperature(margin, correct, t_grid=None):
    """Fit temperature T on (margin, correctness) by minimising NLL.

    Returns T > 0. Uses a deterministic 1-D grid (no SciPy dependency); the grid
    is dense near 1 and extends high enough to flatten extreme overconfidence.
    Returns 1.0 (identity) if there is nothing to fit.
    """
    margin = np.asarray(margin, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if margin.size == 0:
        return 1.0
    if t_grid is None:
        t_grid = np.concatenate([np.linspace(0.05, 5.0, 496),
                                 np.linspace(5.1, 50.0, 450)])
    best_T, best_nll = 1.0, np.inf
    for T in t_grid:
        p = np.clip(_sigmoid(margin / T), 1e-12, 1 - 1e-12)
        nll = -float(np.sum(correct * np.log(p) + (1 - correct) * np.log(1 - p)))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T
