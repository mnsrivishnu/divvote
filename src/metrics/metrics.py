"""Evaluation metrics: SHD, TENE, TERE (and their normalized forms).

Purpose
-------
Structural scoring of a predicted adjacency matrix against the ground-truth
DAG, matching the base paper's exact definitions so our numbers are directly
comparable (reference/src/metrics.py, `evaluate_adjacency_matrices` /
`normalize_metrics`; PLAN.md sections 2.5 / 7.2):

  - SHD  (Structural Hamming Distance) = number of cells where pred and true
    differ. A reversed edge differs at both (i, j) and (j, i), so it counts 2.
    Primary metric; lower is better.
  - TENE (True Edge -> Negative Edge) = true edges the prediction missed
    (true == 1 and pred == 0).
  - TERE (True Edge -> Reverse Edge) = true edges that appear reversed in the
    prediction (true[i, j] == 1 and pred[j, i] == 1).

  Normalized forms (base paper's `normalize_metrics`):
  - SHD  / total cells
  - TENE / number of true edges   (0 if there are no true edges)
  - TERE / number of true edges   (0 if there are no true edges)

Argument order is (pred, true) throughout. SHD is symmetric, but TENE/TERE and
the normalizers depend on which matrix is ground truth.

Note on matrix shape: real adjacency matrices are square, and on square inputs
these functions are identical to the base paper's. TERE additionally
bounds-checks the (j, i) lookup so it does not raise on non-square test
matrices; on square inputs the check is always satisfied and changes nothing.

Maps to: Week 2, Days 1-2 (PLAN.md section 10) — "Implement metrics (SHD,
TERE, TENE) with unit tests."
"""

import numpy as np


def _as_int_array(m):
    return np.asarray(m).astype(int)


def shd(pred, true):
    """Structural Hamming Distance: count of cells where pred and true differ."""
    pred = _as_int_array(pred)
    true = _as_int_array(true)
    return int(np.sum(pred != true))


def tene(pred, true):
    """True-Edge-to-Negative-Edge: true edges missed (true == 1, pred == 0)."""
    pred = _as_int_array(pred)
    true = _as_int_array(true)
    return int(np.sum((true == 1) & (pred == 0)))


def tere(pred, true):
    """True-Edge-to-Reverse-Edge: true edges reversed (true[i,j]=1, pred[j,i]=1)."""
    pred = _as_int_array(pred)
    true = _as_int_array(true)
    rows, cols = pred.shape
    count = 0
    for i, j in np.argwhere(true == 1):
        # Bounds guard is a no-op on square matrices (always satisfied there).
        if j < rows and i < cols and pred[j, i] == 1:
            count += 1
    return int(count)


def n_true_edges(true):
    """Number of true edges (cells equal to 1 in the ground-truth matrix)."""
    return int(np.sum(_as_int_array(true) == 1))


def shd_normalized(pred, true):
    """SHD divided by the total number of cells in the ground-truth matrix."""
    total_cells = _as_int_array(true).size
    return shd(pred, true) / total_cells if total_cells > 0 else 0.0


def tene_normalized(pred, true):
    """TENE divided by the number of true edges (0 if there are none)."""
    denom = n_true_edges(true)
    return tene(pred, true) / denom if denom > 0 else 0.0


def tere_normalized(pred, true):
    """TERE divided by the number of true edges (0 if there are none)."""
    denom = n_true_edges(true)
    return tere(pred, true) / denom if denom > 0 else 0.0


def evaluate(pred, true):
    """Return all six metrics for a (pred, true) pair as a dict."""
    return {
        "SHD": shd(pred, true),
        "SHD_normalized": shd_normalized(pred, true),
        "TENE": tene(pred, true),
        "TENE_normalized": tene_normalized(pred, true),
        "TERE": tere(pred, true),
        "TERE_normalized": tere_normalized(pred, true),
    }
