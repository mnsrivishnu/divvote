"""Graph post-processing for the baseline: cycle removal + mediator removal.

Purpose
-------
The base paper applies two optional graph-cleanup steps after voting
(reference/src/graph_utils.py): removing bidirectional (2-cycle) conflicts, and
mediator removal (transitive reduction). Applying both on top of the raw vote
graph is the base paper's "Pruned" variant; the raw vote graph is "Raw".

  - mediator_removal: a direct port of the reference. Drops a direct edge
    i -> j when some mediator k gives an indirect path i -> k -> j. Edge
    existence for the path test is read from the ORIGINAL matrix (matching the
    reference, which reads A while writing A_new).

  - remove_bidirectional_conflicts: adjacency-matrix adaptation of the
    reference's `remove_true_bidirectional_conflicts`. Where both i -> j and
    j -> i are predicted, it keeps one direction. The reference breaks the tie
    with each answer's probability; here we use the per-direction yes-vote
    count (`scores`) as the stand-in confidence, keeping the higher-voted
    direction. On a tie (or when no scores are given) the i -> j direction with
    i < j is kept, deterministically.

Maps to: Week 2, Days 5-6 (PLAN.md section 10) — base-paper reproduction.
"""

import numpy as np


def mediator_removal(adj):
    """Transitive reduction: drop i->j if an indirect path i->k->j exists.

    Direct port of reference/src/graph_utils.py::mediator_removal.
    """
    A = np.asarray(adj).astype(int)
    n = len(A)
    A_new = A.copy()
    for i in range(n):
        for j in range(n):
            if i != j and A[i][j] == 1:
                has_indirect_path = any(
                    A[i][k] == 1 and A[k][j] == 1
                    for k in range(n) if k != i and k != j
                )
                if has_indirect_path:
                    A_new[i][j] = 0
    return A_new


def remove_bidirectional_conflicts(adj, scores=None):
    """Resolve each i<->j 2-cycle by keeping a single direction.

    Keeps the higher-voted direction (using `scores`, the yes-vote counts);
    on a tie, or when `scores` is None, keeps i->j for i < j.
    """
    A = np.asarray(adj).astype(int).copy()
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1 and A[j, i] == 1:
                s_ij = scores[i, j] if scores is not None else 0
                s_ji = scores[j, i] if scores is not None else 0
                if s_ij < s_ji:
                    A[i, j] = 0
                else:
                    A[j, i] = 0
    return A


def prune(adj, scores=None, cycle_removal=True, mediator=True):
    """Apply the selected post-processing steps (cycle removal, then mediator)."""
    out = np.asarray(adj).astype(int).copy()
    if cycle_removal:
        out = remove_bidirectional_conflicts(out, scores=scores)
    if mediator:
        out = mediator_removal(out)
    return out
