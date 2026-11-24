"""Unit tests for src/metrics/metrics.py.

These use small, hand-built 3x4 adjacency matrices so every expected value can
be checked by hand. The main case deliberately contains one correct edge, one
missing edge, one extra edge, and one reversed edge.

Run from the repo root:
    python -m unittest tests.test_metrics      (stdlib, no deps)
    python -m pytest tests/test_metrics.py     (if pytest is installed)
"""

import os
import sys
import unittest

import numpy as np

# Ensure the repo root (containing `src/`) is importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import metrics  # noqa: E402


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Ground-truth: 3 edges at (0,1), (1,2), (2,0).
        self.true = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
            ]
        )
        # Prediction:
        #   (0,1) correct  -> kept
        #   (1,2) missing   -> pred[1,2] = 0          (TENE)
        #   (2,0) reversed  -> pred[2,0]=0, pred[0,2]=1 (TERE, also TENE)
        #   (0,3) extra     -> pred[0,3] = 1          (SHD only)
        self.pred = np.array(
            [
                [0, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

    def test_shd(self):
        # Differing cells: (0,2),(0,3),(1,2),(2,0) -> 4
        # (the reversed edge accounts for 2 of these: (2,0) and (0,2)).
        self.assertEqual(metrics.shd(self.pred, self.true), 4)

    def test_tene(self):
        # True edges with pred==0 at same cell: (1,2) and (2,0) -> 2.
        self.assertEqual(metrics.tene(self.pred, self.true), 2)

    def test_tere(self):
        # True edge (2,0) has pred[0,2]==1 -> 1 reversed edge.
        self.assertEqual(metrics.tere(self.pred, self.true), 1)

    def test_n_true_edges(self):
        self.assertEqual(metrics.n_true_edges(self.true), 3)

    def test_shd_normalized(self):
        # 4 / (3*4) = 4/12
        self.assertAlmostEqual(
            metrics.shd_normalized(self.pred, self.true), 4 / 12
        )

    def test_tene_normalized(self):
        # 2 / 3 true edges
        self.assertAlmostEqual(
            metrics.tene_normalized(self.pred, self.true), 2 / 3
        )

    def test_tere_normalized(self):
        # 1 / 3 true edges
        self.assertAlmostEqual(
            metrics.tere_normalized(self.pred, self.true), 1 / 3
        )

    def test_evaluate_dict(self):
        result = metrics.evaluate(self.pred, self.true)
        self.assertEqual(result["SHD"], 4)
        self.assertEqual(result["TENE"], 2)
        self.assertEqual(result["TERE"], 1)
        self.assertAlmostEqual(result["SHD_normalized"], 4 / 12)
        self.assertAlmostEqual(result["TENE_normalized"], 2 / 3)
        self.assertAlmostEqual(result["TERE_normalized"], 1 / 3)

    def test_perfect_match(self):
        # Identical matrices -> all metrics zero.
        self.assertEqual(metrics.shd(self.true, self.true), 0)
        self.assertEqual(metrics.tene(self.true, self.true), 0)
        self.assertEqual(metrics.tere(self.true, self.true), 0)
        self.assertAlmostEqual(metrics.shd_normalized(self.true, self.true), 0.0)
        self.assertAlmostEqual(metrics.tene_normalized(self.true, self.true), 0.0)
        self.assertAlmostEqual(metrics.tere_normalized(self.true, self.true), 0.0)

    def test_no_true_edges_guard(self):
        # Empty ground truth -> normalized TENE/TERE guard to 0, no divide error.
        empty = np.zeros((3, 4), dtype=int)
        pred = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        self.assertEqual(metrics.n_true_edges(empty), 0)
        self.assertEqual(metrics.tene(pred, empty), 0)
        self.assertEqual(metrics.tere(pred, empty), 0)
        self.assertAlmostEqual(metrics.tene_normalized(pred, empty), 0.0)
        self.assertAlmostEqual(metrics.tere_normalized(pred, empty), 0.0)
        # One extra edge differs -> SHD 1, normalized 1/12.
        self.assertEqual(metrics.shd(pred, empty), 1)
        self.assertAlmostEqual(metrics.shd_normalized(pred, empty), 1 / 12)


if __name__ == "__main__":
    unittest.main()
