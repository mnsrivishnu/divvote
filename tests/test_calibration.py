"""Unit tests for src/metrics/calibration.py (ECE), fully offline.

Hand-built confidence/correctness arrays with ECE worked out by hand:
  conf=[0.95, 0.95, 0.05], correct=[1, 0, 0]
    bin [0.9,1.0]: 2 edges, mean_conf 0.95, acc 0.5, gap 0.45, weight 2/3
    bin [0.0,0.1): 1 edge,  mean_conf 0.05, acc 0.0, gap 0.05, weight 1/3
    ECE = 2/3*0.45 + 1/3*0.05 = 0.316666...

Run from the repo root:
    python -m unittest tests.test_calibration -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import calibration  # noqa: E402


class TestECE(unittest.TestCase):
    def test_perfect_calibration_is_zero(self):
        # Confidence 1.0 everywhere and always correct -> no gap.
        conf = np.array([1.0, 1.0, 1.0, 1.0])
        correct = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(calibration.ece(conf, correct), 0.0)

    def test_fully_confident_but_half_wrong(self):
        # One bin at conf 1.0 but only 50% correct -> ECE = |0.5 - 1.0| = 0.5.
        conf = np.array([1.0, 1.0, 1.0, 1.0])
        correct = np.array([1.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(calibration.ece(conf, correct), 0.5)

    def test_known_two_bin_value(self):
        conf = np.array([0.95, 0.95, 0.05])
        correct = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(calibration.ece(conf, correct), 2/3*0.45 + 1/3*0.05)

    def test_empty_prediction_is_zero(self):
        self.assertEqual(calibration.ece(np.array([]), np.array([])), 0.0)

    def test_confidence_one_lands_in_last_bin(self):
        # conf exactly 1.0 must be counted (last bin is closed on the right).
        rows = calibration.reliability_table(np.array([1.0]), np.array([1.0]))
        self.assertEqual(rows[-1]["count"], 1)
        self.assertEqual(sum(r["count"] for r in rows), 1)

    def test_bin_counts_and_gaps(self):
        conf = np.array([0.95, 0.95, 0.05])
        correct = np.array([1.0, 0.0, 0.0])
        rows = calibration.reliability_table(conf, correct, n_bins=10)
        self.assertEqual(rows[0]["count"], 1)            # [0.0,0.1): the 0.05
        self.assertAlmostEqual(rows[0]["accuracy"], 0.0)
        self.assertEqual(rows[-1]["count"], 2)           # [0.9,1.0]: the 0.95s
        self.assertAlmostEqual(rows[-1]["mean_confidence"], 0.95)
        self.assertAlmostEqual(rows[-1]["accuracy"], 0.5)
        self.assertAlmostEqual(rows[-1]["gap"], 0.45)
        # empty bins report NaN, not a spurious 0.
        self.assertTrue(np.isnan(rows[5]["accuracy"]))


class TestECEForPrediction(unittest.TestCase):
    def test_extracts_predicted_edges_and_scores_against_truth(self):
        # 3 nodes. Predicted edges: 0->1 (true), 0->2 (false), 1->0 (false).
        true_adj = np.array([[0, 1, 0],
                             [0, 0, 0],
                             [0, 0, 0]])
        pred_adj = np.array([[0, 1, 1],
                             [1, 0, 0],
                             [0, 0, 0]])
        confidence = np.array([[0.0, 0.95, 0.95],
                              [0.05, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])
        conf, correct = calibration.confidence_and_correctness(
            pred_adj, confidence, true_adj)
        self.assertTrue(np.allclose(sorted(conf), [0.05, 0.95, 0.95]))
        self.assertAlmostEqual(correct.sum(), 1.0)       # only 0->1 is correct
        e, rows = calibration.ece_for_prediction(pred_adj, confidence, true_adj)
        self.assertAlmostEqual(e, 2/3*0.45 + 1/3*0.05)
        self.assertEqual(sum(r["count"] for r in rows), 3)


class TestTemperatureScaling(unittest.TestCase):
    def test_apply_temperature_t1_is_sigmoid(self):
        # margin = confidence * k; at k=5, confidence 1.0 -> margin 5.
        m = calibration.margin_from_confidence(np.array([1.0]), k=5)
        self.assertAlmostEqual(float(m[0]), 5.0)
        p = calibration.apply_temperature(m, T=1.0)
        self.assertAlmostEqual(float(p[0]), 1.0 / (1.0 + np.exp(-5.0)))

    def test_fit_softens_overconfident_data(self):
        # All edges unanimous (margin 5) but only half correct -> raw conf 1.0,
        # accuracy 0.5. Fitting T should push calibrated confidence toward 0.5
        # and cut ECE sharply.
        margin = np.full(20, 5.0)
        correct = np.array([1.0, 0.0] * 10)
        raw_conf = margin / 5.0                      # == 1.0 everywhere
        pre = calibration.ece(raw_conf, correct)
        self.assertAlmostEqual(pre, 0.5)
        T = calibration.fit_temperature(margin, correct)
        self.assertGreater(T, 1.0)                   # softening
        post_conf = calibration.apply_temperature(margin, T)
        post = calibration.ece(post_conf, correct)
        self.assertLess(post, pre)
        self.assertLess(post, 0.1)

    def test_fit_recovers_true_temperature_and_calibrates(self):
        # Labels generated exactly from sigmoid(margin / T_true), T_true = 2:
        # margins 1/3/5 correct at sigmoid(0.5/1.5/2.5). Fit should recover
        # T ~= 2 and drive post-ECE near 0, well below the raw-margin ECE.
        n = 1000
        margins, correct = [], []
        for m in (1.0, 3.0, 5.0):
            p = calibration._sigmoid(m / 2.0)
            n_ones = int(round(n * p))
            margins += [m] * n
            correct += [1.0] * n_ones + [0.0] * (n - n_ones)
        margin = np.array(margins)
        correct = np.array(correct)
        pre = calibration.ece(margin / 5.0, correct)       # raw margin-confidence
        T = calibration.fit_temperature(margin, correct)
        self.assertAlmostEqual(T, 2.0, places=1)
        post = calibration.ece(calibration.apply_temperature(margin, T), correct)
        self.assertLess(post, 0.02)
        self.assertLess(post, pre)

    def test_empty_fit_returns_identity(self):
        self.assertEqual(calibration.fit_temperature(np.array([]),
                                                      np.array([])), 1.0)


if __name__ == "__main__":
    unittest.main()
