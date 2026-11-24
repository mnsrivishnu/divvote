"""Unit tests for src/voting/divvote.py (oracle mock, no API/network).

Uses an injected `vote_fn(cause, effect, medoid) -> 0/1`:
  - an oracle that returns the ground-truth edges -> SHD/TENE/TERE all 0;
  - a noisy variant (drop one true edge, add its reverse) -> SHD 2.
Plus a focused check of uniform aggregation and the confidence formula.

Run from the repo root:
    python -m unittest tests.test_divvote -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voting import divvote, baseline  # noqa: E402
from src.metrics import metrics  # noqa: E402


class TestDivVoteOracle(unittest.TestCase):
    def setUp(self):
        self.true_adj, self.nodes = baseline.load_ground_truth("asia")
        self.idx = {n: i for i, n in enumerate(self.nodes)}

    def _oracle(self, cause, effect, medoid):
        # Every medoid votes exactly the true edges -> perfect reconstruction.
        return int(self.true_adj[self.idx[cause], self.idx[effect]] == 1)

    def test_oracle_matches_ground_truth(self):
        res = divvote.run_divvote("asia", vote_fn=self._oracle, verbose=False)
        self.assertTrue(np.array_equal(res["pred_adj"], self.true_adj))
        self.assertEqual(res["SHD"], 0)
        self.assertEqual(res["TENE"], 0)
        self.assertEqual(res["TERE"], 0)
        # Unanimous votes -> edge_score is 0 or 1 -> confidence is 1 everywhere.
        conf_on_edges = res["confidence"][res["pred_adj"] == 1]
        self.assertTrue(np.allclose(conf_on_edges, 1.0))

    def test_noisy_case_shd_two(self):
        i, j = np.argwhere(self.true_adj == 1)[0]

        def noisy(cause, effect, medoid):
            a, b = self.idx[cause], self.idx[effect]
            if (a, b) == (i, j):
                return 0           # drop a true edge  -> missing (TENE)
            if (a, b) == (j, i):
                return 1           # add its reverse   -> reversed (TERE)
            return int(self.true_adj[a, b] == 1)

        res = divvote.run_divvote("asia", vote_fn=noisy, verbose=False)
        self.assertEqual(res["SHD"], 2)
        self.assertEqual(res["TENE"], 1)
        self.assertEqual(res["TERE"], 1)


class TestUniformAggregation(unittest.TestCase):
    def test_mean_threshold_and_confidence(self):
        nodes = ["A", "B"]
        medoids = ["m0", "m1", "m2", "m3", "m4"]  # k = 5

        def vote_fn(cause, effect, medoid):
            # A->B: 3 of 5 medoids say yes; B->A: none.
            if (cause, effect) == ("A", "B") and medoid in ("m0", "m1", "m2"):
                return 1
            return 0

        adj, confidence, edge_score = divvote.run_divvote_voting(
            nodes, medoids, vote_fn=vote_fn)

        # A->B: mean 3/5 = 0.6 -> edge, confidence |2*0.6-1| = 0.2
        self.assertAlmostEqual(edge_score[0, 1], 0.6)
        self.assertEqual(adj[0, 1], 1)
        self.assertAlmostEqual(confidence[0, 1], 0.2)
        # B->A: mean 0 -> no edge, confidence 1.0
        self.assertEqual(adj[1, 0], 0)
        self.assertAlmostEqual(confidence[1, 0], 1.0)


if __name__ == "__main__":
    unittest.main()
