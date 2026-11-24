"""Unit tests for src/voting/baseline.py and src/voting/postprocess.py.

These use an injected `vote_fn` (an oracle / noisy stand-in LLM), so no API key
or network is needed. The oracle votes exactly the ground-truth edges; a noisy
variant drops one true edge and adds its reverse.

Run from the repo root:
    python -m unittest tests.test_baseline -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.voting import baseline, postprocess  # noqa: E402
from src.metrics import metrics  # noqa: E402


class TestBaselineOracle(unittest.TestCase):
    def setUp(self):
        self.true_adj, self.nodes = baseline.load_ground_truth("asia")
        self.idx = {n: i for i, n in enumerate(self.nodes)}

    def _oracle(self, cause, effect, verb):
        # Every verb votes yes exactly on true edges -> perfect reconstruction.
        return int(self.true_adj[self.idx[cause], self.idx[effect]] == 1)

    def test_oracle_matches_ground_truth(self):
        res = baseline.run_baseline("asia", vote_fn=self._oracle, verbose=False)
        self.assertTrue(np.array_equal(res["pred_adj"], self.true_adj))
        self.assertEqual(res["SHD"], 0)
        self.assertEqual(res["TENE"], 0)
        self.assertEqual(res["TERE"], 0)
        self.assertEqual(res["variant"], "Raw")

    def test_noisy_case_shd_two(self):
        # Drop one true edge (i->j) and add its reverse (j->i): a reversed edge.
        i, j = np.argwhere(self.true_adj == 1)[0]

        def noisy(cause, effect, verb):
            a, b = self.idx[cause], self.idx[effect]
            if (a, b) == (i, j):
                return 0           # missing true edge
            if (a, b) == (j, i):
                return 1           # reversed edge added
            return int(self.true_adj[a, b] == 1)

        pred = baseline.run_voting(self.nodes, vote_fn=noisy)
        # One reversed edge differs at both (i,j) and (j,i) -> SHD 2.
        self.assertEqual(metrics.shd(pred, self.true_adj), 2)
        self.assertEqual(metrics.tene(pred, self.true_adj), 1)
        self.assertEqual(metrics.tere(pred, self.true_adj), 1)


class TestPostProcess(unittest.TestCase):
    def test_mediator_removal_drops_transitive_edge(self):
        # 0->1, 1->2, and direct 0->2. Mediator 1 removes 0->2.
        adj = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
        ])
        out = postprocess.mediator_removal(adj)
        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])
        self.assertTrue(np.array_equal(out, expected))

    def test_mediator_removal_keeps_non_transitive(self):
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ])
        self.assertTrue(np.array_equal(postprocess.mediator_removal(adj), adj))

    def test_bidirectional_removal_keeps_higher_voted(self):
        # 0<->1 conflict; scores favor 1->0, so 0->1 is dropped.
        adj = np.array([
            [0, 1],
            [1, 0],
        ])
        scores = np.array([
            [0, 3],
            [6, 0],
        ])
        out = postprocess.remove_bidirectional_conflicts(adj, scores=scores)
        self.assertEqual(out[0, 1], 0)
        self.assertEqual(out[1, 0], 1)

    def test_bidirectional_removal_tie_keeps_upper(self):
        # Tie (no scores) -> keep i->j for i<j (0->1).
        adj = np.array([
            [0, 1],
            [1, 0],
        ])
        out = postprocess.remove_bidirectional_conflicts(adj)
        self.assertEqual(out[0, 1], 1)
        self.assertEqual(out[1, 0], 0)


if __name__ == "__main__":
    unittest.main()
