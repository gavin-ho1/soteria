from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from streaming_predictor.delong import delong_roc_test


class DeLongTests(unittest.TestCase):
    def test_auc_matches_sklearn_for_interleaved_labels(self) -> None:
        labels = np.array([0, 1, 0, 1, 0, 1])
        scores_a = np.array([0.10, 0.90, 0.40, 0.80, 0.70, 0.60])
        scores_b = np.array([0.20, 0.75, 0.30, 0.65, 0.55, 0.85])

        result = delong_roc_test(labels, scores_a, scores_b)

        self.assertAlmostEqual(result["auc_a"], roc_auc_score(labels, scores_a))
        self.assertAlmostEqual(result["auc_b"], roc_auc_score(labels, scores_b))

    def test_result_is_invariant_to_example_order(self) -> None:
        labels = np.array([0, 1, 0, 1, 0, 1])
        scores_a = np.array([0.10, 0.90, 0.40, 0.80, 0.70, 0.60])
        scores_b = np.array([0.20, 0.75, 0.30, 0.65, 0.55, 0.85])
        permutation = np.array([5, 0, 3, 2, 1, 4])

        original = delong_roc_test(labels, scores_a, scores_b)
        reordered = delong_roc_test(
            labels[permutation], scores_a[permutation], scores_b[permutation]
        )

        for key in ("auc_a", "auc_b", "z", "p", "var"):
            self.assertAlmostEqual(original[key], reordered[key])


if __name__ == "__main__":
    unittest.main()
