from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from common.data import Seizure, is_preonset_evaluable
from streaming_predictor.config import Config
from streaming_predictor.data import (PreonsetClipDataset, ViewInstance,
                                      select_eval_view_instances)
from streaming_predictor.evaluate import _gauss_emit, _tokenize_clip


class PreonsetTokenizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)

    def test_positive_clip_never_falls_back_inside_exclusion_zone(self) -> None:
        feat = np.arange(12, dtype=np.float32).reshape(3, 4)
        times = np.array([0.2, 0.6, 0.9], dtype=np.float32)

        token_feat, token_times = _tokenize_clip(
            feat, times, onset=1.0, label=1, N=4, rng=self.rng
        )

        self.assertIsNone(token_feat)
        self.assertIsNone(token_times)

    def test_zero_onset_positive_clip_is_unusable(self) -> None:
        feat = np.arange(16, dtype=np.float32).reshape(4, 4)
        times = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)

        token_feat, token_times = _tokenize_clip(
            feat, times, onset=0.0, label=1, N=4, rng=self.rng
        )

        self.assertIsNone(token_feat)
        self.assertIsNone(token_times)

    def test_positive_clip_uses_only_strictly_pre_exclusion_tokens(self) -> None:
        feat = np.arange(20, dtype=np.float32).reshape(5, 4)
        times = np.array([0.1, 0.3, 0.49, 0.5, 0.9], dtype=np.float32)

        _, token_times = _tokenize_clip(
            feat, times, onset=1.0, label=1, N=3, rng=self.rng
        )

        self.assertTrue(np.all(token_times < 0.5))

    def test_negative_clip_is_not_preonset_truncated(self) -> None:
        feat = np.arange(16, dtype=np.float32).reshape(4, 4)
        times = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)

        _, token_times = _tokenize_clip(
            feat, times, onset=-1.0, label=0, N=4, rng=self.rng
        )

        np.testing.assert_array_equal(token_times, times)


class PreonsetTrainingDatasetTests(unittest.TestCase):
    def test_excludes_positive_instance_without_preonset_context(self) -> None:
        sequences = {
            "bad_01_L": {
                "features": torch.arange(16, dtype=torch.float32).reshape(4, 4),
                "frame_indices": torch.tensor([0, 5, 10, 20]),
            },
            "good_01_L": {
                "features": torch.arange(16, dtype=torch.float32).reshape(4, 4),
                "frame_indices": torch.tensor([0, 5, 10, 20]),
            },
        }
        instances = [
            ViewInstance("bad_01", "L", 1, 0.0, 2.0),
            ViewInstance("good_01", "L", 1, 2.0, 2.0),
        ]
        cfg = Config()
        cfg.model.feature_dim = 4
        cfg.model.num_tokens = 3

        dataset = PreonsetClipDataset(
            instances, sequences, cfg, train=False, seed=0
        )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["seq_key"], "good_01_L")

    def test_invalid_positive_is_removed_before_splitting(self) -> None:
        sequences = {
            "bad_01_L": {"frame_indices": torch.tensor([0, 5, 10, 20])},
            "good_01_L": {"frame_indices": torch.tensor([0, 5, 10, 20])},
            "negative_01_L": {"frame_indices": torch.tensor([0, 5, 10, 20])},
        }
        bad = Seizure("bad_01", "bad", 1, 0.0, ("L",), 2.0)
        good = Seizure("good_01", "good", 1, 2.0, ("L",), 2.0)
        negative = Seizure("negative_01", "negative", 0, None, ("L",), 2.0)

        self.assertFalse(is_preonset_evaluable(bad, sequences))
        self.assertTrue(is_preonset_evaluable(good, sequences))
        self.assertTrue(is_preonset_evaluable(negative, sequences))

    def test_evaluation_view_is_s_preferred_and_order_independent(self) -> None:
        l_view = ViewInstance("001_01", "L", 0, -1.0, 2.0)
        s_view = ViewInstance("001_01", "S", 0, -1.0, 2.0)
        l_only = ViewInstance("002_01", "L", 0, -1.0, 2.0)

        forward = select_eval_view_instances([l_view, s_view, l_only])
        reverse = select_eval_view_instances([s_view, l_view, l_only])

        self.assertEqual([(x.ssid, x.view) for x in forward],
                         [("001_01", "S"), ("002_01", "L")])
        self.assertEqual([(x.ssid, x.view) for x in reverse],
                         [("001_01", "S"), ("002_01", "L")])


class GaussianEmissionTests(unittest.TestCase):
    def test_uses_normalized_gaussian_density(self) -> None:
        sigma = 0.2
        expected = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        self.assertAlmostEqual(_gauss_emit(0.4, 0.4, sigma), expected, places=12)


if __name__ == "__main__":
    unittest.main()
