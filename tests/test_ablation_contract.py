from __future__ import annotations

import unittest

import numpy as np

from streaming_predictor.ablation_study import ABLATION_CONDITIONS, _BARE
from streaming_predictor.config import Config
from streaming_predictor.evaluate import (_tcs_transition_matrix, alarm_records,
                                          calibrate_threshold)

PAPER_DECODER_MODES = (
    "threshold", "hmm4_early_skip_ssem", "hmm4_early_ssem", "hmm4_early",
    "hmm4_early_full_ssem", "hmm4_early_skip_regressive_ssem",
    "hmm4_skip_ssem", "hmm4_ssem", "hmm",
)


class ObjectiveBuildUpTests(unittest.TestCase):
    def test_bare_condition_really_has_no_auxiliary_losses(self) -> None:
        expected_zero = {
            "train.alarm_margin_weight",
            "train.alarm_loss_weight",
            "train.hazard_loss_weight",
            "train.cvar_weight",
        }
        self.assertTrue(expected_zero.issubset(_BARE))
        for key in expected_zero:
            self.assertEqual(_BARE[key], 0.0, key)

    def test_default_condition_is_the_complete_objective(self) -> None:
        # The CVaR row of the build-up is the `default` checkpoint replayed with the threshold decoder, so `default` must resolve to Config()
        self.assertEqual(ABLATION_CONDITIONS["default"]["overrides"], {})


class PaperDecoderModeTests(unittest.TestCase):
    def test_every_paper_decoder_mode_calibrates_and_decides(self) -> None:
        rng = np.random.default_rng(0)
        streams = {}
        for i in range(12):
            label = i % 2
            times = np.arange(1.0, 21.0, dtype=np.float32)
            base = 0.2 + 0.5 * label * times / 20.0
            risk = np.clip(base + rng.normal(0.0, 0.05, len(times)), 0.01, 0.99)
            streams[f"{i:03d}_01"] = {
                "risk": risk.astype(np.float32), "times": times, "label": label,
                "onset": 21.0 if label else -1.0, "length_sec": 21.0,
            }
        cfg = Config()
        for mode in PAPER_DECODER_MODES:
            cfg.alarm.mode = mode
            params = calibrate_threshold(streams, cfg)
            records = alarm_records(streams, params, cfg)
            self.assertEqual(len(records), len(streams), mode)


class TcsEmissionFitTransitionTests(unittest.TestCase):
    def test_last_state_is_absorbing_during_em_fit(self) -> None:
        transition = _tcs_transition_matrix(3, a_stay=0.85)

        np.testing.assert_allclose(transition.sum(axis=1), 1.0)
        np.testing.assert_allclose(transition[-1], [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
