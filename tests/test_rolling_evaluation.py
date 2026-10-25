import unittest

import numpy as np
import torch

from streaming_predictor.rolling_evaluation import (
    build_rolling_windows,
    config_from_dict,
    rolling_risk_stream,
    rolling_summary,
)
from streaming_predictor.config import Config


class RollingWindowContractTests(unittest.TestCase):
    def test_warmup_starts_before_full_window_and_never_uses_future(self):
        times = np.arange(0.5, 10.5, 0.5, dtype=np.float32)

        windows = build_rolling_windows(
            times,
            end_time=9.5,
            window_sec=20.0,
            cadence_sec=1.0,
            num_tokens=24,
        )

        self.assertGreater(len(windows), 0)
        self.assertLess(windows[0].decision_time, 20.0)
        for window in windows:
            sampled_times = times[window.sample_indices]
            self.assertLessEqual(float(sampled_times.max()), window.decision_time)
            self.assertEqual(float(sampled_times[-1]), window.decision_time)

    def test_window_discards_observations_older_than_lookback(self):
        times = np.arange(0.0, 41.0, 1.0, dtype=np.float32)

        windows = build_rolling_windows(
            times,
            end_time=40.0,
            window_sec=20.0,
            cadence_sec=5.0,
            num_tokens=6,
        )

        last = windows[-1]
        sampled_times = times[last.sample_indices]
        self.assertEqual(last.decision_time, 40.0)
        self.assertGreaterEqual(float(sampled_times.min()), 20.0)
        self.assertEqual(float(sampled_times[-1]), 40.0)

    def test_positive_end_time_is_a_hard_future_boundary(self):
        times = np.arange(0.0, 31.0, 1.0, dtype=np.float32)

        windows = build_rolling_windows(
            times,
            end_time=19.5,
            window_sec=20.0,
            cadence_sec=3.0,
            num_tokens=8,
        )

        self.assertLessEqual(max(w.decision_time for w in windows), 19.5)
        self.assertLessEqual(
            max(float(times[w.sample_indices].max()) for w in windows),
            19.5,
        )


    def test_positive_replay_excludes_embedding_exactly_on_boundary(self):
        # Onset 20.5 s puts the exclusion boundary at exactly 20.0 s, where an embedding exists, "strictly earlier" means it must not be replayed
        times = np.arange(0.0, 31.0, 1.0, dtype=np.float32)
        feat = np.ones((len(times), Config().model.feature_dim), dtype=np.float32)

        class ZeroModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[:2])

        _, decision_times = rolling_risk_stream(
            ZeroModel(), Config(), feat, times,
            label=1, onset=20.5, device="cpu", cadence_sec=1.0,
        )
        self.assertEqual(float(decision_times.max()), 19.0)

class RollingMetricContractTests(unittest.TestCase):
    def test_summary_reports_rolling_auc_and_evaluated_negative_exposure(self):
        streams = {
            "p": {
                "risk": np.array([0.1, 0.9]),
                "times": np.array([1.0, 2.0]),
                "label": 1,
                "onset": 3.0,
            },
            "n": {
                "risk": np.array([0.2, 0.3, 0.4]),
                "times": np.array([1.0, 2.0, 3.0]),
                "label": 0,
                "onset": None,
            },
        }
        records = [
            {"is_tcs": True, "alarmed": True, "lead_sec": -1.0},
            {"is_tcs": False, "alarmed": False, "lead_sec": None},
        ]

        summary = rolling_summary(records, streams, cadence_sec=1.0)

        self.assertEqual(summary["rolling_peak_auc"], 1.0)
        self.assertAlmostEqual(summary["evaluated_negative_hours"], 3.0 / 3600.0)
        self.assertEqual(summary["false_alarm_recordings_per_evaluated_hour"], 0.0)
        self.assertEqual(summary["false_alarm_rate_scope"], "first_alarm_per_recording")


class CheckpointConfigContractTests(unittest.TestCase):
    def test_round_trip_requires_complete_exact_schema(self):
        raw = Config().to_dict()
        rebuilt = config_from_dict(raw)
        self.assertEqual(rebuilt.to_dict(), raw)

    def test_missing_key_fails_instead_of_using_default(self):
        raw = Config().to_dict()
        del raw["model"]["window_sec"]
        with self.assertRaisesRegex(KeyError, "missing=.*window_sec"):
            config_from_dict(raw)

    def test_unknown_key_fails_instead_of_being_ignored(self):
        raw = Config().to_dict()
        raw["train"]["renamed_old_field"] = 1
        with self.assertRaisesRegex(KeyError, "unknown=.*renamed_old_field"):
            config_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
