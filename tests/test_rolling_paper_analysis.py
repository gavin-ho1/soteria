from __future__ import annotations

import unittest

import numpy as np

from streaming_predictor.analyze_rolling_paper_results import (
    DELONG_FAMILIES,
    EXPECTED_SEEDS,
    SOURCES,
    WILCOXON_FAMILIES,
    _holm,
)


class RollingPaperAnalysisContractTests(unittest.TestCase):
    def test_seed_contract_has_fifteen_unique_seeds(self) -> None:
        self.assertEqual(len(EXPECTED_SEEDS), 15)

    def test_every_statistical_pair_has_a_declared_source(self) -> None:
        for pairs, metrics in WILCOXON_FAMILIES.values():
            self.assertTrue(metrics)
            for a, b in pairs:
                self.assertIn(a, SOURCES)
                self.assertIn(b, SOURCES)
        for pairs in DELONG_FAMILIES.values():
            for a, b in pairs:
                self.assertIn(a, SOURCES)
                self.assertIn(b, SOURCES)

    def test_decoder_rows_reuse_one_checkpoint_replay(self) -> None:
        decoder_paths = {
            source.relative_path for source in SOURCES.values()
            if source.group == "decoder"
        }
        self.assertEqual(decoder_paths,
                         {"worker_posthoc/decoder/default_all_modes.json"})

    def test_cvar_and_hmm_are_separate_cumulative_steps(self) -> None:
        self.assertEqual(SOURCES["+cvar"].mode, "threshold")
        self.assertEqual(SOURCES["default"].mode, "hmm4_early_skip_ssem")
        self.assertEqual(SOURCES["+cvar"].relative_path,
                         SOURCES["default"].relative_path)

    def test_holm_is_monotone_in_sorted_raw_p_values(self) -> None:
        raw = [0.03, 0.001, np.nan, 0.02]
        adjusted = _holm(raw)
        self.assertTrue(np.isnan(adjusted[2]))
        ordered = sorted((raw[i], adjusted[i]) for i in (0, 1, 3))
        self.assertLessEqual(ordered[0][1], ordered[1][1])
        self.assertLessEqual(ordered[1][1], ordered[2][1])
        for p_raw, p_adjusted in ordered:
            self.assertGreaterEqual(p_adjusted, p_raw)


if __name__ == "__main__":
    unittest.main()
