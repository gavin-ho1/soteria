"""Paired per-seed test: does the HMM decoder's realized sensitivity, at its own false-alarm rate, exceed what the raw peak-risk threshold ROC curve achieves at that same false-alarm rate?

Fig. 2 plots the HMM decoder's (FAR, sensitivity) point on top of the peak-risk threshold ROC curve, which the decoder doesn't sweep
This script tests that visual comparison per seed: for each of the 15 seeds, it interpolates the seed's peak-risk ROC curve (threshold decoder mode) at its own HMM decoder false-alarm rate, then compares the interpolated TPR against the HMM decoder's realized sensitivity
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve

from common.paths import resolve_project_path

EXPECTED_SEEDS = sorted({
    12811957, 1661938183, 1697828733, 1344571277, 320942036,
    1508945547, 2004173725, 1694056806, 1456124910, 1604011579,
    1814110808, 1871455221, 1205112118, 198655547, 1904331482,
})

DECODER_MODE = "hmm4_early_skip_ssem"
THRESHOLD_MODE = "threshold"
RELATIVE_PATH = "worker_posthoc/decoder/default_all_modes.json"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def _seed_diff(mode_results: dict) -> tuple[float, float, float, float]:
    hmm_records = mode_results[DECODER_MODE]["rolling_records"]
    thr_records = mode_results[THRESHOLD_MODE]["rolling_records"]

    labels = np.asarray([int(r["label"]) for r in thr_records])
    scores = np.asarray([float(r["peak_risk"]) for r in thr_records])
    fpr, tpr, _ = roc_curve(labels, scores)

    hmm_labels = np.asarray([int(r["label"]) for r in hmm_records])
    hmm_alarmed = np.asarray([bool(r["alarmed"]) for r in hmm_records])
    tp = int(np.sum((hmm_labels == 1) & hmm_alarmed))
    fn = int(np.sum((hmm_labels == 1) & ~hmm_alarmed))
    fp = int(np.sum((hmm_labels == 0) & hmm_alarmed))
    tn = int(np.sum((hmm_labels == 0) & ~hmm_alarmed))
    hmm_sens = tp / (tp + fn)
    hmm_far = fp / (fp + tn)

    curve_tpr_at_far = float(np.interp(hmm_far, fpr, tpr))
    return hmm_far, hmm_sens, curve_tpr_at_far, hmm_sens - curve_tpr_at_far


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    root, _ = resolve_project_path(args.result_root, field="--result-root")
    out_dir, _ = resolve_project_path(args.out_dir, field="--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads((root / RELATIVE_PATH).read_text())
    seed_map = {int(s["seed"]): s["modes"] for s in payload["seeds"]}
    if set(seed_map) != set(EXPECTED_SEEDS):
        raise ValueError(f"seed set mismatch: {sorted(seed_map)}")

    rows = []
    for seed in EXPECTED_SEEDS:
        far, sens, curve_tpr, diff = _seed_diff(seed_map[seed])
        rows.append({
            "seed": seed, "decoder_far": far, "decoder_sensitivity": sens,
            "threshold_curve_tpr_at_far": curve_tpr, "diff": diff,
        })
    _write_csv(out_dir / "operating_point_vs_curve_per_seed.csv",
               list(rows[0]), rows)

    diffs = np.asarray([r["diff"] for r in rows])
    n = len(diffs)
    mean = float(diffs.mean())
    half = float(stats.t.ppf(0.975, n - 1) * diffs.std(ddof=1) / np.sqrt(n))
    tstat, pval = stats.ttest_1samp(diffs, 0.0)
    wstat, wp = stats.wilcoxon(diffs)

    summary = {
        "n_seeds": n,
        "mean_diff": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "paired_t": float(tstat),
        "paired_t_p": float(pval),
        "wilcoxon_w": float(wstat),
        "wilcoxon_p": float(wp),
        "n_seeds_nonnegative_diff": int((diffs >= 0).sum()),
        "n_seeds_strictly_positive_diff": int((diffs > 0).sum()),
    }
    _write_csv(out_dir / "operating_point_vs_curve_summary.csv",
               list(summary), [summary])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
