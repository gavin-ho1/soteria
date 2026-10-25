"""Create paper-ready statistics from the checkpointed rolling-only sweep"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from streaming_predictor.delong import delong_roc_test
from common.paths import resolve_project_path


EXPECTED_SEEDS = {
    12811957, 1661938183, 1697828733, 1344571277, 320942036,
    1508945547, 2004173725, 1694056806, 1456124910, 1604011579,
    1814110808, 1871455221, 1205112118, 198655547, 1904331482,
}


@dataclass(frozen=True)
class Source:
    relative_path: str
    mode: str
    label: str
    group: str


SOURCES = {
    "bare": Source("worker_posthoc/objective/bare.json", "threshold", "Bare LRU", "objective"),
    "+chan_drop": Source("worker_posthoc/objective/+chan_drop.json", "threshold", "+ channel dropout", "objective"),
    "+hazard": Source("worker_posthoc/objective/+hazard.json", "threshold", "+ hazard loss", "objective"),
    "+alarm_loss": Source("worker_posthoc/objective/+alarm_loss.json", "threshold", "+ absolute margin", "objective"),
    "+focal": Source("worker_posthoc/objective/+focal.json", "threshold", "+ focal weighting", "objective"),
    "+groupdro": Source("worker_posthoc/objective/+groupdro.json", "threshold", "+ Group DRO", "objective"),
    "+margin": Source("worker_posthoc/objective/+margin.json", "threshold", "+ pairwise margin", "objective"),
    "+cvar": Source("worker_posthoc/decoder/default_all_modes.json", "threshold", "+ CVaR", "objective"),
    "default": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_early_skip_ssem", "LRU (full)", "architecture"),
    "gru": Source("worker_arch/rolling/gru.json", "hmm4_early_skip_ssem", "GRU", "architecture"),
    "lstm": Source("worker_arch/rolling/lstm.json", "hmm4_early_skip_ssem", "LSTM", "architecture"),
    "tcn": Source("worker_arch/rolling/tcn.json", "hmm4_early_skip_ssem", "TCN", "architecture"),
    "transformer": Source("worker_arch/rolling/transformer.json", "hmm4_early_skip_ssem", "Causal Transformer", "architecture"),
    "lru_pure": Source("worker_posthoc/objective/bare.json", "threshold", "LRU (BCE only)", "pure_architecture"),
    "gru_pure": Source("worker_misc/rolling/gru_pure.json", "threshold", "GRU (BCE only)", "pure_architecture"),
    "lstm_pure": Source("worker_misc/rolling/lstm_pure.json", "threshold", "LSTM (BCE only)", "pure_architecture"),
    "decoder_default": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_early_skip_ssem", "HMM4 early skip SSEM", "decoder"),
    "decoder_noskip": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_early_ssem", "HMM4 early SSEM", "decoder"),
    "decoder_no_ssem": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_early", "HMM4 early, no SSEM", "decoder"),
    "decoder_full": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_early_full_ssem", "HMM4 early full SSEM", "decoder"),
    "decoder_regressive": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_early_skip_regressive_ssem", "HMM4 early skip regressive SSEM", "decoder"),
    "decoder_ictal_skip": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_skip_ssem", "HMM4 highest-risk, skip", "decoder"),
    "decoder_ictal_noskip": Source("worker_posthoc/decoder/default_all_modes.json", "hmm4_ssem", "HMM4 highest-risk, no skip", "decoder"),
    "decoder_hmm2": Source("worker_posthoc/decoder/default_all_modes.json", "hmm", "HMM2", "decoder"),
    "decoder_threshold": Source("worker_posthoc/decoder/default_all_modes.json", "threshold", "Raw threshold", "decoder"),
}

METRICS = (
    "rolling_peak_auc", "rolling_peak_pr_auc", "f1", "accuracy",
    "balanced_accuracy", "positive_recording_sensitivity",
    "negative_recording_false_alarm_rate", "median_first_alarm_offset_sec",
    "false_alarm_recordings_per_evaluated_hour",
)

WILCOXON_FAMILIES = {
    "objective": (
        (("+chan_drop", "bare"), ("+hazard", "+chan_drop"),
         ("+alarm_loss", "+hazard"), ("+focal", "+alarm_loss"),
         ("+groupdro", "+focal"), ("+margin", "+groupdro"),
         ("+cvar", "+margin")),
        METRICS,
    ),
    "objective_total": (
        ((("+cvar", "bare")),),
        ("rolling_peak_auc", "rolling_peak_pr_auc"),
    ),
    "objective_decoder": (
        (("default", "+cvar"),),
        ("f1", "accuracy", "balanced_accuracy", "positive_recording_sensitivity",
         "negative_recording_false_alarm_rate", "median_first_alarm_offset_sec"),
    ),
    "architecture": (
        (("tcn", "default"), ("lstm", "default"), ("gru", "default"),
         ("transformer", "default")),
        METRICS,
    ),
    "pure_architecture": (
        (("lstm_pure", "lru_pure"), ("gru_pure", "lru_pure")), METRICS,
    ),
    "decoder": (
        tuple((key, "decoder_default") for key in (
            "decoder_noskip", "decoder_no_ssem", "decoder_full",
            "decoder_regressive", "decoder_ictal_skip",
            "decoder_ictal_noskip", "decoder_hmm2", "decoder_threshold",
        )),
        ("f1", "accuracy", "balanced_accuracy", "positive_recording_sensitivity",
         "negative_recording_false_alarm_rate", "median_first_alarm_offset_sec"),
    ),
}

DELONG_FAMILIES = {
    # AUC comparisons used in the paper, decoder variants aren't here: they share the same raw risk stream, so they're compared with paired alarm-metric tests instead of DeLong
    "architecture": (
        ("default", "tcn"), ("default", "lstm"),
        ("default", "gru"), ("default", "transformer"),
    ),
    "pure_architecture": (
        ("lru_pure", "lstm_pure"), ("lru_pure", "gru_pure"),
    ),
    "objective": (
        ("+chan_drop", "bare"), ("+hazard", "+chan_drop"),
        ("+alarm_loss", "+hazard"), ("+focal", "+alarm_loss"),
        ("+groupdro", "+focal"), ("+margin", "+groupdro"),
        ("+cvar", "+margin"),
    ),
    "objective_total": (("+cvar", "bare"),),
}
ROC_CONDITIONS = ("default", "tcn", "lstm", "gru", "transformer")


def _halfwidth(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(values) - 1)
                 * values.std(ddof=1) / np.sqrt(len(values)))


def _holm(p_values: list[float]) -> list[float]:
    """NaNs stay NaN"""
    out = np.full(len(p_values), np.nan, dtype=float)
    finite = [i for i, p in enumerate(p_values) if np.isfinite(p)]
    ordered = sorted(finite, key=lambda i: p_values[i])
    running = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[idx]))
        out[idx] = running
    return out.tolist()


def _clustered_auc_delta(labels: np.ndarray, subjects: np.ndarray,
                         scores_a: np.ndarray, scores_b: np.ndarray,
                         rng: np.random.Generator, n_boot: int) -> dict:
    unique = np.unique(subjects)
    index = {subject: np.flatnonzero(subjects == subject) for subject in unique}
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[s] for s in sampled])
        y = labels[idx]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(roc_auc_score(y, scores_a[idx]) - roc_auc_score(y, scores_b[idx]))
    values = np.asarray(deltas, dtype=float)
    if len(values) < int(0.95 * n_boot):
        raise ValueError(f"only {len(values)}/{n_boot} valid clustered bootstrap samples")
    p = min(1.0, 2.0 * min((np.count_nonzero(values <= 0) + 1) / (len(values) + 1),
                           (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)))
    return {"cluster_boot_n": len(values),
            "cluster_delta_mean": float(values.mean()),
            "cluster_ci95_low": float(np.percentile(values, 2.5)),
            "cluster_ci95_high": float(np.percentile(values, 97.5)),
            "cluster_p_two_sided": float(p)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty output {path}")
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def _metrics(records: list[dict], rolling: dict) -> dict[str, float]:
    labels = np.asarray([int(record["label"]) for record in records])
    scores = np.asarray([float(record["peak_risk"]) for record in records])
    alarms = np.asarray([bool(record["alarmed"]) for record in records])
    tp = int(np.sum((labels == 1) & alarms))
    fp = int(np.sum((labels == 0) & alarms))
    fn = int(np.sum((labels == 1) & ~alarms))
    tn = int(np.sum((labels == 0) & ~alarms))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    offsets = [float(record["lead_sec"]) for record in records
               if record["label"] == 1 and record["alarmed"]
               and record.get("lead_sec") is not None]
    return {
        "rolling_peak_auc": float(roc_auc_score(labels, scores)),
        "rolling_peak_pr_auc": float(average_precision_score(labels, scores)),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "accuracy": (tp + tn) / len(records),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "positive_recording_sensitivity": sensitivity,
        "negative_recording_false_alarm_rate": fp / (fp + tn),
        "median_first_alarm_offset_sec": float(np.median(offsets)) if offsets else float("nan"),
        "false_alarm_recordings_per_evaluated_hour": float(
            rolling["false_alarm_recordings_per_evaluated_hour"]),
    }


def _load_sources(root: Path) -> tuple[dict, list[int], dict[str, str]]:
    cache: dict[Path, dict] = {}
    hashes: dict[str, str] = {}
    conditions: dict = {}
    canonical: dict[str, tuple[str, int]] | None = None
    for key, source in SOURCES.items():
        path = root / source.relative_path
        if path not in cache:
            cache[path] = json.loads(path.read_text())
            hashes[source.relative_path] = _sha256(path)
        payload = cache[path]
        if source.mode not in payload.get("alarm_modes", []):
            raise ValueError(f"{path}: missing requested mode {source.mode}")
        seeds = {}
        for seed_result in payload.get("seeds", []):
            seed = int(seed_result["seed"])
            mode_result = seed_result["modes"][source.mode]
            records = mode_result["rolling_records"]
            if len(records) != 210:
                raise ValueError(f"{key} seed {seed}: expected 210 records, got {len(records)}")
            metadata = {str(r["ssid"]): (str(r["subject"]), int(r["label"])) for r in records}
            if len(metadata) != 210:
                raise ValueError(f"{key} seed {seed}: duplicate recording ID")
            if canonical is None:
                canonical = metadata
            elif metadata != canonical:
                raise ValueError(f"{key} seed {seed}: cohort metadata mismatch")
            seeds[seed] = {
                "records": records,
                "metrics": _metrics(records, mode_result["rolling"]),
            }
        if set(seeds) != EXPECTED_SEEDS:
            raise ValueError(f"{key}: seed set mismatch: {sorted(seeds)}")
        conditions[key] = {"source": source, "seeds": seeds}
    return conditions, sorted(EXPECTED_SEEDS), hashes


def _condition_rows(conditions: dict, seeds: list[int]) -> tuple[list[dict], list[dict]]:
    summary_rows, seed_rows = [], []
    for key, condition in conditions.items():
        source = condition["source"]
        for seed in seeds:
            seed_rows.append({"condition": key, "seed": seed,
                              **condition["seeds"][seed]["metrics"]})
        summary = {"condition": key, "label": source.label, "group": source.group,
                   "decoder_mode": source.mode, "n_seeds": len(seeds)}
        for metric in METRICS:
            values = np.asarray([condition["seeds"][s]["metrics"][metric] for s in seeds])
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(finite.mean())
            summary[f"{metric}_sd"] = float(finite.std(ddof=1))
            summary[f"{metric}_ci95_halfwidth"] = _halfwidth(finite)
        summary_rows.append(summary)
    return summary_rows, seed_rows


def _wilcoxon_rows(conditions: dict, seeds: list[int]) -> list[dict]:
    rows = []
    for family, (pairs, metrics) in WILCOXON_FAMILIES.items():
        for metric in metrics:
            block = []
            for a, b in pairs:
                delta = np.asarray([
                    conditions[a]["seeds"][seed]["metrics"][metric]
                    - conditions[b]["seeds"][seed]["metrics"][metric]
                    for seed in seeds
                ], dtype=float)
                delta = delta[np.isfinite(delta)]
                p_raw = (1.0 if np.allclose(delta, 0.0) else
                         float(stats.wilcoxon(delta, alternative="two-sided").pvalue))
                block.append({
                    "family": family, "metric": metric, "condition_a": a,
                    "condition_b": b, "delta_definition": "condition_a - condition_b",
                    "n_pairs": len(delta), "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "ci95_halfwidth": _halfwidth(delta), "p_raw": p_raw,
                    "positive": int(np.sum(delta > 0)), "negative": int(np.sum(delta < 0)),
                    "zero": int(np.sum(delta == 0)),
                })
            adjusted = _holm([row["p_raw"] for row in block])
            for row, adjusted_p in zip(block, adjusted):
                row["p_holm_within_family_metric"] = adjusted_p
                rows.append(row)
    return rows


def _clip_and_score_rows(conditions: dict, seeds: list[int]) -> tuple[list[dict], list[dict], dict]:
    clip_rows, mean_rows, maps = [], [], {}
    for key, condition in conditions.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        metadata = {}
        for seed in seeds:
            for record in condition["seeds"][seed]["records"]:
                clip_rows.append({"condition": key, "seed": seed, **record})
                ssid = str(record["ssid"])
                grouped[ssid].append(float(record["peak_risk"]))
                metadata[ssid] = (str(record["subject"]), int(record["label"]))
        maps[key] = {}
        for ssid in sorted(grouped):
            values = np.asarray(grouped[ssid])
            subject, label = metadata[ssid]
            row = {"condition": key, "ssid": ssid, "subject": subject,
                   "label": label, "mean_peak_risk": float(values.mean()),
                   "sd_peak_risk": float(values.std(ddof=1)), "n_seeds": len(values)}
            mean_rows.append(row)
            maps[key][ssid] = row
    return clip_rows, mean_rows, maps


def _delong_rows(maps: dict, n_boot: int, bootstrap_seed: int) -> list[dict]:
    rows = []
    rng = np.random.default_rng(bootstrap_seed)
    for family, pairs in DELONG_FAMILIES.items():
        block = []
        for a, b in pairs:
            ssids = sorted(set(maps[a]) & set(maps[b]))
            if len(ssids) != 210:
                raise ValueError(f"{a}/{b}: expected 210 aligned recordings")
            labels = np.asarray([maps[a][s]["label"] for s in ssids])
            subjects = np.asarray([maps[a][s]["subject"] for s in ssids])
            scores_a = np.asarray([maps[a][s]["mean_peak_risk"] for s in ssids])
            scores_b = np.asarray([maps[b][s]["mean_peak_risk"] for s in ssids])
            result = delong_roc_test(labels, scores_a, scores_b)
            result.update(_clustered_auc_delta(labels, subjects, scores_a, scores_b,
                                               rng, n_boot))
            block.append({"family": family, "condition_a": a, "condition_b": b,
                          "score_definition": "per-recording mean rolling peak risk across 15 seeds",
                          **result})
        delong_adjusted = _holm([row["p"] for row in block])
        cluster_adjusted = _holm([row["cluster_p_two_sided"] for row in block])
        for row, delong_p, cluster_p in zip(block, delong_adjusted, cluster_adjusted):
            row["p_holm_within_family"] = delong_p
            row["cluster_p_holm_within_family"] = cluster_p
            rows.append(row)
    return rows


def _roc_rows(conditions: dict, seeds: list[int]) -> tuple[list[dict], list[dict]]:
    grid = np.linspace(0.0, 1.0, 201)
    curves, aucs = [], []
    for key in ROC_CONDITIONS:
        tprs = []
        for seed in seeds:
            records = conditions[key]["seeds"][seed]["records"]
            labels = np.asarray([r["label"] for r in records])
            scores = np.asarray([r["peak_risk"] for r in records])
            fpr, tpr, _ = roc_curve(labels, scores)
            interpolated = np.interp(grid, fpr, tpr)
            interpolated[0], interpolated[-1] = 0.0, 1.0
            tprs.append(interpolated)
            aucs.append({"condition": key, "seed": seed,
                         "auc": float(roc_auc_score(labels, scores))})
        matrix = np.asarray(tprs)
        critical = stats.t.ppf(0.975, len(matrix) - 1)
        sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
        for index, fpr_value in enumerate(grid):
            curves.append({"condition": key, "fpr": float(fpr_value),
                           "tpr_mean": float(matrix[:, index].mean()),
                           "tpr_ci95_halfwidth": float(critical * sem[index])})
    return curves, aucs


def _operating_point(
    conditions: dict, seeds: list[int], key: str = "default"
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Mean (FPR, TPR) of the deployed decoder across seeds, with 95% CI half-widths

    FPR is the negative-recording false-alarm rate, TPR is the positive-recording sensitivity
    This is the decoder's own operating point (Table I), not a point from the peak-risk threshold sweep the ROC curve shows
    """
    sens = np.asarray([conditions[key]["seeds"][s]["metrics"]["positive_recording_sensitivity"]
                        for s in seeds])
    far = np.asarray([conditions[key]["seeds"][s]["metrics"]["negative_recording_false_alarm_rate"]
                       for s in seeds])
    return (float(far.mean()), float(sens.mean())), (float(_halfwidth(far)), float(_halfwidth(sens)))


def _plot_roc(curves: list[dict], aucs: list[dict], out_dir: Path,
              operating_point: tuple[float, float] | None = None,
              operating_point_err: tuple[float, float] | None = None) -> None:
    colors = {"default": "#1f77b4", "tcn": "#ff7f0e", "lstm": "#2ca02c",
              "gru": "#d62728", "transformer": "#9467bd"}
    # "(full)" vs "(BCE only)" only matters in Table II, no BCE-only curve is plotted here, so this is just "LRU"
    plot_labels = {"default": "LRU"}
    # Draw LRU ("default") last so it renders on top of the other curves
    draw_order = [c for c in ROC_CONDITIONS if c != "default"] + ["default"]
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    handles_by_condition = {}
    for condition in draw_order:
        rows = [row for row in curves if row["condition"] == condition]
        fpr = np.asarray([row["fpr"] for row in rows])
        mean = np.asarray([row["tpr_mean"] for row in rows])
        half = np.asarray([row["tpr_ci95_halfwidth"] for row in rows])
        label = plot_labels.get(condition, SOURCES[condition].label)
        (line,) = axis.plot(fpr, mean, linewidth=1.8, label=label,
                             color=colors[condition],
                             zorder=4 if condition == "default" else 3)
        handles_by_condition[condition] = line
        axis.fill_between(fpr, np.clip(mean - half, 0, 1), np.clip(mean + half, 0, 1),
                          alpha=0.1, color=colors[condition], zorder=2)
    (chance_line,) = axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1,
                               color="0.5", label="Chance")
    op_handle = None
    if operating_point is not None:
        op_fpr, op_tpr = operating_point
        if operating_point_err is not None:
            op_xerr, op_yerr = operating_point_err
            axis.errorbar([op_fpr], [op_tpr], xerr=[op_xerr], yerr=[op_yerr],
                          fmt="none", ecolor="black", elinewidth=0.9,
                          capsize=3, capthick=0.9, zorder=5)
        op_handle = axis.scatter([op_fpr], [op_tpr], marker="o", s=55,
                                  facecolor="#1f77b4", edgecolor="black",
                                  linewidth=0.9, zorder=6,
                                  label="SOTERIA operating point")
        # Put the label up-left of the point, in the empty space above the LRU curve, so it doesn't cross any line
        axis.annotate(f"({op_fpr:.3f}, {op_tpr:.3f})",
                       xy=(op_fpr, op_tpr), xycoords="data",
                       xytext=(-4, 4), textcoords="offset points",
                       ha="right", va="bottom", fontsize=8, zorder=6)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate",
             xlim=(0, 1), ylim=(0, 1))
    # LRU first in the legend, though it's drawn last
    legend_order = ["default"] + [c for c in ROC_CONDITIONS if c != "default"]
    handles = [handles_by_condition[c] for c in legend_order] + [chance_line]
    labels = [plot_labels.get(c, SOURCES[c].label) for c in legend_order] + ["Chance"]
    if op_handle is not None:
        handles.append(op_handle)
        labels.append("SOTERIA operating point")
    axis.legend(handles, labels, frameon=True, framealpha=0.92, fontsize=7.5,
                loc="lower right")
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "backbone_rolling_roc.png", dpi=300)
    fig.savefig(out_dir / "backbone_rolling_roc.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cluster-boot", type=int, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    args = parser.parse_args()
    root, _ = resolve_project_path(args.result_root, field="--result-root")
    out_dir, out_dir_relative = resolve_project_path(args.out_dir, field="--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions, seeds, input_hashes = _load_sources(root)
    summary, per_seed = _condition_rows(conditions, seeds)
    wilcoxon = _wilcoxon_rows(conditions, seeds)
    clips, mean_scores, maps = _clip_and_score_rows(conditions, seeds)
    delong = _delong_rows(maps, args.cluster_boot, args.bootstrap_seed)
    curves, roc_aucs = _roc_rows(conditions, seeds)
    operating_point, operating_point_err = _operating_point(conditions, seeds)
    outputs = {
        "condition_summary.csv": summary,
        "condition_per_seed.csv": per_seed,
        "wilcoxon_pairs.csv": wilcoxon,
        "rolling_record_scores.csv": clips,
        "mean_peak_risks.csv": mean_scores,
        "delong_and_clustered_auc_pairs.csv": delong,
        "roc_curves.csv": curves,
        "roc_auc_per_seed.csv": roc_aucs,
    }
    for filename, rows in outputs.items():
        _write_csv(out_dir / filename, rows)
    _plot_roc(curves, roc_aucs, out_dir, operating_point, operating_point_err)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metric_scope": "outcome-independent 1 s rolling replay",
        "conditions": list(SOURCES), "seeds": seeds,
        "n_records_per_condition_seed": 210,
        "cluster_bootstrap_replicates": args.cluster_boot,
        "bootstrap_seed": args.bootstrap_seed,
        "input_sha256": input_hashes,
        "outputs": {},
    }
    for path in sorted(out_dir.iterdir()):
        # A manifest can't contain a stable hash of itself, and excluding it stops a rerun from carrying forward the previous manifest's hash
        if (path.is_file() and path.name != "artifact_manifest.json"
                and not path.name.endswith(".pending")):
            manifest["outputs"][path.name] = {"sha256": _sha256(path),
                                                    "bytes": path.stat().st_size}
    pending = out_dir / "artifact_manifest.json.pending"
    pending.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    pending.replace(out_dir / "artifact_manifest.json")
    print(f"Validated {len(SOURCES)} paper rows x 15 seeds x 210 recordings")
    print(f"Wrote rolling paper artifacts to {out_dir_relative}")


if __name__ == "__main__":
    main()
