r"""Evaluate frozen cross-validation checkpoints by true rolling replay

Example:
    python -u streaming_predictor/run_rolling_evaluation.py \
      --checkpoint-root path/to/checkpoints/default \
      --alarm-modes hmm4_early_ssem hmm4_early_skip_ssem \
      --cadence-sec 1.0 --device cpu \
      --output path/to/rolling_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sp_stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data as cdata
from common.paths import resolve_project_path
from streaming_predictor.data import expand_view_instances
from streaming_predictor.evaluate import (
    ALARM_MODES,
    alarm_records,
    calibrate_threshold,
    stream_clips,
    summarize,
)
from streaming_predictor.model import build_model
from streaming_predictor.rolling_evaluation import (
    config_from_dict,
    rolling_streams_for_instances,
    rolling_summary,
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, default=str))
    pending.replace(path)


_EVALUATOR_SOURCE_PATHS = (
    "common/data.py",
    "streaming_predictor/config.py",
    "streaming_predictor/data.py",
    "streaming_predictor/evaluate.py",
    "streaming_predictor/model.py",
    "streaming_predictor/rolling_evaluation.py",
    "streaming_predictor/run_rolling_evaluation.py",
)


def _evaluator_source_sha256() -> str:
    """Hash of every source file that can change rolling-replay results"""
    repo_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for rel in _EVALUATOR_SOURCE_PATHS:
        raw = (repo_root / rel).read_bytes()
        digest.update(rel.encode("utf-8"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _checkpoint_sha256(seed_dir: Path) -> str:
    folds = sorted(seed_dir.glob("fold_*.pt"))
    digest = hashlib.sha256()
    for fold in folds:
        raw = fold.read_bytes()
        digest.update(fold.name.encode("utf-8"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _complete_seed_dirs(root: Path) -> list[Path]:
    """Every seed directory under root, aborts if any is missing folds"""
    complete = []
    for seed_dir in sorted(root.glob("seed_*")):
        folds = sorted(seed_dir.glob("fold_*.pt"))
        if not folds:
            continue
        saved = torch.load(folds[0], map_location="cpu", weights_only=False)
        expected = int(saved["config"]["eval"]["n_folds"])
        if len(folds) != expected:
            raise SystemExit(
                f"[abort] {seed_dir.name} has {len(folds)}/{expected} fold "
                "checkpoints; finish or remove that seed before replaying.")
        complete.append(seed_dir)
    return complete


def _seed_from_dir(path: Path) -> int:
    return int(path.name.removeprefix("seed_"))


def _selected_metrics(summary: dict) -> dict:
    keys = (
        "auc", "pr_auc", "bacc", "accuracy", "f1", "precision", "recall",
        "tcs_detection_rate", "fos_false_alarm_rate", "median_lead_sec",
        "n_tcs", "n_fos", "n_tp", "n_fp", "n_fn", "n_tn",
    )
    return {key: summary[key] for key in keys if key in summary}


def _record_payload(records: list[dict], streams: dict) -> list[dict]:
    by_id = {r["ssid"]: r for r in records}
    payload = []
    for ssid, stream in sorted(streams.items()):
        record = by_id[ssid]
        payload.append({
            "ssid": ssid,
            "subject": stream.get("subject"),
            "label": int(stream["label"]),
            "peak_risk": float(np.max(stream["risk"])),
            "n_decisions": int(len(stream["risk"])),
            "first_decision_sec": float(stream["times"][0]),
            "last_decision_sec": float(stream["times"][-1]),
            "alarmed": bool(record["alarmed"]),
            "alarm_time_sec": record.get("alarm_time_sec"),
            "onset_sec": record.get("onset_sec"),
            "lead_sec": record.get("lead_sec"),
        })
    return payload


def _mean_ci(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "ci95_half_width": float("nan"), "n": 0}
    half = 0.0 if len(x) == 1 else float(
        sp_stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / math.sqrt(len(x))
    )
    return {"mean": float(np.mean(x)), "ci95_half_width": half, "n": int(len(x))}


def _aggregate(seed_results: list[dict], mode: str) -> dict:
    terminal_keys = seed_results[0]["modes"][mode]["terminal"].keys()
    rolling_keys = seed_results[0]["modes"][mode]["rolling"].keys()
    return {
        "terminal": {
            key: _mean_ci([s["modes"][mode]["terminal"].get(key, float("nan"))
                           for s in seed_results])
            for key in terminal_keys
            if isinstance(seed_results[0]["modes"][mode]["terminal"][key], (int, float))
        },
        "rolling": {
            key: _mean_ci([s["modes"][mode]["rolling"].get(key, float("nan"))
                           for s in seed_results])
            for key in rolling_keys
            if isinstance(seed_results[0]["modes"][mode]["rolling"][key], (int, float))
        },
    }


def evaluate_seed(
    seed_dir: Path,
    sequences: dict,
    seizures_by_ssid: dict,
    *,
    alarm_modes: list[str],
    cadence_sec: float,
    device: str,
) -> dict:
    seed = _seed_from_dir(seed_dir)
    pooled: dict[str, dict[str, dict]] = {
        mode: {
            "terminal_streams": {}, "rolling_streams": {},
            "terminal_records": [], "rolling_records": [],
        }
        for mode in alarm_modes
    }

    for checkpoint_path in sorted(seed_dir.glob("fold_*.pt")):
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = config_from_dict(saved["config"])
        cfg.device = device
        _, checkpoint_path_relative = resolve_project_path(
            checkpoint_path, field="checkpoint")
        print(f"[config] source={checkpoint_path_relative}", flush=True)
        print(json.dumps(cfg.to_dict(), indent=2, sort_keys=True, default=str),
              flush=True)
        model = build_model(cfg)
        model.load_state_dict(saved["state_dict"])
        model.to(device).eval()

        calib_inst = expand_view_instances(saved["calib_ssids"], seizures_by_ssid)
        val_inst = expand_view_instances(saved["val_ssids"], seizures_by_ssid)

        terminal_calib = stream_clips(model, cfg, calib_inst, sequences, device)
        terminal_val = stream_clips(model, cfg, val_inst, sequences, device)
        rolling_calib = rolling_streams_for_instances(
            model, cfg, calib_inst, sequences, device=device,
            cadence_sec=cadence_sec,
        )
        rolling_val = rolling_streams_for_instances(
            model, cfg, val_inst, sequences, device=device,
            cadence_sec=cadence_sec,
        )

        for mode in alarm_modes:
            cfg.alarm.mode = mode
            terminal_params = calibrate_threshold(terminal_calib, cfg)
            rolling_params = calibrate_threshold(rolling_calib, cfg)
            terminal_records = alarm_records(terminal_val, terminal_params, cfg)
            rolling_records = alarm_records(rolling_val, rolling_params, cfg)
            bucket = pooled[mode]
            bucket["terminal_streams"].update(terminal_val)
            bucket["rolling_streams"].update(rolling_val)
            bucket["terminal_records"].extend(terminal_records)
            bucket["rolling_records"].extend(rolling_records)

    result = {"seed": seed, "modes": {}}
    for mode, bucket in pooled.items():
        terminal = _selected_metrics(summarize(bucket["terminal_records"]))
        rolling = rolling_summary(
            bucket["rolling_records"], bucket["rolling_streams"],
            cadence_sec=cadence_sec,
        )
        result["modes"][mode] = {
            "terminal": terminal,
            "rolling": rolling,
            "terminal_records": _record_payload(
                bucket["terminal_records"], bucket["terminal_streams"]),
            "rolling_records": _record_payload(
                bucket["rolling_records"], bucket["rolling_streams"]),
        }
    return result


def _load_resumable(output: Path, checkpoint_root: str, alarm_modes: list[str],
                     cadence_sec: float, device: str, evaluator_sha256: str,
                     data_provenance: dict, root: Path) -> dict | None:
    """Existing payload, or None if absent

    Exits if settings, evaluator source, data, or a completed seed's checkpoints changed: resuming would pair mismatched runs in the aggregate
    """
    if not output.exists():
        return None
    _, output_relative = resolve_project_path(output, field="--output")
    payload = json.loads(output.read_text())
    mismatches = []
    for key, value in (("device", device),
                       ("evaluator_source_sha256", evaluator_sha256),
                       ("data_provenance", data_provenance)):
        if payload.get(key) != value:
            mismatches.append(f"{key} ({payload.get(key)!r} != {value!r})")
    for seed_result in payload["seeds"]:
        seed_dir = root / f"seed_{seed_result['seed']:02d}"
        if _checkpoint_sha256(seed_dir) != seed_result["checkpoint_sha256"]:
            mismatches.append(f"checkpoints of {seed_dir.name} changed")
    if payload.get("checkpoint_root") != checkpoint_root:
        mismatches.append(f"checkpoint_root ({payload.get('checkpoint_root')!r} != {checkpoint_root!r})")
    if payload.get("alarm_modes") != alarm_modes:
        mismatches.append(f"alarm_modes ({payload.get('alarm_modes')!r} != {alarm_modes!r})")
    if payload.get("cadence_sec") != cadence_sec:
        mismatches.append(f"cadence_sec ({payload.get('cadence_sec')!r} != {cadence_sec!r})")
    if mismatches:
        raise SystemExit(
            f"[abort] {output_relative} exists but was produced with different settings: "
            f"{'; '.join(mismatches)}. Use a fresh --output rather than mixing runs."
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alarm-modes", nargs="+", choices=ALARM_MODES, required=True)
    parser.add_argument("--cadence-sec", type=float, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    args = parser.parse_args()

    root, checkpoint_root_relative = resolve_project_path(
        args.checkpoint_root, field="--checkpoint-root")
    output, output_relative = resolve_project_path(args.output, field="--output")
    seed_dirs = _complete_seed_dirs(root)
    if not seed_dirs:
        raise SystemExit(
            "No seeds with all cross-validation folds found under "
            f"{checkpoint_root_relative}")

    sequences = cdata.load_view_sequences(verbose=False)
    seizures = cdata.load_seizures(sequences, verbose=False)
    seizures_by_ssid = {s.ssid: s for s in seizures}
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    evaluator_sha256 = _evaluator_source_sha256()
    data_provenance = {
        "seizures_csv_sha256": hashlib.sha256(cdata.SEIZURES_CSV.read_bytes()).hexdigest(),
        "clean_cache_size": cdata.CLEAN_CACHE.stat().st_size,
    }

    payload = _load_resumable(
        output, checkpoint_root_relative, args.alarm_modes, args.cadence_sec,
        args.device, evaluator_sha256, data_provenance, root)
    if payload is not None:
        done_seeds = {s["seed"] for s in payload["seeds"]}
        remaining = [d for d in seed_dirs if _seed_from_dir(d) not in done_seeds]
        print(f"[resume] {output_relative} already has {len(done_seeds)}/{len(seed_dirs)} seeds "
              f"done; continuing with the remaining {len(remaining)}.", flush=True)
        seed_dirs = remaining
    else:
        payload = {
            "git_sha": git_sha,
            "evaluator_source_sha256": evaluator_sha256,
            "data_provenance": data_provenance,
            "device": args.device,
            "checkpoint_root": checkpoint_root_relative,
            "cadence_sec": args.cadence_sec,
            "alarm_modes": args.alarm_modes,
            "metric_scope": {
                "terminal_auc": "peak raw risk in one endpoint-aligned terminal window",
                "rolling_auc": "peak current-time raw risk across all replayed windows",
                "false_alarm_rate": "first alarm per onset-negative seizure recording",
                "negative_exposure": "onset-negative seizure footage, not ordinary activity",
            },
            "seeds": [],
            "aggregate": {},
        }

    if not seed_dirs:
        print(f"[resume] all seeds already present in {output_relative}; nothing to do.", flush=True)
    for seed_dir in seed_dirs:
        print(f"[seed {_seed_from_dir(seed_dir)}] rolling replay", flush=True)
        seed_result = evaluate_seed(
            seed_dir, sequences, seizures_by_ssid,
            alarm_modes=args.alarm_modes,
            cadence_sec=args.cadence_sec,
            device=args.device,
        )
        seed_result["checkpoint_sha256"] = _checkpoint_sha256(seed_dir)
        payload["seeds"].append(seed_result)
        payload["aggregate"] = {
            mode: _aggregate(payload["seeds"], mode) for mode in args.alarm_modes
        }
        _write_json_atomic(output, payload)
        for mode in args.alarm_modes:
            t_auc = seed_result["modes"][mode]["terminal"]["auc"]
            r_auc = seed_result["modes"][mode]["rolling"]["rolling_peak_auc"]
            print(f"  {mode}: terminal AUC={t_auc:.3f}, rolling AUC={r_auc:.3f}", flush=True)
    print(f"Saved {output_relative}", flush=True)


if __name__ == "__main__":
    main()
