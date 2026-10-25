r"""Train every paper condition under patient-disjoint 10-fold CV

Conditions (see ABLATION_CONDITIONS) map onto the paper's tables:
  A - objective build-up, Table IV: bare -> +chan_drop -> +hazard -> +alarm_loss -> +focal -> +groupdro -> +margin -> default
      The CVaR row is the `default` checkpoint replayed with the threshold decoder
      The "+ HMM decoder (full)" row is the same checkpoint with the HMM decoder
  B - backbone swaps, Table II upper block: gru / lstm / tcn / transformer
  C - BCE-only floors, Table II lower block: gru_pure / lstm_pure (the LRU floor is `bare`)
The decoder variants of Table VI are not separate conditions: they are rolling-replay decoders applied to the `default` checkpoints

All conditions share one seed list, so per-seed deltas are paired
Results are saved incrementally and can be resumed if interrupted
Output is JSON only

The paper's numbers come from one-second rolling replay of the checkpoints this script saves (--save-checkpoints), via run_rolling_evaluation.py
The metrics this script stores are endpoint-aligned terminal-window metrics
The README lists the exact commands

The paper runs used CPU
For a model this small MPS is ~22x slower, and switching device changes float results, so seeds run on different devices couldn't be paired

Usage:
    python streaming_predictor/ablation_study.py --n-seeds 15 --device cpu \
        --seed-list ... --conditions default \
        --save-checkpoints results/paper/checkpoints \
        --output results/paper/train.json
    python streaming_predictor/ablation_study.py --smoke --n-seeds 1 \
        --device cpu --seed-list 12811957 --conditions default \
        --save-checkpoints results/smoke/checkpoints --output results/smoke/train.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as sp_stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data as cdata
from common.paths import resolve_project_path
from common.splits import patient_grouped
from streaming_predictor.data import expand_view_instances
from streaming_predictor.evaluate import (
    alarm_records,
    calibrate_threshold,
    compute_operating_curves,
    stream_clips,
    summarize,
)
from streaming_predictor.train import (
    patient_disjoint_calib_split,
    train_one,
)
from streaming_predictor.config import Config


def _write_json_atomic(path: Path, payload: dict) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with open(pending, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    pending.replace(path)


_TRAINING_SOURCE_PATHS = (
    "common/data.py",
    "common/splits.py",
    "streaming_predictor/ablation_study.py",
    "streaming_predictor/config.py",
    "streaming_predictor/data.py",
    "streaming_predictor/evaluate.py",
    "streaming_predictor/model.py",
    "streaming_predictor/train.py",
)


def _training_source_provenance(repo_root: Path) -> dict:
    """Hash of every file that can change this sweep's results
    The analysis scripts are left out so they can be edited during a long run without breaking resume, paths and byte lengths go into the hash so concatenations are unambiguous
    """
    aggregate = hashlib.sha256()
    per_file: dict[str, str] = {}
    for rel in _TRAINING_SOURCE_PATHS:
        raw = (repo_root / rel).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        per_file[rel] = digest
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(len(raw).to_bytes(8, "big"))
        aggregate.update(raw)
    return {"sha256": aggregate.hexdigest(), "files": per_file}

# Ablation condition registry
#   label: human-readable name
#   group: letter for grouping rows
#   overrides: dict "section.key" → value applied on top of default Config()
#   alarm_mode_override: decoder for the terminal-window metrics stored here

_BARE = {
    "train.focal_gamma":        0.0,
    "train.groupdro":           False,
    "train.feat_channel_drop_p": 0.0,
    "train.alarm_margin_weight": 0.0,
    "train.alarm_loss_weight":  0.0,
    "train.hazard_loss_weight": 0.0,
    "train.cvar_weight":        0.0,
}

ABLATION_CONDITIONS: dict[str, dict] = {
    # A: Bottom-up build sequence
    # Each row = previous row + one component, `default` (overrides={}) is the complete objective
    "bare":        {"label": "Bare LRU  (BCE + threshold only)",  "group": "A",
                    "overrides": _BARE,
                    "alarm_mode_override": "threshold"},
    "+chan_drop":  {"label": "+ channel dropout (p=0.2)",         "group": "A",
                    "overrides": {**_BARE,
                                  "train.feat_channel_drop_p": 0.2},
                    "alarm_mode_override": "threshold"},
    "+hazard":     {"label": "+ hazard loss (w=0.3)",             "group": "A",
                    "overrides": {**_BARE,
                                  "train.feat_channel_drop_p": 0.2,
                                  "train.hazard_loss_weight":  0.3},
                    "alarm_mode_override": "threshold"},
    "+alarm_loss": {"label": "+ absolute margin (w=0.5)",         "group": "A",
                    "overrides": {**_BARE,
                                  "train.feat_channel_drop_p": 0.2,
                                  "train.hazard_loss_weight":  0.3,
                                  "train.alarm_loss_weight":   0.5},
                    "alarm_mode_override": "threshold"},
    "+focal":      {"label": "+ focal loss (γ=2)",                "group": "A",
                    "overrides": {**_BARE,
                                  "train.feat_channel_drop_p": 0.2,
                                  "train.hazard_loss_weight":  0.3,
                                  "train.alarm_loss_weight":   0.5,
                                  "train.focal_gamma":         2.0},
                    "alarm_mode_override": "threshold"},
    "+groupdro":   {"label": "+ GroupDRO",                        "group": "A",
                    "overrides": {**_BARE,
                                  "train.feat_channel_drop_p": 0.2,
                                  "train.hazard_loss_weight":  0.3,
                                  "train.alarm_loss_weight":   0.5,
                                  "train.focal_gamma":         2.0,
                                  "train.groupdro":            True},
                    "alarm_mode_override": "threshold"},
    "+margin":     {"label": "+ pairwise margin (w=0.5)",         "group": "A",
                    "overrides": {**_BARE,
                                  "train.feat_channel_drop_p": 0.2,
                                  "train.hazard_loss_weight":  0.3,
                                  "train.alarm_loss_weight":   0.5,
                                  "train.focal_gamma":         2.0,
                                  "train.groupdro":            True,
                                  "train.alarm_margin_weight": 0.5},
                    "alarm_mode_override": "threshold"},
    # `default` and the backbone rows store terminal-window metrics under the non-skip decoder, the paper's default decoder (hmm4_early_skip_ssem) is applied during rolling replay, so this choice affects no reported number
    "default":     {"label": "Full system (+ HMM4-early-SSEM)",         "group": "A",
                    "overrides": {},
                    "alarm_mode_override": "hmm4_early_ssem"},

    # B: Backbone comparison (full training stack, only backbone swapped)
    "gru":         {"label": "Backbone → GRU",         "group": "B",
                    "overrides": {"model.backbone": "gru"},
                    "alarm_mode_override": "hmm4_early_ssem"},
    "lstm":        {"label": "Backbone → LSTM",        "group": "B",
                    "overrides": {"model.backbone": "lstm"},
                    "alarm_mode_override": "hmm4_early_ssem"},
    "tcn":         {"label": "Backbone → TCN",         "group": "B",
                    "overrides": {"model.backbone": "tcn"},
                    "alarm_mode_override": "hmm4_early_ssem"},
    "transformer": {"label": "Backbone → Transformer", "group": "B",
                    "overrides": {"model.backbone": "transformer"},
                    "alarm_mode_override": "hmm4_early_ssem"},

    # C: Model floors (no aux losses, threshold decoder)
    "gru_pure":    {"label": "Pure GRU  (BCE + threshold only)", "group": "C",
                    "overrides": {**_BARE, "model.backbone": "gru"},
                    "alarm_mode_override": "threshold"},
    "lstm_pure":   {"label": "Pure LSTM  (BCE + threshold only)", "group": "C",
                    "overrides": {**_BARE, "model.backbone": "lstm"},
                    "alarm_mode_override": "threshold"},
}


# Helpers

def _apply_overrides(cfg: Config, overrides: dict) -> None:
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for i, p in enumerate(parts):
            if not hasattr(obj, p):
                raise KeyError(f"unknown config override key {key!r} (no field {p!r})")
            if i < len(parts) - 1:
                obj = getattr(obj, p)
        setattr(obj, parts[-1], val)


def _make_config(seed: int, overrides: dict, alarm_mode_override: str | None,
                 smoke: bool, device: str) -> Config:
    cfg = Config()
    cfg.train.seed = seed
    cfg.device = device
    _apply_overrides(cfg, overrides)
    if alarm_mode_override is not None:
        cfg.alarm.mode = alarm_mode_override
    if smoke:
        cfg.train.epochs = 5
        cfg.eval.n_folds = 2
    return cfg


def _avg_curve(per_fold: list) -> tuple[np.ndarray, np.ndarray]:
    n_pts = len(per_fold[0]["operating_curve"]["thresholds"])
    avg_det = np.zeros(n_pts)
    avg_fa = np.zeros(n_pts)
    for pf in per_fold:
        oc = pf["operating_curve"]
        avg_det += np.array(oc["detection_rates"])
        avg_fa += np.array(oc["fa_rates"])
    avg_det /= len(per_fold)
    avg_fa /= len(per_fold)
    return avg_det, avg_fa


def _curve_metrics(per_fold: list) -> dict:
    avg_det, avg_fa = _avg_curve(per_fold)
    idx_80det = int(np.argmin(np.abs(avg_det - 0.80)))
    has_25fa = np.any(avg_fa >= 0.25)
    idx_25fa = int(np.argmin(np.abs(avg_fa - 0.25))) if has_25fa else -1
    return {
        "fa_at_80det": float(avg_fa[idx_80det]),
        "det_at_80det": float(avg_det[idx_80det]),
        "det_at_25fa": float(avg_det[idx_25fa]) if idx_25fa >= 0 else float("nan"),
        "fa_at_25fa": float(avg_fa[idx_25fa]) if idx_25fa >= 0 else float("nan"),
    }


# Trained condition: one seed
# cfg.alarm.* is read only by calibrate_threshold / alarm_records / compute_operating_curves (evaluate.py), never by train_one or stream_clips, so the expensive part (_train_and_stream_one_seed) is independent of the decoder, and _decode_and_summarize applies the condition's decoder afterwards

def _train_and_stream_one_seed(
    seed: int,
    sequences: dict,
    seizures: list,
    overrides: dict,
    smoke: bool,
    device: str,
    checkpoint_dir: Path,
    training_source_provenance: dict,
) -> tuple[Config, list[dict]]:
    """Returns (base_cfg, per_fold_streams), base_cfg.alarm.mode is not final, _decode_and_summarize sets it per call
    """
    cfg = _make_config(seed, overrides, None, smoke, device)

    seizures_by_ssid = {s.ssid: s for s in seizures}
    ssids = [s.ssid for s in seizures]
    folds = patient_grouped(seizures, cfg.eval.n_folds, cfg.train.seed)

    per_fold_streams: list = []

    for fi, (tr_idx, va_idx) in enumerate(folds):
        train_ssids = [ssids[i] for i in tr_idx]
        val_ssids   = [ssids[i] for i in va_idx]
        trainp, calib = patient_disjoint_calib_split(
            train_ssids, seizures_by_ssid,
            cfg.train.calib_frac, cfg.train.seed + fi,
        )
        train_inst = expand_view_instances(trainp,  seizures_by_ssid)
        calib_inst = expand_view_instances(calib,   seizures_by_ssid)
        val_inst   = expand_view_instances(val_ssids, seizures_by_ssid)

        model = train_one(
            cfg, train_inst, sequences,
            seed=seed + fi * 100, device=cfg.device,
        )

        seed_dir = checkpoint_dir / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_dir / f"fold_{fi:02d}.pt"
        pending_path = checkpoint_path.with_suffix(".pt.pending")
        torch.save({
            "state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "condition_seed": seed,
            "training_seed": seed + fi * 100,
            "fold": fi,
            "train_ssids": trainp,
            "calib_ssids": calib,
            "val_ssids": val_ssids,
            "training_source_provenance": training_source_provenance,
        }, pending_path)
        pending_path.replace(checkpoint_path)

        cs = stream_clips(model, cfg, calib_inst, sequences, cfg.device)
        vs = stream_clips(model, cfg, val_inst,   sequences, cfg.device)

        per_fold_streams.append({"cs": cs, "vs": vs})

    return cfg, per_fold_streams


def _decode_and_summarize(
    cfg: Config, alarm_mode_override: str, per_fold_streams: list[dict],
) -> dict:
    """Decoder calibration and alarm decisions only, no training or forward passes"""
    cfg = copy.deepcopy(cfg)
    cfg.alarm.mode = alarm_mode_override

    pooled_records: list = []
    per_fold: list = []

    for fold_idx, stream in enumerate(per_fold_streams):
        cs, vs = stream["cs"], stream["vs"]
        params = calibrate_threshold(cs, cfg)
        recs   = alarm_records(vs, params, cfg)
        for record in recs:
            record["fold"] = fold_idx
        s_fold = summarize(recs)
        curve  = compute_operating_curves(vs, cfg)

        pooled_records.extend(recs)
        per_fold.append({
            "fold":           fold_idx,
            "params":         params,
            "detection_rate": s_fold["tcs_detection_rate"],
            "fa_rate":        s_fold["fos_false_alarm_rate"],
            "median_lead":    s_fold["median_lead_sec"],
            "operating_curve": curve,
        })

    alarm_sum = summarize(pooled_records)
    return {
        "alarm": alarm_sum,
        "per_fold": per_fold,
        **_curve_metrics(per_fold),
    }


# Helper functions

def _ci95(vals: list[float]) -> tuple[float, float]:
    v = np.array(vals, dtype=float)
    n = len(v)
    if n < 2:
        return (float("nan"), float("nan"))
    se = np.std(v, ddof=1) / np.sqrt(n)
    t  = sp_stats.t.ppf(0.975, df=n - 1)
    m  = v.mean()
    return float(m - t * se), float(m + t * se)


def _stats(vals: list[float]) -> dict:
    v = np.array([x for x in vals if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    m  = float(v.mean())
    s  = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    lo, hi = _ci95(v.tolist())
    return {"mean": m, "std": s, "ci_lo": lo, "ci_hi": hi,
            "min": float(v.min()), "max": float(v.max()), "n": len(v)}


def _summarize_seeds(seed_results: list[dict]) -> dict:
    keys_alarm = [
        "tcs_detection_rate", "fos_false_alarm_rate", "fa_per_hour",
        "f1", "precision", "recall", "auc", "pr_auc", "bacc", "accuracy",
        "median_lead_sec", "pct_seizures_detected_before_onset",
    ]
    keys_top = ["fa_at_80det", "det_at_80det", "det_at_25fa", "fa_at_25fa"]
    out: dict = {}
    for k in keys_alarm:
        out[k] = _stats([r["alarm"][k] for r in seed_results])
    for k in keys_top:
        out[k] = _stats([r[k] for r in seed_results])
    return out


# main

def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation study for streaming seizure predictor")
    ap.add_argument("--n-seeds",    type=int, required=True,
                    help="Number of seeds per condition; must equal the length "
                         "of --seed-list (the paper uses 15; --smoke forces 1)")
    ap.add_argument("--smoke",      action="store_true",
                    help="Smoke-test: 2 folds, 5 epochs, first seed only")
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="Condition keys to run (space-separated)")
    ap.add_argument("--output",     required=True,
                    help="Output JSON path")
    ap.add_argument("--seed-list",  type=int, nargs="+", required=True,
                    help="Condition seeds, in order (the README lists the "
                         "paper's 15).")
    ap.add_argument("--device",     required=True, choices=["cpu", "mps"],
                    help="Torch device. The paper used cpu, which is also ~22x "
                         "faster than mps at this model size")
    ap.add_argument("--save-checkpoints", required=True,
                    help="Directory for per-condition/per-seed/per-fold model "
                         "checkpoints, which run_rolling_evaluation.py replays.")
    args = ap.parse_args()
    device = args.device

    if len(args.seed_list) != args.n_seeds:
        raise SystemExit(
            f"[abort] --n-seeds {args.n_seeds} does not match the "
            f"{len(args.seed_list)} seeds given in --seed-list.")

    out_path, args.output = resolve_project_path(args.output, field="--output")
    checkpoint_root, args.save_checkpoints = resolve_project_path(
        args.save_checkpoints, field="--save-checkpoints")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
        ).decode().strip()
    except Exception:
        git_sha = "unknown"
    try:
        source_diff = subprocess.check_output(
            ["git", "diff", "--binary"],
            cwd=repo_root,
        )
        source_diff_sha256 = hashlib.sha256(source_diff).hexdigest()
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
        )
        git_dirty = bool(git_status)
    except Exception:
        source_diff_sha256 = "unknown"
        git_dirty = None

    if not cdata.CLEAN_CACHE.is_file():
        # On a first run the clean cache doesn't exist yet and the provenance stat() below would fail, so build it now (later load_view_sequences() calls just read it)
        print("[data] clean cache missing; building it now (one-time)…")
        cdata.load_view_sequences(verbose=False)

    cache_stat = cdata.CLEAN_CACHE.stat()
    csv_bytes = cdata.SEIZURES_CSV.read_bytes()
    _, clean_cache_relative = resolve_project_path(
        cdata.CLEAN_CACHE, field="clean feature cache")
    _, seizures_csv_relative = resolve_project_path(
        cdata.SEIZURES_CSV, field="seizures CSV")
    data_provenance = {
        "clean_cache": clean_cache_relative,
        "clean_cache_size": cache_stat.st_size,
        "clean_cache_mtime_ns": cache_stat.st_mtime_ns,
        "seizures_csv": seizures_csv_relative,
        "seizures_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    training_source_provenance = _training_source_provenance(repo_root)
    run_contract = {
        "smoke": args.smoke,
        "device": device,
        "checkpoint_dir": args.save_checkpoints,
    }

    # Load existing incremental results (for restart)
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        incremental: dict = saved["conditions"]
        print(f"[resume] Loaded {len(incremental)} existing condition results from {args.output}")
        # Provenance describes the run that started this file, don't overwrite it on resume
        git_sha = saved["git_sha"]
        git_dirty = saved["git_dirty"]
        source_diff_sha256 = saved["source_diff_sha256"]
    else:
        incremental = {}

    # Results from different code or data would mix incompatible seeds in the paired statistics, so refuse to resume even if the seed list matches
    if incremental:
        saved_source = saved.get("training_source_provenance")
        if saved_source != training_source_provenance:
            raise SystemExit(
                "[abort] The existing sweep was produced by different "
                "training/evaluation source. Use a fresh --output (or restore "
                "the exact recorded source) rather than mixing runs."
            )
        if saved.get("data_provenance") != data_provenance:
            raise SystemExit(
                "[abort] The existing sweep was produced from different data "
                "artifacts. Use a fresh --output rather than mixing runs."
            )
        if saved.get("run_contract") != run_contract:
            raise SystemExit(
                "[abort] The existing sweep was produced with different run "
                f"settings ({saved.get('run_contract')} != {run_contract}). "
                "Use a fresh --output rather than mixing runs."
            )

    # Select conditions
    unknown = [k for k in args.conditions if k not in ABLATION_CONDITIONS]
    if unknown:
        raise SystemExit(
            f"[abort] unknown condition keys: {unknown}. "
            f"Valid keys: {sorted(ABLATION_CONDITIONS)}")
    active_keys = list(args.conditions)

    seeds = args.seed_list[:1] if args.smoke else list(args.seed_list)
    # Refuse to append a foreign seed set
    # A subset or superset of the existing list (e.g. from an earlier, larger --n-seeds run) is still fine
    for _k, _v in incremental.items():
        _done = set(_v.get("completed_seeds", []))
        _this = set(seeds)
        if _done and not (_done.issubset(_this) or _this.issubset(_done)):
            raise SystemExit(
                f"[abort] '{_k}' already holds seeds {sorted(_done)}, which share "
                f"no subset relation with this run's seed list ({seeds}). "
                f"Appending would mix two seed sets and invalidate every paired "
                f"delta. Use --seed-list to match the existing sweep, or write to "
                f"a fresh --output.")
    # The file's seed list is the ordered union of every resumed run's seeds
    recorded_seeds = list(saved["seeds"]) if incremental else []
    recorded_seeds += [s for s in seeds if s not in recorded_seeds]

    # Load shared data once
    print(f"Device: {device}  |  {len(seeds)} seeds: {seeds}")
    print("Loading data …")
    sequences = cdata.load_view_sequences(verbose=False)
    seizures  = cdata.load_seizures(sequences, verbose=False)
    print(f"  {len(seizures)} clips ({sum(s.label for s in seizures)} onset-positive / "
          f"{sum(1 - s.label for s in seizures)} onset-negative)")

    # Run conditions
    total_start = time.time()

    for cond_key in active_keys:
        cond_meta = ABLATION_CONDITIONS[cond_key]
        label     = cond_meta["label"]
        overrides = cond_meta["overrides"]
        alarm_ovr = cond_meta["alarm_mode_override"]

        if cond_key not in incremental:
            incremental[cond_key] = {
                "label":  label,
                "group":  cond_meta["group"],
                "seeds_raw": [],
                "completed_seeds": [],
            }

        resolved = _make_config(0, overrides, alarm_ovr, args.smoke, device)
        resolved_dict = resolved.to_dict()
        resolved_dict["train"].pop("seed")
        saved_resolved = incremental[cond_key].setdefault("resolved_config", resolved_dict)
        if saved_resolved != resolved_dict:
            raise SystemExit(
                f"[abort] '{cond_key}' was produced with a different resolved "
                "config than this run would use. Use a fresh --output rather "
                "than mixing runs.")
        for raw_result in incremental[cond_key]["seeds_raw"]:
            if len(raw_result["per_fold"]) != resolved.eval.n_folds:
                raise SystemExit(
                    f"[abort] '{cond_key}' seed {raw_result['seed']} has "
                    f"{len(raw_result['per_fold'])}/{resolved.eval.n_folds} folds "
                    "recorded. Use a fresh --output.")
        done_seeds = set(incremental[cond_key]["completed_seeds"])

        print(f"\n{'='*70}")
        print(f"[{cond_key}] {label}")
        print(f"  overrides: {overrides or '(none — default config)'}")
        print(f"  alarm_mode_override: {alarm_ovr}")
        print(f"  seeds: {seeds}  (done: {sorted(done_seeds)})")
        print("  resolved config (data: "
              f"{data_provenance['clean_cache']}, {data_provenance['seizures_csv']}):")
        print(json.dumps(resolved_dict, indent=2, sort_keys=True, default=str))

        for seed in seeds:
            if seed in done_seeds:
                print(f"  seed {seed}: skipped (already done)")
                continue

            t0 = time.time()
            base_cfg, streams = _train_and_stream_one_seed(
                seed, sequences, seizures, overrides,
                smoke=args.smoke,
                device=device,
                checkpoint_dir=checkpoint_root / cond_key,
                training_source_provenance=training_source_provenance,
            )
            result = _decode_and_summarize(base_cfg, alarm_ovr, streams)

            elapsed = time.time() - t0
            a = result["alarm"]
            result["seed"] = seed
            print(f"  seed {seed}: det={a['tcs_detection_rate']:.3f} "
                  f"FAR={a['fos_false_alarm_rate']:.3f} "
                  f"f1={a['f1']:.3f}  auc={a['auc']:.3f}  "
                  f"({elapsed:.0f}s)")

            incremental[cond_key]["seeds_raw"].append(result)
            incremental[cond_key]["completed_seeds"].append(seed)

            # Incremental save after every (condition, seed)
            payload = {
                "git_sha":    git_sha,
                "git_dirty":  git_dirty,
                "source_diff_sha256": source_diff_sha256,
                "training_source_provenance": training_source_provenance,
                "data_provenance": data_provenance,
                "run_contract": run_contract,
                "seeds":      recorded_seeds,
                "conditions": incremental,
            }
            _write_json_atomic(out_path, payload)

        raw = incremental[cond_key]["seeds_raw"]
        if raw:
            incremental[cond_key]["summary"] = _summarize_seeds(raw)

    # Final save
    payload = {
        "git_sha":    git_sha,
        "git_dirty":  git_dirty,
        "source_diff_sha256": source_diff_sha256,
        "training_source_provenance": training_source_provenance,
        "data_provenance": data_provenance,
        "run_contract": run_contract,
        "seeds":      recorded_seeds,
        "conditions": incremental,
    }
    _write_json_atomic(out_path, payload)
    print(f"\nFull results saved → {args.output}")

    total_min = (time.time() - total_start) / 60
    print(f"\nTotal runtime: {total_min:.1f} min")


if __name__ == "__main__":
    main()
