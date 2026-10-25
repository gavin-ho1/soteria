"""Outcome-independent rolling replay for frozen SOTERIA checkpoints

Builds past-only windows at explicit wall-clock decision times, emits one risk per window, and passes that risk sequence to the alarm decoder
This module is separate from the terminal-window evaluator, so terminal metrics stay reproducible and are reported alongside the rolling ones, not overwritten
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from common import data as cdata
from streaming_predictor.config import Config
from streaming_predictor.data import (
    _l2norm,
    _preonset_mask,
    segment_sample_indices,
    select_eval_view_instances,
)


@dataclass(frozen=True)
class RollingWindow:
    """One decision time and its absolute source-array indices"""

    decision_time: float
    sample_indices: np.ndarray


def _unique_available_positions(times: np.ndarray, targets: np.ndarray) -> list[int]:
    """Newest observation available at each target time"""
    positions: list[int] = []
    for target in targets:
        pos = int(np.searchsorted(times, target, side="right") - 1)
        if pos >= 1 and (not positions or pos != positions[-1]):
            positions.append(pos)
    return positions


def build_rolling_windows(
    times: np.ndarray,
    *,
    end_time: float,
    window_sec: float,
    cadence_sec: float,
    num_tokens: int,
) -> list[RollingWindow]:
    """Deterministic, past-only windows through end_time

    Warm-up starts once two embeddings exist
    Short histories are resampled with repetition to num_tokens, as in the terminal evaluator
    The newest embedding is always the last token, so each risk belongs to its decision_time
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if len(times) < 2 or not np.all(np.diff(times) >= 0):
        return []
    if cadence_sec <= 0 or window_sec <= 0 or num_tokens < 2:
        raise ValueError("cadence_sec/window_sec must be positive and num_tokens >= 2")

    eligible = np.flatnonzero(times <= float(end_time) + 1e-9)
    if len(eligible) < 2:
        return []
    last_pos = int(eligible[-1])
    usable_times = times[: last_pos + 1]
    start_time = float(usable_times[1])
    final_time = float(usable_times[-1])

    targets = np.arange(start_time, final_time + 1e-9, cadence_sec, dtype=np.float64)
    positions = _unique_available_positions(usable_times, targets)
    # Always include the last observation, even off-cadence
    if not positions or positions[-1] != last_pos:
        positions.append(last_pos)

    rng = np.random.default_rng(0)
    windows: list[RollingWindow] = []
    for pos in positions:
        decision_time = float(times[pos])
        context_positions = np.flatnonzero(
            (times[: pos + 1] >= decision_time - window_sec - 1e-9)
            & (times[: pos + 1] <= decision_time + 1e-9)
        )
        if len(context_positions) < 2:
            continue
        local = segment_sample_indices(len(context_positions), num_tokens, False, rng)
        local[-1] = len(context_positions) - 1
        absolute = context_positions[local].astype(np.int64, copy=False)
        windows.append(RollingWindow(decision_time, absolute))
    return windows


def config_from_dict(raw: dict) -> Config:
    """Rebuild a checkpoint Config

    Missing fields raise instead of taking today's dataclass defaults, and unknown fields raise instead of being dropped, either means the checkpoint came from a different config schema and needs its original source revision
    """
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint config must be a mapping, got {type(raw).__name__}")
    expected_root = {"model", "train", "alarm", "eval", "device"}
    missing_root = sorted(expected_root - raw.keys())
    unknown_root = sorted(raw.keys() - expected_root)
    if missing_root or unknown_root:
        raise KeyError(
            "checkpoint config root schema mismatch: "
            f"missing={missing_root}, unknown={unknown_root}"
        )

    cfg = Config()
    for section_name in ("model", "train", "alarm", "eval"):
        saved_section = raw[section_name]
        if not isinstance(saved_section, dict):
            raise TypeError(
                f"checkpoint config section {section_name!r} must be a mapping, "
                f"got {type(saved_section).__name__}"
            )
        section = getattr(cfg, section_name)
        expected = {f.name for f in fields(section)}
        missing = sorted(expected - saved_section.keys())
        unknown = sorted(saved_section.keys() - expected)
        if missing or unknown:
            raise KeyError(
                f"checkpoint config section {section_name!r} schema mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        for key, value in saved_section.items():
            setattr(section, key, value)
    cfg.device = raw["device"]
    return cfg


def replay_end_time(times: np.ndarray, *, label: int, onset: float | None) -> float | None:
    """Positives stop at the last embedding strictly before onset - 0.5 s (the training mask), negatives run to the end, None if fewer than two qualify
    """
    if label != 1:
        return float(times[-1])
    if onset is None or not np.isfinite(onset):
        return None
    pre = _preonset_mask(times, label, float(onset))
    if pre.sum() < cdata.MIN_PREONSET_EMBEDDINGS:
        return None
    return float(times[np.flatnonzero(pre)[-1]])


@torch.no_grad()
def rolling_risk_stream(
    model,
    cfg: Config,
    feat: np.ndarray,
    times: np.ndarray,
    *,
    label: int,
    onset: float | None,
    device: str,
    cadence_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay one recording and return one final-token risk per decision time"""
    feat = np.asarray(feat, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    if len(feat) != len(times):
        raise ValueError("feat and times lengths differ")

    end_time = replay_end_time(times, label=label, onset=onset)
    if end_time is None:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    windows = build_rolling_windows(
        times,
        end_time=end_time,
        window_sec=float(cfg.model.window_sec),
        cadence_sec=float(cadence_sec),
        num_tokens=int(cfg.model.num_tokens),
    )
    risks: list[float] = []
    risk_times: list[float] = []
    model.eval()
    for window in windows:
        f_in = _l2norm(feat[window.sample_indices]).astype(np.float32, copy=False)
        logits = model(torch.from_numpy(f_in).unsqueeze(0).to(device))
        risks.append(float(torch.sigmoid(logits[0, -1]).cpu()))
        risk_times.append(window.decision_time)
    return np.asarray(risks, dtype=np.float32), np.asarray(risk_times, dtype=np.float32)


@torch.no_grad()
def rolling_streams_for_instances(
    model,
    cfg: Config,
    instances: Iterable,
    sequences: dict,
    *,
    device: str,
    cadence_sec: float,
) -> dict:
    """Replay the evaluation view (S, else L) of every recording"""
    instances = select_eval_view_instances(list(instances))
    out: dict = {}
    for ins in instances:
        key = f"{ins.ssid}_{ins.view}"
        feat, times = cdata._frames(sequences[key])
        subject = ins.ssid.split("_")[0]
        risks, risk_times = rolling_risk_stream(
            model,
            cfg,
            feat,
            times,
            label=ins.label,
            onset=ins.onset_sec,
            device=device,
            cadence_sec=cadence_sec,
        )
        if len(risks) == 0:
            continue
        out[ins.ssid] = {
            "risk": risks,
            "times": risk_times,
            "label": ins.label,
            "onset": ins.onset_sec,
            "view": ins.view,
            "subject": subject,
            "length_sec": ins.length_sec,
        }
    return out


def rolling_summary(records: list[dict], streams: dict, *, cadence_sec: float) -> dict:
    """Summarize raw rolling scores and first-alarm behavior"""
    labels = np.asarray([s["label"] for s in streams.values()], dtype=int)
    peaks = np.asarray([float(np.max(s["risk"])) for s in streams.values()], dtype=float)
    has_both = len(np.unique(labels)) == 2

    positive = [r for r in records if r["is_tcs"]]
    negative = [r for r in records if not r["is_tcs"]]
    positive_alarm = [r for r in positive if r["alarmed"]]
    false_alarm_recordings = sum(bool(r["alarmed"]) for r in negative)
    negative_exposure_sec = sum(
        len(s["risk"]) * float(cadence_sec)
        for s in streams.values() if s["label"] == 0
    )
    negative_hours = negative_exposure_sec / 3600.0
    leads = [float(r["lead_sec"]) for r in positive_alarm if r.get("lead_sec") is not None]

    return {
        "rolling_peak_auc": float(roc_auc_score(labels, peaks)) if has_both else float("nan"),
        "rolling_peak_pr_auc": float(average_precision_score(labels, peaks)) if has_both else float("nan"),
        "positive_recording_sensitivity": (
            len(positive_alarm) / len(positive) if positive else float("nan")
        ),
        "negative_recording_false_alarm_rate": (
            false_alarm_recordings / len(negative) if negative else float("nan")
        ),
        "median_first_alarm_offset_sec": float(np.median(leads)) if leads else float("nan"),
        "evaluated_negative_hours": float(negative_hours),
        "false_alarm_recordings_per_evaluated_hour": (
            false_alarm_recordings / negative_hours if negative_hours > 0 else float("nan")
        ),
        "false_alarm_rate_scope": "first_alarm_per_recording",
        "n_positive_recordings": len(positive),
        "n_negative_recordings": len(negative),
    }


__all__ = [
    "RollingWindow",
    "build_rolling_windows",
    "config_from_dict",
    "replay_end_time",
    "rolling_risk_stream",
    "rolling_streams_for_instances",
    "rolling_summary",
]
