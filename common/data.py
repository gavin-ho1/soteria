"""GESTURES feature loading and seizure table

Data format:
  * Features at data/dataset/features_fpc_8_fps_15/<subj>_<seiz>_<view>/...pth, one (512,) float32 tensor per snippet, the trailing number is the frame index
  * Snippets extracted at FPS = 15 via torchvision.datasets.video_utils.VideoClips with frames_per_clip=8, frames_between_clips=1 (see fepegar/gestures-miccai-2021, dataset.py)
    VideoClips' clip index i covers raw frames [i, i+7] inclusive, so the stored "frame index" is the snippet's start, not its end
    The embedding filed under index i is not causal in i, it depends on frames up to i+7, i.e. up to (frames_per_clip-1)/FPS = 7/15 ~= 0.467 s after its index
    time_sec = (frame_index + frames_per_clip - 1) / FPS is the causal, end-of-snippet timestamp, not frame_index / FPS (causal-contract boundary 1)
  * The first load reads every .pth file once and writes a cache of plain tensors (data/cache/clean_features_v1.pt) that later runs load directly
  * Labels (seizures.csv): onset-positive iff `OnsetClonic` is present, its value is the clonic-onset time in seconds
    The CSV's `GTCS` field is not used, it disagrees with `OnsetClonic` for seven retained recordings
    `Discard` marks unusable views:
        No -> both views usable | Small -> only L | Large -> only S | Yes -> skip seizure

Counts used by the paper after pre-onset eligibility filtering: 210 recordings, 76 onset-positive and 134 onset-negative
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Paths / constants
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data" / "dataset"
FEATURES_DIR = DATASET_DIR / "features_fpc_8_fps_15"
SEIZURES_CSV = DATASET_DIR / "seizures.csv"
CACHE_DIR = REPO_ROOT / "data" / "cache"
CLEAN_CACHE = CACHE_DIR / "clean_features_v1.pt"  # plain tensors, no raw .pth needed

FPS = 15.0
FRAMES_PER_CLIP = 8  # matches features_fpc_8_fps_15, each embedding's raw-frame span
VIEWS = ("L", "S")
PREONSET_EXCLUSION_SEC = 0.5
MIN_PREONSET_EMBEDDINGS = 2


def frame_index_to_time_sec(frame_index):
    """End-of-snippet time, always use this instead of frame_index / FPS

    Embedding i covers frames [i, i+FRAMES_PER_CLIP-1], so i / FPS would leak ~0.47 s of future video
    """
    return (frame_index + FRAMES_PER_CLIP - 1) / FPS


# Feature loading
def _build_clean_cache_from_pth() -> dict:
    """Read all 533k raw .pth files (slow, one-time) and write CLEAN_CACHE"""
    clean: dict[str, dict] = {}
    view_dirs = sorted(p for p in FEATURES_DIR.iterdir() if p.is_dir())
    for d in view_dirs:
        parts = d.name.split("_")
        if len(parts) != 3:
            continue
        subj, seiz, view = parts
        files = sorted(d.glob("*.pth"))
        idxs, feats = [], []
        for f in files:
            idxs.append(int(f.stem.split("_")[-1]))
            feats.append(torch.load(f, map_location="cpu").to(torch.float32))
        order = np.argsort(idxs)
        feat = torch.stack(feats)[order].contiguous()
        fi = torch.tensor([idxs[i] for i in order], dtype=torch.int64)
        clean[d.name] = {"features": feat, "frame_indices": fi,
                         "subject": subj, "seizure": seiz, "view": view}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(clean, CLEAN_CACHE)
    return clean


def load_view_sequences(verbose: bool = True) -> dict[str, dict]:
    """Return {ssid_view: {features (M,512) float32, frame_indices (M,) int64, subject, seizure, view}}

    Builds the clean cache from the raw .pth files on first use
    """
    if CLEAN_CACHE.is_file():
        if verbose:
            print(f"[data] loading clean cache {CLEAN_CACHE.name}")
        return torch.load(CLEAN_CACHE, map_location="cpu", weights_only=False)
    if verbose:
        print("[data] no cache found; building from raw .pth files (slow, one-time)")
    return _build_clean_cache_from_pth()


# Seizure table
@dataclass(frozen=True)
class Seizure:
    ssid: str
    subject: str
    label: int
    onset_sec: float | None
    views: tuple[str, ...]
    length_sec: float


_USABLE_VIEWS = {"No": {"L", "S"}, "Large": {"S"}, "Small": {"L"}, "Yes": set()}


def _usable_views(discard: str) -> set[str]:
    d = (discard or "").strip()
    if d not in _USABLE_VIEWS:
        raise ValueError(f"unknown Discard value {discard!r} in seizures.csv; "
                         f"expected one of {sorted(_USABLE_VIEWS)}")
    return set(_USABLE_VIEWS[d])


def is_preonset_evaluable(seizure: Seizure, sequences: dict[str, dict]) -> bool:
    if seizure.label == 0:
        return True
    if seizure.onset_sec is None or seizure.onset_sec <= 0:
        return False
    cutoff = seizure.onset_sec - PREONSET_EXCLUSION_SEC
    # Use evaluation's view choice (S, else L) here too, so a preferred view that evaluation would drop can't shape folds or class weights
    view = "S" if "S" in seizure.views else "L"
    n_valid = int((frame_index_to_time_sec(
        sequences[f"{seizure.ssid}_{view}"]["frame_indices"].numpy()
    ) < cutoff).sum())
    return n_valid >= MIN_PREONSET_EMBEDDINGS


def load_seizures(sequences: dict[str, dict], verbose: bool = True) -> list[Seizure]:
    out: list[Seizure] = []
    with open(SEIZURES_CSV) as fh:
        for row in csv.DictReader(fh):
            subj = row["Subject"].strip()
            seiz = row["Seizure"].strip()
            ssid = f"{subj}_{seiz}"
            onset_raw = row["OnsetClonic"].strip()
            label = 1 if onset_raw else 0
            onset_sec = float(onset_raw) if onset_raw else None
            allowed = _usable_views(row["Discard"])
            views = tuple(v for v in VIEWS
                          if v in allowed and f"{ssid}_{v}" in sequences)
            if not views:
                continue
            length_sec = max(
                frame_index_to_time_sec(float(sequences[f"{ssid}_{v}"]["frame_indices"][-1].item()))
                for v in views
            )
            seizure = Seizure(ssid, subj, label, onset_sec, views, length_sec)
            if not is_preonset_evaluable(seizure, sequences):
                continue
            out.append(seizure)
    if verbose:
        npos = sum(s.label for s in out)
        nsubj = len({s.subject for s in out})
        print(f"[data] {len(out)} pre-onset-evaluable seizures "
              f"({npos} onset-positive / {len(out)-npos} onset-negative) across {nsubj} subjects")
    return out


# Frame selection
def _frames(seq: dict) -> tuple[np.ndarray, np.ndarray]:
    """(features, times_sec), sorted by frame index"""
    fi = seq["frame_indices"].numpy()
    feat = seq["features"].numpy()
    order = np.argsort(fi, kind="stable")
    return feat[order], frame_index_to_time_sec(fi[order])
