from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data as cdata

PREONSET_EXCLUSION_SEC = cdata.PREONSET_EXCLUSION_SEC


def _preonset_mask(times: np.ndarray, label: int, onset: float) -> np.ndarray:
    """Mask of embeddings before the exclusion boundary

    A positive clip with a non-finite or non-positive onset gets an empty mask

    Callers must drop clips with fewer than two embeddings, not relax the boundary, which would leak later video
    """
    if label != 1:
        return np.ones(len(times), dtype=bool)
    if not np.isfinite(onset) or onset <= 0:
        return np.zeros(len(times), dtype=bool)
    return times < onset - PREONSET_EXCLUSION_SEC


@dataclass(frozen=True)
class ViewInstance:
    ssid: str
    view: str
    label: int
    onset_sec: float
    length_sec: float


def expand_view_instances(ssids, seizures_by_ssid) -> list[ViewInstance]:
    out = []
    for ssid in ssids:
        s = seizures_by_ssid[ssid]
        for v in s.views:
            out.append(ViewInstance(ssid, v, s.label,
                                    s.onset_sec if s.onset_sec is not None else -1.0,
                                    s.length_sec))
    return out


def select_eval_view_instances(instances) -> list[ViewInstance]:
    """One view per seizure: S, else L, regardless of input order"""
    by_ssid = {}
    for ins in instances:
        current = by_ssid.get(ins.ssid)
        if current is None or (ins.view == "S" and current.view != "S"):
            by_ssid[ins.ssid] = ins
    return list(by_ssid.values())


def _channel_dropout(f, drop_p):
    """Same channel mask at every time step"""
    if drop_p <= 0:
        return f
    mask = (np.random.random(f.shape[1]) >= drop_p).astype(np.float32)
    return f * mask[np.newaxis, :]


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def segment_sample_indices(M: int, N: int, train: bool, rng) -> np.ndarray:
    if M >= N:
        bounds = np.linspace(0, M, N + 1)
        idx = np.empty(N, dtype=int)
        for s in range(N):
            lo, hi = int(bounds[s]), max(int(bounds[s + 1]), int(bounds[s]) + 1)
            idx[s] = rng.integers(lo, hi) if train else (lo + hi - 1) // 2
        return idx
    pos = np.linspace(0, M - 1, N)
    if train:
        pos = pos + rng.uniform(-0.5, 0.5, size=N)
    return np.clip(np.round(pos), 0, M - 1).astype(int)


class PreonsetClipDataset(Dataset):
    def __init__(self, instances, sequences, cfg, train: bool, seed: int):
        self.cfg = cfg
        self.train = train
        self.N = cfg.model.num_tokens
        self.rng = np.random.default_rng(seed)

        self.seq_feat = {}
        self.seq_times = {}
        for seq_key, seq in sequences.items():
            feat, times = cdata._frames(seq)
            self.seq_feat[seq_key] = feat.astype(np.float32)
            self.seq_times[seq_key] = times.astype(np.float32)

        self.entries = []
        for ins in instances:
            seq_key = f"{ins.ssid}_{ins.view}"
            times = self.seq_times[seq_key]
            if len(times) < 2:
                continue
            if (_preonset_mask(times, ins.label, ins.onset_sec).sum()
                    < cdata.MIN_PREONSET_EMBEDDINGS):
                continue
            self.entries.append((seq_key, ins.label, ins.onset_sec))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        seq_key, label, onset = self.entries[i]
        feat = self.seq_feat[seq_key]
        times = self.seq_times[seq_key]

        if label == 1:
            pre = _preonset_mask(times, label, onset)
            feat = feat[pre]
            times = times[pre]

        # Cap to the last window_sec before sampling so context width doesn't depend on clip length (as in evaluation)
        if len(times) >= 2:
            t_end = times[-1]
            window_mask = times >= (t_end - self.cfg.model.window_sec)
            if window_mask.sum() >= 2:
                feat = feat[window_mask]
                times = times[window_mask]

        M = len(times)
        idx = segment_sample_indices(M, self.N, self.train, self.rng)
        f = _l2norm(feat[idx].astype(np.float32))
        t = times[idx].astype(np.float32)

        if self.train:
            f = _channel_dropout(f, self.cfg.train.feat_channel_drop_p)

        ramp_W0 = self.cfg.model.ramp_W0
        ramp_scale = self.cfg.model.ramp_scale
        ramp_floor = self.cfg.model.ramp_floor
        z = (t - (onset - ramp_W0)) / ramp_scale
        z = np.clip(z, -10.0, 10.0)
        w = ramp_floor + (1.0 - ramp_floor) * 1.0 / (1.0 + np.exp(-z))
        weight = np.where(label == 1, w, np.ones(self.N, dtype=np.float32))

        return {
            "feat": torch.from_numpy(f),
            "times": torch.from_numpy(t),
            "label": float(label),
            "onset": float(onset),
            "weight": torch.from_numpy(weight),
            "seq_key": seq_key,
        }


def collate(batch):
    feat = torch.stack([b["feat"] for b in batch])
    times = torch.stack([b["times"] for b in batch])
    weight = torch.stack([b["weight"] for b in batch])
    return {
        "feat": feat,
        "times": times,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "onset": torch.tensor([b["onset"] for b in batch], dtype=torch.float32),
        "weight": weight,
        "seq_key": [b["seq_key"] for b in batch],
    }


def make_loader(instances, sequences, cfg, train: bool, seed: int):
    ds = PreonsetClipDataset(instances, sequences, cfg, train, seed=seed)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=train,
                      num_workers=0, collate_fn=collate, drop_last=False)
