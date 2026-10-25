from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from streaming_predictor.data import make_loader
from streaming_predictor.model import build_model


def patient_disjoint_calib_split(train_ssids, seizures_by_ssid, calib_frac, seed):
    by_subj = defaultdict(list)
    for ssid in train_ssids:
        by_subj[seizures_by_ssid[ssid].subject].append(ssid)
    subjects = sorted(by_subj)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_calib = max(1, int(round(calib_frac * len(subjects))))
    calib_subj = subjects[:n_calib]
    train_subj = subjects[n_calib:]
    calib = [s for su in calib_subj for s in by_subj[su]]
    trainp = [s for su in train_subj for s in by_subj[su]]
    assert not (set(calib_subj) & set(train_subj)), "calib leak"
    return trainp, calib


def _groupdro_loss(per_sample, subj_idx, q, eta):
    present = torch.unique(subj_idx)
    Lg = torch.stack([per_sample[subj_idx == g].mean() for g in present])
    with torch.no_grad():
        q[present] = q[present] * torch.exp(eta * Lg.detach())
        q /= q.sum()
        w = q[present] / q[present].sum()
    return (w * Lg).sum()


def _alarm_margin_loss(logits, labels, beta, margin):
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return logits.sum() * 0.0
    smax = torch.logsumexp(beta * logits, dim=1) / beta
    d = smax[pos][:, None] - smax[neg][None, :]
    return (torch.clamp(margin - d, min=0.0) ** 2).mean()


def _cvar_loss(logits, labels, q):
    fos = labels == 0
    if fos.sum() == 0:
        return logits.sum() * 0.0
    per = F.binary_cross_entropy_with_logits(
        logits[fos], torch.zeros_like(logits[fos]), reduction="none"
    ).flatten()
    k = max(1, int(q * per.numel()))
    return torch.topk(per, k).values.mean()


def _alarm_level_loss(logits, times, onset, labels, margin):
    """Clip-level margin loss

    Onset-positive: push the max pre-onset logit above margin (early alarm)
    Onset-negative: push all logits below it (no false alarms)
    """
    loss = 0.0
    n = len(labels)
    for i in range(n):
        if labels[i] > 0 and onset[i] > 0:
            pre = times[i] < onset[i]
            if pre.any():
                loss = loss + F.relu(margin - logits[i, pre].max())
        else:
            loss = loss + F.relu(logits[i].max() - margin)
    return loss / n


def _hazard_loss(logits, times, onset, labels, tau, k):
    """Risk should rise smoothly toward onset

    Onset-positive target: sigmoid(-k * (onset - t - tau))
    Onset-negative target: 0
    """
    loss = 0.0
    n = len(labels)
    for i in range(n):
        if labels[i] > 0 and onset[i] > 0:
            remaining = onset[i] - times[i]
            target = torch.sigmoid(-k * (remaining - tau))
            loss = loss + F.binary_cross_entropy_with_logits(
                logits[i], target, reduction="sum"
            )
        else:
            loss = loss + F.binary_cross_entropy_with_logits(
                logits[i], torch.zeros_like(logits[i]), reduction="sum"
            )
    return loss / n


def _total_loss(logits, labels, weights, times, onset, subj_idx, cfg, q):
    labels_exp = labels.unsqueeze(-1).expand(-1, logits.size(1))
    loss_per = F.binary_cross_entropy_with_logits(
        logits, labels_exp, weight=weights, reduction="none"
    )

    if cfg.train.focal_gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * labels_exp + (1 - p) * (1 - labels_exp)
        loss_per = loss_per * ((1 - p_t) ** cfg.train.focal_gamma)

    if cfg.train.groupdro and len(q) > 1:
        loss = _groupdro_loss(loss_per.mean(dim=1), subj_idx, q, cfg.train.groupdro_eta)
    else:
        loss = loss_per.mean()

    if cfg.train.alarm_margin_weight > 0:
        loss = loss + cfg.train.alarm_margin_weight * _alarm_margin_loss(
            logits, labels, cfg.train.alarm_margin_beta, cfg.train.alarm_margin_margin
        )

    if cfg.train.cvar_weight > 0:
        loss = loss + cfg.train.cvar_weight * _cvar_loss(logits, labels, cfg.train.cvar_q)

    if cfg.train.alarm_loss_weight > 0:
        loss = loss + cfg.train.alarm_loss_weight * _alarm_level_loss(
            logits, times, onset, labels, cfg.train.alarm_loss_margin
        )

    if cfg.train.hazard_loss_weight > 0:
        loss = loss + cfg.train.hazard_loss_weight * _hazard_loss(
            logits, times, onset, labels, cfg.train.hazard_tau, cfg.train.hazard_k
        )

    return loss


def train_one(cfg, train_instances, sequences, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(cfg)
    model.to(device)

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    loader = make_loader(train_instances, sequences, cfg, train=True, seed=seed)

    subjects = sorted({ins.ssid.split("_")[0] for ins in train_instances})
    subj_map = {s: i for i, s in enumerate(subjects)}
    q = torch.ones(len(subjects), device=device) / len(subjects)

    for _ in range(cfg.train.epochs):
        model.train()
        for batch in loader:
            feat = batch["feat"].to(device)
            labels = batch["label"].to(device)
            weights = batch["weight"].to(device)

            ssids = [k.rsplit("_", 1)[0] for k in batch["seq_key"]]
            subj_idx = torch.tensor(
                [subj_map[s.split("_")[0]] for s in ssids],
                device=device,
            )

            times = batch["times"].to(device)
            onset = batch["onset"].to(device)

            logits = model(feat)
            loss = _total_loss(logits, labels, weights, times, onset, subj_idx, cfg, q)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            opt.step()

    return model
