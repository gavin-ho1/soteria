"""Fast DeLong test for comparing correlated ROC AUCs (Sun & Xu, 2014)

Used to compare two conditions scored on the same held-out recordings
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _compute_midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    n = len(x)
    sorted_x = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + 1 + j)
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """scores (k, n), labels (n,) with 1 = onset-positive

    Returns (aucs (k,), DeLong covariance (k, k))
    """
    assert scores.ndim == 2
    k, n = scores.shape
    pos_idx = labels == 1
    neg_idx = labels == 0
    m = int(pos_idx.sum())
    nn = int(neg_idx.sum())
    if m == 0 or nn == 0:
        raise ValueError("DeLong test requires both classes present")

    pos_scores = scores[:, pos_idx]
    neg_scores = scores[:, neg_idx]

    tx = np.empty((k, m))
    ty = np.empty((k, nn))
    tz = np.empty((k, m + nn))
    for r in range(k):
        tx[r] = _compute_midranks(pos_scores[r])
        ty[r] = _compute_midranks(neg_scores[r])
        # The fast DeLong equations assume all positives come before all negatives
        # Labels can arrive in any order (records are sorted by seizure ID), so rank the regrouped scores, not the original vector
        grouped = np.concatenate([pos_scores[r], neg_scores[r]])
        tz[r] = _compute_midranks(grouped)

    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / nn

    v01 = (tz[:, :m] - tx) / nn     # (k, m)
    v10 = 1.0 - (tz[:, m:] - ty) / m  # (k, nn)

    sx = np.cov(v01)   # (k, k)
    sy = np.cov(v10)
    if k == 1:
        sx = sx.reshape(1, 1)
        sy = sy.reshape(1, 1)
    cov = sx / m + sy / nn
    return auc, cov


def delong_roc_test(labels, scores_a, scores_b) -> dict:
    """Paired test on the same examples in the same order

    Returns auc_a, auc_b, z, two-sided p, and var of the AUC difference
    """
    labels = np.asarray(labels).astype(int)
    scores = np.vstack([np.asarray(scores_a, dtype=float),
                         np.asarray(scores_b, dtype=float)])
    auc, cov = _fast_delong(scores, labels)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-12:
        z = 0.0 if np.isclose(auc[0], auc[1]) else float(np.sign(auc[0] - auc[1]) * np.inf)
    else:
        z = float((auc[0] - auc[1]) / np.sqrt(var))
    p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else 0.0
    return {"auc_a": float(auc[0]), "auc_b": float(auc[1]),
            "z": z, "p": p, "var": float(var), "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum())}
