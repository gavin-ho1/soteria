"""Terminal-window streaming evaluation and the alarm decoders

Each clip is window-sampled into N tokens, the backbone emits a risk at each token position, and the calibrated alarm decoder turns that risk stream into a first-alarm decision, rolling replay (rolling_evaluation.py) reuses the same decoders on its one-risk-per-update streams

Metrics:
  - Detection Rate: fraction of onset-positive recordings alarmed before onset
  - False Alarm Rate: fraction of onset-negative recordings with any false alarm
  - Median Lead Time: median (alarm_time - onset) for alarmed onset-positive recordings, negative means the alarm came before onset
  - F1 at alarm level: harmonic mean of alarm precision and detection rate
"""
from __future__ import annotations

import numpy as np
import torch

from common import data as cdata
from streaming_predictor.data import (_l2norm, segment_sample_indices,
                                      _preonset_mask, select_eval_view_instances)


def _tokenize_clip(feat, times, onset, N, rng, window_sec=None, *, label):
    if label == 1:
        pre = _preonset_mask(times, label, onset)
        if pre.sum() < cdata.MIN_PREONSET_EMBEDDINGS:
            return None, None
        feat = feat[pre]
        times = times[pre]

    # Cap lookback so context width doesn't depend on clip length, TSN then samples inside this window
    if window_sec is not None and len(times) >= 2:
        t_end = times[-1]
        window_mask = times >= (t_end - window_sec)
        if window_mask.sum() >= 2:
            feat = feat[window_mask]
            times = times[window_mask]

    if len(times) < 2:
        return None, None

    idx = segment_sample_indices(len(times), N, train=False, rng=rng)
    return feat[idx].astype(np.float32), times[idx].astype(np.float32)


@torch.no_grad()
def stream_clips(model, cfg, instances, sequences, device):
    model.eval()
    instances = select_eval_view_instances(instances)
    out = {}
    N = cfg.model.num_tokens
    rng = np.random.default_rng(0)
    window_sec = cfg.model.window_sec

    seq_cache = {}
    for ins in instances:
        key = f"{ins.ssid}_{ins.view}"
        if key not in seq_cache:
            feat, times = cdata._frames(sequences[key])
            seq_cache[key] = (feat.astype(np.float32), times.astype(np.float32))

    for ins in instances:
        feat_full, times_full = seq_cache[f"{ins.ssid}_{ins.view}"]
        onset = ins.onset_sec

        token_feat, token_times = _tokenize_clip(
            feat_full, times_full, onset, N, rng,
            window_sec=window_sec, label=ins.label,
        )
        if token_feat is None:
            continue

        f_t = torch.from_numpy(_l2norm(token_feat)).unsqueeze(0).to(device)
        logits_t = model(f_t)
        risks = torch.sigmoid(logits_t).cpu().numpy().flatten().astype(np.float32)

        out[ins.ssid] = {
            "risk": risks,
            "times": token_times,
            "label": ins.label,
            "onset": onset,
            "view": ins.view,
            "subject": ins.ssid.split("_")[0],
            "length_sec": ins.length_sec,
        }

    return out


def _first_crossing(statistic, times, h):
    """(alarmed, index, time) of the first statistic >= h"""
    for i, value in enumerate(statistic):
        if value >= h:
            return True, i, float(times[i])
    return False, None, None


HMM_N_MODES = (
    "hmm4_ssem", "hmm4_early_ssem",
    "hmm4_skip_ssem", "hmm4_early_skip_ssem",
    "hmm4_early_skip_regressive_ssem",
    "hmm4_early_full_ssem",
)
ALARM_MODES = ("threshold", "hmm", "hmm4_early", *HMM_N_MODES)


def decide(risk_stream, times, mode, **kwargs):
    if mode == "threshold":
        return _first_crossing(risk_stream, times, kwargs["h"])
    elif mode == "hmm":
        return decide_hmm(risk_stream, times,
                          a00=kwargs["a00"], a11=kwargs["a11"],
                          mu0=kwargs["mu0"], sigma0=kwargs["sigma0"],
                          mu1=kwargs["mu1"], sigma1=kwargs["sigma1"],
                          h=kwargs["h"])
    elif mode == "hmm4_early":
        return decide_hmm4(risk_stream, times,
                           a0stay=kwargs["a0stay"], a1stay=kwargs["a1stay"],
                           a2stay=kwargs["a2stay"],
                           mu0=kwargs["mu0"], sigma0=kwargs["sigma0"],
                           mu1=kwargs["mu1"], sigma1=kwargs["sigma1"],
                           mu2=kwargs["mu2"], sigma2=kwargs["sigma2"],
                           mu3=kwargs["mu3"], sigma3=kwargs["sigma3"],
                           h=kwargs["h"],
                           fire_states=(2, 3))
    elif mode in HMM_N_MODES:
        return decide_hmm_n(risk_stream, times,
                            stay_probs=kwargs["stay_probs"],
                            mus=kwargs["mus"],
                            sigmas=kwargs["sigmas"],
                            h=kwargs["h"],
                            fire_states=tuple(kwargs["fire_states"]),
                            back_probs=kwargs.get("back_probs", None),
                            skip=kwargs.get("skip", False),
                            full=kwargs.get("full", False))
    raise ValueError(f"unknown alarm mode: {mode}")


def _score_records(recs):
    """Alarm-level F1 (the calibration objective)"""
    tcs = [r for r in recs if r["is_tcs"]]
    fos = [r for r in recs if not r["is_tcs"]]
    n_tcs = len(tcs)
    n_fos = len(fos)
    detection = sum(r["alarmed"] for r in tcs) / n_tcs if n_tcs else 0.0
    fa = sum(r["alarmed"] for r in fos) / n_fos if n_fos else 0.0
    prec = detection * n_tcs / (detection * n_tcs + fa * n_fos + 1e-8)
    rec = detection
    return 2 * prec * rec / (prec + rec + 1e-8)


def _calib_threshold(calib_streams, cfg):
    candidates = np.linspace(0.05, 0.99, 96).tolist()
    best_h, best_score = 0.5, -1e9
    for h in candidates:
        recs = alarm_records(calib_streams, {"h": h}, cfg, mode="threshold")
        score = _score_records(recs)
        if score > best_score:
            best_score = score
            best_h = h
    return {"h": best_h}


def compute_operating_curves(streams, cfg, n_thresholds=200):
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    detections = []
    fa_rates = []
    lead_times = []

    for h in thresholds:
        params = {"h": h}
        recs = alarm_records(streams, params, cfg, mode="threshold")
        tcs = [r for r in recs if r["is_tcs"]]
        fos = [r for r in recs if not r["is_tcs"]]
        n_tcs = len(tcs)
        n_fos = len(fos)
        detected = sum(r["alarmed"] for r in tcs) / n_tcs if n_tcs else 0.0
        fa = sum(r["alarmed"] for r in fos) / n_fos if n_fos else 0.0
        leads_list = [r["lead_sec"] for r in recs if r["alarmed"] and r["is_tcs"] and r["lead_sec"] is not None]
        median_lead = float(np.median(leads_list)) if leads_list else None
        detections.append(detected)
        fa_rates.append(fa)
        lead_times.append(median_lead)

    return {
        "thresholds": thresholds.tolist(),
        "detection_rates": detections,
        "fa_rates": fa_rates,
        "lead_times": lead_times,
    }


def _gauss_emit(x, mu, sigma):
    sigma = max(float(sigma), 1e-8)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _hmm_forward(risk_stream, a00, a11, mu0, sigma0, mu1, sigma1):
    """Causal 2-state forward filter (0 baseline, 1 seizure imminent)

    Returns P(state 1 | r[0..t]) for each t
    """
    A = np.array([[a00, 1.0 - a00],
                  [1.0 - a11, a11]])
    alpha = np.array([1.0, 0.0])
    posteriors = np.zeros(len(risk_stream))
    for t, r in enumerate(risk_stream):
        emit = np.array([_gauss_emit(r, mu0, sigma0),
                         _gauss_emit(r, mu1, sigma1)])
        alpha = (A.T @ alpha) * emit
        alpha /= (alpha.sum() + 1e-30)
        posteriors[t] = alpha[1]
    return posteriors


def decide_hmm(risk_stream, times, a00, a11, mu0, sigma0, mu1, sigma1, h):
    posteriors = _hmm_forward(risk_stream, a00, a11, mu0, sigma0, mu1, sigma1)
    return _first_crossing(posteriors, times, h)


# 4-state left-to-right HMM, fixed time-window emissions

def _hmm4_forward(risk_stream, a0stay, a1stay, a2stay,
                  mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3,
                  fire_states):
    """States: 0 baseline, 1 far pre-onset, 2 mid pre-onset, 3 near pre-onset (alarm), strictly left-to-right (0→1→2→3, no skips), 3 is absorbing

    fire_states: states whose posteriors sum to the alarm statistic (hmm4_early uses 2 and 3)
    """
    A = np.array([
        [a0stay,    1.0 - a0stay, 0.0,           0.0          ],
        [0.0,       a1stay,       1.0 - a1stay,  0.0          ],
        [0.0,       0.0,          a2stay,         1.0 - a2stay ],
        [0.0,       0.0,          0.0,            1.0          ],
    ])
    alpha = np.array([1.0, 0.0, 0.0, 0.0])
    post = np.zeros(len(risk_stream))
    for t, r in enumerate(risk_stream):
        emit = np.array([
            _gauss_emit(r, mu0, sigma0),
            _gauss_emit(r, mu1, sigma1),
            _gauss_emit(r, mu2, sigma2),
            _gauss_emit(r, mu3, sigma3),
        ])
        alpha = (A.T @ alpha) * emit
        alpha /= (alpha.sum() + 1e-30)
        post[t] = sum(alpha[s] for s in fire_states)
    return post


def decide_hmm4(risk_stream, times, a0stay, a1stay, a2stay,
                mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3,
                h, fire_states):
    posteriors = _hmm4_forward(risk_stream, a0stay, a1stay, a2stay,
                               mu0, sigma0, mu1, sigma1, mu2, sigma2, mu3, sigma3,
                               fire_states=fire_states)
    return _first_crossing(posteriors, times, h)


def fit_hmm4_params(calib_streams, near_window=3.0, mid_lo=3.0, mid_hi=10.0,
                    far_lo=10.0, far_hi=20.0):
    """State 0 is fit to onset-negative tokens, states 1, 2, 3 are fit to onset-positive tokens in the far, mid and near pre-onset windows given by the keyword args (seconds before onset)
    """
    fos_risks, far_risks, mid_risks, near_risks = [], [], [], []
    for s in calib_streams.values():
        if s["label"] == 0:
            fos_risks.extend(s["risk"].tolist())
        else:
            onset = s.get("onset", -1.0)
            if onset <= 0:
                continue
            risks, times_s = s["risk"], s["times"]
            near_mask = times_s >= (onset - near_window)
            mid_mask  = (times_s >= (onset - mid_hi)) & (times_s < (onset - mid_lo))
            far_mask  = (times_s >= (onset - far_hi)) & (times_s < (onset - far_lo))
            if near_mask.sum() > 0:
                near_risks.extend(risks[near_mask].tolist())
            if mid_mask.sum() > 0:
                mid_risks.extend(risks[mid_mask].tolist())
            if far_mask.sum() > 0:
                far_risks.extend(risks[far_mask].tolist())

    fos  = np.array(fos_risks)  if fos_risks  else np.array([0.10])
    far  = np.array(far_risks)  if far_risks  else np.array([0.30])
    mid  = np.array(mid_risks)  if mid_risks  else np.array([0.50])
    near = np.array(near_risks) if near_risks else np.array([0.75])
    return (
        float(np.mean(fos)),  float(max(0.05, np.std(fos))),
        float(np.mean(far)),  float(max(0.05, np.std(far))),
        float(np.mean(mid)),  float(max(0.05, np.std(mid))),
        float(np.mean(near)), float(max(0.05, np.std(near))),
    )


def _calib_hmm4(calib_streams, cfg):
    mu0, s0, mu1, s1, mu2, s2, mu3, s3 = fit_hmm4_params(calib_streams)
    if mu3 <= mu0:
        fallback = _calib_threshold(calib_streams, cfg)
        fallback["mode"] = "threshold"
        return fallback
    # Enforce strict ordering between the 4 means
    mu1 = float(np.clip(mu1, mu0 + 0.02, mu3 - 0.06))
    mu2 = float(np.clip(mu2, mu1 + 0.02, mu3 - 0.03))

    best_params, best_score = None, -1e9
    for a0stay in [0.90, 0.95, 0.99]:
        for a1stay in [0.70, 0.80, 0.90]:
            for a2stay in [0.70, 0.80, 0.90]:
                for h in [0.30, 0.50, 0.70, 0.90]:
                    params = dict(a0stay=a0stay, a1stay=a1stay, a2stay=a2stay,
                                  mu0=mu0, sigma0=s0, mu1=mu1, sigma1=s1,
                                  mu2=mu2, sigma2=s2, mu3=mu3, sigma3=s3, h=h)
                    recs = alarm_records(calib_streams, params, cfg)
                    score = _score_records(recs)
                    if score > best_score:
                        best_score = score
                        best_params = params
    return best_params



# Generic N-state left-to-right HMM with SSEM emissions
# State 0 is fit directly to onset-negative risks, since onset-negative recordings are known baseline
# States 1..N-1 are fitted by semi-supervised EM on onset-positive risk streams

def _hmm_n_forward(risk_stream, stay_probs, mus, sigmas, fire_states, back_probs=None,
                   skip=False, full=False):
    """stay_probs[k] = P(stay in k), the last state is absorbing (stay_probs[-1] ignored)
    back_probs[k] = P(k -> k-1), ignored for the first and last state, None means no backward moves
    fire_states: states whose posteriors sum to the alarm statistic
    skip: from state j, split forward mass equally over all higher states instead of only j+1
    full: from state j, split it equally over all other states (overrides skip and back_probs)
    """
    N = len(mus)
    if back_probs is None:
        back_probs = np.zeros(N)
    back_probs = np.asarray(back_probs, dtype=float)
    alpha = np.zeros(N)
    alpha[0] = 1.0
    post = np.zeros(len(risk_stream))
    for t, r in enumerate(risk_stream):
        emit = np.array([_gauss_emit(r, mus[k], sigmas[k]) for k in range(N)])
        new_alpha = np.zeros(N)
        if full:
            for k in range(N):
                new_alpha[k] = alpha[k] * stay_probs[k]
                for j in range(N):
                    if j == k:
                        continue
                    remaining = 1.0 - stay_probs[j]
                    if remaining > 0:
                        new_alpha[k] += alpha[j] * remaining / (N - 1)
        elif not skip:
            for k in range(N):
                new_alpha[k] = alpha[k] * stay_probs[k]
                if k > 0:
                    fwd_prob = 1.0 - stay_probs[k - 1] - back_probs[k - 1]
                    if fwd_prob > 0:
                        new_alpha[k] += alpha[k - 1] * fwd_prob
                if k < N - 1:
                    new_alpha[k] += alpha[k + 1] * back_probs[k + 1]
        else:
            for k in range(N):
                new_alpha[k] = alpha[k] * stay_probs[k]
                for j in range(k):
                    fwd_prob = 1.0 - stay_probs[j] - back_probs[j]
                    n_higher = N - j - 1
                    if fwd_prob > 0 and n_higher > 0:
                        new_alpha[k] += alpha[j] * fwd_prob / n_higher
                if k < N - 1:
                    new_alpha[k] += alpha[k + 1] * back_probs[k + 1]
        new_alpha *= emit
        s = new_alpha.sum()
        alpha = new_alpha / (s + 1e-30)
        post[t] = sum(alpha[k] for k in fire_states)
    return post


def decide_hmm_n(risk_stream, times, stay_probs, mus, sigmas,
                 h, fire_states, back_probs=None, skip=False, full=False):
    posteriors = _hmm_n_forward(risk_stream, stay_probs, mus, sigmas, fire_states,
                                back_probs=back_probs, skip=skip, full=full)
    return _first_crossing(posteriors, times, h)


def _tcs_transition_matrix(n_states, a_stay=0.85):
    """Transition matrix for fitting onset-positive emissions

    Left-to-right, the final state is absorbing, as in the deployed HMM
    """
    A = np.zeros((n_states, n_states), dtype=float)
    for k in range(n_states - 1):
        A[k, k] = a_stay
        A[k, k + 1] = 1.0 - a_stay
    A[-1, -1] = 1.0
    return A


def _ssem_tcs(tcs_calib_dicts, n_tcs_states, init_mus, init_sigmas,
              n_iter=30, a_stay=0.85, min_sigma=0.05, blend=0.3):
    """Semi-supervised EM: blend Baum-Welch posteriors with soft labels

    Soft labels are Gaussian kernels on time-to-onset, spaced evenly across states
    blend=0 is pure Baum-Welch, blend=1 is pure labels
    """
    N = n_tcs_states
    A = _tcs_transition_matrix(N, a_stay)
    mus = np.array(init_mus, dtype=float)
    sigmas = np.array(init_sigmas, dtype=float)
    t_centers = np.linspace(1.5, 15.0, N)[::-1]
    t_widths = np.linspace(1.5, 5.0, N)[::-1]

    for _ in range(n_iter):
        acc_w = np.zeros(N)
        acc_wr = np.zeros(N)
        acc_wr2 = np.zeros(N)

        for item in tcs_calib_dicts:
            risk_stream = item["risk"]
            times = item["times"]
            onset = float(item["onset"])
            T = len(risk_stream)
            if T == 0:
                continue

            # Forward-backward
            alpha = np.zeros((T, N))
            alpha[0, 0] = _gauss_emit(risk_stream[0], mus[0], sigmas[0])
            alpha[0] /= (alpha[0].sum() + 1e-30)
            for t in range(1, T):
                for k in range(N):
                    alpha[t, k] = alpha[t - 1, k] * A[k, k]
                    if k > 0:
                        alpha[t, k] += alpha[t - 1, k - 1] * A[k - 1, k]
                    alpha[t, k] *= _gauss_emit(risk_stream[t], mus[k], sigmas[k])
                alpha[t] /= (alpha[t].sum() + 1e-30)

            beta = np.ones((T, N))
            for t in range(T - 2, -1, -1):
                for k in range(N):
                    stay_c = A[k, k] * _gauss_emit(risk_stream[t+1], mus[k], sigmas[k]) * beta[t+1, k]
                    adv_c = 0.0
                    if k < N - 1:
                        adv_c = (A[k, k + 1]
                                 * _gauss_emit(risk_stream[t+1], mus[k+1], sigmas[k+1])
                                 * beta[t+1, k+1])
                    beta[t, k] = stay_c + adv_c
                beta[t] /= (beta[t].sum() + 1e-30)

            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-30)

            # Soft time-window labels
            dt = onset - np.array(times, dtype=float)
            label_w = np.zeros((T, N))
            for k in range(N):
                label_w[:, k] = np.exp(-0.5 * ((dt - t_centers[k]) / t_widths[k]) ** 2)
            row_s = label_w.sum(axis=1, keepdims=True)
            label_w /= (row_s + 1e-30)

            gamma = (1.0 - blend) * gamma + blend * label_w
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-30)

            for k in range(N):
                w = gamma[:, k]
                acc_w[k] += w.sum()
                acc_wr[k] += (w * risk_stream).sum()
                acc_wr2[k] += (w * risk_stream ** 2).sum()

        for k in range(N):
            if acc_w[k] > 1e-6:
                mu = acc_wr[k] / acc_w[k]
                var = max(0.0, acc_wr2[k] / acc_w[k] - mu ** 2)
                mus[k] = mu
                sigmas[k] = float(max(min_sigma, np.sqrt(var)))
        for k in range(1, N):
            if mus[k] <= mus[k - 1] + 0.01:
                mus[k] = mus[k - 1] + 0.02

    return mus, sigmas


def fit_hmm_n_params(calib_streams, n_states, ssem_blend, n_iter=30, min_sigma=0.05):
    """State 0 is fixed to the onset-negative risks (not EM-updated), states 1..N-1 are fitted to onset-positive risks by _ssem_tcs, returns (mus, sigmas)
    """
    fos_risks = [float(r)
                 for s in calib_streams.values() if s["label"] == 0
                 for r in s["risk"]]
    tcs_dicts = [s for s in calib_streams.values() if s["label"] == 1]

    fos = np.array(fos_risks) if fos_risks else np.array([0.10])
    mu0 = float(np.mean(fos))
    s0 = float(max(min_sigma, np.std(fos)))

    n_tcs = n_states - 1
    if not tcs_dicts:
        mus = np.linspace(mu0, 0.85, n_states)
        sigmas = np.full(n_states, 0.10)
        mus[0], sigmas[0] = mu0, s0
        return mus, sigmas

    all_tcs = np.concatenate([d["risk"] for d in tcs_dicts])
    mu_hi = float(np.percentile(all_tcs, 90))
    init_mus = np.linspace(mu0 + 0.10, mu_hi, n_tcs)
    init_sigmas = np.full(n_tcs, max(min_sigma, float(np.std(all_tcs) / max(1, n_tcs - 1))))

    em_mus, em_sigmas = _ssem_tcs(
        tcs_dicts, n_tcs, init_mus, init_sigmas,
        n_iter=n_iter, min_sigma=min_sigma, blend=ssem_blend)

    return np.concatenate([[mu0], em_mus]), np.concatenate([[s0], em_sigmas])


def _calib_hmm_n(calib_streams, cfg, n_states, fire_states, mode_name,
                 back_probs_grid=None):
    """back_probs_grid: list of back_probs arrays to try (see _hmm_n_forward), None means no backward transitions (plain left-to-right)
    mode_name may contain "skip" or "full" (see _hmm_n_forward)
    """
    skip = "skip" in mode_name
    full = "full" in mode_name
    mus, sigmas = fit_hmm_n_params(calib_streams, n_states, cfg.alarm.ssem_blend)

    if mus[-1] <= mus[0]:
        fallback = _calib_threshold(calib_streams, cfg)
        fallback["mode"] = "threshold"
        return fallback

    best_params, best_score = None, -1e9
    if back_probs_grid is None:
        back_probs_grid = [np.zeros(n_states)]
    for a0stay in [0.90, 0.95, 0.99]:
        for a_int in [0.70, 0.80, 0.90]:
            stay_probs = [a0stay] + [a_int] * (n_states - 2) + [1.0]
            for back_probs in back_probs_grid:
                back_probs = np.asarray(back_probs, dtype=float)
                if any(stay_probs[k] + back_probs[k] > 1.0 for k in range(n_states)):
                    continue
                for h in [0.30, 0.50, 0.70, 0.90]:
                    params = dict(
                        mode=mode_name,
                        stay_probs=stay_probs,
                        mus=mus.tolist(),
                        sigmas=sigmas.tolist(),
                        fire_states=list(fire_states),
                        h=float(h),
                    )
                    if skip:
                        params["skip"] = True
                    if full:
                        params["full"] = True
                    if back_probs.sum() > 0:
                        params["back_probs"] = back_probs.tolist()
                    recs = alarm_records(calib_streams, params, cfg)
                    score = _score_records(recs)
                    if score > best_score:
                        best_score = score
                        best_params = params

    return best_params if best_params is not None else _calib_threshold(calib_streams, cfg)


def fit_hmm_params(calib_streams, near_onset_window=3.0):
    """State 0 uses all onset-negative tokens, state 1 only onset-positive tokens within near_onset_window seconds of onset

    The causal model's earlier onset-positive tokens look like onset-negative ones, so averaging them in would collapse the gap between the states
    Returns (mu0, sigma0, mu1, sigma1)
    """
    fos_risks, tcs_near_risks, tcs_all_risks = [], [], []
    for s in calib_streams.values():
        if s["label"] == 0:
            fos_risks.extend(s["risk"].tolist())
        else:
            onset = s.get("onset", -1.0)
            risks = s["risk"]
            times = s["times"]
            tcs_all_risks.extend(risks.tolist())
            if onset > 0:
                near = times >= (onset - near_onset_window)
                if near.sum() > 0:
                    tcs_near_risks.extend(risks[near].tolist())

    fos = np.array(fos_risks) if fos_risks else np.array([0.2])
    # No near-onset tokens: fall back to all onset-positive tokens
    tcs = np.array(tcs_near_risks) if tcs_near_risks else (
        np.array(tcs_all_risks) if tcs_all_risks else np.array([0.8]))
    return (float(np.mean(fos)), max(0.05, float(np.std(fos))),
            float(np.mean(tcs)), max(0.05, float(np.std(tcs))))


def _calib_hmm(calib_streams, cfg):
    """Calibrate transition probabilities and the alarm threshold

    Falls back to threshold mode when mu0 ≈ mu1, which would alarm on every onset-negative recording
    """
    mu0, sigma0, mu1, sigma1 = fit_hmm_params(calib_streams)
    min_sep = max(sigma0, sigma1) * 0.5
    if mu1 - mu0 < min_sep:
        fallback = _calib_threshold(calib_streams, cfg)
        fallback["mode"] = "threshold"
        return fallback

    best_params, best_score = None, -1e9
    for a00 in [0.70, 0.80, 0.90, 0.95]:
        for a11 in [0.70, 0.80, 0.90, 0.95]:
            for h in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
                params = dict(a00=a00, a11=a11,
                              mu0=mu0, sigma0=sigma0,
                              mu1=mu1, sigma1=sigma1, h=h)
                recs = alarm_records(calib_streams, params, cfg)
                score = _score_records(recs)
                if score > best_score:
                    best_score = score
                    best_params = params
    return best_params


def calibrate_threshold(calib_streams, cfg):
    mode = cfg.alarm.mode
    if mode == "threshold":
        return _calib_threshold(calib_streams, cfg)
    elif mode == "hmm":
        return _calib_hmm(calib_streams, cfg)
    elif mode == "hmm4_early":
        return _calib_hmm4(calib_streams, cfg)
    elif mode in ("hmm4_ssem", "hmm4_early_ssem",
                  "hmm4_skip_ssem", "hmm4_early_skip_ssem",
                  "hmm4_early_full_ssem"):
        fire_states = (2, 3) if ("early" in mode) else (3,)
        return _calib_hmm_n(calib_streams, cfg, n_states=4, fire_states=fire_states, mode_name=mode)
    elif mode == "hmm4_early_skip_regressive_ssem":
        grid = [[0.0, bp, bp, 0.0] for bp in [0.0, 0.02, 0.05, 0.10]]
        return _calib_hmm_n(calib_streams, cfg, n_states=4, fire_states=(2, 3),
                            mode_name=mode, back_probs_grid=grid)
    raise ValueError(f"unknown alarm mode: {mode}")


def alarm_records(streams, params, cfg, mode=None):
    records = []
    # params["mode"] is set when a calibrator fell back to another mode (e.g. HMM -> threshold)
    mode = mode if mode is not None else params.get("mode", cfg.alarm.mode)
    p = {k: v for k, v in params.items() if k != "mode"}
    for ssid, s in streams.items():
        if len(s["risk"]) == 0:
            continue
        alarmed, idx, atime = decide(s["risk"], s["times"], mode, **p)
        onset = s["onset"] if s["label"] == 1 else None
        lead = (atime - onset) if (alarmed and onset is not None) else None
        records.append({
            "ssid": ssid,
            "view": s.get("view"),
            "subject": s.get("subject", ssid.split("_")[0]),
            "label": s["label"],
            "is_tcs": s["label"] == 1,
            "alarmed": alarmed,
            "alarm_time_sec": atime,
            "onset_sec": onset,
            "lead_sec": lead,
            "peak_risk": float(s["risk"].max()),
            "length_sec": float(s["length_sec"]),
        })
    return records


def summarize(records):
    tcs = [r for r in records if r["is_tcs"]]
    fos = [r for r in records if not r["is_tcs"]]
    tcs_al = [r for r in tcs if r["alarmed"]]
    leads = np.array([r["lead_sec"] for r in tcs_al if r["lead_sec"] is not None], dtype=float)

    detection_rate = (len(tcs_al) / len(tcs)) if tcs else float("nan")
    fa_rate = (sum(r["alarmed"] for r in fos) / len(fos)) if fos else float("nan")

    median_lead = float(np.median(leads)) if len(leads) > 0 else float("nan")

    n_tp = len(tcs_al)
    n_fp = sum(r["alarmed"] for r in fos)
    n_fn = len(tcs) - n_tp
    n_tn = len(fos) - n_fp
    prec = n_tp / (n_tp + n_fp + 1e-8)
    rec = detection_rate
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    accuracy = (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn + 1e-8)

    # Only one terminal window per onset-negative recording is evaluated
    # Dividing its alarms by the full recording length would claim exposure the model never saw, so there is no per-hour or per-day alarm rate
    total_fos_sec = sum(r["length_sec"] for r in fos)
    fa_per_hour = float("nan")
    fa_per_day = float("nan")

    labels = np.array([r["label"] for r in records], dtype=int)
    scores = np.array([r["peak_risk"] for r in records], dtype=float)
    has_both = len(np.unique(labels)) == 2
    if has_both:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc    = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
    else:
        auc = pr_auc = float("nan")

    # Balanced accuracy of the binary alarmed / not-alarmed decision
    if has_both:
        specificity = n_tn / (n_tn + n_fp + 1e-8)
        bacc = (detection_rate + specificity) / 2.0
    else:
        bacc = float("nan")

    return {
        "tcs_detection_rate": detection_rate,
        "fos_false_alarm_rate": fa_rate,
        "fa_per_hour": fa_per_hour,
        "fa_per_day": fa_per_day,
        "fa_rate_scope": "terminal_window_only",
        "median_lead_sec": median_lead,
        "auc": auc,
        "pr_auc": pr_auc,
        "bacc": bacc,
        "accuracy": accuracy,
        "pct_seizures_detected_before_onset": detection_rate * 100.0,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_tcs": len(tcs),
        "n_fos": len(fos),
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "n_tn": n_tn,
        "total_fos_hours": total_fos_sec / 3600.0,
    }
