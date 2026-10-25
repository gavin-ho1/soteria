import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from streaming_predictor.evaluate import _hmm_n_forward
from streaming_predictor.model import StreamLRU

torch.manual_seed(0)
torch.set_num_threads(1)  # single core

INPUT_DIM = 512
D_MODEL = 64
N_TOKENS = 24
N_WARMUP = 20
N_REPS = 500

model = StreamLRU(input_dim=INPUT_DIM, d_model=D_MODEL, dropout=0.0)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Backbone params: {n_params:,}")

x_full = torch.randn(1, N_TOKENS, INPUT_DIM)

# Full 24-token window, one shot (what the streaming predictor does per alarm check)
with torch.no_grad():
    for _ in range(N_WARMUP):
        model(x_full)
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        model(x_full)
        times.append(time.perf_counter() - t0)

times.sort()
def pct(p):
    return times[int(len(times) * p)] * 1e3

print("\nFull 24-token window forward pass (batch=1, CPU, 1 thread):")
print(f"  mean:   {sum(times)/len(times)*1e3:.3f} ms")
print(f"  median: {pct(0.50):.3f} ms")
print(f"  p95:    {pct(0.95):.3f} ms")
print(f"  p99:    {pct(0.99):.3f} ms")

# Single token (T=1): cost of one recurrence step
# Streaming code recomputes the full window each call so this is only the per-token share of that
x_one = torch.randn(1, 1, INPUT_DIM)
with torch.no_grad():
    for _ in range(N_WARMUP):
        model(x_one)
    times1 = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        model(x_one)
        times1.append(time.perf_counter() - t0)
times1.sort()
def pct1(p):
    return times1[int(len(times1) * p)] * 1e3
print("\nSingle-token forward pass (T=1):")
print(f"  mean:   {sum(times1)/len(times1)*1e3:.3f} ms")
print(f"  median: {pct1(0.50):.3f} ms")

# HMM decoder (hmm4_early_skip_ssem): one forward-filter update per decision
# Parameters are representative calibrated values, the cost doesn't depend on them
HMM_PARAMS = dict(stay_probs=[0.95, 0.80, 0.80, 1.0], mus=[0.10, 0.30, 0.50, 0.75],
                  sigmas=[0.10, 0.10, 0.10, 0.10], fire_states=(2, 3), skip=True)
N_STEPS = 600  # a 10-minute recording at one decision per second
risk_stream = np.random.default_rng(0).uniform(0.0, 1.0, N_STEPS)
for _ in range(N_WARMUP):
    _hmm_n_forward(risk_stream, **HMM_PARAMS)
times_hmm = []
for _ in range(N_REPS // 10):
    t0 = time.perf_counter()
    _hmm_n_forward(risk_stream, **HMM_PARAMS)
    times_hmm.append((time.perf_counter() - t0) / N_STEPS)
times_hmm.sort()
hmm_median = times_hmm[len(times_hmm) // 2] * 1e3
print(f"\nHMM decoder update, 4-state skip SSEM (per decision, over a {N_STEPS}-step stream):")
print(f"  mean:   {sum(times_hmm)/len(times_hmm)*1e3:.4f} ms")
print(f"  median: {hmm_median:.4f} ms")

total = pct(0.50) + hmm_median
print(f"\nFor reference, SOTERIA makes one decision per second, so a "
      f"{total:.3f} ms median LRU + HMM decision is a {total/1000*100:.4f}% duty cycle.")
