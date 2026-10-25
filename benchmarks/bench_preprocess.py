import time
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# GESTURES raw acquisition: 1920x1080 @ 30 FPS (dual full-HD cameras)
RAW_H, RAW_W = 1080, 1920
RAW_FPS = 30
TARGET_FPS = 15
SNIPPET_FRAMES = 8
CROP_FRAC = 0.05
TARGET_SIZE = 112

RAW_FRAMES_NEEDED = SNIPPET_FRAMES * (RAW_FPS // TARGET_FPS)


def preprocess_snippet(raw_frames: torch.Tensor) -> torch.Tensor:
    """raw_frames: (T_raw, H, W, C) uint8"""
    # Resample 30 to 15 FPS by taking every other frame
    step = RAW_FPS // TARGET_FPS
    frames = raw_frames[::step][:SNIPPET_FRAMES]  # (8, H, W, C)

    # Crop 5% off each horizontal edge
    w = frames.shape[2]
    crop = int(w * CROP_FRAC)
    frames = frames[:, :, crop:w - crop, :]

    # Pad vertically to square
    h, w = frames.shape[1], frames.shape[2]
    if h < w:
        pad = (w - h) // 2
        frames = F.pad(frames.permute(0, 3, 1, 2).float(), (0, 0, pad, w - h - pad))
    else:
        frames = frames.permute(0, 3, 1, 2).float()

    # Resize to 112x112 (R(2+1)D-34 input)
    frames = F.interpolate(frames, size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)

    # Normalize (ig65m-pytorch mean/std) and rearrange to (C, T, H, W)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    frames = (frames / 255.0 - mean) / std
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    return frames


# Fake decoded frames, container decode/IO isn't timed here (see bench_decode.py), only resample/crop/pad/resize/normalize
raw_frames = (torch.rand(RAW_FRAMES_NEEDED, RAW_H, RAW_W, 3) * 255).to(torch.uint8)

N_WARMUP = 10
N_REPS = 200

for _ in range(N_WARMUP):
    preprocess_snippet(raw_frames)

times = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    out = preprocess_snippet(raw_frames)
    times.append(time.perf_counter() - t0)

times.sort()
print("Preprocessing (resample+crop+pad+resize+normalize), 8-frame 1080p->112x112 snippet, CPU:")
print(f"  mean:   {sum(times)/len(times)*1e3:.3f} ms")
print(f"  median: {times[len(times)//2]*1e3:.3f} ms")
print(f"  p95:    {times[int(len(times)*0.95)]*1e3:.3f} ms")
print(f"  output shape: {tuple(out.shape)}")
