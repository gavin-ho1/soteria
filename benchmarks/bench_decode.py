"""Requires ffmpeg with libx265 on PATH"""
import atexit
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2

N_FRAMES = 8
N_WARMUP = 5
N_REPS = 100

_tmpdir = tempfile.mkdtemp(prefix="soteria_bench_")
atexit.register(shutil.rmtree, _tmpdir, ignore_errors=True)
PATH = str(Path(_tmpdir) / "gestures_sim.mp4")
subprocess.run(
    ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=10",
     "-c:v", "libx265", "-pix_fmt", "yuv420p", PATH],
    check=True, capture_output=True,
)


def decode_snippet(start_frame: int):
    cap = cv2.VideoCapture(PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


for _ in range(N_WARMUP):
    decode_snippet(0)

times = []
for i in range(N_REPS):
    start = (i * 7) % 130
    t0 = time.perf_counter()
    frames = decode_snippet(start)
    times.append(time.perf_counter() - t0)

times.sort()
print("HEVC decode, 8-frame 320x180 snippet, cv2/ffmpeg backend, CPU:")
print(f"  mean:   {sum(times)/len(times)*1e3:.3f} ms")
print(f"  median: {times[len(times)//2]*1e3:.3f} ms")
print(f"  p95:    {times[int(len(times)*0.95)]*1e3:.3f} ms")
print(f"  frames decoded per call: {len(frames)}")
