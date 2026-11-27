"""Time each stage on real-size synthetic frames (2592 x 1944, Bayer, lossless like MControl).

    uv run python bench/bench.py                  # 4 x 3 frames, quick and quality
    uv run python bench/bench.py --nx 6 --ny 5 --threads 8

Prints ms per frame (unpack + prepare) and ms per pair (registration), and extrapolates to 30x30 and 58x58.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import mstitch as ms  # noqa: E402
from synth import scan  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nx", type=int, default=4)
ap.add_argument("--ny", type=int, default=3)
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()

t = time.perf_counter()
sc = scan(nx=a.nx, ny=a.ny, w=2592, h=1944, jitter_px=20, um_per_px=0.5)
print(f"synthetic scan {a.nx}x{a.ny}: {time.perf_counter() - t:.1f} s to make")
prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(2592, 1944), repeatability_um=8)

for mode in ("quick", "quality"):
    s = ms.Session(prof, mode, threads=a.threads)
    t0 = time.perf_counter()
    for f in sc.frames:
        s.add(f)
    s._s.wait()
    t1 = time.perf_counter()
    rep = s.solve()
    t2 = time.perf_counter()
    cv = s.render()
    t3 = time.perf_counter()
    n, pairs = len(sc.frames), len(rep.edges["i"])
    err = rep.positions_px * 2 ** s.level - sc.truth
    err -= err.mean(axis=0)
    per_f, per_p = (t1 - t0) / n, (t2 - t1) / max(pairs, 1)
    print(f"{mode:8s} frames {n}, pairs {pairs}: prepare {per_f * 1e3:6.1f} ms/frame (wall, all threads), "
          f"register+solve {per_p * 1e3:6.1f} ms/pair, render {t3 - t2:5.2f} s, "
          f"error rms {np.sqrt((np.linalg.norm(err, axis=1) ** 2).mean()):.2f} px")
    for g in (30, 58):
        est = g * g * per_f + 2 * g * (g - 1) * per_p
        print(f"         {g}x{g}: about {est:6.1f} s after the last frame if nothing ran during the scan")
