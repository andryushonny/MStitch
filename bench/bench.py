"""Time each stage on real-size synthetic frames (2592 x 1944, Bayer, lossless like MControl).

    uv run python bench/bench.py                  # 4 x 3 frames, quick and quality
    uv run python bench/bench.py --nx 6 --ny 5 --threads 8

Prints ms per frame (unpack + prepare) and ms per pair (registration) on the CPU and on the GPU (when there is
one), and extrapolates to 30x30 and 58x58.
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
ap.add_argument("--backends", default="cpu,cuda", help="comma list of cpu, cuda")
a = ap.parse_args()
gpu = ms.cuda.info()
print("GPU:", gpu.get("name") if gpu["ok"] else f"none ({gpu['why']})")
backends = [b for b in a.backends.split(",") if b == "cpu" or gpu["ok"]]

t = time.perf_counter()
sc = scan(nx=a.nx, ny=a.ny, w=2592, h=1944, jitter_px=20, um_per_px=0.5)
print(f"synthetic scan {a.nx}x{a.ny}: {time.perf_counter() - t:.1f} s to make")
prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(2592, 1944), repeatability_um=8)

for mode, backend in [(m, b) for m in ("quick", "quality") for b in backends]:
    s = ms.Session(prof, mode, threads=a.threads, backend=backend)
    t0 = time.perf_counter()
    for f in sc.frames:
        s.add(f)
    s._s.wait()
    t1 = time.perf_counter()
    t_prep = t1 - t0
    rep = s.solve()
    t_cold = time.perf_counter() - t1
    t1 = time.perf_counter()
    rep = s.solve()              # again: cuFFT plans and caches are warm
    t2 = time.perf_counter()
    cv = s.render()
    t3 = time.perf_counter()
    n, pairs = len(sc.frames), len(rep.edges["i"])
    err = rep.positions_px * 2 ** s.level - sc.truth
    err -= err.mean(axis=0)
    per_f, per_p = t_prep / n, (t2 - t1) / max(pairs, 1)
    print(f"{mode:8s} {rep.stats['backend']:4s} frames {n}, pairs {pairs}: prepare {per_f * 1e3:6.1f} ms/frame (wall, all threads), "
          f"register+solve {per_p * 1e3:6.2f} ms/pair (first call {t_cold * 1e3 / max(pairs, 1):6.2f}), "
          f"render {t3 - t2:5.2f} s, "
          f"error rms {np.sqrt((np.linalg.norm(err, axis=1) ** 2).mean()):.3f} px")
    if rep.stats["backend"] == "cuda":
        print("         " + rep.stats["backend_note"])
    for g in (30, 58):
        n_g, p_g = g * g, 2 * g * (g - 1)
        after = p_g * per_p + n_g * (t3 - t2) / n     # registration + solve + render, once the last frame is in
        print(f"         {g}x{g}: {after:6.1f} s after the last frame (registration {p_g * per_p:5.1f} s, render "
              f"{n_g * (t3 - t2) / n:5.1f} s); preparing {per_f * 1e3:.0f} ms/frame runs during the scan "
              f"({n_g * per_f:5.1f} s if stitching a saved scan)")
