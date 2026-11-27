"""How long MStitch will take, learned from its own past runs.

The model is one linear layer: inputs x = [1, frames, pairs, map megapixels] of a scan, outputs the seconds of
each phase after the last frame (registration, global solve, map rendering):  t = W x,  W >= 0 except the bias.
It is trained in closed form (ridge least squares, non-negative active set) on every run recorded for the same
setup (mode, backend, frame size, GPU), so it learns the speed of this laptop as it is used. Until a setup has
`min_runs` recorded runs, predict() gives None: the first runs are only remembered.

The inputs of a planned scan (from MControl's list of stage positions) are computed with the same rules the
session uses, so a plan can be priced before a single frame is taken.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .profile import Profile

PHASES = ("register", "solve", "render")
FEATURES = ("bias", "frames", "pairs", "map_mpx")


def default_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "MStitch"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mstitch"
    return base / "timing.json"


def plan_features(profile: Profile, positions_um, mode: str = "quick", level: int | None = None,
                  search_um: float | None = None, min_overlap: float = 0.03) -> dict:
    """Inputs of the model for a planned scan: the pairs and the map size the session would get."""
    from .session import MODES

    lv = MODES[mode] if level is None else level
    xy = np.asarray(positions_um, float).reshape(-1, 2)
    n = len(xy)
    w, h = profile.frame                                # frame size at the level, integer halving as in C++
    for _ in range(lv):
        w, h = w // 2, h // 2
    m = profile.matrix() / 2 ** lv
    p = xy @ m.T                                          # frame centres, working-level pixels
    if search_um is None:
        search_um = 6 * profile.repeatability_um
    search = max(4.0, search_um / profile.um_per_px / 2 ** lv)
    pairs = 0
    for i in range(n - 1):
        d = np.abs(p[i + 1:] - p[i])
        ow = w - np.maximum(0.0, d[:, 0] - search)
        oh = h - np.maximum(0.0, d[:, 1] - search)
        pairs += int(((ow > 0) & (oh > 0) & (ow * oh >= min_overlap * w * h)).sum())
    if n:
        span = p.max(axis=0) - p.min(axis=0) + [w, h]
        mpx = float(span[0] * span[1]) / 1e6
    else:
        mpx = 0.0
    return {"frames": n, "pairs": pairs, "map_mpx": mpx}


@dataclass
class Estimate:
    seconds: float                 # after the last frame: backlog + registration + solve + render
    phases: dict                   # seconds per phase
    runs: int                      # recorded runs the model learned from
    rms: float | None              # model error on those runs (s), None with too few runs

    def __str__(self):
        parts = ", ".join(f"{k} {v:.1f}" for k, v in self.phases.items())
        return f"~{self.seconds:.1f} s ({parts}; from {self.runs} runs)"


class Eta:
    """Timing history + the linear model. One file per user (default_path())."""

    def __init__(self, path: str | Path | None = None, min_runs: int = 2, keep: int = 200):
        self.path = Path(path) if path else default_path()
        self.min_runs = min_runs
        self.keep = keep
        self.runs: list[dict] = []
        try:
            self.runs = json.loads(self.path.read_text())["runs"]
        except (OSError, ValueError, KeyError):
            self.runs = []

    @staticmethod
    def setup_key(mode: str, backend: str, frame, device: str) -> str:
        return f"{mode}|{backend}|{frame[0]}x{frame[1]}|{device}"

    @staticmethod
    def device_name(backend: str) -> str:
        if backend == "cuda":
            from . import _core
            d = _core.cuda_device()
            return d.get("name", "gpu") if d.get("ok") else "gpu"
        return f"{platform.processor() or platform.machine()} x{os.cpu_count()}"

    # --- learning
    def record(self, key: str, features: dict, seconds: dict, prepare_s_per_frame: float | None = None) -> None:
        self.runs.append({"key": key, "t": time.time(), "x": {f: float(features[f]) for f in FEATURES[1:]},
                          "y": {p: float(seconds[p]) for p in PHASES}, "prep": prepare_s_per_frame})
        self.runs = self.runs[-self.keep:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": 1, "runs": self.runs}))
            os.replace(tmp, self.path)
        except OSError:
            pass                                          # timing is a convenience: never fail a stitch on it

    def _fit(self, key: str):
        rs = [r for r in self.runs if r["key"] == key]
        if len(rs) < self.min_runs:
            return None, rs
        x = np.array([[1.0] + [r["x"][f] for f in FEATURES[1:]] for r in rs])
        y = np.array([[r["y"][p] for p in PHASES] for r in rs])
        scale = np.maximum(np.abs(x).max(axis=0), 1e-9)   # features to ~1 so one ridge strength fits all
        xs = x / scale
        lam = 1e-3 * len(rs)
        wts = np.zeros((len(FEATURES), len(PHASES)))
        for k in range(len(PHASES)):
            active = list(range(len(FEATURES)))
            for _ in range(len(FEATURES)):              # non-negative weights (bias may be anything)
                a = xs[:, active]
                w = np.linalg.solve(a.T @ a + lam * np.eye(len(active)), a.T @ y[:, k])
                neg = [active[i] for i, v in enumerate(w) if v < 0 and active[i] != 0]
                if not neg:
                    break
                active = [i for i in active if i not in neg]
            wts[active, k] = w
        wts /= scale[:, None]
        return wts, rs

    def predict_features(self, key: str, features: dict, pending_frames: int = 0) -> Estimate | None:
        wts, rs = self._fit(key)
        if wts is None:
            return None
        x = np.array([1.0] + [float(features[f]) for f in FEATURES[1:]])
        ph = np.maximum(x @ wts, 0.0)
        phases = {p: float(v) for p, v in zip(PHASES, ph)}
        preps = [r["prep"] for r in rs if r.get("prep")]
        if pending_frames and preps:
            phases = {"prepare": pending_frames * float(np.median(preps)), **phases}
        rms = None
        if len(rs) > len(FEATURES):
            xr = np.array([[1.0] + [r["x"][f] for f in FEATURES[1:]] for r in rs])
            yr = np.array([[r["y"][p] for p in PHASES] for r in rs]).sum(axis=1)
            rms = float(np.sqrt(np.mean((np.maximum(xr @ wts, 0).sum(axis=1) - yr) ** 2)))
        return Estimate(float(sum(phases.values())), phases, len(rs), rms)

    def predict(self, profile: Profile, mode: str, positions_um, backend: str = "auto") -> Estimate | None:
        """Seconds MStitch will need after the last frame of this planned scan; None for the first runs of a
        setup (they are only remembered)."""
        from . import cuda as _cuda

        used = backend
        if backend == "auto":
            used = "cuda" if _cuda.info()["ok"] else "cpu"
        f = plan_features(profile, positions_um, mode)
        key = self.setup_key(mode, used, profile.frame, self.device_name(used))
        return self.predict_features(key, f)
