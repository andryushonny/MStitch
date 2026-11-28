"""Lithography crosses in a frame: the arms make one row and one column of a window stand out (darker or
brighter); their sub-pixel positions are the centre (as MControl's cross finder)."""
from __future__ import annotations

import numpy as np


def _peak(p: np.ndarray, frac: float) -> float | None:
    med = np.median(p)
    dev = np.abs(p - med)
    mad = max(float(np.median(dev)), 0.5)
    k = int(np.argmax(dev))
    best = float(dev[k])
    if best < 6.0 * mad or k == 0 or k + 1 == len(p):
        return None
    half = best * frac
    a = b = k
    while a > 0 and dev[a - 1] > half:
        a -= 1
    while b + 1 < len(p) and dev[b + 1] > half:
        b += 1
    w = dev[a:b + 1]
    return float((w * np.arange(a, b + 1)).sum() / w.sum())


def _profiles(g: np.ndarray, u: float, v: float, r: int, frac: float):
    cx, cy = int(np.floor(u)), int(np.floor(v))
    h, w = g.shape
    if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
        return None
    win = g[cy - r:cy + r + 1, cx - r:cx + r + 1]
    y = _peak(win.mean(axis=1), frac)
    x = _peak(win.mean(axis=0), frac)
    if x is None or y is None:
        return None
    return cx - r + x + 0.5, cy - r + y + 0.5


def refine(g: np.ndarray, u: float, v: float, radius: int):
    """Centre of the cross near (u, v) in grey image g (edge-based pixels), or None."""
    c = _profiles(g, u, v, radius, 0.1)
    if c is None:
        return None
    r2 = max(radius * 3 // 10, 4)
    for _ in range(3):
        n = _profiles(g, c[0], c[1], r2, 0.25)
        if n is None:
            return None
        moved = max(abs(n[0] - c[0]), abs(n[1] - c[1]))
        c = n
        if moved < 0.01:
            break
    return c
