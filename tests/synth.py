"""Synthetic scans with known truth: a sample (purple substrate, dust, crystals), a camera with vignetting and
noise, a stage that lands off target, frames packed exactly like MControl (Bayer + lossless)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import zstandard


def world(w: int, h: int, seed: int = 0) -> np.ndarray:
    """RGB float sample, shape h x w x 3."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (0.47, 0.52, 0.92)
    yy, xx = np.ogrid[0:h, 0:w]
    img *= (1 + 0.03 * np.sin(xx / 97.0) * np.cos(yy / 131.0)).astype(np.float32)[..., None]
    n = w * h // 900
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(1.0, 4.5)
        x0, x1 = int(max(0, cx - 4 * r)), int(min(w, cx + 4 * r + 1))
        y0, y1 = int(max(0, cy - 4 * r)), int(min(h, cy + 4 * r + 1))
        g = np.exp(-((xx[:, x0:x1] - cx) ** 2 + (yy[y0:y1] - cy) ** 2) / (2 * r * r))
        img[y0:y1, x0:x1] *= (1 - 0.7 * g)[..., None] * np.array([1.0, 0.95, 1.0], np.float32)
    for _ in range(max(1, n // 400)):          # a few big dark crystals
        cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(15, 40)
        x0, x1, y0, y1 = int(max(0, cx - r)), int(min(w, cx + r + 1)), int(max(0, cy - r)), int(min(h, cy + r + 1))
        m = (xx[:, x0:x1] - cx) ** 2 + (yy[y0:y1] - cy) ** 2 < r * r
        img[y0:y1, x0:x1][m] = (0.25, 0.22, 0.35)
    return np.clip(img, 0, 1)


def vignette(w: int, h: int, k: float = 0.35) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    return (1 - k * r2 / 2).astype(np.float32)


def bayer_rg(rgb: np.ndarray, bits: int = 8) -> np.ndarray:
    """RGGB mosaic, uint8 or uint16."""
    h, w, _ = rgb.shape
    full = (1 << bits) - 1
    out = np.empty((h, w), np.float32)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return np.clip(np.round(out * full), 0, full).astype(np.uint8 if bits <= 8 else np.uint16)


def pack_lossless(px: np.ndarray) -> bytes:
    """Exactly what mc-server does (codec.rs): difference to the same-colour pixel two to the left
    (first two columns: two rows up), wrapping, then zstd."""
    d = px.copy()
    d[:, 2:] = px[:, 2:] - px[:, :-2]
    d[2:, :2] = px[2:, :2] - px[:-2, :2]
    raw = d.astype(px.dtype).tobytes() if px.dtype == np.uint8 else d.astype("<u2").tobytes()
    return zstandard.ZstdCompressor(level=1).compress(raw)


@dataclass
class Frame:          # same shape as mcontrol.Frame for what MStitch reads
    meta: dict
    data: bytes


@dataclass
class Scan:
    frames: list
    truth: np.ndarray        # Nx2 true top-left of each frame in the world (full-res px)
    m: np.ndarray            # true stage->pixel matrix
    w: int
    h: int


def stage_matrix(um_per_px: float, rot_deg: float, flip_y: bool) -> np.ndarray:
    t = math.radians(rot_deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return r @ np.diag([1.0, -1.0 if flip_y else 1.0]) / um_per_px


def scan(nx=6, ny=5, w=640, h=480, overlap=0.25, um_per_px=0.5, rot_deg=0.55, flip_y=True, jitter_px=10.0,
         noise=2.0, bits=8, seed=1, blank: set | None = None, moving: set | None = None) -> Scan:
    rng = np.random.default_rng(seed)
    m = stage_matrix(um_per_px, rot_deg, flip_y)
    minv = np.linalg.inv(m)
    step = np.array([w * (1 - overlap), h * (1 - overlap)])
    # nominal stage targets so that the image grid is axis-aligned
    targets = [minv @ (np.array([i, j]) * step) for j in range(ny) for i in range(nx)]
    margin = 80
    tl = np.array([m @ s - [w / 2, h / 2] for s in targets])
    true_tl = np.round(tl + rng.uniform(-jitter_px, jitter_px, tl.shape))   # frames cut at whole pixels
    off = -true_tl.min(axis=0) + margin                               # whole pixels too
    W = int(np.ceil((true_tl[:, 0] + off[0]).max() + w + margin))
    H = int(np.ceil((true_tl[:, 1] + off[1]).max() + h + margin))
    sample = world(W, H, seed)
    vig = vignette(w, h)[..., None]
    frames = []
    for k, (s, t) in enumerate(zip(targets, true_tl)):
        x0, y0 = int(t[0] + off[0]), int(t[1] + off[1])
        img = sample[y0:y0 + h, x0:x0 + w] * vig
        if blank and k in blank:
            img = np.full_like(img, 0.6) * vig
        img = img + rng.normal(0, noise / 255, img.shape).astype(np.float32)
        px = bayer_rg(np.clip(img, 0, 1), bits)
        meta = {"seq": k, "w": w, "h": h, "fmt": f"bayer_rg{bits}", "bits": bits, "codec": "lossless",
                "raw_bytes": px.nbytes, "xyz_um": [float(s[0]), float(s[1]), 0.0],
                "moving": bool(moving and k in moving)}
        frames.append(Frame(meta, pack_lossless(px)))
    return Scan(frames, true_tl, m, w, h)
