"""calibrate(): measure an objective's profile from a few frames taken at known stage moves.

MControl's prepare() takes the frames (a 3x3 grid on any textured spot, ~30-50 % overlap, approached from one
side, plus repeats of one position) and hands them here. No crosses needed: only texture.
"""
from __future__ import annotations

import numpy as np

from . import _core
from .profile import Profile
from .session import _meta_args


def calibrate(frames, objective: str, level: int = 2, min_ncc: float = 0.5, threads: int = 0) -> Profile:
    """frames: MControl Frames, or (image, x_um, y_um) tuples. Returns a Profile with the stage->pixel matrix,
    the repeatability and the flat-field (if >= 5 frames)."""
    s = _core.Session(level, [1.0, 0.0, 0.0, 1.0], threads)
    pos_um = []
    frame_wh = None
    for f in frames:
        if isinstance(f, tuple):
            img, x, y = f
            s.add_image(np.ascontiguousarray(img), float(x), float(y))
            frame_wh = (img.shape[1], img.shape[0])
            pos_um.append((x, y))
        else:
            a = _meta_args(f.meta)
            s.add_sensor(bytes(f.data), a["w"], a["h"], a["bits"], a["fmt"], a["codec"], a["sx"], a["sy"])
            frame_wh = (a["w"], a["h"])
            pos_um.append((a["sx"], a["sy"]))
    s.wait()
    pos_um = np.array(pos_um, float)
    e = s.register(0.0, False, 8, 0.05)        # no stage prior: every pair, any offset
    ok = e["found"].astype(bool) & (e["ncc"] >= min_ncc)
    ds = pos_um[e["j"]] - pos_um[e["i"]]        # stage moves (um)
    t = e["t"]                                  # measured shifts (working-level px)
    keep = ok & (np.linalg.norm(ds, axis=1) > 1e-6)
    if keep.sum() < 3:
        raise RuntimeError(f"calibrate: only {int(keep.sum())} reliable pairs; take more overlap or texture")
    ml = None
    for _ in range(5):                          # least squares t = M_level @ ds, dropping outliers
        sol, *_ = np.linalg.lstsq(ds[keep], t[keep], rcond=None)
        ml = sol.T
        r = np.linalg.norm(t - ds @ ml.T, axis=1)
        lim = max(1.0, 4 * 1.4826 * np.median(r[keep]))
        new = keep & (r <= lim)
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    m = ml * 2 ** level
    resid_px = np.linalg.norm(t[keep] - ds[keep] @ ml.T, axis=1) * 2 ** level
    um_per_px = 1 / np.sqrt(abs(np.linalg.det(m)))
    p = Profile(objective, tuple(float(v) for v in m.ravel()), tuple(int(v) for v in frame_wh),
                repeatability_um=float(max(0.5, np.sqrt(np.mean(resid_px ** 2)) * um_per_px)))
    p.notes = {"pairs": int(keep.sum()), "resid_px_rms": float(np.sqrt(np.mean(resid_px ** 2))),
               "level": level}
    if len(s) >= 5:
        p.flat = s.estimate_flat(list(range(len(s))))
        p.flat_level = level
    return p
