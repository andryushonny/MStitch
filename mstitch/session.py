"""Stitching session: frames from MControl in, report + canvas out."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from .canvas import Canvas
from .profile import Profile

MODES = {"quick": 2, "quality": 1}   # working level per mode (2 = quarter, 1 = half resolution)


def _meta_args(meta: dict) -> dict:
    fmt = meta.get("fmt", "")
    if fmt == "jpeg":
        raise ValueError("a preview frame (fmt jpeg) cannot be stitched: use Microscope.snap() frames")
    xyz = meta.get("xyz_um") or (0.0, 0.0, 0.0)
    return dict(w=int(meta["w"]), h=int(meta["h"]), bits=int(meta.get("bits", 8)), fmt=fmt,
                codec=meta.get("codec", "raw"), sx=float(xyz[0]), sy=float(xyz[1]), moving=bool(meta.get("moving")))


@dataclass
class Report:
    """What solve() found. Positions are full-resolution pixels in the stage-aligned frame of the profile."""
    positions_px: np.ndarray      # Nx2: solved top-left of each frame (working level)
    predicted_px: np.ndarray      # Nx2: where the stage says it is (working level)
    edges: dict                   # i, j, pred, t, ncc, overlap, found, used, resid
    stats: dict
    poses: list                   # one dict per frame
    level: int

    def reshoot(self) -> list[dict]:
        """Frames to take again in the second pass: where to go (um) and why."""
        return [p["reshoot"] for p in self.poses if p["reshoot"]]

    def summary(self) -> str:
        n = len(self.poses)
        bad = sum(1 for p in self.poses if p["reshoot"])
        return (f"{n} frames, pairs used {self.stats['pairs_used']} (dropped {self.stats['pairs_dropped']}), "
                f"rms {self.stats['rms_px'] * 2 ** self.level:.2f} px full-res, to reshoot: {bad}")


class Session:
    """Collect frames, then solve() and render().

    >>> s = Session(Profile.load("profiles/10x.json"), mode="quick")
    >>> for x, y in plan: s.add(m.snap(x=x, y=y))
    >>> rep = s.solve(); cv = s.render(); cv.save("scans/chip7")
    """

    def __init__(self, profile: Profile, mode: str = "quick", level: int | None = None, threads: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        self.profile = profile
        self.mode = mode
        self.level = MODES[mode] if level is None else level
        self._s = _core.Session(self.level, list(profile.m), threads)
        self.report: Report | None = None

    def __len__(self):
        return len(self._s)

    # --- frames in
    def add(self, frame=None, *, meta: dict | None = None, data: bytes | None = None, row: int = -1,
            col: int = -1) -> int:
        """An MControl Frame (from Microscope.snap / fetch), or its meta + data. Returns the frame index.
        Returns at once: unpacking and preparation run on all cores."""
        if frame is not None:
            meta, data = frame.meta, frame.data
        a = _meta_args(meta)
        return self._s.add_sensor(bytes(data), a["w"], a["h"], a["bits"], a["fmt"], a["codec"], a["sx"], a["sy"],
                                  a["moving"], row, col)

    def add_image(self, image: np.ndarray, x_um: float, y_um: float, moving: bool = False, row: int = -1,
                  col: int = -1) -> int:
        """A plain image (HxW or HxWx3, uint8/uint16, full resolution) taken with the stage at (x, y) um."""
        return self._s.add_image(np.ascontiguousarray(image), float(x_um), float(y_um), moving, row, col)

    # --- solve
    def solve(self, search_um: float | None = None, min_ncc: float = 0.3, min_overlap: float = 0.03) -> Report:
        """Register overlapping pairs, then place all frames. search_um: how far a frame may be from where the
        stage says (default 6 x repeatability)."""
        s = self._s
        s.wait()
        n = len(s)
        infos = [s.tile_info(k) for k in range(n)]
        scale = 2 ** self.level
        if search_um is None:
            search_um = 6 * self.profile.repeatability_um
        search_px = max(4.0, search_um / self.profile.um_per_px / scale)
        s.register(search_px, True, 8, min_overlap)
        pos, stats, edges = s.solve(1e-3, min_ncc, 1.5, 5)
        pred = s.predicted()
        self.report = Report(pos, pred, edges, stats, self._poses(infos, pos, pred, edges, min_ncc), self.level)
        return self.report

    def _poses(self, infos, pos, pred, edges, min_ncc):
        scale = 2 ** self.level
        minv = np.linalg.inv(self.profile.matrix())
        fov = np.array(self.profile.frame, float)
        poses = []
        for k, inf in enumerate(infos):
            mine = (edges["i"] == k) | (edges["j"] == k)
            used = mine & edges["used"].astype(bool)
            best = float(edges["ncc"][used].max()) if used.any() else 0.0
            d_px = (pos[k] - pred[k]) * scale
            d_um = minv @ d_px
            reason = None
            if not inf["ready"]:
                reason = "error: " + inf["error"]
            elif inf["moving"]:
                reason = "taken while the stage was moving"
            elif mine.any() and not used.any():
                reason = "no reliable overlap with any neighbour"
            elif mine.any() and best < max(min_ncc, 0.5):
                reason = f"weak overlap (best NCC {best:.2f})"
            reshoot = None
            if reason:
                # Step a quarter frame towards the neighbours that failed: the same place seen from another spot.
                bad = mine & ~edges["used"].astype(bool)
                off = np.zeros(2)
                if bad.any():
                    sign = np.where(edges["i"][bad] == k, 1.0, -1.0)[:, None]
                    v = (edges["pred"][bad] * sign).mean(axis=0)
                    if np.linalg.norm(v) > 0:
                        off = minv @ (v / np.linalg.norm(v) * 0.25 * fov.min())
                reshoot = {"index": k, "x_um": inf["sx"] + off[0], "y_um": inf["sy"] + off[1], "reason": reason}
            poses.append({"index": k, "row": inf["row"], "col": inf["col"], "x_um": inf["sx"], "y_um": inf["sy"],
                          "x_px": float(pos[k][0] * scale), "y_px": float(pos[k][1] * scale),
                          "dx_px": float(d_px[0]), "dy_px": float(d_px[1]),
                          "dx_um": float(d_um[0]), "dy_um": float(d_um[1]),
                          "pairs": int(used.sum()), "best_ncc": round(best, 4), "reshoot": reshoot})
        return poses

    # --- canvas out
    def render(self, flat: np.ndarray | None | str = "auto") -> Canvas:
        """The mosaic at the working level. flat: 'auto' = the profile's, else estimated from these frames
        (needs >= 8 frames); None = no correction; or an HxWx3 gain."""
        if self.report is None:
            self.solve()
        s, rep = self._s, self.report
        n = len(s)
        idx = list(range(n))
        ready = [k for k in idx if s.tile_info(k)["ready"]]
        if isinstance(flat, str):
            shape = (s.tile_info(ready[0])["h"], s.tile_info(ready[0])["w"]) if ready else (0, 0)
            flat = self.profile.flat_at(self.level, shape)
            if flat is None and len(ready) >= 8:
                step = max(1, len(ready) // 64)
                flat = s.estimate_flat(ready[::step])
        pos = np.ascontiguousarray(rep.positions_px, dtype=np.float64)
        img, owner, ox, oy = s.render(idx, pos, None if flat is None else np.ascontiguousarray(flat, np.float32))
        return Canvas(img, owner, self.level, (ox, oy), self._stage_from_px(ready, pos, ox, oy), rep.poses)

    def _stage_from_px(self, ready, pos, ox, oy):
        """Affine canvas pixel -> stage um, fitted on frame centres (falls back to the profile matrix)."""
        infos = [self._s.tile_info(k) for k in ready]
        uv = np.array([[pos[k][0] - ox + 0.5 * (i["w"] - 1), pos[k][1] - oy + 0.5 * (i["h"] - 1)]
                       for k, i in zip(ready, infos)])
        xy = np.array([[i["sx"], i["sy"]] for i in infos])
        if len(ready) >= 3:
            sv = np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)
            if sv[1] > 0.05 * sv[0] and sv[1] > 10:      # frames spread in 2-D: fit, else the profile decides
                sol, *_ = np.linalg.lstsq(np.c_[uv, np.ones(len(uv))], xy, rcond=None)
                return sol.T
        minv = np.linalg.inv(self.profile.matrix()) * 2 ** self.level
        b = xy.mean(axis=0) - minv @ uv.mean(axis=0) if len(ready) else np.zeros(2)
        return np.c_[minv, b]
