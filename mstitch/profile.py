"""Objective profile: how stage motion maps to camera pixels, and the illumination of the field."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Profile:
    """One objective. `m` maps a stage move (um) to the image shift of the content, in full-resolution pixels:
    (u, v) = m @ (x, y), row-major [[mxx, mxy], [myx, myy]]. Rotation and mirroring of the camera live in m."""

    objective: str
    m: tuple[float, float, float, float]
    frame: tuple[int, int] = (2592, 1944)     # full-resolution frame size (w, h)
    repeatability_um: float = 5.0             # how far a frame may land from where the stage says (1 sigma)
    flat: np.ndarray | None = field(default=None, repr=False)   # HxWx3 gain, mean 1, at flat_level
    flat_level: int = 2
    notes: dict = field(default_factory=dict)

    # --- derived
    @property
    def um_per_px(self) -> float:
        a, b, c, d = self.m
        return 1.0 / math.sqrt(abs(a * d - b * c))

    @property
    def rotation_deg(self) -> float:
        """Angle of the stage x axis in the image. m = R(angle) @ diag(1, -1 if mirrored else 1) / um_per_px."""
        a, b, c, d = self.m
        return math.degrees(math.atan2(c, a))

    @property
    def mirrored(self) -> bool:
        a, b, c, d = self.m
        return bool(a * d - b * c < 0)

    def matrix(self) -> np.ndarray:
        return np.array(self.m, float).reshape(2, 2)

    @classmethod
    def guess(cls, objective: str, um_per_px: float, rotation_deg: float = 0.0, flip_x: bool = False,
              flip_y: bool = False, frame=(2592, 1944), repeatability_um: float = 5.0) -> "Profile":
        """A profile from nominal numbers, before calibrate() has measured the real ones."""
        t = math.radians(rotation_deg)
        r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        f = np.diag([-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0])
        m = (r @ f) / um_per_px
        return cls(objective, tuple(m.ravel()), tuple(frame), repeatability_um)

    # --- files: <name>.json + <name>.flat.npy
    def save(self, path: str | Path) -> None:
        path = Path(path)
        d = {"objective": self.objective, "m": [float(v) for v in self.m], "frame": [int(v) for v in self.frame],
             "repeatability_um": float(self.repeatability_um), "flat_level": int(self.flat_level),
             "notes": self.notes, "um_per_px": float(self.um_per_px), "rotation_deg": float(self.rotation_deg),
             "mirrored": bool(self.mirrored)}
        if self.flat is not None:
            fp = path.with_suffix(".flat.npy")
            np.save(fp, self.flat.astype(np.float32))
            d["flat"] = fp.name
        path.write_text(json.dumps(d, indent=1, default=float))

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        path = Path(path)
        d = json.loads(path.read_text())
        flat = np.load(path.parent / d["flat"]) if d.get("flat") else None
        return cls(d["objective"], tuple(d["m"]), tuple(d["frame"]), d.get("repeatability_um", 5.0), flat,
                   d.get("flat_level", 2), d.get("notes", {}))

    def flat_at(self, level: int, shape: tuple[int, int]) -> np.ndarray | None:
        """The flat-field resampled to a level (nearest / box), or None."""
        if self.flat is None:
            return None
        f = self.flat
        lv = self.flat_level
        while lv < level:                       # halve
            h, w = f.shape[0] // 2 * 2, f.shape[1] // 2 * 2
            f = f[:h, :w].reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
            lv += 1
        while lv > level:                       # double
            f = f.repeat(2, axis=0).repeat(2, axis=1)
            lv -= 1
        if f.shape[:2] != tuple(shape):
            return None
        return np.ascontiguousarray(f, dtype=np.float32)
