"""The stitched map: pixels, and the link between canvas pixels and stage micrometres."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Cell:
    i: int                      # row of the grid
    j: int                      # column
    px: tuple[int, int, int, int]   # x0, y0, w, h on the canvas
    center_um: tuple[float, float]  # stage position of the cell centre

    def image(self, canvas: "Canvas") -> np.ndarray:
        return canvas.region(*self.px)


class Canvas:
    """The map at `level` (1 = half, 2 = quarter of full resolution), rendered on demand:
    `region()`, `read()`, `grid()` and `save()` render only the part they need (GPU or CPU), so a 58x58 map is
    usable as soon as the frames are placed and never has to sit in memory whole. `image` / `owner` render the
    whole map once and keep it.
    owner: frame index each pixel came from (-1: nothing). to_stage / to_px: canvas pixel <-> stage um."""

    def __init__(self, renderer, origin, size, level, stage_from_px, poses, frame_rects):
        self._render = renderer               # (x0, y0, w, h) in working-level coordinates -> (rgb, owner)
        self.origin = origin                  # working-level coordinates of canvas pixel (0, 0)
        self.size = size                      # (W, H)
        self.level = level
        self._a = np.asarray(stage_from_px, float)      # 2x3: stage = A @ [u, v, 1]
        self._ainv = np.linalg.inv(np.vstack([self._a, [0, 0, 1]]))[:2]
        self.poses = poses                    # list of dicts, one per frame (see Report.poses)
        self._frames = np.asarray(frame_rects, float).reshape(-1, 4)   # x, y, w, h of each frame on the canvas
        self._full = None

    # --- pixels
    def region(self, x0: int, y0: int, w: int, h: int, owner: bool = False):
        """Pixels of canvas rectangle [x0, x0+w) x [y0, y0+h) (and the owner of each pixel if owner=True)."""
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        w, h = min(int(w), self.size[0] - x0), min(int(h), self.size[1] - y0)
        if w <= 0 or h <= 0:
            img, own = np.zeros((0, 0, 3), np.uint8), np.zeros((0, 0), np.int32)
        elif self._full is not None:
            img = self._full[0][y0:y0 + h, x0:x0 + w]
            own = self._full[1][y0:y0 + h, x0:x0 + w]
        else:
            img, own = self._render(self.origin[0] + x0, self.origin[1] + y0, w, h)
        return (img, own) if owner else img

    def _whole(self):
        if self._full is None:
            self._full = self._render(self.origin[0], self.origin[1], self.size[0], self.size[1])
        return self._full

    @property
    def image(self) -> np.ndarray:
        return self._whole()[0]

    @property
    def owner(self) -> np.ndarray:
        return self._whole()[1]

    @property
    def shape(self):
        return (self.size[1], self.size[0], 3)

    @property
    def um_per_px(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self._a[:, :2]))))

    def to_stage(self, u, v):
        """Canvas pixel (u = column, v = row; arrays allowed) -> stage (x, y) in um."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        x = self._a[0, 0] * u + self._a[0, 1] * v + self._a[0, 2]
        y = self._a[1, 0] * u + self._a[1, 1] * v + self._a[1, 2]
        return x, y

    def to_px(self, x_um, y_um):
        """Stage (x, y) um -> canvas pixel (u, v)."""
        x, y = np.asarray(x_um, float), np.asarray(y_um, float)
        u = self._ainv[0, 0] * x + self._ainv[0, 1] * y + self._ainv[0, 2]
        v = self._ainv[1, 0] * x + self._ainv[1, 1] * y + self._ainv[1, 2]
        return u, v

    def _bbox(self, x0_um, y0_um, x1_um, y1_um):
        us, vs = self.to_px([x0_um, x1_um, x0_um, x1_um], [y0_um, y0_um, y1_um, y1_um])
        a = max(0, int(np.floor(us.min()))), max(0, int(np.floor(vs.min())))
        b = min(self.size[0], int(np.ceil(us.max()))), min(self.size[1], int(np.ceil(vs.max())))
        return a[0], a[1], max(0, b[0] - a[0]), max(0, b[1] - a[1])

    def read(self, x_um: float, y_um: float, w_um: float, h_um: float) -> np.ndarray:
        """Pixels covering the stage rectangle [x, x+w] x [y, y+h] (axis-aligned box on the canvas)."""
        return self.region(*self._bbox(x_um, y_um, x_um + w_um, y_um + h_um))

    def covered(self, x0: int, y0: int, w: int, h: int) -> bool:
        """Does any frame reach into this canvas rectangle (no rendering)?"""
        f = self._frames
        return bool(((f[:, 0] < x0 + w) & (f[:, 0] + f[:, 2] > x0) & (f[:, 1] < y0 + h) &
                     (f[:, 1] + f[:, 3] > y0)).any())

    def grid(self, cell_um: float):
        """Cut the canvas into square cells of about cell_um; yields Cell (only cells some frame reaches).
        Nothing is rendered until a cell's image() is asked for."""
        n = max(1, int(round(cell_um / self.um_per_px)))
        W, H = self.size
        for i, y0 in enumerate(range(0, H, n)):
            for j, x0 in enumerate(range(0, W, n)):
                w, h = min(n, W - x0), min(n, H - y0)
                if not self.covered(x0, y0, w, h):
                    continue
                cx, cy = self.to_stage(x0 + w / 2, y0 + h / 2)
                yield Cell(i, j, (x0, y0, w, h), (float(cx), float(cy)))

    def save(self, folder: str | Path, tile: int = 2048) -> Path:
        """manifest.json + poses.csv + tiles/<row>_<col>.png (tile x tile pixels)."""
        from PIL import Image

        folder = Path(folder)
        (folder / "tiles").mkdir(parents=True, exist_ok=True)
        W, H = self.size
        names = []
        for ty, y0 in enumerate(range(0, H, tile)):          # one tile at a time: the whole map is never in RAM
            for tx, x0 in enumerate(range(0, W, tile)):
                w, h = min(tile, W - x0), min(tile, H - y0)
                if not self.covered(x0, y0, w, h):
                    continue
                name = f"tiles/{ty}_{tx}.png"
                # RGBA: where no frame reaches, transparent (else it is black and covers what lies under the map)
                img, own = self.region(x0, y0, w, h, owner=True)
                rgba = np.dstack([img, np.where(own >= 0, 255, 0).astype(np.uint8)])
                Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(folder / name)
                names.append({"file": name, "x": x0, "y": y0})
        (folder / "manifest.json").write_text(json.dumps({
            "size": [W, H], "level": self.level, "origin": [float(v) for v in self.origin], "um_per_px": self.um_per_px,
            "stage_from_px": self._a.tolist(), "tile": tile, "tiles": names}, indent=1))
        if self.poses:
            with open(folder / "poses.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.poses[0].keys()))
                w.writeheader()
                w.writerows(self.poses)
        return folder
