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
        x0, y0, w, h = self.px
        return canvas.image[y0:y0 + h, x0:x0 + w]


class Canvas:
    """image: HxWx3 uint8 at `level` (1 = half, 2 = quarter of full resolution).
    owner: HxW frame index each pixel came from (-1: nothing).
    to_stage / to_px: canvas pixel <-> stage um, fitted on all solved frames."""

    def __init__(self, image, owner, level, origin, stage_from_px, poses):
        self.image = image
        self.owner = owner
        self.level = level
        self.origin = origin                  # working-level coordinates of canvas pixel (0, 0)
        self._a = np.asarray(stage_from_px, float)      # 2x3: stage = A @ [u, v, 1]
        self._ainv = np.linalg.inv(np.vstack([self._a, [0, 0, 1]]))[:2]
        self.poses = poses                    # list of dicts, one per frame (see Report.poses)

    @property
    def shape(self):
        return self.image.shape

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
        b = min(self.image.shape[1], int(np.ceil(us.max()))), min(self.image.shape[0], int(np.ceil(vs.max())))
        return a[0], a[1], max(0, b[0] - a[0]), max(0, b[1] - a[1])

    def read(self, x_um: float, y_um: float, w_um: float, h_um: float) -> np.ndarray:
        """Pixels covering the stage rectangle [x, x+w] x [y, y+h] (axis-aligned box on the canvas)."""
        x0, y0, w, h = self._bbox(x_um, y_um, x_um + w_um, y_um + h_um)
        return self.image[y0:y0 + h, x0:x0 + w]

    def grid(self, cell_um: float):
        """Cut the canvas into square cells of about cell_um; yields Cell (only cells with any image)."""
        n = max(1, int(round(cell_um / self.um_per_px)))
        H, W = self.image.shape[:2]
        for i, y0 in enumerate(range(0, H, n)):
            for j, x0 in enumerate(range(0, W, n)):
                w, h = min(n, W - x0), min(n, H - y0)
                if (self.owner[y0:y0 + h, x0:x0 + w] < 0).all():
                    continue
                cx, cy = self.to_stage(x0 + w / 2, y0 + h / 2)
                yield Cell(i, j, (x0, y0, w, h), (float(cx), float(cy)))

    def save(self, folder: str | Path, tile: int = 2048) -> Path:
        """manifest.json + poses.csv + tiles/<row>_<col>.png (tile x tile pixels)."""
        from PIL import Image

        folder = Path(folder)
        (folder / "tiles").mkdir(parents=True, exist_ok=True)
        H, W = self.image.shape[:2]
        names = []
        for ty, y0 in enumerate(range(0, H, tile)):
            for tx, x0 in enumerate(range(0, W, tile)):
                name = f"tiles/{ty}_{tx}.png"
                Image.fromarray(self.image[y0:y0 + tile, x0:x0 + tile]).save(folder / name)
                names.append({"file": name, "x": x0, "y": y0})
        (folder / "manifest.json").write_text(json.dumps({
            "size": [W, H], "level": self.level, "origin": list(self.origin), "um_per_px": self.um_per_px,
            "stage_from_px": self._a.tolist(), "tile": tile, "tiles": names}, indent=1))
        if self.poses:
            with open(folder / "poses.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.poses[0].keys()))
                w.writeheader()
                w.writerows(self.poses)
        return folder
