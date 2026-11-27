"""Command line, for scans saved with MControl's Frame.save() (name.png or name.npy + name.json).

    python -m mstitch calibrate FOLDER --objective 10x --out profiles/10x.json
    python -m mstitch stitch FOLDER --profile profiles/10x.json [--mode quick|quality] --out scans/chip7
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import Profile, Session, calibrate


class _Saved:          # what Session.add / calibrate read from an MControl Frame
    def __init__(self, meta, data):
        self.meta, self.data = meta, data


def load_folder(folder: Path):
    """Yields (kind, item): ('frame', _Saved) for .npy sensor data, ('image', (img, x, y)) for colour images."""
    from PIL import Image

    for js in sorted(folder.glob("*.json")):
        meta = json.loads(js.read_text())
        xyz = meta.get("xyz_um") or [0, 0, 0]
        npy, img = js.with_suffix(".npy"), None
        if npy.exists():
            px = np.load(npy)
            meta = dict(meta, codec="raw")
            yield "frame", _Saved(meta, px.astype("<u2" if px.dtype != np.uint8 else np.uint8).tobytes())
            continue
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            if js.with_suffix(ext).exists():
                img = np.asarray(Image.open(js.with_suffix(ext)).convert("RGB"))
                break
        if img is not None:
            yield "image", (img, float(xyz[0]), float(xyz[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mstitch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate", help="measure an objective's profile")
    c.add_argument("folder", type=Path)
    c.add_argument("--objective", required=True)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("stitch", help="stitch a saved scan")
    s.add_argument("folder", type=Path)
    s.add_argument("--profile", type=Path, required=True)
    s.add_argument("--mode", default="quick", choices=["quick", "quality"])
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    items = list(load_folder(a.folder))
    if not items:
        sys.exit(f"no frames (name.json + name.png/.npy) in {a.folder}")
    if a.cmd == "calibrate":
        prof = calibrate([it if k == "image" else it for k, it in items], a.objective)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        prof.save(a.out)
        print(f"{a.objective}: {prof.um_per_px:.4f} um/px, rotation {prof.rotation_deg:.3f} deg, "
              f"mirrored {prof.mirrored}, repeatability {prof.repeatability_um:.2f} um -> {a.out}")
        return
    t = time.perf_counter()
    ses = Session(Profile.load(a.profile), mode=a.mode)
    for k, it in items:
        ses.add(it) if k == "frame" else ses.add_image(*it)
    rep = ses.solve()
    cv = ses.render()
    cv.save(a.out)
    print(rep.summary(), f"| {time.perf_counter() - t:.1f} s -> {a.out}")
    for r in rep.reshoot():
        print(f"  reshoot frame {r['index']} at ({r['x_um']:.1f}, {r['y_um']:.1f}) um: {r['reason']}")


if __name__ == "__main__":
    main()
