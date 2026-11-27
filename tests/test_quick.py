import numpy as np
import pytest

import mstitch as ms
from synth import Frame, bayer_rg, pack_lossless, scan


def errors(rep, sc, level):
    got = rep.positions_px * 2 ** level
    d = got - sc.truth
    d -= d.mean(axis=0)                     # the map may sit anywhere: compare shapes
    return np.linalg.norm(d, axis=1)


@pytest.mark.parametrize("bits", [8, 12])
def test_lossless_unpack_matches_superpixel(bits):
    rng = np.random.default_rng(3)
    rgb = rng.uniform(0, 1, (64, 96, 3)).astype(np.float32)
    px = bayer_rg(rgb, bits)
    meta = {"w": 96, "h": 64, "fmt": f"bayer_rg{bits}", "bits": bits, "codec": "lossless", "xyz_um": [0, 0, 0]}
    s = ms.Session(ms.Profile.guess("t", 1.0), mode="quality")
    s.add(Frame(meta, pack_lossless(px)))
    s._s.wait()
    got = s._s.tile_rgb(0).astype(float)
    full = (1 << bits) - 1
    p = px.astype(float) / full
    exp = np.dstack([p[0::2, 0::2], (p[0::2, 1::2] + p[1::2, 0::2]) / 2, p[1::2, 1::2]]) * 255
    assert np.abs(got - exp).max() <= 1.0


def test_quick_grid_subpixel():
    sc = scan(nx=6, ny=5, jitter_px=12)
    prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    s = ms.Session(prof, mode="quick")
    for f in sc.frames:
        s.add(f)
    rep = s.solve()
    e = errors(rep, sc, s.level)
    print(rep.summary(), "err rms", np.sqrt((e ** 2).mean()), "max", e.max())
    assert np.sqrt((e ** 2).mean()) < 0.2       # full-res px, from quarter-size images
    assert e.max() < 0.4
    assert rep.reshoot() == []


def test_quality_level_is_finer():
    sc = scan(nx=3, ny=3, jitter_px=12, seed=5)
    prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    s = ms.Session(prof, mode="quality")
    for f in sc.frames:
        s.add(f)
    e = errors(s.solve(), sc, s.level)
    assert np.sqrt((e ** 2).mean()) < 0.05


def test_bad_frames_are_flagged_for_reshoot():
    sc = scan(nx=4, ny=3, blank={5}, moving={9}, seed=2)
    prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    s = ms.Session(prof)
    for f in sc.frames:
        s.add(f)
    rep = s.solve()
    flagged = {r["index"] for r in rep.reshoot()}
    assert flagged == {5, 9}


def test_canvas_maps_pixels_to_stage():
    sc = scan(nx=4, ny=3, seed=4)
    prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    s = ms.Session(prof)
    for f in sc.frames:
        s.add(f)
    s.solve()
    cv = s.render()
    assert (cv.owner >= 0).mean() > 0.9
    # the centre of frame 0 on the canvas maps back to (about) frame 0's stage position
    p = cv.poses[0]
    u = p["x_px"] / 4 - cv.origin[0] + (sc.w / 4 - 1) / 2
    v = p["y_px"] / 4 - cv.origin[1] + (sc.h / 4 - 1) / 2
    x, y = cv.to_stage(u, v)
    assert abs(x - p["x_um"]) < 10 and abs(y - p["y_um"]) < 10
    uu, vv = cv.to_px(x, y)
    assert abs(uu - u) < 1e-6 and abs(vv - v) < 1e-6
    assert abs(cv.um_per_px - 0.5 * 4) < 0.05
    cells = list(cv.grid(200.0))
    assert cells and all(c.image(cv).size for c in cells)


def test_calibrate_finds_matrix():
    sc = scan(nx=3, ny=3, overlap=0.4, jitter_px=1, seed=7)
    prof = ms.calibrate(sc.frames, "sim")
    assert abs(prof.um_per_px - 0.5) / 0.5 < 0.01
    assert abs(prof.rotation_deg - 0.55) < 0.15
    assert prof.mirrored
    assert prof.flat is not None and abs(prof.flat.mean() - 1) < 1e-3
