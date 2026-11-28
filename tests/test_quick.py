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


def test_map_parts_equal_the_whole(tmp_path):
    sc = scan(nx=5, ny=4, seed=41)
    prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    s = ms.Session(prof, backend="cpu", eta=False)
    for f in sc.frames:
        s.add(f)
    s.solve()
    lazy = s.render()
    whole = s.render()
    img, own = whole.image, whole.owner
    W, H = lazy.size
    assert img.shape == (H, W, 3) and lazy.shape == img.shape
    for (x0, y0, w, h) in [(0, 0, 100, 80), (W // 3, H // 4, 257, 133), (W - 50, H - 40, 50, 40)]:
        part, po = lazy.region(x0, y0, w, h, owner=True)
        assert (part == img[y0:y0 + h, x0:x0 + w]).all() and (po == own[y0:y0 + h, x0:x0 + w]).all()
    # cells and saved tiles come out the same as cuts of the whole map
    for c in list(lazy.grid(150.0))[:5]:
        x0, y0, w, h = c.px
        assert (c.image(lazy) == img[y0:y0 + h, x0:x0 + w]).all()
    from PIL import Image
    lazy.save(tmp_path / "m", tile=300)
    import json
    man = json.loads((tmp_path / "m" / "manifest.json").read_text())
    for t in man["tiles"][:6]:
        a = np.asarray(Image.open(tmp_path / "m" / t["file"]))
        assert a.shape[2] == 4
        cut = img[t["y"]:t["y"] + a.shape[0], t["x"]:t["x"] + a.shape[1]]
        assert (a[..., :3] == cut).all()
        # transparent exactly where no frame reaches
        o = own[t["y"]:t["y"] + a.shape[0], t["x"]:t["x"] + a.shape[1]]
        assert ((a[..., 3] == 0) == (o < 0)).all()


def test_objects_are_whole_from_one_frame():
    """Crystals and dust on the map: each object (with its margin) comes from a single frame, the seams run
    around them; the same map with plain nearest-centre seams cuts some of them."""
    sc = scan(nx=5, ny=4, seed=7)
    prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    s = ms.Session(prof, mode="quality", backend="cpu", eta=False)
    for f in sc.frames:
        s.add(f)
    s.solve()
    cv = s.render(seams="objects")
    own = cv.owner
    ents = s.entities
    boxes = ents.boxes()
    assert len(ents) > 10 and ents.whole > 0.8 * len(ents)
    ox, oy = int(round(cv.origin[0])), int(round(cv.origin[1]))
    cut_objects = cut_voronoi = 0
    vor = s.render(seams="voronoi").owner
    for i, (x0, y0, w, h, o) in enumerate(boxes):
        if o < 0:
            continue
        m = ents.mask(i).astype(bool)
        sub = own[y0 - oy:y0 - oy + h, x0 - ox:x0 - ox + w][m]
        cut_objects += int((sub != o).any())
        cut_voronoi += int(len(set(vor[y0 - oy:y0 - oy + h, x0 - ox:x0 - ox + w][m].tolist())) > 1)
    print(len(ents), "objects,", ents.whole, "whole,", ents.too_big, "too big; cut by voronoi seams:", cut_voronoi)
    assert cut_objects == 0
    assert cut_voronoi > 0


def test_metrics():
    sc = scan(nx=4, ny=3, seed=5)
    prof = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    s = ms.Session(prof, mode="quality", backend="cpu", eta=False)
    for f in sc.frames:
        s.add(f)
    s.solve()
    s.render()
    m = s.metrics()
    print(m)
    assert m["frames"] == 12 and m["pairs_used"] >= 15
    assert m["pair_rms_px"] < 0.5 and m["ncc_median"] > 0.5
    assert 0 <= m["seam_step_pct"] < 5 and 0 <= m["background_sigma_pct"] < 10
    assert m["objects_whole"] + m["objects_cut"] == m["objects"]
