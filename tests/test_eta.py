import numpy as np

import mstitch as ms
from synth import scan


def prof(sc):
    return ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)


def test_plan_features_match_the_session():
    sc = scan(nx=5, ny=4, seed=21)
    p = prof(sc)
    xy = [f.meta["xyz_um"][:2] for f in sc.frames]
    f = ms.plan_features(p, xy, "quick")
    s = ms.Session(p, eta=False)
    for fr in sc.frames:
        s.add(fr)
    rep = s.solve()
    cv = s.render()
    assert f["frames"] == 20 and f["pairs"] == len(rep.edges["i"])
    assert abs(f["map_mpx"] - cv.image.shape[0] * cv.image.shape[1] / 1e6) / f["map_mpx"] < 0.1


def test_linear_model_learns_and_waits_for_two_runs(tmp_path):
    eta = ms.Eta(tmp_path / "t.json")
    key = "quick|cuda|2592x1944|test"
    law = lambda f: {"register": 0.02 + 0.001 * f["pairs"], "solve": 0.001 + 1e-5 * f["pairs"],  # noqa: E731
                     "render": 0.05 + 0.08 * f["map_mpx"]}
    rng = np.random.default_rng(0)
    for k in range(2):
        f = {"frames": 100 * (k + 1), "pairs": 180 * (k + 1), "map_mpx": 30.0 * (k + 1)}
        assert eta.predict_features(key, f) is None            # the first runs are only remembered
        eta.record(key, f, law(f))
    for _ in range(6):
        g = int(rng.integers(10, 60))
        f = {"frames": g * g, "pairs": 2 * g * (g - 1), "map_mpx": 0.3 * g * g}
        eta.record(key, f, {k: v * rng.uniform(0.97, 1.03) for k, v in law(f).items()})
    f = {"frames": 58 * 58, "pairs": 2 * 58 * 57, "map_mpx": 0.3 * 58 * 58}
    est = eta.predict_features(key, f)
    true = sum(law(f).values())
    assert est is not None and abs(est.seconds - true) / true < 0.08, (est, true)
    # remembered on disk
    assert ms.Eta(tmp_path / "t.json").predict_features(key, f) is not None


def test_session_learns_its_own_timing(tmp_path):
    eta = ms.Eta(tmp_path / "t.json")
    seen = []
    for k in range(3):
        sc = scan(nx=3 + k, ny=3, seed=30 + k)
        s = ms.Session(prof(sc), backend="cpu", eta=eta)
        for f in sc.frames:
            s.add(f)
        seen.append(s.eta_remaining())
        s.solve()
        s.render()
    assert seen[0] is None and seen[1] is None and seen[2] is not None
    assert seen[2].seconds > 0 and "render" in seen[2].phases
    xy = [f.meta["xyz_um"][:2] for f in sc.frames]
    assert eta.predict(prof(sc), "quick", xy, backend="cpu") is not None
