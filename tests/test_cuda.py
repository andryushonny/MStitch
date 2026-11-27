"""GPU backend. Three levels:
1. the kernels compile with NVRTC (needs only nvidia-cuda-nvrtc-cu12);
2. the GPU path's host logic against the CPU path, on a fake driver + cuFFT (tests/mock_cuda, built with
   -DMSTITCH_MOCK_CUDA=ON; path in MSTITCH_MOCK_CUDA);
3. the real GPU against the CPU (skipped without an NVIDIA GPU)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import mstitch as ms
from synth import scan

HERE = Path(__file__).parent
INFO = ms.cuda.info()
HAS_GPU = INFO["ok"] and "Mock" not in INFO.get("name", "")
HAS_NVRTC = any(line.startswith("nvrtc: ") and "not found" not in line for line in INFO["libraries"])


def prof(sc):
    return ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)


@pytest.mark.skipif(not HAS_NVRTC, reason="NVRTC not installed (pip install nvidia-cuda-nvrtc-cu12)")
@pytest.mark.parametrize("arch", [75, 86, 89])
def test_kernels_compile(arch):
    assert ms.cuda.compile_check(arch) == ""


@pytest.mark.skipif(HAS_GPU, reason="this machine has a GPU")
def test_auto_falls_back_to_cpu():
    sc = scan(nx=3, ny=2, seed=11)
    s = ms.Session(prof(sc), backend="auto")
    for f in sc.frames:
        s.add(f)
    rep = s.solve()
    assert rep.stats["backend"] == "cpu" and rep.stats["backend_note"]
    with pytest.raises(RuntimeError):
        t = ms.Session(prof(sc), backend="cuda")
        for f in sc.frames:
            t.add(f)
        t.solve()


MOCK_SCRIPT = textwrap.dedent("""
    import json, sys
    import numpy as np
    sys.path.insert(0, {here!r})
    from mstitch import _core
    import mstitch as ms
    from mstitch.cuda import _candidates
    drv, nvrtc, cufft = _candidates()
    print(_core.cuda_load([{mock!r}], nvrtc, [{mock!r}]), file=sys.stderr)
    from synth import scan
    sc = scan(nx={nx}, ny={ny}, seed=12)
    p = ms.Profile.guess("sim", 0.5, rotation_deg=0.55, flip_y=True, frame=(sc.w, sc.h), repeatability_um=4)
    out = {{}}
    for b in ("cpu", "cuda"):
        s = ms.Session(p, backend=b, mode={mode!r})
        for f in sc.frames:
            s.add(f)
        rep = s.solve()
        cv = s.render()
        W, H = cv.size
        np.save({here!r} + f"/_mock_{{b}}_part.npy", cv.region(W // 3, H // 4, 300, 200))
        np.save({here!r} + f"/_mock_{{b}}_img.npy", cv.image)
        np.save({here!r} + f"/_mock_{{b}}_owner.npy", cv.owner)
        out[b] = dict(render=s.render_backend, render_note=s.render_note, t=rep.edges["t"].tolist(), ncc=rep.edges["ncc"].tolist(), found=rep.edges["found"].tolist(),
                      pos=rep.positions_px.tolist(), backend=rep.stats["backend"], note=rep.stats["backend_note"])
    print(json.dumps(out))
""")


def run_mock(nx, ny, mode="quick", max_slots=None):
    mock = os.environ.get("MSTITCH_MOCK_CUDA")
    if not mock or not Path(mock).exists():
        pytest.skip("build tests/mock_cuda (-DMSTITCH_MOCK_CUDA=ON) and set MSTITCH_MOCK_CUDA")
    env = dict(os.environ)
    if max_slots:      # tight memory: small spectrum cache, thin map bands (frames recycled between bands)
        env["MSTITCH_GPU_MAX_SLOTS"] = str(max_slots)
        env["MSTITCH_GPU_BAND_ROWS"] = "96"
    code = MOCK_SCRIPT.format(here=str(HERE), mock=mock, nx=nx, ny=ny, mode=mode)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode,max_slots", [("quick", None), ("quality", None), ("quick", 8)])
def test_gpu_path_on_mock_matches_cpu(mode, max_slots):
    out = run_mock(9, 7, mode, max_slots)
    c, g = out["cpu"], out["cuda"]
    assert g["backend"] == "cuda", g["note"]
    print(g["note"])
    assert c["found"] == g["found"]
    np.testing.assert_allclose(g["t"], c["t"], atol=1e-6)
    np.testing.assert_allclose(g["pos"], c["pos"], atol=1e-6)
    # the map: the GPU render path (bands, frame slots, cell lists) gives the CPU's pixels exactly
    assert g["render"] == "cuda", g["render_note"]
    print(g["render_note"])
    ic, ig = np.load(HERE / "_mock_cpu_img.npy"), np.load(HERE / "_mock_cuda_img.npy")
    oc, og = np.load(HERE / "_mock_cpu_owner.npy"), np.load(HERE / "_mock_cuda_owner.npy")
    part = np.load(HERE / "_mock_cuda_part.npy")      # a part rendered alone on the "GPU"
    H, W = ic.shape[:2]
    assert (part == ic[H // 4:H // 4 + part.shape[0], W // 3:W // 3 + part.shape[1]]).all()
    for f in HERE.glob("_mock_*.npy"):
        f.unlink()
    assert ic.shape == ig.shape and (ic == ig).all() and (oc == og).all()
    if max_slots:    # tight memory: spectra had to be evicted and recomputed, and the result is the same
        assert " 0 evictions" not in g["note"]


@pytest.mark.skipif(not HAS_GPU, reason="no NVIDIA GPU")
@pytest.mark.parametrize("mode", ["quick", "quality"])
def test_real_gpu_matches_cpu(mode):
    sc = scan(nx=8, ny=6, seed=13)
    res = {}
    for b in ("cpu", "cuda"):
        s = ms.Session(prof(sc), backend=b, mode=mode)
        for f in sc.frames:
            s.add(f)
        res[b] = s.solve()
    c, g = res["cpu"], res["cuda"]
    assert g.stats["backend"] == "cuda"
    # the map from the same positions: GPU vs CPU pixels
    imgs = {}
    for b in ("cpu", "cuda"):
        s = ms.Session(prof(sc), backend=b, mode=mode)
        for f in sc.frames:
            s.add(f)
        s.solve()
        s.report.positions_px[:] = c.positions_px    # same placement for both
        cv = s.render()
        assert s.render_backend == b, s.render_note
        imgs[b] = (cv.image, cv.owner)
    (ic, oc), (ig, og) = imgs["cpu"], imgs["cuda"]
    assert ic.shape == ig.shape
    diff = np.abs(ic.astype(int) - ig.astype(int))
    assert diff.max() <= 1 and (diff > 0).mean() < 1e-3      # IEEE on both; MSVC /fp:fast may round a few
    assert (oc == og).mean() > 0.999
    print(g.stats["backend_note"])
    # cuFFT and pocketfft round differently: allow a rare different peak order, demand the same placement
    assert (c.edges["found"] == g.edges["found"]).mean() > 0.98
    d = (g.positions_px - c.positions_px) * 2 ** s.level
    assert np.abs(d - d.mean(axis=0)).max() < 0.05
