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
        out[b] = dict(t=rep.edges["t"].tolist(), ncc=rep.edges["ncc"].tolist(), found=rep.edges["found"].tolist(),
                      pos=rep.positions_px.tolist(), backend=rep.stats["backend"], note=rep.stats["backend_note"])
    print(json.dumps(out))
""")


def run_mock(nx, ny, mode="quick", max_slots=None):
    mock = os.environ.get("MSTITCH_MOCK_CUDA")
    if not mock or not Path(mock).exists():
        pytest.skip("build tests/mock_cuda (-DMSTITCH_MOCK_CUDA=ON) and set MSTITCH_MOCK_CUDA")
    env = dict(os.environ)
    if max_slots:
        env["MSTITCH_GPU_MAX_SLOTS"] = str(max_slots)
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
    print(g.stats["backend_note"])
    # cuFFT and pocketfft round differently: allow a rare different peak order, demand the same placement
    assert (c.edges["found"] == g.edges["found"]).mean() > 0.98
    d = (g.positions_px - c.positions_px) * 2 ** s.level
    assert np.abs(d - d.mean(axis=0)).max() < 0.05
