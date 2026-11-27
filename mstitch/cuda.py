"""Finding the CUDA libraries. Nothing CUDA is needed to build or install MStitch: the NVIDIA driver comes with
Windows' graphics driver, NVRTC and cuFFT come from pip (`pip install mstitch[cuda]`, or a CUDA Toolkit)."""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

from . import _core

_loaded: str | None = None


def _site_dirs() -> list[Path]:
    out = []
    for p in sys.path:
        d = Path(p) / "nvidia"
        if d.is_dir():
            out.append(d)
    return out


def _candidates() -> tuple[list[str], list[str], list[str]]:
    win = sys.platform == "win32"
    nv = _site_dirs()
    sub = "bin" if win else "lib"
    toolkit = [Path(os.environ[v]) / ("bin" if win else "lib64") for v in ("CUDA_PATH", "CUDA_HOME")
               if os.environ.get(v)]

    def find(pkg: str, pattern: str) -> list[str]:
        hits = []
        for d in [n / pkg / sub for n in nv] + toolkit:
            hits += sorted(glob.glob(str(d / pattern)), reverse=True)
        return hits

    if win:
        driver = ["nvcuda.dll"]
        nvrtc = find("cuda_nvrtc", "nvrtc64_12*_0.dll") + ["nvrtc64_120_0.dll"]
        cufft = find("cufft", "cufft64_1*.dll") + ["cufft64_11.dll"]
        for f in nvrtc + cufft:                         # NVRTC loads its builtins dll from the same folder
            if os.path.isabs(f) and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(os.path.dirname(f))
    else:
        driver = ["libcuda.so.1", "libcuda.so"]
        nvrtc = [f for f in find("cuda_nvrtc", "libnvrtc.so.1*") if ".alt." not in f] + ["libnvrtc.so.12"]
        cufft = find("cufft", "libcufft.so.1*") + ["libcufft.so.11"]
    return driver, nvrtc, cufft


def load() -> str:
    """Load the libraries once; returns what was found (a line per library)."""
    global _loaded
    if _loaded is None:
        _loaded = _core.cuda_load(*_candidates())
    return _loaded


def info() -> dict:
    """{'ok': True, 'name', 'cc', 'free_mb', 'total_mb'} or {'ok': False, 'why'}; plus 'libraries'."""
    libs = load()
    d = dict(_core.cuda_device())
    d["libraries"] = libs.strip().splitlines()
    return d


def compile_check(arch: int = 86) -> str:
    """Compile the kernels for sm_<arch> with NVRTC (no GPU needed). '' when fine."""
    load()
    return _core.cuda_compile_check(arch)
