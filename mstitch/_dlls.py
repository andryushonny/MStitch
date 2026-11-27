"""Windows: which DLLs a .pyd needs and which of them cannot be found (for a clear error instead of
'DLL load failed while importing _core')."""
from __future__ import annotations

import struct
from pathlib import Path


def imports(path: str | Path) -> list[str]:
    """Names of the DLLs a PE file (dll / pyd / exe) imports."""
    d = Path(path).read_bytes()
    pe = struct.unpack_from("<I", d, 0x3C)[0]
    if d[pe:pe + 4] != b"PE\0\0":
        raise ValueError("not a PE file")
    nsec, = struct.unpack_from("<H", d, pe + 6)
    opt_size, = struct.unpack_from("<H", d, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", d, opt)
    ddir = opt + (112 if magic == 0x20B else 96)          # PE32+ / PE32
    imp_rva, _ = struct.unpack_from("<II", d, ddir + 8)   # data directory 1: imports
    secs = []
    for i in range(nsec):
        s = opt + opt_size + 40 * i
        vsize, va, rsize, raw = struct.unpack_from("<IIII", d, s + 8)
        secs.append((va, max(vsize, rsize), raw))

    def off(rva):
        for va, size, raw in secs:
            if va <= rva < va + size:
                return rva - va + raw
        raise ValueError("bad RVA")

    out = []
    p = off(imp_rva)
    while True:
        name_rva = struct.unpack_from("<I", d, p + 12)[0]
        if name_rva == 0:
            break
        q = off(name_rva)
        out.append(d[q:d.index(b"\0", q)].decode("ascii", "replace"))
        p += 20
    return out


def missing(path: str | Path) -> list[str]:
    """DLLs imported by `path` that Windows cannot load from the default search places."""
    import ctypes

    bad = []
    for name in imports(path):
        if name.lower().startswith(("api-ms-win-", "ext-ms-")):
            continue
        try:
            ctypes.WinDLL(name)
        except OSError:
            bad.append(name)
    return bad


def explain(pkg_dir: Path) -> str:
    pyds = sorted(pkg_dir.glob("_core*.pyd"))
    if not pyds:
        return ("mstitch._core is not built here. Inside the MStitch folder run `uv sync --extra cuda`; "
                "elsewhere install the package (`uv add \"mstitch[cuda] @ git+https://github.com/andryushonny/MStitch\"`).")
    lines = []
    for f in pyds:
        try:
            miss = missing(f)
            lines.append(f"{f.name} imports {imports(f)}; cannot load: {miss or 'none'}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"{f.name}: {e}")
    hint = ("Libraries like libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll mean it was built with MinGW; "
            "msvcp140.dll / vcruntime140*.dll mean the Visual C++ runtime is missing. "
            "Rebuild with the MSVC Build Tools: `uv sync --extra cuda --reinstall-package mstitch`.")
    return "\n".join(lines + [hint])
