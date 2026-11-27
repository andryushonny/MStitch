"""MStitch: stitching of MControl tile scans (C++ core, CPU and CUDA).

    import mstitch as ms
    prof = ms.Profile.load("profiles/10x.json")       # made by MControl's prepare() via ms.calibrate()
    s = ms.Session(prof, mode="quick")
    for x, y in plan:
        s.add(m.snap(x=x, y=y))                       # MControl frames, lossless
    rep = s.solve()
    for t in rep.reshoot():                           # second pass: only the frames that need it
        s.add(m.snap(x=t["x_um"], y=t["y_um"]))
    rep = s.solve()
    cv = s.render()
    cv.to_stage(u, v); cv.to_px(x_um, y_um); cv.read(x, y, w, h); cv.grid(cell_um); cv.save("scans/chip7")
"""
import sys as _sys

try:
    from . import _core  # noqa: F401
except ImportError as _e:
    if _sys.platform == "win32":
        from pathlib import Path as _Path

        from ._dlls import explain as _explain
        raise ImportError(f"{_e}\n{_explain(_Path(__file__).parent)}") from _e
    raise

from . import cuda
from .calibrate import calibrate
from .canvas import Canvas, Cell
from .eta import Estimate, Eta, plan_features
from .profile import Profile
from .session import Report, Session

__version__ = "0.5.1"
__all__ = ["cuda", "Canvas", "Cell", "Estimate", "Eta", "plan_features", "Profile", "Report", "Session", "calibrate", "__version__"]
