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
from . import cuda
from .calibrate import calibrate
from .canvas import Canvas, Cell
from .profile import Profile
from .session import Report, Session

__version__ = "0.2.0"
__all__ = ["cuda", "Canvas", "Cell", "Profile", "Report", "Session", "calibrate", "__version__"]
