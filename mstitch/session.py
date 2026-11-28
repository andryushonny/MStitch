"""Stitching session: frames from MControl in, report + canvas out."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import _core
from . import cuda as _cuda
from .canvas import Canvas
from .eta import Eta, Estimate, plan_features
from .profile import Profile

MODES = {"quick": 2, "quality": 1}   # working level per mode (2 = quarter, 1 = half resolution)


def _meta_args(meta: dict) -> dict:
    fmt = meta.get("fmt", "")
    if fmt == "jpeg":
        raise ValueError("a preview frame (fmt jpeg) cannot be stitched: use Microscope.snap() frames")
    xyz = meta.get("xyz_um") or (0.0, 0.0, 0.0)
    return dict(w=int(meta["w"]), h=int(meta["h"]), bits=int(meta.get("bits", 8)), fmt=fmt,
                codec=meta.get("codec", "raw"), sx=float(xyz[0]), sy=float(xyz[1]), moving=bool(meta.get("moving")))


@dataclass
class Report:
    """What solve() found. Positions are full-resolution pixels in the stage-aligned frame of the profile."""
    positions_px: np.ndarray      # Nx2: solved top-left of each frame (working level)
    predicted_px: np.ndarray      # Nx2: where the stage says it is (working level)
    edges: dict                   # i, j, pred, t, ncc, overlap, found, used, resid
    stats: dict
    poses: list                   # one dict per frame
    level: int

    def reshoot(self) -> list[dict]:
        """Frames to take again in the second pass: where to go (um) and why."""
        return [p["reshoot"] for p in self.poses if p["reshoot"]]

    def summary(self) -> str:
        n = len(self.poses)
        bad = sum(1 for p in self.poses if p["reshoot"])
        return (f"{n} frames, pairs used {self.stats['pairs_used']} (dropped {self.stats['pairs_dropped']}), "
                f"rms {self.stats['rms_px'] * 2 ** self.level:.2f} px full-res, to reshoot: {bad}, "
                f"registration on {self.stats.get('backend', '?')}")


class Session:
    """Collect frames, then solve() and render().

    >>> s = Session(Profile.load("profiles/10x.json"), mode="quick")
    >>> for x, y in plan: s.add(m.snap(x=x, y=y))
    >>> rep = s.solve(); cv = s.render(); cv.save("scans/chip7")
    """

    def __init__(self, profile: Profile, mode: str = "quick", level: int | None = None, threads: int = 0,
                 backend: str = "auto", eta: Eta | bool = True):
        """backend: 'auto' (GPU when there is one, else CPU), 'cuda' (GPU or an error), 'cpu'.
        eta: the timing model this session learns into and predicts from (True: the user's default file,
        False: none)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}")
        if backend not in ("auto", "cuda", "cpu"):
            raise ValueError("backend must be auto, cuda or cpu")
        if backend != "cpu":
            _cuda.load()
            _core.cuda_device()      # start the GPU and compile the kernels now, not in the first solve()
        self.backend = backend
        self.profile = profile
        self.mode = mode
        self.level = MODES[mode] if level is None else level
        self._s = _core.Session(self.level, list(profile.m), threads)
        self.report: Report | None = None
        self.eta: Eta | None = Eta() if eta is True else (eta or None)
        self.timings: dict = {}           # seconds of the last solve() / render(), by phase
        self.render_times: dict = {}      # the last render(): flat-field estimate, map, the rest
        self.flat: np.ndarray | None = None   # flat-field estimated from this scan (when the profile has none)
        self.flat_level = self.level
        self.entities = None              # objects kept whole by the last render() (seams='objects')
        self._last_flat = None

    def __len__(self):
        return len(self._s)

    # --- frames in
    def add(self, frame=None, *, meta: dict | None = None, data: bytes | None = None, row: int = -1,
            col: int = -1) -> int:
        """An MControl Frame (from Microscope.snap / fetch), or its meta + data. Returns the frame index.
        Returns at once: unpacking and preparation run on all cores."""
        if frame is not None:
            meta, data = frame.meta, frame.data
        a = _meta_args(meta)
        return self._s.add_sensor(bytes(data), a["w"], a["h"], a["bits"], a["fmt"], a["codec"], a["sx"], a["sy"],
                                  a["moving"], row, col)

    def add_image(self, image: np.ndarray, x_um: float, y_um: float, moving: bool = False, row: int = -1,
                  col: int = -1) -> int:
        """A plain image (HxW or HxWx3, uint8/uint16, full resolution) taken with the stage at (x, y) um."""
        return self._s.add_image(np.ascontiguousarray(image), float(x_um), float(y_um), moving, row, col)

    # --- solve
    def solve(self, search_um: float | None = None, min_ncc: float = 0.3, min_overlap: float = 0.03,
              anchors_um=None, anchor_radius_um: float | None = None, anchor_weight: float = 4.0) -> Report:
        """Register overlapping pairs, then place all frames. search_um: how far a frame may be from where the
        stage says (default 6 x repeatability).
        anchors_um: control points, Nx2 stage um: lithography crosses whose true places are known (from the
        drawing, through the lattice). Each is looked for in every frame that holds it (within anchor_radius_um,
        default 3 x repeatability) and pinned there, strongly (anchor_weight against a perfect pair's 1): drift
        along the map cannot pile up between crosses. One found far off the others' consensus is dropped."""
        s = self._s
        t0 = time.perf_counter()
        s.wait()
        t1 = time.perf_counter()
        n = len(s)
        infos = [s.tile_info(k) for k in range(n)]
        scale = 2 ** self.level
        if search_um is None:
            search_um = 6 * self.profile.repeatability_um
        search_px = max(4.0, search_um / self.profile.um_per_px / scale)
        s.register(search_px, True, 8, min_overlap, self.backend)
        t2 = time.perf_counter()
        stats_backend = {"backend": s.backend_used, "backend_note": s.backend_note}
        pos, stats, edges = s.solve(1e-3, min_ncc, 1.5, 5)
        if anchors_um is not None and len(anchors_um):
            pos, stats, edges = self._anchored(pos, stats, edges, np.asarray(anchors_um, float), infos, min_ncc,
                                               anchor_radius_um, anchor_weight)
        t3 = time.perf_counter()
        self.timings = {"prepare_wait": t1 - t0, "register": t2 - t1, "solve": t3 - t2}
        stats.update(stats_backend)
        pred = s.predicted()
        self.report = Report(pos, pred, edges, stats, self._poses(infos, pos, pred, edges, min_ncc), self.level)
        return self.report

    def _anchored(self, pos, stats, edges, targets_um, infos, min_ncc, radius_um, weight):
        from .crosses import refine
        s = self._s
        scale = 2 ** self.level
        m = self.profile.matrix()
        tpx = (targets_um @ m.T) / scale              # where each cross must be (working level, absolute)
        r_um = radius_um if radius_um is not None else max(3.0 * self.profile.repeatability_um, 5.0)
        r_px = max(6, int(round(r_um / self.profile.um_per_px / scale)))
        greys = {}
        rows = []
        for t in tpx:
            for k, inf in enumerate(infos):
                if not inf["ready"]:
                    continue
                u, v = t - pos[k]                         # in the frame, as placed by the solve so far
                if u < r_px + 2 or v < r_px + 2 or u > inf["w"] - r_px - 3 or v > inf["h"] - r_px - 3:
                    continue
                if k not in greys:
                    greys[k] = s.tile_rgb(k).astype(np.float32).mean(axis=2)
                c = refine(greys[k], u, v, r_px)
                if c is not None:
                    rows.append([k, c[0], c[1], t[0], t[1], weight])
        before = None
        if rows:
            a = np.array(rows)
            # how far the crosses sat from their places before pinning (the drift the pins take out)
            d0 = pos[a[:, 0].astype(int)] + a[:, 1:3] - a[:, 3:5]
            d0 -= np.median(d0, axis=0)
            before = float(np.sqrt((d0 ** 2).sum(axis=1).mean())) * scale
            for _ in range(2):
                pos, stats, edges = s.solve(1e-3, min_ncc, 1.5, 5, np.ascontiguousarray(a))
                r = np.linalg.norm(pos[a[:, 0].astype(int)] + a[:, 1:3] - a[:, 3:5], axis=1)
                lim = max(1.0 / scale, 5 * 1.4826 * float(np.median(r)))
                keep = r <= lim
                if keep.all() or keep.sum() < 2:
                    break
                a = a[keep]
            r = np.linalg.norm(pos[a[:, 0].astype(int)] + a[:, 1:3] - a[:, 3:5], axis=1) * scale
            # the camera's scale against the drawing: two crosses in one frame, their distance in the frame against
            # their distance by the drawing (what pins cannot fix: a frame is only moved, not stretched)
            ratios = []
            for k in np.unique(a[:, 0]):
                q = a[a[:, 0] == k]
                for i in range(len(q)):
                    for j in range(i + 1, len(q)):
                        dt = np.hypot(*(q[i, 3:5] - q[j, 3:5]))
                        if dt > 20:
                            ratios.append(np.hypot(*(q[i, 1:3] - q[j, 1:3])) / dt)
            if ratios:
                stats["frame_scale_ppm"] = float((np.median(ratios) - 1) * 1e6)
            stats.update(anchors_used=int(len(a)), anchors_found=len(rows), anchor_rms_px=float(np.sqrt((r ** 2).mean())),
                         anchor_drift_before_px=before)
        else:
            stats.update(anchors_used=0, anchors_found=0)
        stats["anchors_given"] = int(len(targets_um))
        return pos, stats, edges

    def _poses(self, infos, pos, pred, edges, min_ncc):
        scale = 2 ** self.level
        minv = np.linalg.inv(self.profile.matrix())
        fov = np.array(self.profile.frame, float)
        poses = []
        for k, inf in enumerate(infos):
            mine = (edges["i"] == k) | (edges["j"] == k)
            used = mine & edges["used"].astype(bool)
            best = float(edges["ncc"][used].max()) if used.any() else 0.0
            d_px = (pos[k] - pred[k]) * scale
            d_um = minv @ d_px
            reason = None
            if not inf["ready"]:
                reason = "error: " + inf["error"]
            elif inf["moving"]:
                reason = "taken while the stage was moving"
            elif mine.any() and not used.any():
                reason = "no reliable overlap with any neighbour"
            elif mine.any() and best < max(min_ncc, 0.5):
                reason = f"weak overlap (best NCC {best:.2f})"
            reshoot = None
            if reason:
                # Step a quarter frame towards the neighbours that failed: the same place seen from another spot.
                bad = mine & ~edges["used"].astype(bool)
                off = np.zeros(2)
                if bad.any():
                    sign = np.where(edges["i"][bad] == k, 1.0, -1.0)[:, None]
                    v = (edges["pred"][bad] * sign).mean(axis=0)
                    if np.linalg.norm(v) > 0:
                        off = minv @ (v / np.linalg.norm(v) * 0.25 * fov.min())
                reshoot = {"index": k, "x_um": inf["sx"] + off[0], "y_um": inf["sy"] + off[1], "reason": reason}
            poses.append({"index": k, "row": inf["row"], "col": inf["col"], "x_um": inf["sx"], "y_um": inf["sy"],
                          "x_px": float(pos[k][0] * scale), "y_px": float(pos[k][1] * scale),
                          "dx_px": float(d_px[0]), "dy_px": float(d_px[1]),
                          "dx_um": float(d_um[0]), "dy_um": float(d_um[1]),
                          "pairs": int(used.sum()), "best_ncc": round(best, 4), "reshoot": reshoot})
        return poses

    # --- canvas out
    def render(self, flat: np.ndarray | None | str = "auto", seams: str = "objects", blend_px: float = 16.0) -> Canvas:
        """The map at the working level, rendered on demand (see Canvas). flat: 'auto' = the profile's, else
        estimated from these frames (needs >= 8 frames); None = no correction; or an HxWx3 gain.
        seams: 'objects' = objects on the sample (crystals, crosses, flakes) are found and each is taken whole
        from the one frame that holds it, nearest its centre: seams run only through the quiet background and
        no object is cut or pieced together; 'voronoi' = each pixel from the frame whose centre is nearest.
        blend_px: near a seam, in the background, frames are blended over about this many pixels (working level),
        so a small brightness difference fades instead of stepping; objects are never blended. 0: hard seams."""
        if seams not in ("objects", "voronoi"):
            raise ValueError("seams must be objects or voronoi")
        if self.report is None:
            self.solve()
        t_render = time.perf_counter()
        s, rep = self._s, self.report
        n = len(s)
        idx = list(range(n))
        infos = [s.tile_info(k) for k in idx]
        ready = [k for k in idx if infos[k]["ready"]]
        t_flat = time.perf_counter()
        if isinstance(flat, str):
            shape = (infos[ready[0]]["h"], infos[ready[0]]["w"]) if ready else (0, 0)
            flat = self.profile.flat_at(self.level, shape)
            if flat is None:
                # estimated once per session from up to 24 frames spread over the scan (a pixel's median over
                # 24 frames ignores dust and crystals that cover it in fewer than half of them):
                # save it into the profile (`profile.flat = s.flat`) to skip this next time
                if self.flat is None and len(ready) >= 8:
                    pick = np.linspace(0, len(ready) - 1, min(24, len(ready))).round().astype(int)
                    self.flat = s.estimate_flat([ready[i] for i in sorted(set(pick))])
                    self.flat_level = self.level
                flat = self.flat
        t_box = time.perf_counter()
        pos = np.ascontiguousarray(rep.positions_px, dtype=np.float64)
        flat32 = None if flat is None else np.ascontiguousarray(flat, np.float32)
        self._last_flat = flat32
        # the map's box: whole pixels around all frames (as core/render.cpp computes it)
        if ready:
            x0 = float(np.floor(min(pos[k][0] for k in ready)))
            y0 = float(np.floor(min(pos[k][1] for k in ready)))
            W = int(np.ceil(max(pos[k][0] + infos[k]["w"] for k in ready) - x0))
            H = int(np.ceil(max(pos[k][1] + infos[k]["h"] for k in ready) - y0))
        else:
            x0 = y0 = 0.0
            W = H = 0
        rects = [(pos[k][0] - x0, pos[k][1] - y0, infos[k]["w"], infos[k]["h"]) for k in ready]
        t_obj = time.perf_counter()
        self.entities = s.find_entities(idx, pos, flat32) if seams == "objects" and ready else None
        self.render_times_objects = time.perf_counter() - t_obj
        self.render_backend, self.render_note = None, ""

        def renderer(rx, ry, rw, rh):
            t = time.perf_counter()
            img, owner, _, _, used, note = s.render(idx, pos, flat32, float(rx), float(ry), int(rw), int(rh),
                                                    backend=self.backend, entities=self.entities,
                                                    blend_px=float(blend_px))
            self.render_backend, self.render_note = used, note
            self.render_times["map"] = self.render_times.get("map", 0.0) + time.perf_counter() - t
            return img, owner

        cv = Canvas(renderer, (x0, y0), (W, H), self.level, self._stage_from_px(ready, pos, x0, y0), rep.poses,
                    rects)
        t_end = time.perf_counter()
        self.render_times = {"flat": t_box - t_flat, "objects": self.render_times_objects, "map": 0.0,
                             "other": (t_flat - t_render) + (t_end - t_box) - self.render_times_objects}
        self.timings["render"] = t_end - t_render       # until the map can be used; pixels come on demand
        self._learn(ready, (H, W))
        return cv

    # --- how well it went
    def metrics(self) -> dict:
        """Numbers to watch, scan after scan (MControl's Summary keeps their history and limits). After render().
        Pixels are full resolution.
          pair_rms_px / pair_p95_px / pair_max_px: how far each overlapping pair stays from its measured offset
            after the solve (the stitching's own consistency);
          ncc_median: how alike overlaps are (texture, focus);
          stage_dev_rms_um: how far frames landed from where the stage said, the common offset taken out
            (the stage's repeatability as seen);
          objects / objects_whole / objects_cut: objects kept whole from one frame;
          seam_step_pct: brightness difference of two frames over their overlap (flat field, exposure);
          background_sigma_pct: unevenness of each frame's background after the flat field (vignetting left)."""
        rep, s = self.report, self._s
        if rep is None:
            raise RuntimeError("solve() and render() first")
        scale = 2 ** self.level
        e = rep.edges
        used = e["used"].astype(bool)
        res = e["resid"][used] * scale
        m: dict = {"frames": len(rep.poses), "level": self.level, "pairs_used": int(used.sum()),
                   "pairs_dropped": int((~used & e["found"].astype(bool)).sum()),
                   "reshoot": sum(1 for p in rep.poses if p["reshoot"])}
        if res.size:
            m.update(pair_rms_px=float(np.sqrt((res ** 2).mean())), pair_p95_px=float(np.percentile(res, 95)),
                     pair_max_px=float(res.max()), ncc_median=float(np.median(e["ncc"][used])))
        for k in ("anchors_given", "anchors_found", "anchors_used", "anchor_rms_px", "anchor_drift_before_px",
                  "frame_scale_ppm"):
            if rep.stats.get(k) is not None:
                m[k] = rep.stats[k]
        d = np.array([[p["dx_um"], p["dy_um"]] for p in rep.poses])
        if len(d) >= 2:
            dev = d - np.median(d, axis=0)
            m["stage_dev_rms_um"] = float(np.sqrt((dev ** 2).sum(axis=1).mean()))
        if self.entities is not None:
            m.update(objects=len(self.entities), objects_whole=int(self.entities.whole),
                     objects_cut=int(self.entities.too_big))
        n = len(s)
        ready = [k for k in range(n) if s.tile_info(k)["ready"]]
        flat = self._last_flat
        # each frame's background: medians of 8 x 6 blocks (an object in a block does not move its median)
        sig = []
        views = {}
        for k in ready[:: max(1, len(ready) // 40)]:
            v = s.tile_rgb(k).astype(np.float32)
            if flat is not None and flat.shape == v.shape:
                v = v / np.maximum(flat, 0.05)
            g = v.mean(axis=2)
            views[k] = g
            h, w = g.shape
            bh, bw = h // 6, w // 8
            med = np.array([[np.median(g[j * bh:(j + 1) * bh, i * bw:(i + 1) * bw]) for i in range(8)] for j in range(6)])
            if med.mean() > 1:
                sig.append(med.std() / med.mean())
        if sig:
            m["background_sigma_pct"] = float(100 * np.median(sig))
        # brightness step over overlaps: medians of both frames over their common area
        pos = rep.positions_px
        steps = []
        idx = np.flatnonzero(used)
        for q in idx[:: max(1, len(idx) // 60)]:
            i, j = int(e["i"][q]), int(e["j"][q])
            gi = views.get(i)
            gj = views.get(j)
            if gi is None:
                gi = views[i] = self._grey(i, flat)
            if gj is None:
                gj = views[j] = self._grey(j, flat)
            dx, dy = (np.round(pos[j] - pos[i])).astype(int)
            h, w = gi.shape
            x0, x1, y0, y1 = max(0, dx), min(w, w + dx), max(0, dy), min(h, h + dy)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            a = np.median(gi[y0:y1, x0:x1])
            b = np.median(gj[y0 - dy:y1 - dy, x0 - dx:x1 - dx])
            if a + b > 2:
                steps.append(abs(a - b) / ((a + b) / 2))
        if steps:
            m["seam_step_pct"] = float(100 * np.median(steps))
        m["timings_s"] = {k: round(float(v), 3) for k, v in {**self.timings, **self.render_times}.items()}
        return m

    def _grey(self, k, flat):
        v = self._s.tile_rgb(k).astype(np.float32)
        if flat is not None and flat.shape == v.shape:
            v = v / np.maximum(flat, 0.05)
        return v.mean(axis=2)

    # --- how long it takes (eta.py)
    def _key(self, backend: str) -> str:
        inf = self._s.tile_info(0) if len(self._s) else {"full_w": self.profile.frame[0],
                                                          "full_h": self.profile.frame[1]}
        return Eta.setup_key(self.mode, backend, (inf["full_w"], inf["full_h"]), Eta.device_name(backend))

    def _learn(self, ready, shape) -> None:
        if self.eta is None or not {"register", "solve", "render"} <= self.timings.keys():
            return
        preps = [self._s.tile_info(k)["prep_ms"] for k in ready]
        prep = float(np.median(preps)) / 1e3 / max(1, self._s.threads) if preps else None
        feats = {"frames": len(ready), "pairs": int(len(self.report.edges["i"])),
                 "map_mpx": shape[0] * shape[1] / 1e6}
        self.eta.record(self._key(self.report.stats["backend"]), feats, self.timings, prep)

    def eta_remaining(self) -> Estimate | None:
        """Seconds still needed if the frames added so far were the whole scan: frames still being prepared +
        registration + solve + render. None until this setup has been timed min_runs times."""
        if self.eta is None or not len(self._s):
            return None
        backend = self.backend
        if backend == "auto":
            backend = "cuda" if _cuda.info()["ok"] else "cpu"
        xy = [(i["sx"], i["sy"]) for i in (self._s.tile_info(k) for k in range(len(self._s)))]
        f = plan_features(self.profile, xy, self.mode, self.level)
        return self.eta.predict_features(self._key(backend), f, pending_frames=self._s.pending())

    def _stage_from_px(self, ready, pos, ox, oy):
        """Affine canvas pixel -> stage um, fitted on frame centres (falls back to the profile matrix)."""
        infos = [self._s.tile_info(k) for k in ready]
        uv = np.array([[pos[k][0] - ox + 0.5 * (i["w"] - 1), pos[k][1] - oy + 0.5 * (i["h"] - 1)]
                       for k, i in zip(ready, infos)])
        xy = np.array([[i["sx"], i["sy"]] for i in infos])
        if len(ready) >= 3:
            sv = np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)
            if sv[1] > 0.05 * sv[0] and sv[1] > 10:      # frames spread in 2-D: fit, else the profile decides
                sol, *_ = np.linalg.lstsq(np.c_[uv, np.ones(len(uv))], xy, rcond=None)
                return sol.T
        minv = np.linalg.inv(self.profile.matrix()) * 2 ** self.level
        b = xy.mean(axis=0) - minv @ uv.mean(axis=0) if len(ready) else np.zeros(2)
        return np.c_[minv, b]
