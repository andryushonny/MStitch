# MStitch

Stitching for **MControl** scans: frames and stage positions in, one large map out, with the shift of every frame from where the stage said it was. C++ core, a Python module loaded into the MControl operator process (no extra server); CUDA backend next.

* **Any set of frames.** Pairs are found from the stage positions, not from a grid index; crosses are not needed.
* **Exact input.** Takes MControl snaps as they come (Bayer, `lossless` or `raw`), unpacks them itself in C++ on all cores; `add()` returns at once.
* **Two modes.** `quick` works at 1/4 size for maps of 30×30 … 58×58 frames; `quality` works at 1/2 size for small areas (≤ 15×15).
* **No double images.** Every map pixel comes from one frame (the one whose centre is nearest), so a residual error shows as a step on a seam, never as a ghost.
* **Two passes.** After the first pass `solve()` lists the frames to take again (no reliable overlap, taken while moving, …) with a spot a quarter frame aside; the second pass adds them as extra frames.
* **Profile per objective** (4x, 10x, 20x, 50x, 100x): stage→pixel matrix (scale, rotation, mirroring), repeatability, flat-field. MControl's `prepare()` takes the frames, `mstitch.calibrate()` computes it.

## Quick start (Windows, operator PC)

Needs Python ≥ 3.10 and the *MSVC Build Tools* (C++). CMake, zstd and pocketfft are fetched by the build.
```
uv add "mstitch @ git+https://github.com/andryushonny/MStitch"
```
```python
import mstitch as ms
from mcontrol import Microscope

with Microscope() as m:
    prof = ms.Profile.load("profiles/10x.json")       # from prepare(), see below
    s = ms.Session(prof, mode="quick")
    for x, y in plan:                                 # approach every frame from the same side
        s.add(m.snap(x=x, y=y, settle_ms=30))
    rep = s.solve()
    print(rep.summary())
    for t in rep.reshoot():                           # second pass: only what needs it
        s.add(m.snap(x=t["x_um"], y=t["y_um"], settle_ms=30))
    rep = s.solve()
    cv = s.render()                                   # Canvas
    cv.save("scans/chip7")                            # manifest.json + poses.csv + tiles/*.png
    u, v = cv.to_px(x_um, y_um); x, y = cv.to_stage(u, v)
    for cell in cv.grid(200.0): ...                   # cut the map into ~200 um cells
```

### Profile: `prepare()` in MControl, `calibrate()` here
```python
def prepare(m, objective, step_um, n=3, back_um=20):
    x0, y0, _ = m.stage.position
    frames = []
    for j in range(n):
        for i in range(n):
            x, y = x0 + i * step_um, y0 + j * step_um
            m.stage.goto(x - back_um, y - back_um)    # one-side approach: no backlash in the data
            frames.append(m.snap(x=x, y=y, settle_ms=50))
    prof = ms.calibrate(frames, objective)            # step: 50-70 % of the field, any textured spot
    prof.save(f"profiles/{objective}.json")
    return prof
```
The rotation and scale come out as accurate as the stage moves: error ≈ repeatability / (step · √pairs). Use a 5×5 grid if the stage is noisy.

## Python API

| | |
|---|---|
| `Profile.guess(objective, um_per_px, rotation_deg, flip_x, flip_y)` | nominal profile before calibration |
| `Profile.load(path)`, `.save(path)` | `name.json` + `name.flat.npy` |
| `calibrate(frames, objective)` | frames: MControl `Frame`s or `(image, x_um, y_um)` |
| `Session(profile, mode="quick"\|"quality")` | `add(frame)`, `add_image(img, x_um, y_um)`, `solve()`, `render()` |
| `Report` | `summary()`, `reshoot()`, `poses` (per frame: position, shift from the stage in px and um, pairs, best NCC), `edges`, `stats` |
| `Canvas` | `image` (H×W×3 uint8), `owner` (frame per pixel), `to_stage(u, v)`, `to_px(x, y)`, `read(x, y, w, h)`, `grid(cell_um)`, `save(folder)` |

Command line, for scans saved with `Frame.save()`:
```
python -m mstitch calibrate FOLDER --objective 10x --out profiles/10x.json
python -m mstitch stitch FOLDER --profile profiles/10x.json --mode quick --out scans/chip7
```

## How it works

1. **Unpack** (C++, all cores): zstd → undo MControl's same-colour difference → 2×2 Bayer superpixel (exact, no interpolation) → 1/2 or 1/4 size.
2. **Prepare**: grey divided by its local mean (removes vignetting and illumination), zero mean.
3. **Pairs**: every two frames whose stage positions overlap. Each frame's FFT is computed once and freed after its last pair (as in Blattner et al. 2014).
4. **Register**: phase correlation gives candidate shifts (8 peaks × 4 wrap-arounds); candidates must agree with the stage within `search_um` (default 6 × repeatability); the normalized cross-correlation (NCC) chooses, climbs to the integer maximum and a parabola gives the sub-pixel shift.
5. **Solve**: weighted least squares over all pairs (weight NCC²) with a weak pull to the stage positions; pairs that disagree with the consensus are dropped and the solve repeats.
6. **Render**: each map pixel from the nearest frame centre, bilinear, flat-field corrected (the profile's, or estimated from the scan's own frames).

## Accuracy and speed (synthetic scans, known truth)

| | error rms (full-res px) |
|---|---|
| quick, 1/4 size | 0.05 |
| quality, 1/2 size | 0.007 |

Synthetic frames carry vignetting, sensor noise, dust and crystals and a stage that lands ±20 px off target; real frames will be worse by whatever the optics add (distortion: next milestone). Time per stage on full 2592×1944 frames: `python bench/bench.py` (prints ms per frame and per pair and the estimate for 30×30 and 58×58). On a 2-core cloud VM, CPU only: quick ≈ 15 ms per frame and 7 ms per pair; the CUDA backend is there to cut the per-pair part.

## Layout

| folder | |
|---|---|
| `io/` | frames from MControl: unpacking, Bayer → RGB at a level |
| `core/` | tiles, pair registration, global solve, render, session, thread pool |
| `cpu/` | FFT on the CPU (pocketfft) |
| `cuda/` | CUDA backend (planned: see `cuda/README.md`) |
| `features/` | automatic search of features for the quality mode (planned) |
| `bindings/` | the Python module `mstitch._core` (nanobind) |
| `mstitch/` | the Python package |
| `tests/`, `bench/` | synthetic scans packed exactly like MControl; benchmarks |

## Build and test
```
uv run --with pytest --with zstandard pytest      # or: pip install -e .[test] && pytest
cmake -S . -B build -DMSTITCH_PYTHON=ON && cmake --build build   # C++ only
```
