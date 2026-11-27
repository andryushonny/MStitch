# MStitch

Stitching for **MControl** scans: frames and stage positions in, one large map out, with the shift of every frame from where the stage said it was. C++ core, a Python module loaded into the MControl operator process (no extra server); registration on the GPU (CUDA) and the CPU at the same time.

* **Any set of frames.** Pairs are found from the stage positions, not from a grid index; crosses are not needed.
* **Exact input.** Takes MControl snaps as they come (Bayer, `lossless` or `raw`), unpacks them itself in C++ on all cores; `add()` returns at once.
* **Two modes.** `quick` works at 1/4 size for maps of 30×30 … 58×58 frames; `quality` works at 1/2 size for small areas (≤ 15×15).
* **No double images.** Every map pixel comes from one frame (the one whose centre is nearest), so a residual error shows as a step on a seam, never as a ghost.
* **Two passes.** After the first pass `solve()` lists the frames to take again (no reliable overlap, taken while moving, …) with a spot a quarter frame aside; the second pass adds them as extra frames.
* **GPU and CPU together.** cuFFT computes the FFTs, cross-power spectra and peaks of a batch of pairs while the CPU cores refine the previous batch (Blattner et al. 2014). Nothing CUDA is needed to build; without a GPU it runs on the CPU.
* **The map on demand, on the GPU.** `render()` returns as soon as the frames are placed; each part MControl asks for (`region`, `read`, `grid` cells, `save` tiles) is rendered then, on the GPU, pixel for pixel the CPU's result. A 58×58 map never sits in RAM whole.
* **Knows how long it will take.** A linear model learns, from the user's own runs, the seconds after the last frame for a planned scan (frames, pairs, map size); MControl can show it before and during the scan. The first two runs of a setup are only remembered.
* **Profile per objective** (4x, 10x, 20x, 50x, 100x): stage→pixel matrix (scale, rotation, mirroring), repeatability, flat-field. MControl's `prepare()` takes the frames, `mstitch.calibrate()` computes it.

## Quick start (Windows, operator PC)

Needs Python ≥ 3.10 and the *Visual Studio Build Tools* with "Desktop development with C++". CMake, zstd and pocketfft are fetched by the build.

In the operator's project (the one that uses `mcontrol`):
```
uv add "mstitch[cuda] @ git+https://github.com/andryushonny/MStitch"     # [cuda]: NVRTC + cuFFT from pip
uv run python -c "import mstitch; print(mstitch.cuda.info())"           # which GPU, how much memory
```
Inside a clone of MStitch itself (development, tests, bench) do not `uv add` it; build it in place:
```
uv sync --extra cuda --extra test
uv run python -c "import mstitch; print(mstitch.cuda.info())"
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

### How long the stitching will take
```python
eta = ms.Eta()                                        # the user's timing history (%LOCALAPPDATA%\MStitch)
est = eta.predict(prof, "quick", plan)                # plan: the (x, y) um MControl will visit
print(est or "learning: no estimate for the first runs of this setup")
...
s = ms.Session(prof, "quick")                         # learns into the same history after render()
print(s.eta_remaining())                              # during the scan: frames still preparing + the rest
```
`Estimate.seconds` is the time after the last frame; `Estimate.phases` splits it (prepare backlog, register, solve, render). The model is one linear layer, `t = W·[1, frames, pairs, map Mpx]`, weights ≥ 0, fitted in closed form on the runs of the same setup (mode, backend, frame size, GPU/CPU).

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
| `Session(profile, mode="quick"\|"quality", backend="auto"\|"cuda"\|"cpu")` | `add(frame)`, `add_image(img, x_um, y_um)`, `solve()`, `render()` |
| `cuda.info()`, `cuda.compile_check(arch)` | the GPU MStitch would use (or why none); compile the kernels for `sm_<arch>` |
| `Report` | `summary()`, `reshoot()`, `poses` (per frame: position, shift from the stage in px and um, pairs, best NCC), `edges`, `stats` |
| `Eta(path=None, min_runs=2)` | `predict(profile, mode, plan_um)`, `predict_features(key, features)`, `record(...)`; `plan_features(profile, plan_um, mode)` |
| `Canvas` | `region(x, y, w, h)`, `read(x_um, y_um, w_um, h_um)`, `grid(cell_um)`, `save(folder)` render their part on demand; `image` / `owner` the whole map; `to_stage(u, v)`, `to_px(x, y)`, `covered(...)`, `size` |

Command line, for scans saved with `Frame.save()`:
```
python -m mstitch calibrate FOLDER --objective 10x --out profiles/10x.json
python -m mstitch stitch FOLDER --profile profiles/10x.json --mode quick --out scans/chip7
```

## How it works

1. **Unpack** (C++, all cores): zstd → undo MControl's same-colour difference → 2×2 Bayer superpixel (exact, no interpolation) → 1/2 or 1/4 size.
2. **Prepare**: grey divided by its local mean (removes vignetting and illumination), zero mean.
3. **Pairs**: every two frames whose stage positions overlap. Each frame's FFT is computed once and freed after its last pair (as in Blattner et al. 2014).
4. **Peaks** (GPU, or CPU): normalized cross-power spectrum → inverse FFT → the 8 highest peaks. See `cuda/README.md`.
5. **Choose** (CPU pool): each peak gives 4 candidate shifts (wrap-arounds); candidates must agree with the stage within `search_um` (default 6 × repeatability); the normalized cross-correlation (NCC) chooses, climbs to the integer maximum and a parabola gives the sub-pixel shift.
6. **Solve**: weighted least squares over all pairs (weight NCC²) with a weak pull to the stage positions; pairs that disagree with the consensus are dropped and the solve repeats.
7. **Render** (GPU, or CPU): each map pixel from the nearest frame centre, bilinear, flat-field corrected (the profile's, or estimated from the scan's own frames). On the GPU in bands of rows: only the frames a band touches are in VRAM, a cell index tells each pixel which frames can cover it.
8. **Objects whole** (`render(seams="objects")`, the default; `--seams voronoi` to switch off): crystals, crosses, flakes and dust are found in every frame (what stands out of the frame's slowly varying background by more than 6 noise sigma, specks under 9 px dropped, grown by a 6 px margin), the pieces seen in different frames are joined on the map, and each object is taken whole from the one frame that holds it and has it nearest its centre. The seams then run only through the quiet background: no object is cut or pieced together from two frames, its shape and size are those of one frame, and the distances between objects are those the solve placed (sub-pixel). Objects larger than any frame holds keep the usual seams (counted in the CLI output).

## Accuracy and speed (synthetic scans, known truth)

| | error rms (full-res px) |
|---|---|
| quick, 1/4 size | 0.05 |
| quality, 1/2 size | 0.007 |

Synthetic frames carry vignetting, sensor noise, dust and crystals and a stage that lands ±20 px off target; real frames will be worse by whatever the optics add (distortion: next milestone). The GPU path gives the same shifts as the CPU path (tests).

Time per stage on full 2592×1944 frames, CPU and GPU side by side: `python bench/bench.py` (ms per frame and per pair, estimate for 30×30 and 58×58). On a 2-core cloud VM, CPU only: quick ≈ 15 ms per frame and 9 ms per pair, quality ≈ 26 and 38 ms.

## Layout

| folder | |
|---|---|
| `io/` | frames from MControl: unpacking, Bayer → RGB at a level |
| `core/` | tiles, pair registration, global solve, render, session, thread pool |
| `cpu/` | FFT on the CPU (pocketfft) |
| `cuda/` | CUDA backend: run-time loading, NVRTC kernels, batches and spectrum cache (`cuda/README.md`) |
| `features/` | automatic search of features for the quality mode (planned) |
| `bindings/` | the Python module `mstitch._core` (nanobind) |
| `mstitch/` | the Python package |
| `tests/`, `bench/` | synthetic scans packed exactly like MControl; a fake CUDA driver for the GPU path; benchmarks |

## Build and test
```
cd tests && uv run --with pytest --with zstandard pytest     # GPU tests run when there is a GPU
cmake -S . -B build/mock -DMSTITCH_MOCK_CUDA=ON && cmake --build build/mock --target mock_cuda
MSTITCH_MOCK_CUDA=build/mock/libmock_cuda.so pytest tests     # the GPU path on a fake driver
cmake -S . -B build -DMSTITCH_PYTHON=ON && cmake --build build   # C++ only
```
