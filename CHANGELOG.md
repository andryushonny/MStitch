# Changelog

## 0.6.5
- Windows wheels (Python 3.10–3.13) published with every release: no compiler needed to install; MControl installs and updates MStitch from here by itself.

## 0.6.4
- CLI: objects.csv next to the map: every object found (centre and size in stage um, whole from one frame or
  not), for MControl to plan the next round of a scan so that objects sit well inside frames.

## 0.6.3
- Cross finder: with a small search window (as for the drawing's crosses) the fine window was narrower than about
  two arm widths, the arm filled it and nothing stood out: no cross found. The fine window is now several arm
  widths across (the arm's width measured at half its peak in the first pass).

## 0.6.2
- anchors.csv may carry a third column, r_um: how far around each cross's place to look. MControl sends a small
  one for the drawing's crosses (7 um: the arms reach 5 um), so the numbers written next to every cross are not
  taken for it; before, the default (3 x the stage's repeatability, 15 um) took them in, and most crosses were
  found wrong and dropped.

## 0.6.1
- Seams hardly seen. Gain compensation before rendering (`render(gain="plane")`, the default; CLI `--gain
  plane|scalar|none`): every frame's brightness times exp(a + b x + c y) per colour, found from all overlaps at
  once (least squares in log, medians of small blocks so objects do not count, a weak pull to 1): frames a few %
  brighter or darker, or lit with a tilt of their own, are evened out. Then the soft seams blend wider (40 px
  default, was 16). On a synthetic scan with frames +-6 % and tilted light: the step across seams 6.2 -> 2.0 grey
  levels, as even frames with hard seams give (2.2: the texture itself). Objects still whole from one frame.
  Metric `gain_max_pct`: the largest correction.

## 0.6.0
- Control points: `solve(anchors_um=...)` (CLI: `anchors.csv` next to the frames, or `--anchors`). Lithography crosses
  whose true places are known (MControl: from the drawing, through the lattice) are looked for in every frame that
  holds them and pinned there strongly in the global solve, so drift cannot pile up along the map; one far off
  the others' consensus is dropped. Reported: crosses given / found / pinned, what is left (px), the drift before
  pinning, and `frame_scale_ppm`: the camera's scale against the drawing from crosses two by two within a frame
  (what pins cannot fix: frames are moved, not stretched; MControl corrects um/px with it).
  On a synthetic scan with the profile's scale 0.3 % off: free, the crosses drift up to 2.6 px; pinned, rms
  0.6 px (the frames' own 0.3 %), and frame_scale_ppm 3007 for 3000.
- `mstitch.crosses.refine`: the cross finder (as MControl's).

## 0.5.3
- Soft seams in the background (`render(blend_px=16)`, CLI `--blend 16`, 0 = hard): near a seam the frames are
  blended, each weighted by how much farther its centre is than the nearest (and fading to its own edge), so a
  small brightness difference fades instead of stepping. Objects (with their margin) are never blended: whole
  from one frame as before. CPU pass after either render.

## 0.5.2
- `Session.metrics()` and `metrics.json` next to the map (CLI): pairs after the solve (rms, p95, max px), overlap
  NCC median, frames to reshoot, the stage's landing (rms um), objects whole / cut, the brightness step over
  overlaps (%), each frame's background unevenness after the flat field (%), timings. MControl's Summary keeps
  their history.

## 0.5.1
- Saved map tiles are RGBA: transparent where no frame reaches (the map's ragged edges). Before they were black and
  covered whatever lay under the map (black strips between selections in MControl).

## 0.5.0
- Seams around objects, not through them (`render(seams="objects")`, the default; CLI `--seams objects|voronoi`):
  objects on the sample (crystals, crosses, flakes, dust: above 6 noise sigma over the frame's local background,
  >= 9 px, plus a 6 px margin) are found in every frame, joined across frames on the map, and each is taken whole
  from the one frame that holds it nearest its centre. Seams run only through the quiet background; an object's
  shape and size are those of one frame. On a synthetic 5 x 4 scan: 643 objects, 602 whole, none cut (nearest-
  centre seams cut 92); objects larger than a frame holds keep the usual seams. Works after the CPU and the GPU
  render (a pass over each object's pixels); `Session.entities` (`boxes()`, `mask(i)`, `whole`, `too_big`).

## 0.4.0
- The map is rendered on demand: `render()` returns as soon as the frames are placed; `Canvas.region()`,
  `read()`, `grid()` cells and `save()` render only their part (GPU or CPU), `image` / `owner` render the whole
  map once. A 58×58 map never has to be in RAM whole (it was ~4 GB with the owner map), and `save()` streams
  tile by tile.
- Owner ids are written by the render itself (no pass over the map afterwards).
- `Canvas.covered(x, y, w, h)` from the frame rectangles; `grid()` uses it instead of the owner map.

## 0.3.1
- The flat-field estimated from a scan (profile without one) is computed once per session from ≤ 24 frames
  spread over the scan; it was recomputed on every render from up to 64 and took most of the render time.
  `Session.flat` keeps it (save it into the profile to skip it altogether).
- `Session.render_times`: flat-field, map, rest; bench prints them and scales only the map with the scan.

## 0.3.0
- The map is rendered on the GPU (`render_band`): bands of rows, frames in VRAM only while a band needs them,
  a cell index per band; the same pixels as the CPU render (kernels compiled without fast math and FMA).
- Timing model (`mstitch.Eta`): one linear layer `t = W·[1, frames, pairs, map Mpx]` per phase, weights ≥ 0,
  learned from the user's runs per setup; `Eta.predict(profile, mode, plan)` before a scan,
  `Session.eta_remaining()` during it; no estimate for the first two runs of a setup.
- Per-frame preparation time and the frames still being prepared (`pending`) are tracked.

## 0.2.2
- GPU: cuFFT plans cached per frame size (planning took most of a short run); the GPU starts and compiles its
  kernels when the Session is made, not in the first solve().
- The GPU note in `Report.stats["backend_note"]` gives setup, GPU and CPU (NCC) times.
- bench: per-pair time from a second, warm solve(); the first call shown next to it.

## 0.2.1
- Windows: the module links the C++ runtime statically (MSVC `/MT`, MinGW `-static`), so it needs no
  msvcp140 / libstdc++ DLLs; when `_core` still fails to load, the error lists the DLLs it cannot find.

## 0.2.0
- CUDA backend: batched cuFFT for the frame FFTs, cross-power spectra and inverse FFTs, top-8 peaks on the GPU;
  the CPU refines the previous batch at the same time. Spectra cached in VRAM, freed after their last pair,
  evicted and recomputed when VRAM is short.
- Nothing CUDA needed to build: the driver, NVRTC and cuFFT are loaded at run time and the kernels compiled
  for the GPU found; `pip install mstitch[cuda]` brings NVRTC and cuFFT.
- `Session(backend="auto"|"cuda"|"cpu")`, `Report.stats["backend"]`, `mstitch.cuda.info()`.
- Tests: kernels compile for sm_75/86/89; the GPU path on a fake driver gives the CPU path's results exactly;
  GPU vs CPU on real hardware when present. `bench/bench.py` compares CPU and GPU.

## 0.1.0
- First version: C++ core with a Python module (nanobind), CPU backend.
- Takes MControl snaps as they come (Bayer `lossless` / `raw`, 8 and 12 bit), unpacks in C++ on all cores.
- Pairs from stage positions; phase correlation + NCC with sub-pixel refinement; robust weighted least squares.
- Modes `quick` (1/4 size) and `quality` (1/2 size); Voronoi render without double images; flat-field.
- `calibrate()` for objective profiles (stage→pixel matrix, repeatability, flat-field); reshoot list for the second pass.
- `Canvas`: pixel ↔ stage um, crops, grid cells, saving as tiles + manifest + poses.
