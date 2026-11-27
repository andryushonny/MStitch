# Changelog

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
