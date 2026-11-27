# Changelog

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
