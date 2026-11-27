# CUDA backend

Built into every wheel; nothing CUDA is needed to build. At run time MStitch loads the NVIDIA driver
(`nvcuda.dll`, comes with the graphics driver), NVRTC and cuFFT (`pip install mstitch[cuda]`, or a CUDA 12
Toolkit) and compiles its kernels for the GPU it finds (cubin for its `sm_XY`). Without them `backend="auto"`
runs on the CPU and says why in `Report.stats["backend_note"]`.

Target: RTX 2050 laptop (Ampere GA107, `sm_86`, 4 GB, 112 GB/s). Windows driver ≥ 528 (CUDA 12).

## What runs where (per Blattner et al. 2014)

| stage | where |
|---|---|
| unpack (zstd), Bayer → level, high-pass | CPU pool, while frames arrive |
| int16 → float, FFT of each frame | GPU: batched cuFFT R2C, 16 frames per call; spectra cached in VRAM |
| cross-power, inverse FFT, top-8 peaks | GPU: batches of up to 32 pairs (`cross_power`, batched C2R, `topk_peaks`) |
| choose the candidate, NCC, sub-pixel | CPU pool, for batch *b* while the GPU computes batch *b + 1* |
| global solve, render | CPU |

Only the peaks (8 integers per pair) come back from the GPU.

**Spectrum cache.** Pairs are sorted by their later frame; every frame's spectrum is computed once and its slot
is freed after its last pair, so only a band of frames (about a row of the scan) lives in VRAM. The cache takes
~60 % of the free VRAM (`MSTITCH_GPU_MAX_SLOTS` caps it); when it is too small, spectra not needed by the current
batch are evicted and recomputed later — same result, more FFTs.

## Files

| file | |
|---|---|
| `api.hpp/.cpp` | run-time loading of the driver API, NVRTC and cuFFT (only the functions used) |
| `kernels.hpp` | the three kernels, as source for NVRTC |
| `gpu.hpp/.cpp` | device set-up, the batch loop, the spectrum cache |

## Tests

* `test_kernels_compile`: the kernels compile for sm_75/86/89 (NVRTC only).
* `test_gpu_path_on_mock_matches_cpu`: the whole GPU path (batches, cache, evictions, peaks back to their pairs)
  on a fake driver + cuFFT that runs on the CPU (`tests/mock_cuda`, `-DMSTITCH_MOCK_CUDA=ON`): identical edges and
  positions to the CPU path.
* `test_real_gpu_matches_cpu`: on a machine with an NVIDIA GPU, GPU vs CPU on a synthetic 8×6 scan.

## Next

The NCC refinement still runs on the CPU. If `bench/bench.py` shows it is the limit on the laptop, it moves to a
kernel (block reductions over the overlap for the 4 × 8 candidates and the 3 × 3 climb).
