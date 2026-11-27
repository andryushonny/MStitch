# CUDA backend (planned, milestone 2)

Target: NVIDIA RTX 2050 laptop (Ampere GA107, `sm_86`, 4 GB GDDR6, 112 GB/s, PCIe 3.0 x8), CUDA 12.x, MSVC.

What moves to the GPU, per Blattner et al. 2014 (pipeline over CPU and GPU, compute overlapped with copies):

| stage | now (CPU) | GPU |
|---|---|---|
| unpack (zstd) | pool | stays on the CPU |
| Bayer → level, high-pass | pool | kernel after the upload (pinned buffers, 2–3 streams) |
| FFT of each frame | pocketfft | batched cuFFT R2C, one plan per frame size, spectra kept in VRAM with a reference count |
| cross-power, inverse, top-k peaks | per pair | one kernel + batched C2R + CUB select, only the peaks go back |
| NCC of candidates, sub-pixel | per pair | block reductions over the overlap |
| solve | CPU | stays on the CPU (milliseconds) |
| render | CPU bands | kernel writing tiles of the map |

Interface: the same `tile_spectrum` / `register_pair` contract as `core/register.hpp`, batched; the session picks
the backend at start (`MSTITCH_CUDA=ON` in CMake, CPU when no GPU is found).

VRAM budget, quick on 58×58: spectra of 648×486 frames ≈ 1.3 MB each, about two rows alive ≈ 150 MB.
