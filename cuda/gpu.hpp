// GPU part of the registration (Blattner et al. 2014 on one laptop GPU):
// FFT of every frame once (batched cuFFT R2C, spectra cached in VRAM with reference counts),
// cross-power + inverse FFT + top-k peaks for batches of pairs; only the peaks come back.
// The CPU picks and refines the shift (NCC) of one batch while the GPU computes the next.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/register.hpp"
#include "core/render.hpp"

namespace ms::cuda {

struct Device {
    std::string name;
    int cc_major = 0, cc_minor = 0;
    size_t free_bytes = 0, total_bytes = 0;
};

// Initialise once (driver, device 0, kernels). nullptr and *why when there is no usable GPU.
class Gpu;
Gpu* gpu(std::string* why);
Device device_info(Gpu*);

// Compile the kernels for sm_<arch> with NVRTC only (no GPU needed): "" when fine, else the log.
std::string compile_check(int arch);

struct BatchStats {
    size_t pairs = 0, batches = 0, ffts = 0, evictions = 0, slots = 0, pairs_per_batch = 0;
    double setup_ms = 0;   // memory plan, cuFFT plans (cached after the first call per frame size), allocations
    double gpu_ms = 0;     // uploads, FFTs, kernels, peaks back (the batch loop without the hand-off to the CPU)
};

// Peaks for edges (all frames h x w). on_batch(e0, e1, peaks) gets peaks[(e - e0) * k + i] = y * w + x and
// is called from this thread; it should hand the work to other threads and return.
void gpu_peaks(Gpu* g, const std::vector<const Tile*>& tiles, const std::vector<Edge>& edges, int k,
               const std::function<void(size_t, size_t, const std::vector<int>&)>& on_batch, BatchStats* st);

struct RenderStats {
    int bands = 0, rows_per_band = 0, slots = 0;
    double upload_mb = 0, ms = 0;
};

// The map on the GPU, in bands of rows; the same pixels as ms::render (core/render.cpp).
CanvasImage gpu_render(Gpu* g, const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                       const std::vector<float>& flat, double x0, double y0, int w, int h, RenderStats* st);

}  // namespace ms::cuda
