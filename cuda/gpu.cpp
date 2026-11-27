#include "cuda/gpu.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>
#include <tuple>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "cuda/api.hpp"
#include "cuda/kernels.hpp"

namespace ms::cuda {

namespace {

void check(CUresult r, const char* what) {
    if (r == 0) return;
    const char* s = nullptr;
    if (api().cuGetErrorString) api().cuGetErrorString(r, &s);
    throw std::runtime_error(std::string("CUDA ") + what + ": " + (s ? s : std::to_string(r)));
}
void checkfft(cufftResult r, const char* what) {
    if (r != 0) throw std::runtime_error(std::string("cuFFT ") + what + ": error " + std::to_string(r));
}

std::string compile(int arch, std::vector<char>* cubin) {
    Api& a = api();
    if (!a.has_nvrtc) return "NVRTC not loaded";
    nvrtcProgram prog = nullptr;
    if (a.nvrtcCreateProgram(&prog, kKernels, "mstitch.cu", 0, nullptr, nullptr) != 0) return "nvrtcCreateProgram failed";
    const std::string opt = "--gpu-architecture=sm_" + std::to_string(arch);
    // IEEE arithmetic, no fused multiply-add: the render kernel must give the CPU's pixels
    const char* opts[] = {opt.c_str(), "--std=c++17", "--fmad=false"};
    const nvrtcResult r = a.nvrtcCompileProgram(prog, 3, opts);
    std::string log;
    size_t n = 0;
    a.nvrtcGetProgramLogSize(prog, &n);
    if (n > 1) {
        log.resize(n);
        a.nvrtcGetProgramLog(prog, log.data());
        log.resize(n - 1);
    }
    if (r != 0) {
        a.nvrtcDestroyProgram(&prog);
        return "NVRTC compile failed (" + std::to_string(r) + "): " + log;
    }
    if (cubin) {
        a.nvrtcGetCUBINSize(prog, &n);
        cubin->resize(n);
        a.nvrtcGetCUBIN(prog, cubin->data());
    }
    a.nvrtcDestroyProgram(&prog);
    return "";
}

}  // namespace

class Gpu {
public:
    CUcontext ctx = nullptr;
    CUfunction k_i16 = nullptr, k_cross = nullptr, k_topk = nullptr, k_render = nullptr;
    Device dev;
    std::mutex m;   // one registration at a time on the GPU
    std::map<std::tuple<int, int, int, int>, cufftHandle> plans;   // (h, w, type, batch): planning is slow

    void bind() { check(api().cuCtxSetCurrent(ctx), "cuCtxSetCurrent"); }
    cufftHandle plan(int h, int w, int type, int batch);
};

cufftHandle Gpu::plan(int h, int w, int type, int batch) {
    const auto key = std::make_tuple(h, w, type, batch);
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;
    cufftHandle p = 0;
    int dims[2] = {h, w};
    checkfft(api().cufftPlanMany(&p, 2, dims, nullptr, 1, 0, nullptr, 1, 0, type, batch), "plan");
    plans[key] = p;
    return p;
}

Gpu* gpu(std::string* why) {
    static std::once_flag once;
    static std::unique_ptr<Gpu> g;
    static std::string err;
    std::call_once(once, [] {
        try {
            Api& a = api();
            if (!a.has_driver) throw std::runtime_error("no NVIDIA driver library");
            if (!a.has_nvrtc) throw std::runtime_error("NVRTC not found (pip install nvidia-cuda-nvrtc-cu12)");
            if (!a.has_cufft) throw std::runtime_error("cuFFT not found (pip install nvidia-cufft-cu12)");
            check(a.cuInit(0), "cuInit");
            int n = 0;
            check(a.cuDeviceGetCount(&n), "cuDeviceGetCount");
            if (n < 1) throw std::runtime_error("no CUDA device");
            auto p = std::make_unique<Gpu>();
            CUdevice d;
            check(a.cuDeviceGet(&d, 0), "cuDeviceGet");
            char name[256] = {0};
            a.cuDeviceGetName(name, 255, d);
            p->dev.name = name;
            a.cuDeviceGetAttribute(&p->dev.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
            a.cuDeviceGetAttribute(&p->dev.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);
            check(a.cuDevicePrimaryCtxRetain(&p->ctx, d), "cuDevicePrimaryCtxRetain");
            p->bind();
            std::vector<char> cubin;
            const std::string e = compile(p->dev.cc_major * 10 + p->dev.cc_minor, &cubin);
            if (!e.empty()) throw std::runtime_error(e);
            CUmodule mod;
            check(a.cuModuleLoadData(&mod, cubin.data()), "cuModuleLoadData");
            check(a.cuModuleGetFunction(&p->k_i16, mod, "i16_to_f32"), "i16_to_f32");
            check(a.cuModuleGetFunction(&p->k_cross, mod, "cross_power"), "cross_power");
            check(a.cuModuleGetFunction(&p->k_topk, mod, "topk_peaks"), "topk_peaks");
            check(a.cuModuleGetFunction(&p->k_render, mod, "render_band"), "render_band");
            check(a.cuMemGetInfo(&p->dev.free_bytes, &p->dev.total_bytes), "cuMemGetInfo");
            g = std::move(p);
        } catch (const std::exception& ex) {
            err = ex.what();
        }
    });
    if (!g && why) *why = err;
    return g.get();
}

Device device_info(Gpu* g) {
    Device d = g->dev;
    std::lock_guard<std::mutex> lk(g->m);
    g->bind();
    api().cuMemGetInfo(&d.free_bytes, &d.total_bytes);
    return d;
}

std::string compile_check(int arch) { return compile(arch, nullptr); }

namespace {

// Device buffer, freed on scope exit.
struct Buf {
    CUdeviceptr p = 0;
    size_t bytes = 0;
    void alloc(size_t n) {
        bytes = n;
        check(api().cuMemAlloc(&p, n), "cuMemAlloc");
    }
    void reset() {
        if (p) api().cuMemFree(p);
        p = 0;
        bytes = 0;
    }
    ~Buf() { reset(); }
};

void launch(CUfunction f, unsigned gx, unsigned gy, unsigned bx, void** args) {
    check(api().cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel");
}

}  // namespace

void gpu_peaks(Gpu* g, const std::vector<const Tile*>& tiles, const std::vector<Edge>& edges, int k,
               const std::function<void(size_t, size_t, const std::vector<int>&)>& on_batch, BatchStats* st) {
    if (edges.empty()) return;
    std::lock_guard<std::mutex> lk(g->m);
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    double handoff_ms = 0;
    g->bind();
    Api& a = api();
    const Tile* t0 = tiles[edges[0].i];
    const int H = t0->h, W = t0->w;
    const long long n = (long long)H * W;          // real pixels per frame
    const long long len = (long long)H * (W / 2 + 1);  // complex values per spectrum
    const int Bt = 16;                             // frames per R2C batch

    // Memory plan: pairs per batch from the free VRAM, then the cuFFT plans (they take work space),
    // then ~60 % of what is left for the fixed buffers and the cached spectra.
    size_t freeb = 0, total = 0;
    check(a.cuMemGetInfo(&freeb, &total), "cuMemGetInfo");
    int Bp = 32;                                   // pairs per batch
    auto fixed = [&](int bp) {
        return size_t(Bt) * (n * 2 + n * 4 + len * 8) + size_t(bp) * (len * 8 + n * 4) + size_t(bp) * (k + 2) * 4;
    };
    while (Bp > 4 && fixed(Bp) + size_t(2 * Bp + 8) * len * 8 > size_t(0.5 * double(freeb))) Bp /= 2;
    const cufftHandle r2c = g->plan(H, W, CUFFT_R2C, Bt);
    const cufftHandle c2r = g->plan(H, W, CUFFT_C2R, Bp);
    check(a.cuMemGetInfo(&freeb, &total), "cuMemGetInfo");
    const size_t budget = size_t(0.6 * double(freeb));
    std::vector<int> refs(tiles.size(), 0);        // pairs left per frame
    for (auto& e : edges) { refs[e.i]++; refs[e.j]++; }
    size_t needed = 0;
    for (int r : refs) needed += r > 0;
    long long S = (long long)((budget > fixed(Bp) ? budget - fixed(Bp) : 0) / size_t(len * 8));
    S = std::min<long long>(S, (long long)needed);
    if (const char* cap = std::getenv("MSTITCH_GPU_MAX_SLOTS"))   // limit the spectrum cache (VRAM)
        S = std::min<long long>(S, std::max(4LL, std::atoll(cap)));
    if (S < std::min<long long>(4, (long long)needed)) throw std::runtime_error("not enough GPU memory for these frames");
    // a batch touches at most 2 frames per pair: keep that within the cache
    const int Bq = int(std::max<long long>(1, std::min<long long>(Bp, (S - 2) / 2)));

    Buf staging, fbuf, tmpc, spec, pc, pr, sa, sb, peaks;
    staging.alloc(size_t(Bt) * n * 2);
    fbuf.alloc(size_t(Bt) * n * 4);
    tmpc.alloc(size_t(Bt) * len * 8);
    spec.alloc(size_t(S) * len * 8);
    pc.alloc(size_t(Bp) * len * 8);
    pr.alloc(size_t(Bp) * n * 4);
    sa.alloc(size_t(Bp) * 4);
    sb.alloc(size_t(Bp) * 4);
    peaks.alloc(size_t(Bp) * k * 4);
    std::unordered_map<int, int> slot_of;          // frame -> cache slot
    std::vector<int> free_slots;
    for (long long s = S - 1; s >= 0; --s) free_slots.push_back(int(s));
    BatchStats stats;
    stats.slots = size_t(S);
    stats.pairs_per_batch = size_t(Bq);
    std::vector<int16_t> host_stage(size_t(Bt) * n);

    auto spectra_for = [&](const std::vector<int>& want, const std::vector<int>& keep) {
        // frames still missing, computed Bt at a time
        std::vector<int> miss;
        for (int f : want)
            if (!slot_of.count(f)) miss.push_back(f);
        for (size_t b0 = 0; b0 < miss.size(); b0 += Bt) {
            const size_t nb = std::min<size_t>(Bt, miss.size() - b0);
            for (size_t q = 0; q < nb; ++q)
                std::copy(tiles[miss[b0 + q]]->reg.begin(), tiles[miss[b0 + q]]->reg.end(),
                          host_stage.begin() + q * n);
            check(a.cuMemcpyHtoD(staging.p, host_stage.data(), nb * n * 2), "upload frames");
            long long cnt = (long long)nb * n;
            float inv = 1.0f / kRegScale;
            void* args1[] = {&staging.p, &fbuf.p, &cnt, &inv};
            launch(g->k_i16, unsigned((cnt + 255) / 256), 1, 256, args1);
            checkfft(a.cufftExecR2C(r2c, reinterpret_cast<float*>(fbuf.p), reinterpret_cast<void*>(tmpc.p)),
                     "R2C");
            stats.ffts += nb;
            for (size_t q = 0; q < nb; ++q) {
                if (free_slots.empty()) {   // evict a cached spectrum this batch does not need
                    for (auto it = slot_of.begin(); it != slot_of.end(); ++it)
                        if (std::find(keep.begin(), keep.end(), it->first) == keep.end()) {
                            free_slots.push_back(it->second);
                            slot_of.erase(it);
                            stats.evictions++;
                            break;
                        }
                    if (free_slots.empty()) throw std::runtime_error("spectrum cache too small");
                }
                const int s = free_slots.back();
                free_slots.pop_back();
                slot_of[miss[b0 + q]] = s;
                check(a.cuMemcpyDtoD(spec.p + CUdeviceptr(s) * len * 8, tmpc.p + CUdeviceptr(q) * len * 8,
                                     size_t(len) * 8), "copy spectrum");
            }
        }
    };

    stats.setup_ms = std::chrono::duration<double, std::milli>(clk::now() - t_start).count();
    const auto t_loop = clk::now();
    std::vector<int> hsa(Bp), hsb(Bp), hpk;
    for (size_t e0 = 0; e0 < edges.size(); e0 += Bq) {
        const size_t e1 = std::min(edges.size(), e0 + Bq), nb = e1 - e0;
        std::vector<int> want;
        for (size_t e = e0; e < e1; ++e) { want.push_back(edges[e].i); want.push_back(edges[e].j); }
        std::sort(want.begin(), want.end());
        want.erase(std::unique(want.begin(), want.end()), want.end());
        spectra_for(want, want);
        for (size_t e = e0; e < e1; ++e) {
            hsa[e - e0] = slot_of.at(edges[e].i);
            hsb[e - e0] = slot_of.at(edges[e].j);
        }
        check(a.cuMemcpyHtoD(sa.p, hsa.data(), nb * 4), "upload slots");
        check(a.cuMemcpyHtoD(sb.p, hsb.data(), nb * 4), "upload slots");
        int ilen = int(len);
        void* args2[] = {&spec.p, &sa.p, &sb.p, &ilen, &pc.p};
        launch(g->k_cross, unsigned((len + 255) / 256), unsigned(nb), 256, args2);
        checkfft(a.cufftExecC2R(c2r, reinterpret_cast<void*>(pc.p), reinterpret_cast<float*>(pr.p)), "C2R");
        int h = H, w = W, kk = k, rad = 3;
        void* args3[] = {&pr.p, &h, &w, &kk, &rad, &peaks.p};
        launch(g->k_topk, unsigned(nb), 1, 256, args3);
        hpk.resize(nb * k);
        check(a.cuMemcpyDtoH(hpk.data(), peaks.p, nb * k * 4), "download peaks");   // synchronizes
        stats.batches++;
        stats.pairs += nb;
        const auto th = clk::now();
        on_batch(e0, e1, hpk);
        handoff_ms += std::chrono::duration<double, std::milli>(clk::now() - th).count();
        // spectra no later pair needs go back to the free list
        for (size_t e = e0; e < e1; ++e)
            for (int f : {edges[e].i, edges[e].j})
                if (--refs[f] == 0) {
                    auto it = slot_of.find(f);
                    if (it != slot_of.end()) { free_slots.push_back(it->second); slot_of.erase(it); }
                }
    }
    stats.gpu_ms = std::chrono::duration<double, std::milli>(clk::now() - t_loop).count() - handoff_ms;
    if (st) *st = stats;
}

CanvasImage gpu_render(Gpu* g, const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                       const std::vector<float>& flat, double x0, double y0, int w, int h, RenderStats* st) {
    CanvasImage c;
    if (tiles.empty()) return c;
    const int tw = tiles[0]->w, th = tiles[0]->h;
    for (auto* t : tiles)
        if (t->w != tw || t->h != th) throw std::runtime_error("frames of different sizes");
    canvas_box(tiles, pos, x0, y0, w, h);
    c.w = w; c.h = h; c.ox = x0; c.oy = y0;
    c.rgb.resize(size_t(w) * h * 3);
    c.owner.resize(size_t(w) * h);

    std::lock_guard<std::mutex> lk(g->m);
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    g->bind();
    Api& a = api();
    const int n = int(tiles.size());
    // frame k on the canvas: top-left (px, py) relative to the canvas, as the CPU computes it
    std::vector<double> px(n), py(n);
    for (int k = 0; k < n; ++k) { px[k] = pos[k][0] - x0; py[k] = pos[k][1] - y0; }
    const int cell = 32;
    const size_t tile_bytes = size_t(tw) * th * 3;
    // band height: ~192 MB of output per band, a whole number of cells
    int rows = int(std::min<long long>(h, std::max<long long>(cell, 192000000LL / (7LL * w))));
    if (const char* r = std::getenv("MSTITCH_GPU_BAND_ROWS")) rows = std::min(rows, std::max(cell, std::atoi(r)));
    rows = std::max(cell, rows / cell * cell);
    const int nbands = (h + rows - 1) / rows;
    auto rows_of = [&](int k, int& r0, int& r1) {   // canvas rows a frame covers
        r0 = std::max(0, int(std::ceil(py[k])));
        r1 = std::min(h - 1, int(std::floor(py[k] + th - 1)));
    };
    // frames per band, and the most at once (pool size)
    std::vector<std::vector<int>> in_band(nbands);
    for (int k = 0; k < n; ++k) {
        int r0, r1;
        rows_of(k, r0, r1);
        if (r1 < r0) continue;
        for (int b = r0 / rows; b <= r1 / rows; ++b) in_band[b].push_back(k);
    }
    size_t cap = 1;
    for (auto& v : in_band) cap = std::max(cap, v.size());
    size_t freeb = 0, total = 0;
    check(a.cuMemGetInfo(&freeb, &total), "cuMemGetInfo");
    const size_t band_bytes = size_t(rows) * w * 7;
    if (cap * tile_bytes + band_bytes + flat.size() * 4 > size_t(0.8 * double(freeb)))
        throw std::runtime_error("not enough GPU memory for the map");

    Buf pool, dpx, dpy, dsid, dcs, dcl, dflat, dout, down;
    pool.alloc(cap * tile_bytes);
    dpx.alloc(cap * 8); dpy.alloc(cap * 8); dsid.alloc(cap * 4);
    const int ncx = (w + cell - 1) / cell, ncy = rows / cell;
    dcs.alloc(size_t(ncx) * ncy * 4 + 4);
    dout.alloc(size_t(rows) * w * 3);
    down.alloc(size_t(rows) * w * 4);
    int has_flat = flat.size() == tile_bytes ? 1 : 0;
    dflat.alloc(has_flat ? flat.size() * 4 : 4);
    if (has_flat) check(a.cuMemcpyHtoD(dflat.p, flat.data(), flat.size() * 4), "upload flat");

    RenderStats stats;
    stats.bands = nbands; stats.rows_per_band = rows; stats.slots = int(cap);
    std::vector<int> slot_of(n, -1), free_slots;
    for (int s = int(cap) - 1; s >= 0; --s) free_slots.push_back(s);
    std::vector<double> hpx(cap, 0), hpy(cap, 0);
    std::vector<int> hsid(cap, -1), cstart, clist;
    size_t clist_cap = 0;
    for (int b = 0; b < nbands; ++b) {
        const int y0b = b * rows, nrows = std::min(rows, h - y0b);
        // frames leaving: free their slots (frames are needed by consecutive bands only)
        std::vector<char> want(n, 0);
        for (int k : in_band[b]) want[k] = 1;
        for (int k = 0; k < n; ++k)
            if (slot_of[k] >= 0 && !want[k]) { free_slots.push_back(slot_of[k]); slot_of[k] = -1; }
        for (int k : in_band[b])
            if (slot_of[k] < 0) {
                const int s = free_slots.back();
                free_slots.pop_back();
                slot_of[k] = s;
                hpx[s] = px[k]; hpy[s] = py[k]; hsid[s] = k;
                check(a.cuMemcpyHtoD(pool.p + CUdeviceptr(s) * tile_bytes, tiles[k]->rgb.data(), tile_bytes),
                      "upload frame");
                stats.upload_mb += double(tile_bytes) / 1e6;
            }
        check(a.cuMemcpyHtoD(dpx.p, hpx.data(), cap * 8), "upload px");
        check(a.cuMemcpyHtoD(dpy.p, hpy.data(), cap * 8), "upload py");
        check(a.cuMemcpyHtoD(dsid.p, hsid.data(), cap * 4), "upload ids");
        // cell lists, frames in index order (in_band[b] is sorted)
        const int nc = ncx * ncy;
        std::vector<int> cnt(nc + 1, 0);
        auto cells_of = [&](int k, int& cx0, int& cx1, int& cy0, int& cy1) {
            const int xa = std::max(0, int(std::ceil(px[k]))), xb = std::min(w - 1, int(std::floor(px[k] + tw - 1)));
            const int ya = std::max(y0b, int(std::ceil(py[k]))),
                      yb = std::min(y0b + nrows - 1, int(std::floor(py[k] + th - 1)));
            if (xb < xa || yb < ya) return false;
            cx0 = xa / cell; cx1 = xb / cell; cy0 = (ya - y0b) / cell; cy1 = (yb - y0b) / cell;
            return true;
        };
        for (int k : in_band[b]) {
            int cx0, cx1, cy0, cy1;
            if (!cells_of(k, cx0, cx1, cy0, cy1)) continue;
            for (int cy = cy0; cy <= cy1; ++cy)
                for (int cx = cx0; cx <= cx1; ++cx) cnt[cy * ncx + cx + 1]++;
        }
        cstart.assign(nc + 1, 0);
        for (int q = 0; q < nc; ++q) cstart[q + 1] = cstart[q] + cnt[q + 1];
        clist.assign(size_t(cstart[nc]) + 1, 0);
        std::vector<int> fill(cstart.begin(), cstart.end() - 1);
        for (int k : in_band[b]) {
            int cx0, cx1, cy0, cy1;
            if (!cells_of(k, cx0, cx1, cy0, cy1)) continue;
            for (int cy = cy0; cy <= cy1; ++cy)
                for (int cx = cx0; cx <= cx1; ++cx) clist[fill[cy * ncx + cx]++] = slot_of[k];
        }
        if (clist.size() > clist_cap) {
            clist_cap = clist.size() * 2;
            dcl.reset();
            dcl.alloc(clist_cap * 4);
        }
        check(a.cuMemcpyHtoD(dcs.p, cstart.data(), cstart.size() * 4), "upload cells");
        check(a.cuMemcpyHtoD(dcl.p, clist.data(), clist.size() * 4), "upload cells");
        int itw = tw, ith = th, icell = cell, incx = ncx, iW = w, iy0 = y0b, irows = nrows;
        void* args[] = {&pool.p, &itw, &ith, &dpx.p, &dpy.p, &dsid.p, &dcs.p, &dcl.p, &icell, &incx,
                        &dflat.p, &has_flat, &iW, &iy0, &irows, &dout.p, &down.p};
        check(a.cuLaunchKernel(g->k_render, unsigned((w + 31) / 32), unsigned((nrows + 7) / 8), 1, 32, 8, 1, 0,
                               nullptr, args, nullptr), "render_band");
        check(a.cuMemcpyDtoH(c.rgb.data() + size_t(y0b) * w * 3, dout.p, size_t(nrows) * w * 3), "download map");
        check(a.cuMemcpyDtoH(c.owner.data() + size_t(y0b) * w, down.p, size_t(nrows) * w * 4), "download owner");
    }
    stats.ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    if (st) *st = stats;
    return c;
}

}  // namespace ms::cuda
