// A fake CUDA driver + cuFFT on the CPU, for testing the GPU path's host logic (batches, spectrum cache,
// evictions, reference counts, peaks back to the right pairs) on machines without a GPU.
// Kernels are replaced by CPU code with the same meaning; NVRTC is the real one.
// MOCK_CUDA_FREE_MB sets the "free VRAM" (small values force evictions).
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "pocketfft_hdronly.h"

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

using ull = unsigned long long;
static int g_fn[3];   // function handles: addresses of these

EXPORT int cuInit(unsigned) { return 0; }
EXPORT int cuDeviceGetCount(int* n) { *n = 1; return 0; }
EXPORT int cuDeviceGet(int* d, int) { *d = 0; return 0; }
EXPORT int cuDeviceGetName(char* s, int n, int) { std::strncpy(s, "Mock GPU", n); return 0; }
EXPORT int cuDeviceGetAttribute(int* v, int attr, int) { *v = attr == 75 ? 8 : 6; return 0; }
EXPORT int cuDevicePrimaryCtxRetain(void** c, int) { static int ctx; *c = &ctx; return 0; }
EXPORT int cuCtxSetCurrent(void*) { return 0; }
EXPORT int cuCtxSynchronize() { return 0; }
EXPORT int cuMemGetInfo_v2(size_t* f, size_t* t) {
    const char* e = std::getenv("MOCK_CUDA_FREE_MB");
    *f = size_t(e ? std::atof(e) : 3000.0) * 1000000;
    *t = 4000000000ull;
    return 0;
}
EXPORT int cuMemAlloc_v2(ull* p, size_t n) { *p = ull(std::malloc(n ? n : 1)); return *p ? 0 : 2; }
EXPORT int cuMemFree_v2(ull p) { std::free((void*)p); return 0; }
EXPORT int cuMemcpyHtoD_v2(ull d, const void* s, size_t n) { std::memcpy((void*)d, s, n); return 0; }
EXPORT int cuMemcpyDtoH_v2(void* d, ull s, size_t n) { std::memcpy(d, (void*)s, n); return 0; }
EXPORT int cuMemcpyDtoD_v2(ull d, ull s, size_t n) { std::memmove((void*)d, (void*)s, n); return 0; }
EXPORT int cuModuleLoadData(void** m, const void*) { static int mod; *m = &mod; return 0; }
EXPORT int cuModuleGetFunction(void** f, void*, const char* name) {
    int k = !std::strcmp(name, "i16_to_f32") ? 0 : !std::strcmp(name, "cross_power") ? 1 : 2;
    *f = &g_fn[k];
    return 0;
}
EXPORT int cuGetErrorString(int, const char** s) { *s = "mock error"; return 0; }

template <class T> static T arg(void** a, int i) { return *static_cast<T*>(a[i]); }

EXPORT int cuLaunchKernel(void* f, unsigned gx, unsigned gy, unsigned, unsigned bx, unsigned, unsigned, unsigned,
                          void*, void** a, void**) {
    if (f == &g_fn[0]) {
        auto* in = (const int16_t*)arg<ull>(a, 0);
        auto* out = (float*)arg<ull>(a, 1);
        const long long n = arg<long long>(a, 2);
        const float inv = arg<float>(a, 3);
        for (long long i = 0; i < n && i < (long long)gx * bx; ++i) out[i] = in[i] * inv;
    } else if (f == &g_fn[1]) {
        auto* spec = (const float*)arg<ull>(a, 0);
        auto* sa = (const int*)arg<ull>(a, 1);
        auto* sb = (const int*)arg<ull>(a, 2);
        const int len = arg<int>(a, 3);
        auto* out = (float*)arg<ull>(a, 4);
        for (unsigned p = 0; p < gy; ++p)
            for (int k = 0; k < len; ++k) {
                const float* x = spec + 2ll * len * sa[p] + 2ll * k;
                const float* y = spec + 2ll * len * sb[p] + 2ll * k;
                const float re = x[0] * y[0] + x[1] * y[1], im = x[1] * y[0] - x[0] * y[1];
                const float m = std::sqrt(re * re + im * im);
                float* o = out + 2ll * len * p + 2ll * k;
                o[0] = m > 1e-12f ? re / m : 0.f;
                o[1] = m > 1e-12f ? im / m : 0.f;
            }
    } else {
        auto* r = (float*)arg<ull>(a, 0);
        const int H = arg<int>(a, 1), W = arg<int>(a, 2), k = arg<int>(a, 3), rad = arg<int>(a, 4);
        auto* out = (int*)arg<ull>(a, 5);
        for (unsigned p = 0; p < gx; ++p) {
            float* s = r + (long long)p * H * W;
            for (int it = 0; it < k; ++it) {
                const int best = int(std::max_element(s, s + (long long)H * W) - s);
                out[p * k + it] = best;
                const int u = best / W, v = best % W;
                for (int dy = -rad; dy <= rad; ++dy)
                    for (int dx = -rad; dx <= rad; ++dx) s[((u + dy + H) % H) * W + (v + dx + W) % W] = -3.0e38f;
            }
        }
    }
    return 0;
}

struct PlanInfo { int h, w, type, batch; };
static std::map<int, PlanInfo> g_plans;
static int g_next = 1;

EXPORT int cufftPlanMany(int* plan, int rank, int* n, int*, int, int, int*, int, int, int type, int batch) {
    if (rank != 2) return 1;
    *plan = g_next++;
    g_plans[*plan] = {n[0], n[1], type, batch};
    return 0;
}
EXPORT int cufftExecR2C(int plan, float* in, void* out) {
    auto p = g_plans.at(plan);
    const size_t wc = p.w / 2 + 1;
    for (int b = 0; b < p.batch; ++b)
        pocketfft::r2c({size_t(p.h), size_t(p.w)}, {ptrdiff_t(4 * p.w), 4}, {ptrdiff_t(8 * wc), 8}, {0, 1},
                       pocketfft::FORWARD, in + size_t(b) * p.h * p.w,
                       (std::complex<float>*)out + size_t(b) * p.h * wc, 1.0f, 1);
    return 0;
}
EXPORT int cufftExecC2R(int plan, void* in, float* out) {
    auto p = g_plans.at(plan);
    const size_t wc = p.w / 2 + 1;
    for (int b = 0; b < p.batch; ++b)
        pocketfft::c2r({size_t(p.h), size_t(p.w)}, {ptrdiff_t(8 * wc), 8}, {ptrdiff_t(4 * p.w), 4}, {0, 1},
                       pocketfft::BACKWARD, (std::complex<float>*)in + size_t(b) * p.h * wc,
                       out + size_t(b) * p.h * p.w, 1.0f, 1);
    return 0;
}
EXPORT int cufftDestroy(int plan) { g_plans.erase(plan); return 0; }
EXPORT int cufftGetVersion(int* v) { *v = 11000; return 0; }
