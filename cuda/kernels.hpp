// CUDA kernels, compiled at run time by NVRTC for the GPU in the machine (cubin for its sm_XY).
#pragma once

namespace ms::cuda {

inline const char* kKernels = R"CUDA(
#define TPB 256

// int16 high-passed frames -> float, n values
extern "C" __global__ void i16_to_f32(const short* __restrict__ in, float* __restrict__ out, long long n, float inv) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * inv;
}

// Normalized cross-power spectrum of pair p: a * conj(b) / |a * conj(b)|.
// spec: spectra cache, slot s at spec + 2*len*s (interleaved re, im); sa, sb: slots of the pair's frames.
extern "C" __global__ void cross_power(const float* __restrict__ spec, const int* __restrict__ sa,
                                       const int* __restrict__ sb, int len, float* __restrict__ out) {
    const int p = blockIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= len) return;
    const float* a = spec + 2ll * len * sa[p] + 2ll * k;
    const float* b = spec + 2ll * len * sb[p] + 2ll * k;
    const float re = a[0] * b[0] + a[1] * b[1];
    const float im = a[1] * b[0] - a[0] * b[1];
    const float m = sqrtf(re * re + im * im);
    float* o = out + 2ll * len * p + 2ll * k;
    if (m > 1e-12f) { o[0] = re / m; o[1] = im / m; } else { o[0] = 0.f; o[1] = 0.f; }
}

// k highest peaks of each correlation surface (one block per pair), suppressing a (2 rad + 1)^2 square around
// each found peak (wrapping). Ties: the lowest index, as on the CPU. out[p * k + i] = y * W + x.
extern "C" __global__ void topk_peaks(float* r, int H, int W, int k, int rad, int* out) {
    __shared__ float sv[TPB];
    __shared__ int si[TPB];
    const int p = blockIdx.x, t = threadIdx.x, n = H * W, side = 2 * rad + 1;
    float* a = r + (long long)p * n;
    for (int it = 0; it < k; ++it) {
        float bv = -3.0e38f;
        int bi = 0x7fffffff;
        for (int i = t; i < n; i += TPB) {
            const float v = a[i];
            if (v > bv) { bv = v; bi = i; }
        }
        sv[t] = bv; si[t] = bi;
        __syncthreads();
        for (int s = TPB / 2; s > 0; s >>= 1) {
            if (t < s && (sv[t + s] > sv[t] || (sv[t + s] == sv[t] && si[t + s] < si[t]))) {
                sv[t] = sv[t + s]; si[t] = si[t + s];
            }
            __syncthreads();
        }
        const int best = si[0] == 0x7fffffff ? 0 : si[0];
        if (t == 0) out[p * k + it] = best;
        __syncthreads();
        const int u = best / W, v = best % W;
        for (int c = t; c < side * side; c += TPB) {
            const int yy = (u + c / side - rad + H) % H, xx = (v + c % side - rad + W) % W;
            a[yy * W + xx] = -3.0e38f;
        }
        __syncthreads();
    }
}
)CUDA";

}  // namespace ms::cuda
