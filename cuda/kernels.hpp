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

// One band of the map: rows [y0, y0 + rows). Every pixel takes the frame whose centre is nearest (ties: the
// lower frame index), bilinear, divided by the flat-field gain. The same arithmetic as core/render.cpp, so the
// result is the CPU's. Frames live in slots of `pool`; cell lists (cell x cell pixels of the band) name the slots
// that can cover a pixel, in frame order.
extern "C" __global__ void render_band(const unsigned char* __restrict__ pool, int tw, int th,
                                       const double* __restrict__ spx, const double* __restrict__ spy,
                                       const int* __restrict__ sid, const int* __restrict__ cstart,
                                       const int* __restrict__ clist, int cell, int ncx,
                                       const float* __restrict__ flat, int has_flat, int W, int y0, int rows,
                                       unsigned char* __restrict__ out, int* __restrict__ owner) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int yb = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || yb >= rows) return;
    const int y = y0 + yb;
    const int c = (yb / cell) * ncx + x / cell;
    float best = 3.402823466e38f;
    int bs = -1;
    for (int q = cstart[c]; q < cstart[c + 1]; ++q) {
        const int s = clist[q];
        const double px = spx[s], py = spy[s];
        if (x < px || y < py || x > px + (tw - 1) || y > py + (th - 1)) continue;
        const double cx = px + 0.5 * (tw - 1), cy = py + 0.5 * (th - 1);
        const float d = (float)((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (d < best) { best = d; bs = s; }
    }
    const long long o = (long long)yb * W + x;
    if (bs < 0) {
        out[3 * o] = out[3 * o + 1] = out[3 * o + 2] = 0;
        owner[o] = -1;
        return;
    }
    const double sx = x - spx[bs], sy = y - spy[bs];
    const int iy = min((int)sy, th - 2), ix = min((int)sx, tw - 2);
    const float fy = (float)(sy - iy), fx = (float)(sx - ix);
    const unsigned char* t = pool + (long long)bs * tw * th * 3;
    for (int ch = 0; ch < 3; ++ch) {
        const long long i00 = ((long long)iy * tw + ix) * 3 + ch, i01 = i00 + 3;
        const long long i10 = i00 + (long long)tw * 3, i11 = i10 + 3;
        float a = (float)t[i00], b = (float)t[i01], cc = (float)t[i10], dd = (float)t[i11];
        if (has_flat) {
            a /= fmaxf(flat[i00], 0.05f); b /= fmaxf(flat[i01], 0.05f);
            cc /= fmaxf(flat[i10], 0.05f); dd /= fmaxf(flat[i11], 0.05f);
        }
        const float v = (1 - fy) * ((1 - fx) * a + fx * b) + fy * ((1 - fx) * cc + fx * dd);
        out[3 * o + ch] = (unsigned char)fminf(fmaxf(v + 0.5f, 0.0f), 255.0f);
    }
    owner[o] = sid[bs];
}
)CUDA";

}  // namespace ms::cuda
