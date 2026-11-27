#include "cpu/fft.hpp"

#include "pocketfft_hdronly.h"

namespace ms {

void fft_forward(const float* img, int h, int w, Spectrum& out) {
    out.h = h;
    out.w = w;
    const size_t wc = size_t(w) / 2 + 1;
    out.c.resize(size_t(h) * wc);
    pocketfft::shape_t shape{size_t(h), size_t(w)};
    pocketfft::stride_t sin{ptrdiff_t(sizeof(float) * w), ptrdiff_t(sizeof(float))};
    pocketfft::stride_t sout{ptrdiff_t(sizeof(cf) * wc), ptrdiff_t(sizeof(cf))};
    pocketfft::r2c(shape, sin, sout, pocketfft::shape_t{0, 1}, pocketfft::FORWARD, img, out.c.data(), 1.0f, 1);
}

void fft_inverse(const Spectrum& in, float* img) {
    const size_t wc = size_t(in.w) / 2 + 1;
    pocketfft::shape_t shape{size_t(in.h), size_t(in.w)};
    pocketfft::stride_t sin{ptrdiff_t(sizeof(cf) * wc), ptrdiff_t(sizeof(cf))};
    pocketfft::stride_t sout{ptrdiff_t(sizeof(float) * in.w), ptrdiff_t(sizeof(float))};
    pocketfft::c2r(shape, sin, sout, pocketfft::shape_t{0, 1}, pocketfft::BACKWARD, in.c.data(), img, 1.0f, 1);
}

}  // namespace ms
