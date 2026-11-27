// 2-D real FFT on the CPU (pocketfft). The CUDA backend (cuda/) implements the same calls with cuFFT.
#pragma once
#include <complex>
#include <vector>

namespace ms {

using cf = std::complex<float>;

struct Spectrum {
    int h = 0, w = 0;          // real size; the spectrum is h x (w/2+1)
    std::vector<cf> c;
};

// Forward r2c of an h x w float image.
void fft_forward(const float* img, int h, int w, Spectrum& out);
// Inverse c2r to an h x w float image (unnormalized: multiply by 1/(h*w) if needed).
void fft_inverse(const Spectrum& in, float* img);

}  // namespace ms
