#include "core/tile.hpp"

#include <algorithm>
#include <cmath>

namespace ms {

namespace {
// One box-blur pass of radius r along rows then columns, edges clamped.
void box_pass(std::vector<float>& a, int w, int h, int r) {
    std::vector<float> tmp(a.size());
    const float inv = 1.0f / float(2 * r + 1);
    for (int y = 0; y < h; ++y) {
        const float* row = &a[size_t(y) * w];
        float s = 0;
        for (int k = -r; k <= r; ++k) s += row[std::clamp(k, 0, w - 1)];
        for (int x = 0; x < w; ++x) {
            tmp[size_t(y) * w + x] = s * inv;
            s += row[std::min(x + r + 1, w - 1)] - row[std::max(x - r, 0)];
        }
    }
    for (int x = 0; x < w; ++x) {
        float s = 0;
        for (int k = -r; k <= r; ++k) s += tmp[size_t(std::clamp(k, 0, h - 1)) * w + x];
        for (int y = 0; y < h; ++y) {
            a[size_t(y) * w + x] = s * inv;
            s += tmp[size_t(std::min(y + r + 1, h - 1)) * w + x] - tmp[size_t(std::max(y - r, 0)) * w + x];
        }
    }
}
}  // namespace

void prepare_tile(Tile& t, const Rgbf& img) {
    t.w = img.w;
    t.h = img.h;
    const size_t n = size_t(img.w) * img.h;
    t.rgb.resize(n * 3);
    std::vector<float> g(n);
    for (size_t i = 0; i < n; ++i) {
        const float* p = &img.px[i * 3];
        for (int c = 0; c < 3; ++c) t.rgb[i * 3 + c] = uint8_t(std::clamp(p[c] * 255.0f + 0.5f, 0.0f, 255.0f));
        g[i] = 0.25f * p[0] + 0.5f * p[1] + 0.25f * p[2];
    }
    // High-pass: divide by the local mean (removes vignetting and illumination), zero mean.
    std::vector<float> m = g;
    const int r = std::max(3, std::min(img.w, img.h) / 40);
    for (int k = 0; k < 3; ++k) box_pass(m, img.w, img.h, r);
    double mean = 0;
    for (size_t i = 0; i < n; ++i) {
        g[i] = m[i] > 1e-4f ? g[i] / m[i] - 1.0f : 0.0f;
        mean += g[i];
    }
    mean /= double(n);
    t.reg.resize(n);
    for (size_t i = 0; i < n; ++i)
        t.reg[i] = int16_t(std::clamp((g[i] - float(mean)) * kRegScale, -32767.0f, 32767.0f));
    t.ready = true;
}

}  // namespace ms
