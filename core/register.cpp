#include "core/register.hpp"

#include <algorithm>
#include <cmath>

namespace ms {

void tile_spectrum(const Tile& t, Spectrum& out) {
    std::vector<float> f(t.reg.size());
    for (size_t i = 0; i < f.size(); ++i) f[i] = float(t.reg[i]) / kRegScale;
    fft_forward(f.data(), t.h, t.w, out);
}

double ncc_at(const Tile& a, const Tile& b, int tx, int ty, long* n) {
    const int x0 = std::max(0, -tx), x1 = std::min(b.w, a.w - tx);
    const int y0 = std::max(0, -ty), y1 = std::min(b.h, a.h - ty);
    if (n) *n = 0;
    if (x1 <= x0 || y1 <= y0) return -1;
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (int y = y0; y < y1; ++y) {
        const int16_t* pb = &b.reg[size_t(y) * b.w];
        const int16_t* pa = &a.reg[size_t(y + ty) * a.w + tx];
        long long ra = 0, rb = 0, raa = 0, rbb = 0, rab = 0;
        for (int x = x0; x < x1; ++x) {
            const long long va = pa[x], vb = pb[x];
            ra += va; rb += vb; raa += va * va; rbb += vb * vb; rab += va * vb;
        }
        sa += double(ra); sb += double(rb); saa += double(raa); sbb += double(rbb); sab += double(rab);
    }
    const double cnt = double(x1 - x0) * double(y1 - y0);
    if (n) *n = long(cnt);
    const double va = saa - sa * sa / cnt, vb = sbb - sb * sb / cnt;
    if (va <= 0 || vb <= 0) return -1;
    return (sab - sa * sb / cnt) / std::sqrt(va * vb);
}

namespace {
// Vertex of the parabola through (-1, m), (0, z), (+1, p), limited to [-0.5, 0.5].
double parabola(double m, double z, double p) {
    const double d = m - 2 * z + p;
    if (d >= 0) return 0;
    return std::clamp(0.5 * (m - p) / d, -0.5, 0.5);
}
}  // namespace

void register_pair(const Tile& a, const Tile& b, const Spectrum& fa, const Spectrum& fb, Edge& e,
                   const RegParams& p) {
    e.found = false;
    if (a.w != b.w || a.h != b.h) return;
    const int H = a.h, W = a.w;
    // Normalized cross-power spectrum -> correlation surface; its peak sits at t mod (W, H).
    Spectrum c;
    c.h = H;
    c.w = W;
    c.c.resize(fa.c.size());
    for (size_t k = 0; k < c.c.size(); ++k) {
        cf v = fa.c[k] * std::conj(fb.c[k]);
        const float m = std::abs(v);
        c.c[k] = m > 1e-12f ? v / m : cf(0, 0);
    }
    std::vector<float> r(size_t(H) * W);
    fft_inverse(c, r.data());

    const long min_n = long(p.min_overlap * double(W) * double(H));
    double best = -2;
    int bx = 0, by = 0;
    long bn = 0;
    for (int k = 0; k < p.peaks; ++k) {
        size_t im = size_t(std::max_element(r.begin(), r.end()) - r.begin());
        const int u = int(im / W), v = int(im % W);
        for (int dy = -3; dy <= 3; ++dy)  // suppress this peak (wrapping)
            for (int dx = -3; dx <= 3; ++dx)
                r[size_t((u + dy + H) % H) * W + (v + dx + W) % W] = -1e30f;
        for (int ty : {u, u - H})
            for (int tx : {v, v - W}) {
                if (p.gated && (std::abs(tx - e.pred_x) > p.search_px || std::abs(ty - e.pred_y) > p.search_px))
                    continue;
                long n = 0;
                const double s = ncc_at(a, b, tx, ty, &n);
                if (n < min_n) continue;
                if (s > best) { best = s; bx = tx; by = ty; bn = n; }
            }
    }
    if (best <= -1) return;
    // Local refinement: climb to the integer NCC maximum (the peak can be off by a pixel), then a parabola.
    for (int it = 0; it < 8; ++it) {
        int nx = bx, ny = by;
        double nb = best;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (!dx && !dy) continue;
                long n = 0;
                const double s = ncc_at(a, b, bx + dx, by + dy, &n);
                if (n >= min_n && s > nb) { nb = s; nx = bx + dx; ny = by + dy; }
            }
        if (nx == bx && ny == by) break;
        bx = nx; by = ny; best = nb;
    }
    const double xm = ncc_at(a, b, bx - 1, by, nullptr), xp = ncc_at(a, b, bx + 1, by, nullptr);
    const double ym = ncc_at(a, b, bx, by - 1, nullptr), yp = ncc_at(a, b, bx, by + 1, nullptr);
    e.tx = bx + parabola(xm, best, xp);
    e.ty = by + parabola(ym, best, yp);
    e.ncc = best;
    e.overlap = bn;
    e.found = true;
}

}  // namespace ms
