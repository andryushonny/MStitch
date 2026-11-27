#include "core/render.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/pool.hpp"

namespace ms {

void canvas_box(const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos, double& x0, double& y0, int& w,
                int& h) {
    if (w <= 0 || h <= 0) {
        double mx = 1e300, my = 1e300, Mx = -1e300, My = -1e300;
        for (size_t k = 0; k < tiles.size(); ++k) {
            mx = std::min(mx, pos[k][0]);
            my = std::min(my, pos[k][1]);
            Mx = std::max(Mx, pos[k][0] + tiles[k]->w);
            My = std::max(My, pos[k][1] + tiles[k]->h);
        }
        x0 = std::floor(mx);
        y0 = std::floor(my);
        w = int(std::ceil(Mx - x0));
        h = int(std::ceil(My - y0));
    }
}

CanvasImage render(const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                   const std::vector<float>& flat, double x0, double y0, int w, int h) {
    CanvasImage c;
    if (tiles.empty()) return c;
    canvas_box(tiles, pos, x0, y0, w, h);
    c.w = w;
    c.h = h;
    c.ox = x0;
    c.oy = y0;
    c.rgb.assign(size_t(w) * h * 3, 0);
    c.owner.assign(size_t(w) * h, -1);

    const int band = 64;
    const int nb = (h + band - 1) / band;
    parallel_for(size_t(nb), [&](size_t bi) {
        const int ya = int(bi) * band, yb = std::min(h, ya + band);
        std::vector<float> best(size_t(w) * (yb - ya), std::numeric_limits<float>::max());
        for (size_t k = 0; k < tiles.size(); ++k) {
            const Tile& t = *tiles[k];
            const double px = pos[k][0] - x0, py = pos[k][1] - y0;  // tile top-left on the canvas
            const double cx = px + 0.5 * (t.w - 1), cy = py + 0.5 * (t.h - 1);
            const int ty0 = std::max(ya, int(std::ceil(py))), ty1 = std::min(yb, int(std::floor(py + t.h - 1)) + 1);
            const int tx0 = std::max(0, int(std::ceil(px))), tx1 = std::min(w, int(std::floor(px + t.w - 1)) + 1);
            const bool has_flat = flat.size() == size_t(t.w) * t.h * 3;
            for (int y = ty0; y < ty1; ++y) {
                const double sy = y - py;
                const int iy = std::min(int(sy), t.h - 2);
                const float fy = float(sy - iy);
                for (int x = tx0; x < tx1; ++x) {
                    const float d = float((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    float& b = best[size_t(y - ya) * w + x];
                    if (d >= b) continue;
                    b = d;
                    const double sx = x - px;
                    const int ix = std::min(int(sx), t.w - 2);
                    const float fx = float(sx - ix);
                    uint8_t* out = &c.rgb[(size_t(y) * w + x) * 3];
                    for (int ch = 0; ch < 3; ++ch) {
                        auto at = [&](int yy, int xx) {
                            const size_t i = (size_t(yy) * t.w + xx) * 3 + ch;
                            float v = float(t.rgb[i]);
                            if (has_flat) v /= std::max(flat[i], 0.05f);
                            return v;
                        };
                        const float v = (1 - fy) * ((1 - fx) * at(iy, ix) + fx * at(iy, ix + 1)) +
                                        fy * ((1 - fx) * at(iy + 1, ix) + fx * at(iy + 1, ix + 1));
                        out[ch] = uint8_t(std::clamp(v + 0.5f, 0.0f, 255.0f));
                    }
                    c.owner[size_t(y) * w + x] = int32_t(k);
                }
            }
        }
    });
    return c;
}

std::vector<float> estimate_flat(const std::vector<const Tile*>& tiles) {
    if (tiles.empty()) return {};
    const int w = tiles[0]->w, h = tiles[0]->h;
    std::vector<const Tile*> ts;
    std::vector<std::array<double, 3>> mean;
    for (auto* t : tiles) {
        if (t->w != w || t->h != h) continue;
        std::array<double, 3> m{0, 0, 0};
        for (size_t i = 0; i < size_t(w) * h; ++i)
            for (int c = 0; c < 3; ++c) m[c] += t->rgb[i * 3 + c];
        for (int c = 0; c < 3; ++c) m[c] = std::max(m[c] / (double(w) * h), 1.0);
        ts.push_back(t);
        mean.push_back(m);
    }
    std::vector<float> flat(size_t(w) * h * 3, 1.0f);
    if (ts.size() < 3) return flat;
    parallel_for(size_t(h), [&](size_t y) {
        std::vector<float> v(ts.size());
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                const size_t i = (y * w + x) * 3 + c;
                for (size_t k = 0; k < ts.size(); ++k) v[k] = float(ts[k]->rgb[i] / mean[k][c]);
                std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                flat[i] = v[v.size() / 2];
            }
    });
    // Normalise each channel to mean 1.
    for (int c = 0; c < 3; ++c) {
        double s = 0;
        for (size_t i = 0; i < size_t(w) * h; ++i) s += flat[i * 3 + c];
        const float m = float(s / (double(w) * h));
        for (size_t i = 0; i < size_t(w) * h; ++i) flat[i * 3 + c] /= m;
    }
    return flat;
}

}  // namespace ms
