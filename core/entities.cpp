#include "core/entities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "core/pool.hpp"

namespace ms {

namespace {

// A piece of an object as one frame sees it, on the map.
struct Piece {
    int x0, y0, x1, y1;          // box, absolute, x1 / y1 exclusive
    std::vector<uint8_t> mask;   // (x1-x0)*(y1-y0)
};

// Box mean of v (w x h) with radius r, cut at the edges.
std::vector<float> box_mean(const std::vector<float>& v, int w, int h, int r) {
    std::vector<double> s(size_t(w + 1) * (h + 1), 0.0);
    for (int y = 0; y < h; ++y) {
        double row = 0;
        for (int x = 0; x < w; ++x) {
            row += v[size_t(y) * w + x];
            s[size_t(y + 1) * (w + 1) + x + 1] = s[size_t(y) * (w + 1) + x + 1] + row;
        }
    }
    std::vector<float> o(v.size());
    for (int y = 0; y < h; ++y) {
        const int ya = std::max(0, y - r), yb = std::min(h, y + r + 1);
        for (int x = 0; x < w; ++x) {
            const int xa = std::max(0, x - r), xb = std::min(w, x + r + 1);
            const double sum = s[size_t(yb) * (w + 1) + xb] - s[size_t(ya) * (w + 1) + xb] -
                               s[size_t(yb) * (w + 1) + xa] + s[size_t(ya) * (w + 1) + xa];
            o[size_t(y) * w + x] = float(sum / double((xb - xa) * (yb - ya)));
        }
    }
    return o;
}

// 8-connected components of m (w x h): label per pixel (0 = none) and the count.
int label(const std::vector<uint8_t>& m, int w, int h, std::vector<int>& lab) {
    lab.assign(m.size(), 0);
    int n = 0;
    std::vector<int> stack;
    for (int i = 0; i < w * h; ++i) {
        if (!m[i] || lab[i]) continue;
        lab[i] = ++n;
        stack.push_back(i);
        while (!stack.empty()) {
            const int q = stack.back();
            stack.pop_back();
            const int qx = q % w, qy = q / w;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = qx + dx, y = qy + dy;
                    if (x < 0 || y < 0 || x >= w || y >= h) continue;
                    const int j = y * w + x;
                    if (m[j] && !lab[j]) {
                        lab[j] = n;
                        stack.push_back(j);
                    }
                }
        }
    }
    return n;
}

// The objects of one frame, as pieces on the map.
std::vector<Piece> pieces(const Tile& t, const Vec2& p, const std::vector<float>& flat, const EntityParams& prm) {
    const int w = t.w, h = t.h;
    const bool has_flat = flat.size() == size_t(w) * h * 3;
    std::vector<float> g(size_t(w) * h);
    for (size_t i = 0; i < g.size(); ++i) {
        float s = 0;
        for (int c = 0; c < 3; ++c) {
            float v = t.rgb[i * 3 + c];
            if (has_flat) v /= std::max(flat[i * 3 + c], 0.05f);
            s += v;
        }
        g[i] = s / 3.0f;
    }
    const int r = prm.bg_radius > 0 ? prm.bg_radius : std::max(8, std::max(w, h) / 16);
    const auto bg = box_mean(g, w, h, r);
    std::vector<float> c(g.size());
    for (size_t i = 0; i < g.size(); ++i) c[i] = std::fabs(g[i] - bg[i]);
    // noise: the median contrast is the background's (objects cover far less than half a frame)
    std::vector<float> sample;
    for (size_t i = 0; i < c.size(); i += 7) sample.push_back(c[i]);
    std::nth_element(sample.begin(), sample.begin() + sample.size() / 2, sample.end());
    const float sigma = 1.4826f * sample[sample.size() / 2];
    const float thr = std::max(float(prm.k) * sigma, prm.min_contrast);
    std::vector<uint8_t> m(g.size());
    for (size_t i = 0; i < c.size(); ++i) m[i] = c[i] > thr;
    // specks out
    std::vector<int> lab;
    const int n = label(m, w, h, lab);
    std::vector<int> area(n + 1, 0);
    for (int l : lab) area[l]++;
    std::vector<float> keep(g.size());
    for (size_t i = 0; i < m.size(); ++i) keep[i] = (lab[i] && area[lab[i]] >= prm.min_area) ? 1.0f : 0.0f;
    // grown by the margin (any kept pixel within it)
    const auto grown = box_mean(keep, w, h, prm.margin);
    std::vector<uint8_t> d(g.size());
    for (size_t i = 0; i < d.size(); ++i) d[i] = grown[i] > 1e-6f;
    const int nd = label(d, w, h, lab);
    std::vector<std::array<int, 4>> box(nd + 1, {w, h, -1, -1});
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (int l = lab[size_t(y) * w + x]) {
                auto& b = box[l];
                b[0] = std::min(b[0], x);
                b[1] = std::min(b[1], y);
                b[2] = std::max(b[2], x);
                b[3] = std::max(b[3], y);
            }
    // on the map: frame pixel (x, y) at p + (x, y), rounded
    const int ox = int(std::lround(p[0])), oy = int(std::lround(p[1]));
    std::vector<Piece> out;
    for (int l = 1; l <= nd; ++l) {
        const auto& b = box[l];
        Piece pc{ox + b[0], oy + b[1], ox + b[2] + 1, oy + b[3] + 1, {}};
        const int bw = b[2] - b[0] + 1, bh = b[3] - b[1] + 1;
        pc.mask.assign(size_t(bw) * bh, 0);
        for (int y = b[1]; y <= b[3]; ++y)
            for (int x = b[0]; x <= b[2]; ++x)
                if (lab[size_t(y) * w + x] == l) pc.mask[size_t(y - b[1]) * bw + (x - b[0])] = 1;
        out.push_back(std::move(pc));
    }
    return out;
}

bool masks_touch(const Piece& a, const Piece& b) {
    const int x0 = std::max(a.x0, b.x0), x1 = std::min(a.x1, b.x1);
    const int y0 = std::max(a.y0, b.y0), y1 = std::min(a.y1, b.y1);
    if (x0 >= x1 || y0 >= y1) return false;
    const int aw = a.x1 - a.x0, bw = b.x1 - b.x0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x)
            if (a.mask[size_t(y - a.y0) * aw + (x - a.x0)] && b.mask[size_t(y - b.y0) * bw + (x - b.x0)]) return true;
    return false;
}

int find(std::vector<int>& par, int i) {
    while (par[i] != i) i = par[i] = par[par[i]];
    return i;
}

}  // namespace

Entities find_entities(const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                       const std::vector<float>& flat, const EntityParams& p) {
    Entities out;
    std::vector<std::vector<Piece>> per(tiles.size());
    parallel_for(tiles.size(), [&](size_t k) { per[k] = pieces(*tiles[k], pos[k], flat, p); });
    std::vector<Piece> all;
    for (auto& v : per)
        for (auto& pc : v) all.push_back(std::move(pc));
    // the same object seen in several frames: pieces whose masks overlap are one
    std::vector<int> order(all.size()), par(all.size());
    std::iota(order.begin(), order.end(), 0);
    std::iota(par.begin(), par.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return all[a].x0 < all[b].x0; });
    for (size_t i = 0; i < order.size(); ++i) {
        const Piece& a = all[order[i]];
        for (size_t j = i + 1; j < order.size() && all[order[j]].x0 < a.x1; ++j) {
            const Piece& b = all[order[j]];
            if (b.y0 >= a.y1 || a.y0 >= b.y1) continue;
            if (find(par, order[i]) != find(par, order[j]) && masks_touch(a, b))
                par[find(par, order[i])] = find(par, order[j]);
        }
    }
    std::vector<std::vector<int>> groups(all.size());
    for (size_t i = 0; i < all.size(); ++i) groups[find(par, int(i))].push_back(int(i));
    for (auto& gr : groups) {
        if (gr.empty()) continue;
        Entity e;
        int x0 = all[gr[0]].x0, y0 = all[gr[0]].y0, x1 = all[gr[0]].x1, y1 = all[gr[0]].y1;
        for (int i : gr) {
            x0 = std::min(x0, all[i].x0);
            y0 = std::min(y0, all[i].y0);
            x1 = std::max(x1, all[i].x1);
            y1 = std::max(y1, all[i].y1);
        }
        e.x0 = x0;
        e.y0 = y0;
        e.w = x1 - x0;
        e.h = y1 - y0;
        e.mask.assign(size_t(e.w) * e.h, 0);
        double sx = 0, sy = 0, n = 0;
        for (int i : gr) {
            const Piece& pc = all[i];
            const int pw = pc.x1 - pc.x0;
            for (int y = pc.y0; y < pc.y1; ++y)
                for (int x = pc.x0; x < pc.x1; ++x)
                    if (pc.mask[size_t(y - pc.y0) * pw + (x - pc.x0)]) {
                        uint8_t& m = e.mask[size_t(y - y0) * e.w + (x - x0)];
                        if (!m) {
                            m = 1;
                            sx += x;
                            sy += y;
                            n += 1;
                        }
                    }
        }
        e.cx = sx / std::max(n, 1.0);
        e.cy = sy / std::max(n, 1.0);
        // the frame that holds it whole (one pixel inside its edge, for the bilinear neighbour), its centre nearest
        double best = 1e300;
        for (size_t k = 0; k < tiles.size(); ++k) {
            const Tile& t = *tiles[k];
            const double px = pos[k][0], py = pos[k][1];
            if (x0 < px || y0 < py || x1 - 1 > px + t.w - 2 || y1 - 1 > py + t.h - 2) continue;
            const double dx = e.cx - (px + 0.5 * (t.w - 1)), dy = e.cy - (py + 0.5 * (t.h - 1));
            const double d = dx * dx + dy * dy;
            if (d < best) {
                best = d;
                e.owner = t.id;
            }
        }
        if (e.owner >= 0)
            out.whole++;
        else
            out.too_big++;
        out.list.push_back(std::move(e));
    }
    return out;
}

void apply_entities(CanvasImage& c, const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                    const std::vector<float>& flat, const Entities& ents) {
    if (c.w <= 0 || c.h <= 0) return;
    const int cx0 = int(std::lround(c.ox)), cy0 = int(std::lround(c.oy));
    for (const Entity& e : ents.list) {
        if (e.owner < 0) continue;
        if (e.x0 >= cx0 + c.w || e.y0 >= cy0 + c.h || e.x0 + e.w <= cx0 || e.y0 + e.h <= cy0) continue;
        size_t k = 0;
        while (k < tiles.size() && tiles[k]->id != e.owner) ++k;
        if (k == tiles.size()) continue;
        const Tile& t = *tiles[k];
        const bool has_flat = flat.size() == size_t(t.w) * t.h * 3;
        for (int my = 0; my < e.h; ++my) {
            const int Y = e.y0 + my, y = Y - cy0;
            if (y < 0 || y >= c.h) continue;
            for (int mx = 0; mx < e.w; ++mx) {
                if (!e.mask[size_t(my) * e.w + mx]) continue;
                const int X = e.x0 + mx, x = X - cx0;
                if (x < 0 || x >= c.w) continue;
                int32_t& own = c.owner[size_t(y) * c.w + x];
                if (own == t.id) continue;
                // the same arithmetic as render: canvas pixel X at the frame's X - pos
                const double sx = (c.ox + x) - pos[k][0], sy = (c.oy + y) - pos[k][1];
                if (sx < 0 || sy < 0 || sx > t.w - 1 || sy > t.h - 1) continue;
                const int ix = std::min(int(sx), t.w - 2), iy = std::min(int(sy), t.h - 2);
                const float fx = float(sx - ix), fy = float(sy - iy);
                uint8_t* out = &c.rgb[(size_t(y) * c.w + x) * 3];
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
                own = t.id;
            }
        }
    }
}

void feather_seams(CanvasImage& c, const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                   const std::vector<float>& flat, const Entities* ents, double band) {
    if (band <= 0 || c.w <= 0 || c.h <= 0) return;
    const size_t n = size_t(c.w) * c.h;
    const int cx0 = int(std::lround(c.ox)), cy0 = int(std::lround(c.oy));
    // objects: kept as they are
    std::vector<uint8_t> keep(n, 0);
    if (ents)
        for (const Entity& e : ents->list) {
            if (e.x0 >= cx0 + c.w || e.y0 >= cy0 + c.h || e.x0 + e.w <= cx0 || e.y0 + e.h <= cy0) continue;
            for (int my = 0; my < e.h; ++my) {
                const int y = e.y0 + my - cy0;
                if (y < 0 || y >= c.h) continue;
                for (int mx = 0; mx < e.w; ++mx) {
                    const int x = e.x0 + mx - cx0;
                    if (x >= 0 && x < c.w && e.mask[size_t(my) * e.w + mx]) keep[size_t(y) * c.w + x] = 1;
                }
            }
        }
    // the distance to the nearest frame's centre (that frame is the pixel's owner where nothing was kept)
    std::vector<float> dmin(n, std::numeric_limits<float>::max());
    auto centre = [&](size_t k) { return Vec2{pos[k][0] + 0.5 * (tiles[k]->w - 1), pos[k][1] + 0.5 * (tiles[k]->h - 1)}; };
    for (size_t k = 0; k < tiles.size(); ++k) {
        const Tile& t = *tiles[k];
        const Vec2 ct = centre(k);
        const double px = pos[k][0] - c.ox, py = pos[k][1] - c.oy;
        const int ty0 = std::max(0, int(std::ceil(py))), ty1 = std::min(c.h, int(std::floor(py + t.h - 1)) + 1);
        const int tx0 = std::max(0, int(std::ceil(px))), tx1 = std::min(c.w, int(std::floor(px + t.w - 1)) + 1);
        for (int y = ty0; y < ty1; ++y)
            for (int x = tx0; x < tx1; ++x) {
                const double dx = c.ox + x - ct[0], dy = c.oy + y - ct[1];
                float& d = dmin[size_t(y) * c.w + x];
                d = std::min(d, float(std::sqrt(dx * dx + dy * dy)));
            }
    }
    // blend: every frame within the band of the nearest, weighted
    std::vector<float> acc(n * 3, 0.0f), wsum(n, 0.0f);
    for (size_t k = 0; k < tiles.size(); ++k) {
        const Tile& t = *tiles[k];
        const bool has_flat = flat.size() == size_t(t.w) * t.h * 3;
        const Vec2 ct = centre(k);
        const double px = pos[k][0] - c.ox, py = pos[k][1] - c.oy;
        const int ty0 = std::max(0, int(std::ceil(py))), ty1 = std::min(c.h, int(std::floor(py + t.h - 1)) + 1);
        const int tx0 = std::max(0, int(std::ceil(px))), tx1 = std::min(c.w, int(std::floor(px + t.w - 1)) + 1);
        for (int y = ty0; y < ty1; ++y) {
            const double sy = y - py;
            const int iy = std::min(int(sy), t.h - 2);
            const float fy = float(sy - iy);
            for (int x = tx0; x < tx1; ++x) {
                const size_t i = size_t(y) * c.w + x;
                if (keep[i] || c.owner[i] < 0) continue;
                const double dx = c.ox + x - ct[0], dy = c.oy + y - ct[1];
                const float wgt = float(band) - (float(std::sqrt(dx * dx + dy * dy)) - dmin[i]);
                if (wgt <= 0) continue;
                // near the frame's own edge its weight goes to zero too (no hard edge where it ends)
                const float edge = float(std::min(std::min(sy, t.h - 1 - sy), std::min(x - px, t.w - 1 - (x - px))));
                const float w = wgt * std::clamp(edge / float(band), 0.0f, 1.0f);
                if (w <= 0) continue;
                const double sx = x - px;
                const int ix = std::min(int(sx), t.w - 2);
                const float fx = float(sx - ix);
                for (int ch = 0; ch < 3; ++ch) {
                    auto at = [&](int yy, int xx) {
                        const size_t q = (size_t(yy) * t.w + xx) * 3 + ch;
                        float v = float(t.rgb[q]);
                        if (has_flat) v /= std::max(flat[q], 0.05f);
                        return v;
                    };
                    const float v = (1 - fy) * ((1 - fx) * at(iy, ix) + fx * at(iy, ix + 1)) +
                                    fy * ((1 - fx) * at(iy + 1, ix) + fx * at(iy + 1, ix + 1));
                    acc[i * 3 + ch] += w * v;
                }
                wsum[i] += w;
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        if (keep[i] || wsum[i] <= 0) continue;
        for (int ch = 0; ch < 3; ++ch) c.rgb[i * 3 + ch] = uint8_t(std::clamp(acc[i * 3 + ch] / wsum[i] + 0.5f, 0.0f, 255.0f));
    }
}

}  // namespace ms
