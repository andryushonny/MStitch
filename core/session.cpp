#include "core/session.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>

namespace ms {

Session::Session(int level, std::array<double, 4> m, unsigned threads) : level_(level), m_(m), pool_(threads) {
    if (level < 0 || level > 5) throw std::runtime_error("level must be 0..5");
}

int Session::add_sensor(std::string data, int w, int h, int bits, std::string fmt, std::string codec, double sx,
                        double sy, bool moving, int row, int col) {
    Tile* t;
    int id;
    {
        std::lock_guard<std::mutex> lk(add_m_);
        id = int(tiles_.size());
        tiles_.push_back(std::make_unique<Tile>());
        t = tiles_.back().get();
    }
    t->id = id; t->sx = sx; t->sy = sy; t->moving = moving; t->row = row; t->col = col;
    t->full_w = w; t->full_h = h;
    const int level = std::max(1, level_);
    pool_.submit([t, data = std::move(data), w, h, bits, fmt = std::move(fmt), codec = std::move(codec), level] {
        try {
            auto px = unpack(reinterpret_cast<const uint8_t*>(data.data()), data.size(), w, h, bits, codec);
            prepare_tile(*t, sensor_to_rgb(px, w, h, bits, fmt, level));
        } catch (const std::exception& e) {
            t->error = e.what();
        }
    });
    return id;
}

int Session::add_image(std::vector<uint8_t> data, int w, int h, int channels, int bps, double sx, double sy,
                       bool moving, int row, int col) {
    Tile* t;
    int id;
    {
        std::lock_guard<std::mutex> lk(add_m_);
        id = int(tiles_.size());
        tiles_.push_back(std::make_unique<Tile>());
        t = tiles_.back().get();
    }
    t->id = id; t->sx = sx; t->sy = sy; t->moving = moving; t->row = row; t->col = col;
    t->full_w = w; t->full_h = h;
    const int level = level_;
    pool_.submit([t, data = std::move(data), w, h, channels, bps, level] {
        try {
            prepare_tile(*t, image_to_rgb(data.data(), w, h, channels, bps, level));
        } catch (const std::exception& e) {
            t->error = e.what();
        }
    });
    return id;
}

void Session::wait() { pool_.wait(); }

std::vector<Vec2> Session::predicted() const {
    const double s = 1.0 / double(1 << level_);
    std::vector<Vec2> p(tiles_.size());
    for (size_t k = 0; k < tiles_.size(); ++k) {
        const Tile& t = *tiles_[k];
        const double fx = m_[0] * t.sx + m_[1] * t.sy, fy = m_[2] * t.sx + m_[3] * t.sy;  // full-res px
        p[k] = {fx * s - 0.5 * t.w, fy * s - 0.5 * t.h};
    }
    return p;
}

std::vector<const Tile*> Session::ready_tiles() const {
    std::vector<const Tile*> v;
    for (auto& t : tiles_) if (t->ready) v.push_back(t.get());
    return v;
}

const std::vector<Edge>& Session::register_pairs(const RegParams& p) {
    wait();
    const auto pred = predicted();
    const int n = int(tiles_.size());
    edges_.clear();
    for (int i = 0; i < n; ++i) {
        const Tile& a = *tiles_[i];
        if (!a.ready) continue;
        for (int j = i + 1; j < n; ++j) {
            const Tile& b = *tiles_[j];
            if (!b.ready || a.w != b.w || a.h != b.h) continue;
            Edge e;
            e.i = i; e.j = j;
            e.pred_x = pred[j][0] - pred[i][0];
            e.pred_y = pred[j][1] - pred[i][1];
            if (p.gated) {
                const double ow = a.w - std::max(0.0, std::abs(e.pred_x) - p.search_px);
                const double oh = a.h - std::max(0.0, std::abs(e.pred_y) - p.search_px);
                if (ow <= 0 || oh <= 0 || ow * oh < p.min_overlap * a.w * a.h) continue;
            }
            edges_.push_back(e);
        }
    }
    // Each frame's spectrum is computed once and freed after its last pair (Blattner et al. 2014):
    // pairs sorted by their later frame keep only a band of spectra alive.
    std::sort(edges_.begin(), edges_.end(), [](const Edge& x, const Edge& y) {
        return std::max(x.i, x.j) != std::max(y.i, y.j) ? std::max(x.i, x.j) < std::max(y.i, y.j) : x.i < y.i;
    });
    std::vector<std::atomic<int>> refs(n);
    for (auto& r : refs) r = 0;
    for (auto& e : edges_) { refs[e.i]++; refs[e.j]++; }
    std::vector<std::shared_ptr<Spectrum>> spec(n);
    std::vector<std::once_flag> once(n);
    std::mutex sm;
    auto get = [&](int k) {
        std::call_once(once[k], [&] {
            auto s = std::make_shared<Spectrum>();
            tile_spectrum(*tiles_[k], *s);
            std::lock_guard<std::mutex> lk(sm);
            spec[k] = s;
        });
        std::lock_guard<std::mutex> lk(sm);
        return spec[k];
    };
    auto release = [&](int k) {
        if (--refs[k] == 0) {
            std::lock_guard<std::mutex> lk(sm);
            spec[k].reset();
        }
    };
    parallel_for(edges_.size(), [&](size_t k) {
        Edge& e = edges_[k];
        auto fa = get(e.i), fb = get(e.j);
        register_pair(*tiles_[e.i], *tiles_[e.j], *fa, *fb, e, p);
        fa.reset(); fb.reset();
        release(e.i);
        release(e.j);
    }, pool_.size());
    return edges_;
}

std::vector<Vec2> Session::solve(const SolveParams& p, SolveStats* stats) {
    return solve_positions(predicted(), edges_, p, stats);
}

}  // namespace ms
