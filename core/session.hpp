// A stitching session: frames in (any order, any thread), registration, global solve, mosaic out.
#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/pool.hpp"
#include "core/register.hpp"
#include "core/render.hpp"
#include "core/solve.hpp"

namespace ms {

class Session {
public:
    // level: working level (1 = half size, 2 = quarter, ...); m: full-resolution pixels per um of stage motion,
    // row-major 2x2 [[mxx, mxy], [myx, myy]]; threads: 0 = all cores.
    Session(int level, std::array<double, 4> m, unsigned threads);

    int level() const { return level_; }
    std::array<double, 4> stage_matrix() const { return m_; }

    // Queue a frame; decoding and preparation run on the pool. Returns the frame index.
    int add_sensor(std::string data, int w, int h, int bits, std::string fmt, std::string codec, double sx, double sy,
                   bool moving, int row, int col);
    int add_image(std::vector<uint8_t> data, int w, int h, int channels, int bytes_per_sample, double sx, double sy,
                  bool moving, int row, int col);
    void wait();

    size_t size() const { return tiles_.size(); }
    size_t pending() const;
    unsigned threads() const { return pool_.size(); }   // frames still being unpacked / prepared
    const Tile& tile(size_t i) const { return *tiles_[i]; }
    // Brightness of a frame evened out: each channel c times exp(g[3c] + g[3c+1] x + g[3c+2] y), x and y from
    // -1 to 1 across the frame (gain compensation between frames).
    void gain_tile(size_t i, const std::array<double, 9>& g);

    // Stage-predicted top-left of each frame at the working level (frame centre = stage position).
    std::vector<Vec2> predicted() const;
    // Find overlapping pairs (from the stage prediction, or all pairs when !p.gated) and register them.
    // backend: "cpu", "cuda" (fails without a GPU) or "auto" (GPU when there is one).
    const std::vector<Edge>& register_pairs(const RegParams& p, const std::string& backend = "auto");
    // What the last registration ran on, and why (GPU name, batches, or the reason for the CPU).
    const std::string& backend_used() const { return backend_used_; }
    const std::string& backend_note() const { return backend_note_; }
    const std::vector<Edge>& edges() const { return edges_; }
    std::vector<Edge>& edges_mut() { return edges_; }

    std::vector<Vec2> solve(const SolveParams& p, SolveStats* stats);
    std::vector<const Tile*> ready_tiles() const;

private:
    int level_;
    std::array<double, 4> m_;
    Pool pool_;
    std::vector<std::unique_ptr<Tile>> tiles_;
    std::mutex add_m_;
    std::vector<Edge> edges_;
    std::string backend_used_, backend_note_;

    void register_cpu(const RegParams& p);
    void register_gpu(const RegParams& p, void* gpu);
};

}  // namespace ms
