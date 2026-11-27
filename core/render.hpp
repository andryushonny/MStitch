// Mosaic from solved positions. Each canvas pixel comes from ONE frame: the one whose centre is nearest
// (Voronoi seams), so a residual misalignment gives a step on a seam, never a double image.
#pragma once
#include <vector>

#include "core/solve.hpp"
#include "core/tile.hpp"

namespace ms {

struct CanvasImage {
    int w = 0, h = 0;
    double ox = 0, oy = 0;          // canvas pixel (0, 0) is at this position (working level)
    std::vector<uint8_t> rgb;       // w*h*3
    std::vector<int32_t> owner;     // frame (Tile::id) each pixel came from, -1 = none
};

// Bounding box of all frames when w <= 0 (whole pixels), else the given region.
void canvas_box(const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos, double& x0, double& y0, int& w,
                int& h);

// flat: per-pixel gain of a frame (w*h*3, mean 1) or empty; the view is divided by it.
// region: x0, y0, w, h in working-level coordinates, or w <= 0 for the bounding box of all frames.
CanvasImage render(const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                   const std::vector<float>& flat, double x0, double y0, int w, int h);

// Per-pixel median of view / mean(view) over the given frames: the illumination (flat-field), mean 1.
std::vector<float> estimate_flat(const std::vector<const Tile*>& tiles);

}  // namespace ms
