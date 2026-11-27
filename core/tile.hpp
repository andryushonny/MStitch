// One frame of a scan, prepared at the working level.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "io/frame.hpp"

namespace ms {

constexpr float kRegScale = 8192.0f;  // reg is stored as int16: value * kRegScale

struct Tile {
    int id = -1;
    double sx = 0, sy = 0;       // stage position (um) at mid-exposure, as MControl reports it
    bool moving = false;         // the stage was moving during the exposure
    int row = -1, col = -1;      // grid index if the caller has one (informational)
    int w = 0, h = 0;            // size at the working level
    int full_w = 0, full_h = 0;  // size at full resolution
    std::vector<uint8_t> rgb;    // colour view at the level, not flat-corrected, w*h*3
    std::vector<int16_t> reg;    // high-passed grey for registration, zero mean, w*h
    bool ready = false;
    std::string error;
};

// Fill tile.rgb and tile.reg from a colour image at the working level.
void prepare_tile(Tile& t, const Rgbf& img);

}  // namespace ms
