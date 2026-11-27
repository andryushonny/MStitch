// Frames from MControl (packed Bayer / mono) or plain images -> RGB at a pyramid level.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ms {

// Colour image, float in [0, 1], interleaved RGB, w*h*3.
struct Rgbf {
    int w = 0, h = 0;
    std::vector<float> px;
};

// Sensor pixels (uint16, also for 8-bit sensors) from MControl's packing (meta "codec"):
// raw = as the sensor gives it; lossless = zstd of the difference to the same-colour pixel two to the left
// (first two columns: two rows up), wrapping. `bits` <= 8 -> 1 byte per pixel, else 2 (little endian).
std::vector<uint16_t> unpack(const uint8_t* data, size_t n, int w, int h, int bits, const std::string& codec);

// Sensor pixels -> RGB at level >= 1 (level 1 = 2x2 superpixel, exact; each further level halves by a box filter).
// fmt: "bayer_rg8", "bayer_gb12", "mono8", ...
Rgbf sensor_to_rgb(const std::vector<uint16_t>& px, int w, int h, int bits, const std::string& fmt, int level);

// Plain image (uint8 or uint16, 1 or 3 channels) -> RGB at level >= 0 (level k = 2^k box filter).
Rgbf image_to_rgb(const void* data, int w, int h, int channels, int bytes_per_sample, int level);

// Halve by a 2x2 box filter.
Rgbf halve(const Rgbf& a);

}  // namespace ms
