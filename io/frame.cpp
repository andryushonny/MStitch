#include "io/frame.hpp"

#include <stdexcept>
#include <zstd.h>

namespace ms {

std::vector<uint16_t> unpack(const uint8_t* data, size_t n, int w, int h, int bits, const std::string& codec) {
    const size_t npx = size_t(w) * h;
    const int bpp = bits > 8 ? 2 : 1;
    const size_t nbytes = npx * bpp;
    std::vector<uint8_t> buf;
    if (codec == "raw") {
        if (n < nbytes) throw std::runtime_error("raw frame shorter than w*h*bpp");
        buf.assign(data, data + nbytes);
    } else if (codec == "lossless") {
        // The server falls back to the bare difference when zstd fails: check the zstd magic number.
        const bool zst = n >= 4 && data[0] == 0x28 && data[1] == 0xB5 && data[2] == 0x2F && data[3] == 0xFD;
        if (zst) {
            buf.resize(nbytes);
            size_t got = ZSTD_decompress(buf.data(), nbytes, data, n);
            if (ZSTD_isError(got) || got != nbytes) throw std::runtime_error("lossless frame: zstd failed");
        } else {
            if (n < nbytes) throw std::runtime_error("lossless frame shorter than w*h*bpp");
            buf.assign(data, data + nbytes);
        }
    } else {
        throw std::runtime_error("codec '" + codec + "' is not exact: snap with codec lossless or raw");
    }

    std::vector<uint16_t> out(npx);
    if (bpp == 1) {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t i = size_t(y) * w + x;
                uint8_t v = buf[i];
                if (codec == "lossless") {
                    uint8_t p = x >= 2 ? uint8_t(out[i - 2]) : (y >= 2 ? uint8_t(out[i - 2 * size_t(w)]) : 0);
                    v = uint8_t(v + p);
                }
                out[i] = v;
            }
    } else {
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t i = size_t(y) * w + x;
                uint16_t v = uint16_t(buf[2 * i] | (buf[2 * i + 1] << 8));
                if (codec == "lossless") {
                    uint16_t p = x >= 2 ? out[i - 2] : (y >= 2 ? out[i - 2 * size_t(w)] : 0);
                    v = uint16_t(v + p);
                }
                out[i] = v;
            }
    }
    return out;
}

Rgbf halve(const Rgbf& a) {
    Rgbf o;
    o.w = a.w / 2;
    o.h = a.h / 2;
    o.px.resize(size_t(o.w) * o.h * 3);
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x)
            for (int c = 0; c < 3; ++c) {
                auto at = [&](int yy, int xx) { return a.px[(size_t(yy) * a.w + xx) * 3 + c]; };
                o.px[(size_t(y) * o.w + x) * 3 + c] =
                    0.25f * (at(2 * y, 2 * x) + at(2 * y, 2 * x + 1) + at(2 * y + 1, 2 * x) + at(2 * y + 1, 2 * x + 1));
            }
    return o;
}

Rgbf sensor_to_rgb(const std::vector<uint16_t>& px, int w, int h, int bits, const std::string& fmt, int level) {
    if (level < 1) throw std::runtime_error("sensor frames: level must be >= 1 (2x2 superpixel)");
    const float scale = 1.0f / float((1u << (bits > 0 ? bits : 8)) - 1);
    Rgbf o;
    o.w = w / 2;
    o.h = h / 2;
    o.px.resize(size_t(o.w) * o.h * 3);
    if (fmt.rfind("bayer_", 0) == 0 && fmt.size() >= 8) {
        // (row, col) of R and B in the 2x2 cell; the two G are the other two
        const std::string p = fmt.substr(6, 2);
        int ry, rx, by, bx;
        if (p == "rg") { ry = 0; rx = 0; by = 1; bx = 1; }
        else if (p == "gr") { ry = 0; rx = 1; by = 1; bx = 0; }
        else if (p == "gb") { ry = 1; rx = 0; by = 0; bx = 1; }
        else if (p == "bg") { ry = 1; rx = 1; by = 0; bx = 0; }
        else throw std::runtime_error("unknown Bayer pattern in " + fmt);
        const int g1y = ry, g1x = 1 - rx, g2y = 1 - ry, g2x = rx;
        for (int y = 0; y < o.h; ++y)
            for (int x = 0; x < o.w; ++x) {
                auto at = [&](int dy, int dx) { return float(px[size_t(2 * y + dy) * w + 2 * x + dx]); };
                float* d = &o.px[(size_t(y) * o.w + x) * 3];
                d[0] = at(ry, rx) * scale;
                d[1] = 0.5f * (at(g1y, g1x) + at(g2y, g2x)) * scale;
                d[2] = at(by, bx) * scale;
            }
    } else if (fmt.rfind("mono", 0) == 0) {
        for (int y = 0; y < o.h; ++y)
            for (int x = 0; x < o.w; ++x) {
                auto at = [&](int dy, int dx) { return float(px[size_t(2 * y + dy) * w + 2 * x + dx]); };
                float v = 0.25f * (at(0, 0) + at(0, 1) + at(1, 0) + at(1, 1)) * scale;
                float* d = &o.px[(size_t(y) * o.w + x) * 3];
                d[0] = d[1] = d[2] = v;
            }
    } else {
        throw std::runtime_error("unknown pixel format " + fmt);
    }
    for (int l = 1; l < level; ++l) o = halve(o);
    return o;
}

Rgbf image_to_rgb(const void* data, int w, int h, int channels, int bytes_per_sample, int level) {
    if (channels != 1 && channels != 3) throw std::runtime_error("image must have 1 or 3 channels");
    Rgbf o;
    o.w = w;
    o.h = h;
    o.px.resize(size_t(w) * h * 3);
    const float scale = bytes_per_sample == 1 ? 1.0f / 255.0f : 1.0f / 65535.0f;
    for (size_t i = 0; i < size_t(w) * h; ++i)
        for (int c = 0; c < 3; ++c) {
            size_t s = i * channels + (channels == 3 ? c : 0);
            float v = bytes_per_sample == 1 ? float(static_cast<const uint8_t*>(data)[s])
                                            : float(static_cast<const uint16_t*>(data)[s]);
            o.px[i * 3 + c] = v * scale;
        }
    for (int l = 0; l < level; ++l) o = halve(o);
    return o;
}

}  // namespace ms
