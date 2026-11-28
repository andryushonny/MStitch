// Objects on the map (crystals, crosses, flakes, dust) kept whole: each is taken from ONE frame, the one that
// holds all of it and has it nearest its centre, so no seam runs through it and its shape and size are those
// of a single frame. Seams then run only through the quiet background between objects.
//
// Found per frame at the working level: grey (flat-corrected), its slowly varying background (box mean) taken
// out, what stands out of the noise by more than k sigma is an object; specks smaller than min_area dropped;
// grown by `margin` pixels (the seam keeps that far away). The pieces seen in different frames are joined on
// the map (overlapping masks), each joined object goes to the frame that contains it whole.
#pragma once
#include <cstdint>
#include <vector>

#include "core/render.hpp"

namespace ms {

struct EntityParams {
    double k = 6.0;            // object: differs from the background by more than k * noise sigma
    float min_contrast = 6.0f; // ... and by at least this (0..255 grey)
    int margin = 6;            // pixels kept clear around an object (working level)
    int min_area = 9;          // smaller specks are not objects (pixels)
    int bg_radius = 0;         // background box radius; 0 = max(w, h) / 16 of a frame
};

struct Entity {
    int owner = -1;                  // Tile::id it is taken from; -1 = no frame holds it whole
    int x0 = 0, y0 = 0, w = 0, h = 0;  // box on the map (working-level pixels, absolute)
    std::vector<uint8_t> mask;       // w*h, 1 = part of the object (with its margin)
    double cx = 0, cy = 0;           // centre (absolute)
};

struct Entities {
    std::vector<Entity> list;
    int whole = 0;      // taken whole from one frame
    int too_big = 0;    // larger than any frame holds: seams as usual there
};

Entities find_entities(const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                       const std::vector<float>& flat, const EntityParams& p);

// Over a rendered canvas: each object's pixels from its own frame (same arithmetic as render).
void apply_entities(CanvasImage& c, const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                    const std::vector<float>& flat, const Entities& e);

// Soft seams in the quiet background: near a seam (where another frame's centre is less than `band` pixels
// farther than the nearest one's) the frames are blended, weight band - (d_k - d_nearest), so a small brightness
// difference between frames fades instead of stepping. Objects (with their margin) are never blended: they stay
// whole from one frame. band <= 0: nothing.
void feather_seams(CanvasImage& c, const std::vector<const Tile*>& tiles, const std::vector<Vec2>& pos,
                   const std::vector<float>& flat, const Entities* e, double band);

}  // namespace ms
