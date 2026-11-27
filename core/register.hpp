// Pairwise registration: phase correlation for candidates, normalized cross-correlation to choose and refine.
#pragma once
#include "core/tile.hpp"
#include "cpu/fft.hpp"

namespace ms {

// Measured offset of tile j relative to tile i at the working level: pixel (x, y) of j shows the same place
// as pixel (x + tx, y + ty) of i, i.e. pos_j - pos_i = (tx, ty).
struct Edge {
    int i = -1, j = -1;
    double pred_x = 0, pred_y = 0;  // offset predicted from the stage
    double tx = 0, ty = 0;          // measured offset (subpixel)
    double ncc = 0;                 // normalized cross-correlation at the chosen offset
    long overlap = 0;               // overlap in pixels
    bool found = false;             // a candidate passed the checks
    bool used = false;              // kept by the global solve
    double resid = 0;               // |pos_j - pos_i - t| after the solve
};

struct RegParams {
    double search_px = 40;          // accepted |t - pred| per axis (ignored when gated = false)
    bool gated = true;              // false: no stage prior (calibration)
    int peaks = 8;                  // phase-correlation peaks tried
    double min_overlap = 0.03;      // minimal overlap as a fraction of the tile area
};

void tile_spectrum(const Tile& t, Spectrum& out);
// The k highest peaks of the phase correlation of two spectra (index y * w + x), as the GPU computes them.
void correlation_peaks(const Spectrum& fa, const Spectrum& fb, int k, int* peaks);
// From the peaks: candidates (4 wrap-arounds each), stage gate, NCC choice, climb, sub-pixel. Fills e.
void choose_from_peaks(const Tile& a, const Tile& b, const int* peaks, int k, Edge& e, const RegParams& p);
// Both of the above on the CPU.
void register_pair(const Tile& a, const Tile& b, const Spectrum& fa, const Spectrum& fb, Edge& e,
                   const RegParams& p);
// NCC of b against a at integer offset (tx, ty); overlap pixel count in *n.
double ncc_at(const Tile& a, const Tile& b, int tx, int ty, long* n);

}  // namespace ms
