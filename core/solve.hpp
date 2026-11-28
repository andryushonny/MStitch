// Global positions from pairwise offsets: weighted least squares with a weak pull to the stage positions,
// robust re-weighting against wrong pairs.
#pragma once
#include <array>
#include <vector>

#include "core/register.hpp"

namespace ms {

using Vec2 = std::array<double, 2>;

// A point of a frame that must land at a known place of the map (a lithography cross: control point).
struct Anchor {
    int tile = -1;          // frame index
    double u = 0, v = 0;    // where it is in the frame (working-level pixels)
    double x = 0, y = 0;    // where it must be on the map (same coordinates as the positions)
    double weight = 4.0;    // relative to a perfect pair (1)
};

struct SolveParams {
    std::vector<Anchor> anchors;
    double prior_weight = 1e-3;  // pull to the stage position, relative to a perfect pair (weight 1)
    double min_ncc = 0.3;        // pairs below are not used
    double outlier_px = 1.5;     // residual above max(outlier_px, 5 * robust sigma) -> pair dropped
    int rounds = 5;
};

struct SolveStats {
    int pairs_used = 0, pairs_dropped = 0, rounds = 0;
    double rms_px = 0;
};

// pred: stage-predicted top-left of each tile (working level). Returns solved top-left positions and
// marks edges used / resid.
std::vector<Vec2> solve_positions(const std::vector<Vec2>& pred, std::vector<Edge>& edges,
                                  const SolveParams& p, SolveStats* stats);

}  // namespace ms
