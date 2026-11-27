#include "core/solve.hpp"

#include <algorithm>
#include <cmath>

namespace ms {

namespace {
// Solve (w0 I + sum_e w_e (e_i - e_j)(e_i - e_j)^T) x = b by conjugate gradients; x holds the start.
void cg(int n, const std::vector<Edge>& edges, const std::vector<double>& w, double w0, const std::vector<double>& b,
        std::vector<double>& x) {
    auto mul = [&](const std::vector<double>& v, std::vector<double>& out) {
        for (int k = 0; k < n; ++k) out[k] = w0 * v[k];
        for (size_t e = 0; e < edges.size(); ++e) {
            if (w[e] <= 0) continue;
            const int i = edges[e].i, j = edges[e].j;
            const double d = w[e] * (v[j] - v[i]);
            out[j] += d;
            out[i] -= d;
        }
    };
    std::vector<double> r(n), q(n), d(n);
    mul(x, q);
    double rr = 0, bb = 0;
    for (int k = 0; k < n; ++k) {
        r[k] = b[k] - q[k];
        d[k] = r[k];
        rr += r[k] * r[k];
        bb += b[k] * b[k];
    }
    const double tol = 1e-20 * std::max(bb, 1.0);
    for (int it = 0; it < 20 * n + 100 && rr > tol; ++it) {
        mul(d, q);
        double dq = 0;
        for (int k = 0; k < n; ++k) dq += d[k] * q[k];
        const double a = rr / dq;
        double rr2 = 0;
        for (int k = 0; k < n; ++k) {
            x[k] += a * d[k];
            r[k] -= a * q[k];
            rr2 += r[k] * r[k];
        }
        const double beta = rr2 / rr;
        rr = rr2;
        for (int k = 0; k < n; ++k) d[k] = r[k] + beta * d[k];
    }
}
}  // namespace

std::vector<Vec2> solve_positions(const std::vector<Vec2>& pred, std::vector<Edge>& edges,
                                  const SolveParams& p, SolveStats* stats) {
    const int n = int(pred.size());
    std::vector<Vec2> pos = pred;
    for (auto& e : edges) e.used = e.found && e.ncc >= p.min_ncc;
    int round = 0;
    for (; round < p.rounds; ++round) {
        std::vector<double> w(edges.size(), 0.0);
        for (size_t k = 0; k < edges.size(); ++k)
            if (edges[k].used) w[k] = edges[k].ncc * edges[k].ncc;
        for (int axis = 0; axis < 2; ++axis) {
            std::vector<double> b(n), x(n);
            for (int k = 0; k < n; ++k) {
                b[k] = p.prior_weight * pred[k][axis];
                x[k] = pos[k][axis];
            }
            for (size_t k = 0; k < edges.size(); ++k) {
                if (w[k] <= 0) continue;
                const double t = axis == 0 ? edges[k].tx : edges[k].ty;
                b[edges[k].j] += w[k] * t;
                b[edges[k].i] -= w[k] * t;
            }
            cg(n, edges, w, p.prior_weight, b, x);
            for (int k = 0; k < n; ++k) pos[k][axis] = x[k];
        }
        // Residuals; drop pairs far off the consensus.
        std::vector<double> rs;
        for (auto& e : edges) {
            if (!e.found) continue;
            const double dx = pos[e.j][0] - pos[e.i][0] - e.tx, dy = pos[e.j][1] - pos[e.i][1] - e.ty;
            e.resid = std::hypot(dx, dy);
            if (e.used) rs.push_back(e.resid);
        }
        if (rs.empty()) break;
        std::vector<double> s = rs;
        std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
        const double sigma = 1.4826 * s[s.size() / 2];
        const double lim = std::max(p.outlier_px, 5 * sigma);
        bool changed = false;
        for (auto& e : edges)
            if (e.used && e.resid > lim) { e.used = false; changed = true; }
        if (!changed) { ++round; break; }
    }
    if (stats) {
        *stats = {};
        stats->rounds = round;
        double ss = 0;
        for (auto& e : edges) {
            if (e.used) { stats->pairs_used++; ss += e.resid * e.resid; }
            else if (e.found && e.ncc >= p.min_ncc) stats->pairs_dropped++;
        }
        stats->rms_px = stats->pairs_used ? std::sqrt(ss / stats->pairs_used) : 0;
    }
    return pos;
}

}  // namespace ms
