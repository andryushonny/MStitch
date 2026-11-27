// Python module mstitch._core (nanobind). The Python package (mstitch/) wraps it.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>

#include "core/session.hpp"
#include "cuda/api.hpp"
#include "cuda/gpu.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Hand a vector to numpy without copying it again.
template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& v, std::initializer_list<size_t> shape) {
    auto* heap = new std::vector<T>(std::move(v));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(heap->data(), shape, owner);
}

nb::dict edges_dict(const std::vector<ms::Edge>& es) {
    const size_t n = es.size();
    std::vector<int32_t> i(n), j(n);
    std::vector<double> pred(2 * n), t(2 * n), ncc(n), resid(n);
    std::vector<int64_t> ov(n);
    std::vector<uint8_t> found(n), used(n);
    for (size_t k = 0; k < n; ++k) {
        const auto& e = es[k];
        i[k] = e.i; j[k] = e.j;
        pred[2 * k] = e.pred_x; pred[2 * k + 1] = e.pred_y;
        t[2 * k] = e.tx; t[2 * k + 1] = e.ty;
        ncc[k] = e.ncc; resid[k] = e.resid; ov[k] = e.overlap;
        found[k] = e.found; used[k] = e.used;
    }
    nb::dict d;
    d["i"] = to_numpy(std::move(i), {n});
    d["j"] = to_numpy(std::move(j), {n});
    d["pred"] = to_numpy(std::move(pred), {n, 2});
    d["t"] = to_numpy(std::move(t), {n, 2});
    d["ncc"] = to_numpy(std::move(ncc), {n});
    d["overlap"] = to_numpy(std::move(ov), {n});
    d["found"] = to_numpy(std::move(found), {n});
    d["used"] = to_numpy(std::move(used), {n});
    d["resid"] = to_numpy(std::move(resid), {n});
    return d;
}

std::vector<ms::Vec2> to_vec2(nb::ndarray<double, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu> a) {
    std::vector<ms::Vec2> v(a.shape(0));
    for (size_t k = 0; k < v.size(); ++k) v[k] = {a(k, 0), a(k, 1)};
    return v;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "MStitch core (C++). Use the mstitch package, not this module directly.";

    nb::class_<ms::Session>(m, "Session")
        .def(nb::init<int, std::array<double, 4>, unsigned>(), "level"_a, "m"_a, "threads"_a = 0)
        .def_prop_ro("level", &ms::Session::level)
        .def_prop_ro("stage_matrix", &ms::Session::stage_matrix)
        .def("__len__", &ms::Session::size)
        .def("add_sensor",
             [](ms::Session& s, nb::bytes data, int w, int h, int bits, std::string fmt, std::string codec, double sx,
                double sy, bool moving, int row, int col) {
                 std::string buf(data.c_str(), data.size());
                 return s.add_sensor(std::move(buf), w, h, bits, std::move(fmt), std::move(codec), sx, sy, moving, row,
                                     col);
             },
             "data"_a, "w"_a, "h"_a, "bits"_a, "fmt"_a, "codec"_a, "sx"_a, "sy"_a, "moving"_a = false, "row"_a = -1,
             "col"_a = -1)
        .def("add_image",
             [](ms::Session& s, nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> img, double sx, double sy, bool moving,
                int row, int col) {
                 if (img.ndim() != 2 && img.ndim() != 3) throw nb::value_error("image must be HxW or HxWx3");
                 const int h = int(img.shape(0)), w = int(img.shape(1));
                 const int ch = img.ndim() == 3 ? int(img.shape(2)) : 1;
                 int bps;
                 if (img.dtype() == nb::dtype<uint8_t>()) bps = 1;
                 else if (img.dtype() == nb::dtype<uint16_t>()) bps = 2;
                 else throw nb::type_error("image must be uint8 or uint16");
                 std::vector<uint8_t> buf(img.nbytes());
                 std::memcpy(buf.data(), img.data(), buf.size());
                 return s.add_image(std::move(buf), w, h, ch, bps, sx, sy, moving, row, col);
             },
             "image"_a, "sx"_a, "sy"_a, "moving"_a = false, "row"_a = -1, "col"_a = -1)
        .def("wait", &ms::Session::wait, nb::call_guard<nb::gil_scoped_release>())
        .def("pending", &ms::Session::pending)
        .def("tile_info",
             [](const ms::Session& s, size_t i) {
                 const auto& t = s.tile(i);
                 nb::dict d;
                 d["w"] = t.w; d["h"] = t.h; d["full_w"] = t.full_w; d["full_h"] = t.full_h;
                 d["sx"] = t.sx; d["sy"] = t.sy; d["moving"] = t.moving; d["row"] = t.row; d["col"] = t.col;
                 d["ready"] = t.ready.load(); d["error"] = t.failed.load() ? t.error : std::string();
                 d["prep_ms"] = t.prep_ms;
                 return d;
             })
        .def("tile_rgb",
             [](const ms::Session& s, size_t i) {
                 const auto& t = s.tile(i);
                 std::vector<uint8_t> v = t.rgb;
                 return to_numpy(std::move(v), {size_t(t.h), size_t(t.w), 3});
             })
        .def("predicted",
             [](const ms::Session& s) {
                 auto p = s.predicted();
                 std::vector<double> v(2 * p.size());
                 for (size_t k = 0; k < p.size(); ++k) { v[2 * k] = p[k][0]; v[2 * k + 1] = p[k][1]; }
                 return to_numpy(std::move(v), {p.size(), 2});
             })
        .def("register",
             [](ms::Session& s, double search_px, bool gated, int peaks, double min_overlap, std::string backend) {
                 ms::RegParams p;
                 p.search_px = search_px; p.gated = gated; p.peaks = peaks; p.min_overlap = min_overlap;
                 {
                     nb::gil_scoped_release r;
                     s.register_pairs(p, backend);
                 }
                 return edges_dict(s.edges());
             },
             "search_px"_a, "gated"_a = true, "peaks"_a = 8, "min_overlap"_a = 0.03, "backend"_a = "auto")
        .def_prop_ro("threads", [](const ms::Session& s) { return s.threads(); })
        .def_prop_ro("backend_used", &ms::Session::backend_used)
        .def_prop_ro("backend_note", &ms::Session::backend_note)
        .def("edges", [](const ms::Session& s) { return edges_dict(s.edges()); })
        .def("solve",
             [](ms::Session& s, double prior_weight, double min_ncc, double outlier_px, int rounds) {
                 ms::SolveParams p;
                 p.prior_weight = prior_weight; p.min_ncc = min_ncc; p.outlier_px = outlier_px; p.rounds = rounds;
                 ms::SolveStats st;
                 std::vector<ms::Vec2> pos;
                 {
                     nb::gil_scoped_release r;
                     pos = s.solve(p, &st);
                 }
                 std::vector<double> v(2 * pos.size());
                 for (size_t k = 0; k < pos.size(); ++k) { v[2 * k] = pos[k][0]; v[2 * k + 1] = pos[k][1]; }
                 nb::dict stats;
                 stats["pairs_used"] = st.pairs_used; stats["pairs_dropped"] = st.pairs_dropped;
                 stats["rounds"] = st.rounds; stats["rms_px"] = st.rms_px;
                 return nb::make_tuple(to_numpy(std::move(v), {pos.size(), 2}), stats, edges_dict(s.edges()));
             },
             "prior_weight"_a = 1e-3, "min_ncc"_a = 0.3, "outlier_px"_a = 1.5, "rounds"_a = 5)
        .def("estimate_flat",
             [](const ms::Session& s, std::vector<int> idx) {
                 std::vector<const ms::Tile*> ts;
                 for (int k : idx)
                     if (s.tile(k).ready) ts.push_back(&s.tile(k));
                 std::vector<float> f;
                 {
                     nb::gil_scoped_release r;
                     f = ms::estimate_flat(ts);
                 }
                 const size_t h = ts.empty() ? 0 : ts[0]->h, w = ts.empty() ? 0 : ts[0]->w;
                 return to_numpy(std::move(f), {h, w, 3});
             })
        .def("render",
             [](const ms::Session& s, std::vector<int> idx,
                nb::ndarray<double, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu> pos,
                std::optional<nb::ndarray<float, nb::c_contig, nb::device::cpu>> flat, double x0, double y0, int w,
                int h, std::string backend) {
                 std::vector<const ms::Tile*> ts;
                 std::vector<ms::Vec2> p;
                 auto all = to_vec2(pos);
                 for (int k : idx)
                     if (s.tile(k).ready) { ts.push_back(&s.tile(k)); p.push_back(all.at(k)); }
                 std::vector<float> fl;
                 if (flat) fl.assign(flat->data(), flat->data() + flat->size());
                 ms::CanvasImage c;
                 std::string used = "cpu", note;
                 {
                     nb::gil_scoped_release r;
                     if (backend != "cpu") {
                         std::string why;
                         if (auto* g = ms::cuda::gpu(&why)) {
                             try {
                                 ms::cuda::RenderStats st;
                                 c = ms::cuda::gpu_render(g, ts, p, fl, x0, y0, w, h, &st);
                                 used = "cuda";
                                 note = std::to_string(st.bands) + " bands of " + std::to_string(st.rows_per_band) +
                                        " rows, " + std::to_string(st.slots) + " frame slots, " +
                                        std::to_string(int(st.upload_mb)) + " MB up, " + std::to_string(int(st.ms)) +
                                        " ms";
                             } catch (const std::exception& e) {
                                 why = e.what();
                             }
                         }
                         if (used == "cpu") {
                             if (backend == "cuda") throw std::runtime_error("CUDA render unavailable: " + why);
                             note = why;
                         }
                     }
                     if (used == "cpu") c = ms::render(ts, p, fl, x0, y0, w, h);
                 }
                 const size_t cw = c.w, ch = c.h;
                 return nb::make_tuple(to_numpy(std::move(c.rgb), {ch, cw, 3}), to_numpy(std::move(c.owner), {ch, cw}),
                                       c.ox, c.oy, used, note);
             },
             "idx"_a, "pos"_a, "flat"_a.none(), "x0"_a = 0.0, "y0"_a = 0.0, "w"_a = 0, "h"_a = 0,
             "backend"_a = "auto");

    m.def("cuda_load", &ms::cuda::load, "driver"_a, "nvrtc"_a, "cufft"_a,
          "Load the CUDA driver, NVRTC and cuFFT from the first path that opens in each list.");
    m.def("cuda_device", []() -> nb::object {
        std::string why;
        ms::cuda::Gpu* g;
        {
            nb::gil_scoped_release r;
            g = ms::cuda::gpu(&why);
        }
        nb::dict d;
        if (!g) {
            d["ok"] = false;
            d["why"] = why;
            return d;
        }
        auto dev = ms::cuda::device_info(g);
        d["ok"] = true;
        d["name"] = dev.name;
        d["cc"] = nb::make_tuple(dev.cc_major, dev.cc_minor);
        d["free_mb"] = double(dev.free_bytes) / 1e6;
        d["total_mb"] = double(dev.total_bytes) / 1e6;
        return d;
    });
    m.def("cuda_compile_check", &ms::cuda::compile_check, "arch"_a,
          "Compile the kernels for sm_<arch> with NVRTC only (no GPU needed): '' when fine, else the log.");
}
