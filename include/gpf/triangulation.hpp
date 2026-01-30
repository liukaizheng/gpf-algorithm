#pragma once
#include <gpf/mesh.hpp>
#include <initializer_list>
#include <predicates/generic_point_2d.hpp>
#include <predicates/predicates.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

namespace gpf {
namespace triangulation {

namespace ranges = std::ranges;
namespace views = std::views;

// Constants
inline constexpr std::size_t INVALID_IND = std::numeric_limits<std::size_t>::max();

// Helper functions
[[nodiscard]] inline constexpr std::size_t
twin_index(std::size_t idx) noexcept
{
    return idx ^ 1;
}

[[nodiscard]] inline constexpr bool
is_negative(std::size_t mark) noexcept
{
    return (mark & 1) != 0;
}

inline double
dot(const double* a, const double* b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Orientation helper functions (since predicates::Orientation is a simple enum)
[[nodiscard]] inline bool
is_positive(predicates::Orientation ori) noexcept
{
    return ori == predicates::Orientation::Positive;
}

[[nodiscard]] inline bool
is_negative_ori(predicates::Orientation ori) noexcept
{
    return ori == predicates::Orientation::Negative;
}

[[nodiscard]] inline bool
is_zero(predicates::Orientation ori) noexcept
{
    return ori == predicates::Orientation::Zero;
}

// Mesh type used for triangulation
using MeshType = gpf::ManifoldMesh<gpf::Empty, gpf::Empty, gpf::Empty, gpf::Empty>;

// Forward declarations
struct Triangulation;
struct CDT;

// Triangulation struct - implements divide-and-conquer Delaunay triangulation
struct Triangulation
{
    std::span<const double> points;
    MeshType mesh;
    std::vector<VertexId> sorted_vertices;

    explicit Triangulation(std::span<const double> pts)
      : points(pts)
    {
    }

    // Create a new edge between two vertices
    [[nodiscard]] HalfedgeId new_edge_by_vertices(VertexId va, VertexId vb)
    {
        HalfedgeId hid = mesh.new_edge();
        HalfedgeId twin_hid = mesh.he_twin(hid);
        mesh.halfedge_data(hid).vertex = vb;
        mesh.halfedge_data(twin_hid).vertex = va;
        if (va.valid()) {
            mesh.vertex_data(va).halfedge = hid;
        }
        if (vb.valid()) {
            mesh.vertex_data(vb).halfedge = twin_hid;
        }
        return hid;
    }

    // Set vertex halfedge
    void set_v_halfedge(VertexId vid, HalfedgeId hid)
    {
        if (vid.valid()) {
            mesh.vertex_data(vid).halfedge = hid;
        }
    }

    // Main triangulation entry point
    std::pair<HalfedgeId, MeshType> triangulate(bool is_horizontal)
    {
        const std::size_t n_points = points.size() >> 1;
        mesh.new_vertices(n_points);

        auto [bdy_hid, _] = div_conq_recurse(0, n_points, is_horizontal);
        return { bdy_hid, std::move(mesh) };
    }

    // Divide and conquer recursion
    std::array<HalfedgeId, 2> div_conq_recurse(std::size_t start, std::size_t end, bool is_horizontal)
    {
        const std::size_t len = end - start;

        if (len == 2) {
            VertexId va = sorted_vertices[start];
            VertexId vb = sorted_vertices[start + 1];
            auto [ha, hb, hc] = new_triangle(va, vb, VertexId{});
            HalfedgeId twin_ha = mesh.he_twin(ha);
            HalfedgeId twin_hb = mesh.he_twin(hb);
            HalfedgeId twin_hc = mesh.he_twin(hc);
            mesh.new_face_by_halfedges({ twin_hc, twin_hb, twin_ha });

            if (cmp(va, vb, !is_horizontal) < 0) {
                return { twin_hc, twin_hb };
            } else {
                return { hb, hc };
            }
        } else if (len == 3) {
            VertexId va = sorted_vertices[start];
            VertexId vb = sorted_vertices[start + 1];
            VertexId vc = sorted_vertices[start + 2];
            double area = counterclockwise(va, vb, vc);

            if (area > 0.0 || area < 0.0) {
                // Non-collinear points
                std::array<HalfedgeId, 3> halfedges;
                if (area > 0.0) {
                    halfedges = new_triangle(va, vb, vc);
                } else {
                    halfedges = new_triangle(va, vc, vb);
                }

                std::array<HalfedgeId, 3> side_halfedges;
                for (std::size_t i = 0; i < 3; ++i) {
                    VertexId vid = mesh.he_from(halfedges[i]);
                    side_halfedges[i] = new_edge_by_vertices(vid, VertexId{});
                }

                // Create boundary faces (circular tuple windows)
                for (std::size_t i = 0; i < 3; ++i) {
                    HalfedgeId ha = side_halfedges[i];
                    HalfedgeId hb_twin = side_halfedges[(i + 1) % 3];
                    HalfedgeId hc_twin = halfedges[i];
                    mesh.new_face_by_halfedges({ ha, mesh.he_twin(hb_twin), mesh.he_twin(hc_twin) });
                }

                // Find min and max vertices
                std::size_t min_idx = 0, max_idx = 0;
                for (std::size_t i = 1; i < 3; ++i) {
                    if (cmp(mesh.he_from(side_halfedges[i]), mesh.he_from(side_halfedges[min_idx]), !is_horizontal) <
                        0) {
                        min_idx = i;
                    }
                    if (cmp(mesh.he_from(side_halfedges[i]), mesh.he_from(side_halfedges[max_idx]), !is_horizontal) >
                        0) {
                        max_idx = i;
                    }
                }

                return { side_halfedges[min_idx], mesh.he_twin(side_halfedges[max_idx]) };
            } else {
                // Collinear points
                HalfedgeId ha = new_edge_by_vertices(va, vb);
                HalfedgeId hb = new_edge_by_vertices(vb, vc);
                std::array<HalfedgeId, 4> halfedges = { ha, hb, mesh.he_twin(hb), mesh.he_twin(ha) };

                std::array<HalfedgeId, 4> side_halfedges;
                for (std::size_t i = 0; i < 4; ++i) {
                    VertexId vid = mesh.he_from(halfedges[i]);
                    side_halfedges[i] = new_edge_by_vertices(vid, VertexId{});
                }

                for (std::size_t i = 0; i < 4; ++i) {
                    HalfedgeId ha_side = side_halfedges[i];
                    HalfedgeId hb_twin = side_halfedges[(i + 1) % 4];
                    HalfedgeId hc_twin = halfedges[i];
                    mesh.new_face_by_halfedges({ ha_side, mesh.he_twin(hb_twin), mesh.he_twin(hc_twin) });
                }

                if (cmp(va, vc, !is_horizontal) < 0) {
                    return { side_halfedges[0], mesh.he_twin(side_halfedges[2]) };
                } else {
                    return { side_halfedges[2], mesh.he_twin(side_halfedges[0]) };
                }
            }
        } else {
            // Divide and merge
            std::size_t mid = start + (len >> 1);
            auto [_, lr_hid] = div_conq_recurse(start, mid, !is_horizontal);
            auto [rl_hid, __] = div_conq_recurse(mid, end, !is_horizontal);
            return merge_hulls(lr_hid, rl_hid, is_horizontal);
        }
    }

    // Get vertices [dest, apex] from halfedge
    [[nodiscard]] std::array<VertexId, 2> he_dest_apex(HalfedgeId hid) const
    {
        HalfedgeId next_hid = mesh.he_next(hid);
        return { mesh.he_to(hid), mesh.he_to(next_hid) };
    }

    // Get vertices [apex, org] from halfedge
    [[nodiscard]] std::array<VertexId, 2> he_apex_org(HalfedgeId hid) const
    {
        HalfedgeId next_hid = mesh.he_next(hid);
        HalfedgeId nnext_hid = mesh.he_next(next_hid);
        return { mesh.he_to(next_hid), mesh.he_to(nnext_hid) };
    }

    // Get halfedge vertices
    [[nodiscard]] std::array<VertexId, 2> he_vertices(HalfedgeId hid) const
    {
        return { mesh.he_from(hid), mesh.he_to(hid) };
    }

    // Merge two convex hulls
    std::array<HalfedgeId, 2> merge_hulls(HalfedgeId lr_hid, HalfedgeId rl_hid, bool is_horizontal)
    {
        auto [lt_vid, lb_vid] = he_dest_apex(lr_hid);
        auto [rb_vid, rt_vid] = he_apex_org(rl_hid);

        // Find lower tangent
        while (true) {
            bool stop = true;
            if (counterclockwise(lt_vid, lb_vid, rt_vid) > 0.0) {
                stop = false;
                lr_hid = mesh.he_prev_twin(lr_hid);
                lt_vid = lb_vid;
                lb_vid = mesh.he_to_to(lr_hid);
            }

            if (counterclockwise(rb_vid, rt_vid, lt_vid) > 0.0) {
                stop = false;
                rl_hid = mesh.he_next_twin(rl_hid);
                rt_vid = rb_vid;
                rb_vid = mesh.he_to_to(rl_hid);
            }

            if (stop) {
                break;
            }
        }

        HalfedgeId left_hid = mesh.he_twin(lr_hid);
        HalfedgeId right_hid = mesh.he_twin(rl_hid);
        lb_vid = lt_vid;
        rb_vid = rt_vid;

        lt_vid = mesh.he_to_to(left_hid);
        rt_vid = mesh.he_to_to(right_hid);

        HalfedgeId bottom_hid = new_edge_by_vertices(rb_vid, lb_vid);
        {
            HalfedgeId ha = new_edge_by_vertices(lb_vid, VertexId{});
            HalfedgeId hb = new_edge_by_vertices(VertexId{}, rb_vid);
            mesh.new_face_by_halfedges({ bottom_hid, ha, hb });

            mesh.he_replace(lr_hid, mesh.he_twin(ha));
            mesh.he_replace(rl_hid, mesh.he_twin(hb));
        }
        HalfedgeId top_hid = mesh.he_twin(bottom_hid);

        // Merge loop
        while (true) {
            bool left_finished = counterclockwise(lt_vid, lb_vid, rb_vid) <= 0.0;
            bool right_finished = counterclockwise(rb_vid, rt_vid, lb_vid) <= 0.0;
            if (left_finished && right_finished) {
                break;
            }

            if (!left_finished) {
                HalfedgeId curr_hid = mesh.he_prev_twin(left_hid);
                while (true) {
                    VertexId apex_vid = mesh.he_to_to(curr_hid);
                    if (!apex_vid.valid()) {
                        break;
                    }

                    if (incircle(lb_vid, rb_vid, lt_vid, apex_vid) <= 0.0) {
                        break;
                    }

                    mesh.flip(curr_hid);
                    lt_vid = apex_vid;
                    curr_hid = mesh.he_next_twin(curr_hid);
                }
                left_hid = mesh.he_next(mesh.he_twin(curr_hid));
            }

            if (!right_finished) {
                HalfedgeId curr_hid = mesh.he_next_twin(right_hid);
                while (true) {
                    VertexId apex_vid = mesh.he_to_to(curr_hid);
                    if (!apex_vid.valid()) {
                        break;
                    }

                    if (incircle(lb_vid, rb_vid, rt_vid, apex_vid) <= 0.0) {
                        break;
                    }

                    mesh.flip(curr_hid);
                    rt_vid = apex_vid;
                    curr_hid = mesh.he_prev_twin(mesh.he_twin(curr_hid));
                }
                right_hid = mesh.he_prev(mesh.he_twin(curr_hid));
            }

            if (left_finished || (!right_finished && incircle(lt_vid, lb_vid, rb_vid, rt_vid) > 0.0)) {
                HalfedgeId prev_right_hid = mesh.he_prev(right_hid);
                mesh.he_replace(right_hid, top_hid);
                HalfedgeId new_hid = new_edge_by_vertices(rt_vid, lb_vid);
                mesh.he_replace(prev_right_hid, new_hid);

                top_hid = mesh.he_twin(new_hid);
                right_hid = mesh.he_twin(prev_right_hid);

                rb_vid = rt_vid;
                rt_vid = mesh.he_to_to(right_hid);
            } else {
                HalfedgeId next_left_hid = mesh.he_next(left_hid);
                mesh.he_replace(left_hid, top_hid);
                HalfedgeId new_hid = new_edge_by_vertices(rb_vid, lt_vid);
                mesh.he_replace(next_left_hid, new_hid);

                top_hid = mesh.he_twin(new_hid);
                left_hid = mesh.he_twin(next_left_hid);

                lb_vid = lt_vid;
                lt_vid = mesh.he_to_to(left_hid);
            }
        }

        mesh.new_face_by_halfedges({ top_hid, mesh.he_twin(right_hid), mesh.he_twin(left_hid) });

        if (is_horizontal) {
            top_hid =
              rotate_prev(top_hid, [](auto pa, auto pb) { return pa[1] < pb[1] || (pa[1] == pb[1] && pa[0] > pb[0]); });
            top_hid = rotate_next(
              top_hid, [](auto pa, auto pb) { return pa[1] >= pb[1] && (pa[1] != pb[1] || pa[0] <= pb[0]); });

            bottom_hid = rotate_next(
              bottom_hid, [](auto pa, auto pb) { return pa[1] <= pb[1] && (pa[1] != pb[1] || pa[0] >= pb[0]); });
            bottom_hid = rotate_prev(
              bottom_hid, [](auto pa, auto pb) { return pa[1] > pb[1] || (pa[1] == pb[1] && pa[0] < pb[0]); });

            return { mesh.he_next(bottom_hid), mesh.he_prev(top_hid) };
        } else {
            top_hid = rotate_next(
              top_hid, [](auto pa, auto pb) { return pa[0] <= pb[0] && (pa[0] != pb[0] || pa[1] <= pb[1]); });
            top_hid =
              rotate_prev(top_hid, [](auto pa, auto pb) { return pa[0] > pb[0] || (pa[0] == pb[0] && pa[1] > pb[1]); });

            bottom_hid = rotate_prev(
              bottom_hid, [](auto pa, auto pb) { return pa[0] < pb[0] || (pa[0] == pb[0] && pa[1] < pb[1]); });
            bottom_hid = rotate_next(
              bottom_hid, [](auto pa, auto pb) { return pa[0] >= pb[0] && (pa[0] != pb[0] || pa[1] >= pb[1]); });

            return { mesh.he_next(top_hid), mesh.he_prev(bottom_hid) };
        }
    }

    // Create a new triangle face
    [[nodiscard]] std::array<HalfedgeId, 3> new_triangle(VertexId va, VertexId vb, VertexId vc)
    {
        HalfedgeId ha = new_edge_by_vertices(va, vb);
        HalfedgeId hb = new_edge_by_vertices(vb, vc);
        HalfedgeId hc = new_edge_by_vertices(vc, va);
        std::array<HalfedgeId, 3> halfedges{ { ha, hb, hc } };
        mesh.new_face_by_halfedges(halfedges);
        return halfedges;
    }

    // Compare vertices
    [[nodiscard]] int cmp(VertexId va, VertexId vb, bool is_horizontal) const
    {
        const auto pa = predicates::point<2>(points.data(), va.idx);
        const auto pb = predicates::point<2>(points.data(), vb.idx);
        if (is_horizontal) {
            if (pa[0] != pb[0])
                return (pa[0] < pb[0]) ? -1 : 1;
            if (pa[1] != pb[1])
                return (pa[1] < pb[1]) ? -1 : 1;
            return 0;
        } else {
            if (pa[1] != pb[1])
                return (pa[1] < pb[1]) ? -1 : 1;
            if (-pa[0] != -pb[0])
                return (-pa[0] < -pb[0]) ? -1 : 1;
            return 0;
        }
    }

    // Rotate halfedge backwards until stop condition
    template<typename StopFn>
    [[nodiscard]] HalfedgeId rotate_prev(HalfedgeId hid, StopFn stop_fn) const
    {
        auto [va, vb] = he_vertices(hid);
        auto pa = predicates::point<2>(points.data(), va.idx);
        auto pb = predicates::point<2>(points.data(), vb.idx);

        while (!stop_fn(pa, pb)) {
            hid = mesh.he_prev(mesh.he_prev_twin(hid));
            pb = pa;
            pa = predicates::point<2>(points.data(), mesh.he_from(hid).idx);
        }
        return hid;
    }

    // Rotate halfedge forwards until stop condition
    template<typename StopFn>
    [[nodiscard]] HalfedgeId rotate_next(HalfedgeId hid, StopFn stop_fn) const
    {
        auto [va, vb] = he_vertices(hid);
        auto pa = predicates::point<2>(points.data(), va.idx);
        auto pb = predicates::point<2>(points.data(), vb.idx);

        while (!stop_fn(pa, pb)) {
            hid = mesh.he_next(mesh.he_next_twin(hid));
            pa = pb;
            pb = predicates::point<2>(points.data(), mesh.he_to(hid).idx);
        }
        return hid;
    }

    // Orient2d test (counterclockwise)
    [[nodiscard]] double counterclockwise(VertexId va, VertexId vb, VertexId vc) const
    {
        return predicates::orient2d(predicates::point<2>(points.data(), va.idx).data(),
                                    predicates::point<2>(points.data(), vb.idx).data(),
                                    predicates::point<2>(points.data(), vc.idx).data());
    }

    // Incircle test
    [[nodiscard]] double incircle(VertexId va, VertexId vb, VertexId vc, VertexId vd) const
    {
        return predicates::incircle(predicates::point<2>(points.data(), va.idx).data(),
                                    predicates::point<2>(points.data(), vb.idx).data(),
                                    predicates::point<2>(points.data(), vc.idx).data(),
                                    predicates::point<2>(points.data(), vd.idx).data());
    }
};

// Constrained Delaunay Triangulation
struct CDT
{
    std::span<const double> points;
    std::span<const std::size_t> segments;
    std::vector<predicates::Point2D> point_indices;
    MeshType mesh;
    std::vector<std::size_t> halfedge_marks;

    // Create a new edge between two vertices
    [[nodiscard]] HalfedgeId new_edge_by_vertices(VertexId va, VertexId vb)
    {
        HalfedgeId hid = mesh.new_edge();
        HalfedgeId twin_hid = mesh.he_twin(hid);
        mesh.halfedge_data(hid).vertex = vb;
        mesh.halfedge_data(twin_hid).vertex = va;
        if (va.valid()) {
            mesh.vertex_data(va).halfedge = hid;
        }
        if (vb.valid()) {
            mesh.vertex_data(vb).halfedge = twin_hid;
        }
        return hid;
    }

    // Set vertex halfedge
    void set_v_halfedge(VertexId vid, HalfedgeId hid)
    {
        if (vid.valid()) {
            mesh.vertex_data(vid).halfedge = hid;
        }
    }

    // Get he_to_to
    [[nodiscard]] VertexId he_to_to(HalfedgeId hid) const { return mesh.he_to(mesh.he_next(hid)); }

    // Get he_prev_twin
    [[nodiscard]] HalfedgeId he_prev_twin(HalfedgeId hid) const { return mesh.he_prev(mesh.he_twin(hid)); }

    // Get he_next_twin
    [[nodiscard]] HalfedgeId he_next_twin(HalfedgeId hid) const { return mesh.he_next(mesh.he_twin(hid)); }

    // Get he_twin_next
    [[nodiscard]] HalfedgeId he_twin_next(HalfedgeId hid) const { return mesh.he_twin(mesh.he_next(hid)); }

    // Get halfedge vertices
    [[nodiscard]] std::array<VertexId, 2> he_vertices(HalfedgeId hid) const
    {
        return { mesh.he_from(hid), mesh.he_to(hid) };
    }

    // Perform CDT
    void perform()
    {
        for (std::size_t idx = 0; idx < segments.size() / 2; ++idx) {
            VertexId va{ segments[idx * 2] };
            VertexId vb{ segments[idx * 2 + 1] };
            if (va != vb) {
                insert_segment(va, vb, idx << 1);
            }
        }
    }

    // Insert a constrained segment
    void insert_segment(VertexId va, VertexId vb, std::size_t mark)
    {
        HalfedgeId start_hid = scout_segment(mesh.v_halfedge(va), vb, mark);
        va = mesh.he_from(start_hid);
        if (va == vb)
            return;

        HalfedgeId end_hid = scout_segment(mesh.v_halfedge(vb), va, twin_index(mark));
        vb = mesh.he_from(end_hid);
        if (vb == va)
            return;

        constrain(start_hid, vb, mark);
    }

    // Scout for segment path
    HalfedgeId scout_segment(HalfedgeId hid, VertexId vb, std::size_t mark)
    {
        VertexId right_vid = mesh.he_to(hid);
        if (right_vid == vb) {
            set_edge_mark(hid, mark);
            return mesh.v_halfedge(vb);
        }

        VertexId va = mesh.he_from(hid);
        VertexId left_vid = he_to_to(hid);

        predicates::Orientation left_ori = orient(vb, va, left_vid);
        predicates::Orientation right_ori = orient(vb, va, right_vid);

        while (!is_positive(left_ori) && !is_positive(right_ori)) {
            hid = he_twin_next(hid);
            left_ori = right_ori;
            right_vid = mesh.he_to(hid);
            right_ori = orient(vb, va, right_vid);
        }

        while (is_negative_ori(right_ori) || is_positive(left_ori)) {
            hid = he_prev_twin(hid);
            right_ori = left_ori;
            left_vid = he_to_to(hid);
            left_ori = orient(vb, va, left_vid);
        }

        bool collinear = false;
        if (is_zero(left_ori)) {
            hid = he_prev_twin(hid);
            collinear = true;
        } else if (is_zero(right_ori)) {
            collinear = true;
        }

        if (collinear) {
            set_edge_mark(hid, mark);
            VertexId end_vid = mesh.he_to(hid);
            if (end_vid == vb) {
                return mesh.v_halfedge(vb);
            } else {
                return scout_segment(mesh.v_halfedge(end_vid), vb, mark);
            }
        } else {
            HalfedgeId split_hid = mesh.he_next(hid);
            if (halfedge_marks[split_hid.idx] == INVALID_IND) {
                return hid;
            } else {
                auto [_, new_hid] = split_edge(split_hid, mark);
                return scout_segment(new_hid, vb, mark);
            }
        }
    }

    // Apply constraint
    void constrain(HalfedgeId bottom_right_hid, VertexId vb, std::size_t mark)
    {
        VertexId va = mesh.he_from(bottom_right_hid);
        HalfedgeId flip_hid = mesh.he_next(bottom_right_hid);
        mesh.flip(flip_hid);

        while (true) {
            VertexId top_vid = mesh.he_from(flip_hid);
            if (top_vid == vb) {
                HalfedgeId fixup_hid = he_twin_next(flip_hid);
                fixup(flip_hid, false);
                fixup(fixup_hid, true);
                set_edge_mark(flip_hid, twin_index(mark));
                break;
            }

            predicates::Orientation ori = orient(va, vb, top_vid);
            if (is_zero(ori)) {
                HalfedgeId fixup_hid = he_twin_next(flip_hid);
                fixup(flip_hid, false);
                fixup(fixup_hid, true);
                set_edge_mark(flip_hid, twin_index(mark));

                HalfedgeId scout_hid = scout_segment(he_prev_twin(flip_hid), vb, mark);
                if (mesh.he_from(scout_hid) != vb) {
                    constrain(scout_hid, vb, mark);
                }
                break;
            } else {
                if (is_positive(ori)) {
                    HalfedgeId fixup_hid = he_twin_next(flip_hid);
                    fixup(fixup_hid, true);
                    flip_hid = mesh.he_prev(flip_hid);
                } else {
                    fixup(flip_hid, false);
                    flip_hid = he_twin_next(flip_hid);
                }

                if (halfedge_marks[flip_hid.idx] != INVALID_IND) {
                    auto [new_hid1, new_hid2] = split_edge(flip_hid, mark);
                    fixup(mesh.he_twin(new_hid1), false);
                    fixup(mesh.he_next(new_hid1), true);
                    HalfedgeId scout_hid = scout_segment(new_hid2, vb, mark);
                    if (mesh.he_from(scout_hid) != vb) {
                        constrain(scout_hid, vb, mark);
                    }
                    break;
                } else {
                    mesh.flip(flip_hid);
                }
            }
        }
    }

    // Fixup Delaunay edges
    void fixup(HalfedgeId hid, bool left_side)
    {
        HalfedgeId flip_hid = mesh.he_next(hid);
        if (halfedge_marks[flip_hid.idx] != INVALID_IND)
            return;

        HalfedgeId twin_flip_hid = mesh.he_twin(flip_hid);

        VertexId bottom_vid = he_to_to(twin_flip_hid);
        if (!bottom_vid.valid())
            return;

        VertexId top_vid = mesh.he_from(hid);
        auto [left_vid, right_vid] = he_vertices(flip_hid);

        if (left_side) {
            if (!is_positive(orient(top_vid, left_vid, bottom_vid)))
                return;
        } else {
            if (!is_positive(orient(bottom_vid, right_vid, top_vid)))
                return;
        }

        if (is_positive(orient(left_vid, bottom_vid, right_vid))) {
            if (!is_positive(incircle(left_vid, bottom_vid, right_vid, top_vid)))
                return;
        }

        mesh.flip(flip_hid);
        fixup(hid, left_side);
        fixup(twin_flip_hid, left_side);
    }

    // Split edge at intersection
    std::array<HalfedgeId, 2> split_edge(HalfedgeId hid, std::size_t input_mark)
    {
        HalfedgeId twin_hid = mesh.he_twin(hid);
        FaceId fid = mesh.he_face(hid);
        FaceId twin_fid = mesh.he_face(twin_hid);
        VertexId bottom_vid = he_to_to(hid);
        VertexId top_vid = he_to_to(twin_hid);

        VertexId left_vid = mesh.he_to(hid);

        std::size_t edge_mark = halfedge_marks[hid.idx];
        assert(edge_mark != INVALID_IND);

        VertexId new_vid = mesh.split_edge(mesh.he_edge(hid));

        const std::size_t* vab = &segments[(input_mark >> 1) << 1];
        const std::size_t* vcd = &segments[(edge_mark >> 1) << 1];

        point_indices.emplace_back(predicates::ImplicitPointSSI(vab[0], vab[1], vcd[0], vcd[1]));

        HalfedgeId new_hid = mesh.v_halfedge(new_vid);
        new_edge_callback(new_hid);

        if (mesh.he_to(new_hid) == left_vid) {
            set_edge_mark(new_hid, edge_mark);
        } else {
            set_edge_mark(new_hid, twin_index(edge_mark));
        }

        HalfedgeId new_hid1 = mesh.split_face(fid, bottom_vid, new_vid);
        new_edge_callback(new_hid1);
        set_edge_mark(new_hid1, input_mark);

        HalfedgeId new_hid2 = mesh.split_face(twin_fid, new_vid, top_vid);
        new_edge_callback(new_hid2);

        return { new_hid1, new_hid2 };
    }

    // Edge callback for new edges
    void new_edge_callback(HalfedgeId new_hid)
    {
        if (new_hid.idx >= halfedge_marks.size()) {
            halfedge_marks.resize(halfedge_marks.size() + 2, INVALID_IND);
        }
    }

    // Set edge mark
    void set_edge_mark(HalfedgeId hid, std::size_t mark)
    {
        halfedge_marks[hid.idx] = mark;
        halfedge_marks[mesh.he_twin(hid).idx] = twin_index(mark);
    }

    // Orient2d for Point2D
    [[nodiscard]] predicates::Orientation orient(VertexId va, VertexId vb, VertexId vc) const
    {
        return predicates::orient2d(point_indices[va.idx], point_indices[vb.idx], point_indices[vc.idx], points.data());
    }

    [[nodiscard]] predicates::Orientation incircle(VertexId va, VertexId vb, VertexId vc, VertexId vd) const
    {
        return predicates::incircle(
          point_indices[va.idx], point_indices[vb.idx], point_indices[vc.idx], point_indices[vd.idx], points.data());
    }

    // Check if face is a ghost (has invalid vertices)
    [[nodiscard]] bool face_is_ghost(FaceId fid) const
    {
        for (const auto& he : mesh.face(fid).halfedges()) {
            if (!he.to().id.valid())
                return true;
        }
        return false;
    }

    // Extract valid faces (not ghost, respecting constraint winding)
    [[nodiscard]] std::vector<FaceId> extract_invalid_faces() const
    {
        std::vector<FaceId> result;
        result.reserve(mesh.n_faces_capacity());

        std::vector<bool> visited(mesh.n_faces_capacity(), false);

        for (const auto& face : mesh.faces()) {
            FaceId fid = face.id;
            if (visited[fid.idx])
                continue;
            visited[fid.idx] = true;

            std::vector<FaceId> queue;
            queue.push_back(fid);
            bool keep = !face_is_ghost(fid);
            std::size_t queue_idx = 0;

            while (queue_idx < queue.size()) {
                FaceId curr_fid = queue[queue_idx];
                ++queue_idx;

                for (const auto& he : mesh.face(curr_fid).halfedges()) {
                    HalfedgeId hid = he.id;
                    std::size_t mark = halfedge_marks[hid.idx];

                    if (mark == INVALID_IND) {
                        FaceId adj_fid = he.twin().face().id;
                        if (visited[adj_fid.idx])
                            continue;
                        visited[adj_fid.idx] = true;

                        if (keep && face_is_ghost(adj_fid)) {
                            keep = false;
                        }
                        queue.push_back(adj_fid);
                    } else {
                        if (is_negative(halfedge_marks[hid.idx])) {
                            keep = false;
                        }
                    }
                }
            }

            if (keep) {
                result.insert(result.end(), queue.begin(), queue.end());
            }
        }
        return result;
    }
};

// Alternating axes sorting for divide-and-conquer
inline void
alternate_axes(const double* points, std::span<VertexId> indices, bool is_horizontal)
{
    using namespace predicates;
    const std::size_t len = indices.size();
    const std::size_t divider = len >> 1;

    bool actual_horizontal = (len <= 3) ? true : is_horizontal;

    if (actual_horizontal) {
        std::nth_element(
          indices.begin(), indices.begin() + divider, indices.end(), [points](const VertexId& i, const VertexId& j) {
              return ranges::lexicographical_compare(point<2>(points, i.idx), point<2>(points, j.idx));
          });
    } else {
        std::nth_element(
          indices.begin(), indices.begin() + divider, indices.end(), [points](const VertexId& i, const VertexId& j) {
              const auto pa = point<2>(points, i.idx);
              const auto pb = point<2>(points, j.idx);
              return ranges::lexicographical_compare(std::initializer_list{ pa[1], -pa[0] },
                                                     std::initializer_list{ pb[1], -pb[0] });
          });
    }

    if (len - divider >= 2) {
        if (divider >= 2) {
            alternate_axes(points, indices.subspan(0, divider), !actual_horizontal);
        }
        alternate_axes(points, indices.subspan(divider), !actual_horizontal);
    }
}

// Set boundary vertex halfedges
inline void
set_boundary_vertex_halfedges(MeshType& mesh, HalfedgeId first_hid)
{
    HalfedgeId curr_hid = first_hid;
    do {
        HalfedgeId prev_hid = mesh.he_prev(curr_hid);
        VertexId va = mesh.he_to(prev_hid);
        if (va.valid()) {
            mesh.vertex_data(va).halfedge = mesh.he_twin(prev_hid);
        }
        curr_hid = mesh.he_next(mesh.he_twin(curr_hid));
    } while (curr_hid != first_hid);
}

// Get triangulated mesh
inline std::pair<HalfedgeId, MeshType>
get_triangulated_mesh(std::span<const double> points, bool is_horizontal)
{
    const std::size_t n_points = points.size() >> 1;

    auto sorted_vertices = ranges::iota_view{ 0ul, n_points } |
                           views::transform([](auto idx) { return gpf::VertexId{ idx }; }) | ranges::to<std::vector>();

    std::sort(sorted_vertices.begin(), sorted_vertices.end(), [&points](VertexId i, VertexId j) {
        return ranges::lexicographical_compare(predicates::point<2>(points.data(), i.idx),
                                               predicates::point<2>(points.data(), j.idx));
    });

    alternate_axes(points.data(), sorted_vertices, is_horizontal);

    Triangulation tri(points);
    tri.sorted_vertices = std::move(sorted_vertices);
    return tri.triangulate(is_horizontal);
}

// Simple triangulation of points
[[nodiscard]] inline std::vector<std::size_t>
triangulate_points(std::span<const double> points, bool is_horizontal)
{
    auto [_, mesh] = get_triangulated_mesh(points, is_horizontal);

    std::vector<std::size_t> result;
    result.reserve(mesh.n_faces() * 3);

    for (const auto& f : mesh.faces()) {
        FaceId fid = f.id;
        HalfedgeId ha = mesh.f_halfedge(fid);
        HalfedgeId hb = mesh.he_next(ha);
        HalfedgeId hc = mesh.he_next(hb);
        VertexId va = mesh.he_to(ha);
        VertexId vb = mesh.he_to(hb);
        VertexId vc = mesh.he_to(hc);

        if (va.valid() && vb.valid() && vc.valid()) {
            result.push_back(va.idx);
            result.push_back(vb.idx);
            result.push_back(vc.idx);
        }
    }
    return result;
}

// CDT triangulation
[[nodiscard]] inline std::vector<std::size_t>
triangulate1(std::span<const double> points, std::span<const std::size_t> segments, bool is_horizontal)
{
    auto [bdy_hid, mesh] = get_triangulated_mesh(points, is_horizontal);
    set_boundary_vertex_halfedges(mesh, bdy_hid);

    std::vector<std::size_t> halfedge_marks(mesh.n_halfedges_capacity(), INVALID_IND);

    std::vector<predicates::Point2D> point_indices;
    point_indices.reserve(mesh.n_vertices_capacity());
    for (std::size_t i = 0; i < mesh.n_vertices_capacity(); ++i) {
        point_indices.emplace_back(i);
    }

    CDT cdt;
    cdt.points = points;
    cdt.segments = segments;
    cdt.point_indices = std::move(point_indices);
    cdt.mesh = std::move(mesh);
    cdt.halfedge_marks = std::move(halfedge_marks);

    cdt.perform();

    std::vector<FaceId> valid_faces = cdt.extract_invalid_faces();

    std::vector<std::size_t> result;
    result.reserve(valid_faces.size() * 3);

    for (FaceId fid : valid_faces) {
        for (const auto& he : cdt.mesh.face(fid).halfedges()) {
            result.push_back(he.to().id.idx);
        }
    }
    return result;
}

// CDT with new intersection points
[[nodiscard]] inline std::pair<std::vector<double>, std::vector<std::size_t>>
triangulate_with_new_points(std::span<const double> points, std::span<const std::size_t> segments, bool is_horizontal)
{
    const std::size_t n_old_points = points.size() >> 1;
    auto [bdy_hid, mesh] = get_triangulated_mesh(points, is_horizontal);
    set_boundary_vertex_halfedges(mesh, bdy_hid);

    std::vector<std::size_t> halfedge_marks(mesh.n_halfedges_capacity(), INVALID_IND);

    std::vector<predicates::Point2D> point_indices;
    point_indices.reserve(mesh.n_vertices_capacity());
    for (std::size_t i = 0; i < mesh.n_vertices_capacity(); ++i) {
        point_indices.emplace_back(i);
    }

    CDT cdt;
    cdt.points = points;
    cdt.segments = segments;
    cdt.point_indices = std::move(point_indices);
    cdt.mesh = std::move(mesh);
    cdt.halfedge_marks = std::move(halfedge_marks);

    cdt.perform();

    std::vector<double> new_points;
    for (std::size_t i = n_old_points; i < cdt.mesh.n_vertices_capacity(); ++i) {
        auto pt = predicates::to_explicit(cdt.point_indices[i], cdt.points.data());
        new_points.push_back(pt[0]);
        new_points.push_back(pt[1]);
    }

    std::vector<FaceId> valid_faces = cdt.extract_invalid_faces();

    std::vector<std::size_t> triangles;
    triangles.reserve(valid_faces.size() * 3);

    for (FaceId fid : valid_faces) {
        for (const auto& he : cdt.mesh.face(fid).halfedges()) {
            triangles.push_back(he.to().id.idx);
        }
    }

    return { std::move(new_points), std::move(triangles) };
}

// Helper: unique indices deduplication
[[nodiscard]] inline std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
unique_indices(std::span<const std::size_t> indices)
{
    std::size_t count = 0;
    std::unordered_map<std::size_t, std::size_t> map;
    map.reserve(indices.size());

    std::vector<std::size_t> result;
    result.reserve(indices.size());

    for (std::size_t old : indices) {
        auto it = map.find(old);
        if (it != map.end()) {
            result.push_back(it->second);
        } else {
            map.emplace(old, count);
            result.push_back(count);
            ++count;
        }
    }

    std::vector<std::size_t> new_to_ori_map(map.size());
    for (const auto& [k, v] : map) {
        new_to_ori_map[v] = k;
    }

    return { std::move(result), std::move(new_to_ori_map) };
}

// 3D polygon triangulation
[[nodiscard]] inline std::vector<std::size_t>
triangulate_polygon(std::span<const double> points,
                    std::span<const std::size_t> segments,
                    const double* o,
                    const double* x,
                    const double* y)
{
    auto [new_segments, new_to_ori_map] = unique_indices(segments);

    std::vector<double> points_2d;
    points_2d.reserve(new_to_ori_map.size() * 2);

    for (std::size_t idx : new_to_ori_map) {
        auto p = predicates::point<3>(points.data(), idx);
        double v[3]{ p[0] - o[0], p[1] - o[1], p[2] - o[2] };
        points_2d.push_back(dot(v, x));
        points_2d.push_back(dot(v, y));
    }

    std::vector<std::size_t> triangles =
      triangulate1(std::span<const double>(points_2d), std::span<const std::size_t>(new_segments), true);

    std::vector<std::size_t> result;
    result.reserve(triangles.size());

    for (std::size_t idx : triangles) {
        result.push_back(new_to_ori_map[idx]);
    }

    return result;
}

// Multi-polygon triangulation
[[nodiscard]] inline std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
triangulate_polygon_soup(std::span<const double> points,
                         const std::vector<std::vector<std::size_t>>& edges,
                         std::span<const double> axes)
{
    std::vector<std::size_t> triangles;
    std::vector<std::size_t> parents;

    for (std::size_t idx = 0; idx < edges.size(); ++idx) {
        const std::vector<std::size_t>& face_segments = edges[idx];
        const double* axis_data = axes.data() + idx * 9;

        std::vector<std::size_t> face_triangles = triangulate_polygon(
          points, std::span<const std::size_t>(face_segments), axis_data, axis_data + 3, axis_data + 6);

        parents.resize(parents.size() + face_triangles.size() / 3, idx);
        triangles.insert(triangles.end(), face_triangles.begin(), face_triangles.end());
    }

    return { std::move(triangles), std::move(parents) };
}

} // namespace triangulation
} // namespace gpf
