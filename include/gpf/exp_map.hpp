#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <expected>
#include <functional>
#include <limits>
#include <queue>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gpf/ids.hpp>
#include <gpf/manifold_mesh.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

namespace gpf {

enum class ExpMapFailure
{
    EmptyPatch,         // No triangular faces qualify.
    MissingCenter,      // No retained face is incident to the projected center.
    DisconnectedPatch,  // Retained faces are not connected through edges.
    NonManifoldVertex,  // A used vertex has separated retained face fans.
    NotTopologicalDisk, // The connected, vertex-manifold patch has Euler characteristic other than one.
    ProjectionFailed    // Center projection failed before updating its one-ring properties.
};

/// A nonempty, connected triangular topological disk in the post-projection source mesh.
/// face_ids are unique and in traversal order. vertex_ids are exactly their incident vertices, in ascending ID order,
/// with vertex_ids[i] corresponding to raw uvs[i]. The center is included at UV (0, 0).
/// Later mesh edits can invalidate these IDs; a topological disk does not guarantee an injective or unflipped UV map.
struct ExpMapResult
{
    gpf::VertexId center_vertex{};
    std::vector<gpf::VertexId> vertex_ids;
    std::vector<std::array<double, 2>> uvs;
    std::vector<gpf::FaceId> face_ids;
};

namespace detail {

struct ExpMapPatch
{
    std::vector<gpf::FaceId> face_ids;
    std::size_t n_vertices = 0;
};

// The source is a closed, orientable triangular manifold; kept_faces marks only live faces.
// Both masks are capacity-indexed. used_vertices reuses the propagation settlement buffer.
template<typename Mesh>
[[nodiscard]] std::expected<ExpMapPatch, ExpMapFailure>
validate_exp_map_patch(const Mesh& mesh,
                       const VertexId center_vertex,
                       const std::vector<bool>& kept_faces,
                       std::vector<bool>& used_vertices)
{
    assert(kept_faces.size() == mesh.n_faces_capacity());
    assert(used_vertices.size() == mesh.n_vertices_capacity());
    std::size_t n_faces = 0;
    FaceId start_face{};
    for (const auto face : mesh.faces()) {
        if (!kept_faces[face.id.idx]) {
            continue;
        }
        ++n_faces;
        if (!start_face.valid()) {
            for (const auto he : face.halfedges()) {
                if (he.from().id == center_vertex) {
                    start_face = face.id;
                    break;
                }
            }
        }
    }
    if (n_faces == 0) {
        return std::unexpected(ExpMapFailure::EmptyPatch);
    }
    if (!start_face.valid()) {
        return std::unexpected(ExpMapFailure::MissingCenter);
    }

    std::vector<bool> visited_faces(mesh.n_faces_capacity(), false);
    std::fill(used_vertices.begin(), used_vertices.end(), false);
    ExpMapPatch patch;
    patch.face_ids.reserve(n_faces);
    patch.face_ids.push_back(start_face);
    visited_faces[start_face.idx] = true;
    std::size_t n_edges = 0;
    // The returned face list is also the FIFO; marking on enqueue keeps every face unique.
    for (std::size_t cursor = 0; cursor < patch.face_ids.size(); ++cursor) {
        const auto face_id = patch.face_ids[cursor];
        for (const auto he : mesh.face(face_id).halfedges()) {
            const auto vertex_id = he.from().id;
            if (!used_vertices[vertex_id.idx]) {
                used_vertices[vertex_id.idx] = true;
                ++patch.n_vertices;
            }
            const auto neighbor_id = he.twin().face().id;
            if (!kept_faces[neighbor_id.idx] || face_id < neighbor_id) {
                ++n_edges;
            }
            if (kept_faces[neighbor_id.idx] && !visited_faces[neighbor_id.idx]) {
                visited_faces[neighbor_id.idx] = true;
                patch.face_ids.push_back(neighbor_id);
            }
        }
    }
    if (patch.face_ids.size() != n_faces) {
        return std::unexpected(ExpMapFailure::DisconnectedPatch);
    }

    for (const auto vertex : mesh.vertices()) {
        if (!used_vertices[vertex.id.idx]) {
            continue;
        }
        const auto start = vertex.halfedge();
        auto he = start;
        bool previous_kept = kept_faces[he.face().id.idx];
        std::size_t retained_runs = 0;
        do {
            he = he.twin().next();
            const bool current_kept = kept_faces[he.face().id.idx];
            if (current_kept && !previous_kept) {
                ++retained_runs;
            }
            previous_kept = current_kept;
        } while (he.id != start.id);
        // Including the closing transition permits a boundary fan that wraps around the start.
        // Zero runs means the entire ambient ring is retained; one run is an interval link.
        if (retained_runs > 1) {
            return std::unexpected(ExpMapFailure::NonManifoldVertex);
        }
    }

    // For a connected orientable 2-manifold, chi = 2 - 2g - b = 1 precisely for a disk.
    // Rearranging V - E + F = 1 avoids unsigned-subtraction underflow.
    if (patch.n_vertices + n_faces != n_edges + 1) {
        return std::unexpected(ExpMapFailure::NotTopologicalDisk);
    }
    return patch;
}

template<typename Mesh>
void
update_exp_map_properties_around_vertex(Mesh& mesh, const VertexId center_vid)
{
    auto center_vertex = mesh.vertex(center_vid);
    for (const auto he : center_vertex.incoming_halfedges()) {
        update_edge_length<3>(he.edge());
    }

    for (const auto he : center_vertex.incoming_halfedges()) {
        update_corner_angles_on_face(he.face());
    }
    update_vertex_angle_sum(center_vertex);
    update_halfedge_signpost_angles_at_vertex(center_vertex);

    for (auto he : center_vertex.incoming_halfedges()) {
        const auto prev = he.twin().next();
        he.prop().signpost_angle =
          std::fmod(prev.prop().signpost_angle + prev.prop().angle, he.from().prop().angle_sum);
    }

    for (const auto he : center_vertex.incoming_halfedges()) {
        update_halfedge_vector(he);
        update_halfedge_vector(he.twin());
    }
}

} // namespace detail

/// Requires a closed, orientable triangular manifold with valid geometry and initialized mesh-derived properties.
/// Computes raw coordinates in mesh units with radius-controlled propagation, including the unconditional center
/// one-ring and final frontier beyond the radius. Rejects the induced face patch unless it is a topological disk.
/// Projection may insert the center and can fail before its one-ring properties are updated. After successful
/// projection, those properties are updated even if patch validation fails. Properties outside the one-ring stay
/// unchanged. Failures are nontransactional: mesh mutations are not rolled back.
template<typename VP, typename HP, typename EP, typename FP>
    requires HasPositionProperty<VertexHandle<ManifoldMesh<VP, HP, EP, FP>, false>, 3> &&
             HasAngleSumProperty<VertexHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasAngleProperty<HalfedgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasSignpostAngleProperty<HalfedgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasVectorProperty<HalfedgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>> &&
             HasLengthProperty<EdgeHandle<ManifoldMesh<VP, HP, EP, FP>, false>>
[[nodiscard]] std::expected<ExpMapResult, ExpMapFailure>
exp_map(const std::span<const double, 3> center_pt,
        gpf::ManifoldMesh<VP, HP, EP, FP>& mesh,
        const double max_pseudo_geodesic_dist)
{
    using Mesh = gpf::ManifoldMesh<VP, HP, EP, FP>;
    static_assert(gpf::mesh_position_dim_v<Mesh> == 3);

    // Projection reuses a nearby vertex or retriangulates the containing face around a newly inserted center.
    std::vector<std::array<double, 3>> center_points{ { center_pt[0], center_pt[1], center_pt[2] } };
    const auto projected_vertices = detail::project_points_on_mesh<3>(center_points, mesh, 1e-3);
    if (!projected_vertices) {
        return std::unexpected(ExpMapFailure::ProjectionFailed);
    }

    const gpf::VertexId center_vertex = projected_vertices->front();
    detail::update_exp_map_properties_around_vertex(mesh, center_vertex);

    // Mesh IDs can contain gaps, so propagation state is indexed by storage capacity rather than active count.
    const std::size_t n_vertices = mesh.n_vertices_capacity();
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> distances(n_vertices, infinity);
    std::vector<std::array<double, 2>> uvs(n_vertices, { 0.0, 0.0 });
    std::vector<bool> settled(n_vertices, false);

    using QueueEntry = std::tuple<double, VertexId, HalfedgeId, std::array<double, 2>, std::array<double, 2>>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> pq;
    distances[center_vertex.idx] = 0.0;
    settled[center_vertex.idx] = true;

    for (const auto he : mesh.vertex(center_vertex).outgoing_halfedges()) {
        const auto vid = he.to().id;
        distances[vid.idx] = he.edge().prop().len;
        uvs[vid.idx] = he.prop().vector;
        settled[vid.idx] = true;
    }

    auto enqueue = [&mesh, &uvs, &settled, &pq](
                     const auto vb_id, const HalfedgeId hab_id, const std::array<double, 2>& vab_data) noexcept {
        const auto vob{ Eigen::Vector2d::Map(uvs[vb_id.idx].data()) };
        const auto vab = Eigen::Vector2d::Map(vab_data.data()).normalized().eval();
        const auto hba_id = mesh.he_twin(hab_id);
        const auto& local_vba = mesh.halfedge_prop(hba_id).vector;
        Eigen::Vector2d local_vab{ -local_vba[0], -local_vba[1] };
        local_vab.normalize();
        Eigen::Matrix2d mat = Eigen::Matrix2d{ { vab[0], -vab[1] }, { vab[1], vab[0] } } *
                              Eigen::Matrix2d{ { local_vab[0], local_vab[1] }, { -local_vab[1], local_vab[0] } };
        mat.col(0).normalize();
        mat(0, 1) = -mat(1, 0);
        mat(1, 1) = mat(0, 0);

        for (const auto hbc : mesh.vertex(vb_id).outgoing_halfedges()) {
            const auto vc_id = hbc.to().id;
            if (settled[vc_id.idx]) {
                continue;
            }

            std::array<double, 2> vbc_data{};
            auto vbc{ Eigen::Vector2d::Map(vbc_data.data()) };
            vbc = mat * Eigen::Vector2d::Map(hbc.prop().vector.data());
            std::array<double, 2> voc_data{};
            auto voc{ Eigen::Vector2d::Map(voc_data.data()) };
            voc = vob + vbc;
            const auto len = voc.norm();

            pq.emplace(len, vc_id, hbc.id, std::move(vbc_data), std::move(voc_data));
        }
    };

    for (const auto he : mesh.vertex(center_vertex).outgoing_halfedges()) {
        const auto vid = he.to().id;
        enqueue(vid, he.id, he.prop().vector);
    }

    while (!pq.empty()) {
        const auto [len, vb, hab, vab, vob] = pq.top();
        pq.pop();
        if (settled[vb.idx]) {
            continue;
        }
        settled[vb.idx] = true;
        distances[vb.idx] = len;
        uvs[vb.idx] = vob;
        if (len < max_pseudo_geodesic_dist) {
            enqueue(vb, hab, vab);
        }
    }

    std::vector<bool> kept_faces(mesh.n_faces_capacity(), false);
    for (const auto face : mesh.faces()) {
        const auto he = face.halfedge();
        kept_faces[face.id.idx] = std::isfinite(distances[he.from().id.idx]) &&
                                  std::isfinite(distances[he.next().from().id.idx]) &&
                                  std::isfinite(distances[he.next().next().from().id.idx]);
    }
    auto patch = detail::validate_exp_map_patch(mesh, center_vertex, kept_faces, settled);
    if (!patch) {
        return std::unexpected(patch.error());
    }

    ExpMapResult result;
    result.center_vertex = center_vertex;
    result.face_ids = std::move(patch->face_ids);
    // settled now marks only face-incident vertices, dropping dangling reached vertices without trimming faces.
    // mesh.vertices() yields active vertices in ascending ID order, establishing the public result ordering.
    result.vertex_ids.reserve(patch->n_vertices);
    result.uvs.reserve(patch->n_vertices);
    for (const auto vertex : mesh.vertices()) {
        const gpf::VertexId vertex_id = vertex.id;
        if (settled[vertex_id.idx]) {
            result.vertex_ids.push_back(vertex_id);
            result.uvs.push_back(uvs[vertex_id.idx]);
        }
    }
    return result;
}

} // namespace gpf
