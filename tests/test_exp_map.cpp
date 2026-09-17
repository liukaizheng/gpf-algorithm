#include "read_off.hpp"

#include <gpf/exp_map.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ExpMapVertexProp
{
    std::array<double, 3> pt{};
    double angle_sum = 0.0;
};

struct HalfedgeProp
{
    double angle = 0.0;
    double signpost_angle = 0.0;
    std::array<double, 2> vector{};
};

struct EdgeProp
{
    double len = 0.0;
};

using ExpMapMesh = gpf::ManifoldMesh<ExpMapVertexProp, HalfedgeProp, EdgeProp, gpf::Empty>;

void
write_off(const char* path,
          const std::vector<std::array<double, 3>>& positions,
          const std::vector<std::vector<std::size_t>>& faces)
{
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(std::string{ "Cannot open file: " } + path);
    }
    out << "OFF\n" << positions.size() << ' ' << faces.size() << " 0\n";
    out << std::setprecision(17);
    for (const auto& position : positions) {
        out << position[0] << ' ' << position[1] << ' ' << position[2] << '\n';
    }
    for (const auto& face : faces) {
        out << face.size();
        for (const auto index : face) {
            out << ' ' << index;
        }
        out << '\n';
    }
    out.close();
    if (!out) {
        throw std::runtime_error(std::string{ "Cannot write file: " } + path);
    }
}

// Faces of the exp-map chart: faces whose vertices all carry a UV. Vertex slot indices are kept, so the
// resulting file shares its vertex indexing with write_off of the corresponding 3D positions.
std::vector<std::vector<std::size_t>>
chart_faces(const ExpMapMesh& mesh, const std::vector<bool>& vertex_has_uv)
{
    std::vector<std::vector<std::size_t>> faces;
    for (const auto face : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto he : face.halfedges()) {
            const auto vertex_index = he.from().id.idx;
            if (!vertex_has_uv[vertex_index]) {
                face_vertices.clear();
                break;
            }
            face_vertices.push_back(vertex_index);
        }
        if (!face_vertices.empty()) {
            faces.push_back(std::move(face_vertices));
        }
    }
    return faces;
}

// Writes the flattened exp-map chart: every chart vertex at its UV (z = 0).
void
write_exp_map_off(const char* path, const ExpMapMesh& mesh, const gpf::ExpMapResult& result)
{
    assert(result.vertex_ids.size() == result.uvs.size());
    std::vector<std::array<double, 2>> vertex_uvs(mesh.n_vertices_capacity());
    std::vector<bool> vertex_has_uv(mesh.n_vertices_capacity(), false);
    for (std::size_t i = 0; i < result.vertex_ids.size(); ++i) {
        const auto vertex_id = result.vertex_ids[i];
        assert(vertex_id.idx < vertex_uvs.size());
        vertex_uvs[vertex_id.idx] = result.uvs[i];
        vertex_has_uv[vertex_id.idx] = true;
    }

    std::vector<std::array<double, 3>> positions(mesh.n_vertices_capacity());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        positions[i] = { vertex_uvs[i][0], vertex_uvs[i][1], 0.0 };
    }
    write_off(path, positions, chart_faces(mesh, vertex_has_uv));
}

// Writes the 3D counterpart of write_exp_map_off: the same chart faces over the same vertex slot
// indices, with every chart vertex at its position on the mesh surface instead of its UV.
void
write_exp_map_3d_off(const char* path, const ExpMapMesh& mesh, const gpf::ExpMapResult& result)
{
    std::vector<bool> vertex_has_uv(mesh.n_vertices_capacity(), false);
    for (const auto vertex_id : result.vertex_ids) {
        assert(vertex_id.idx < vertex_has_uv.size());
        vertex_has_uv[vertex_id.idx] = true;
    }

    std::vector<std::array<double, 3>> positions(mesh.n_vertices_capacity());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (vertex_has_uv[i]) {
            positions[i] = mesh.vertex(gpf::VertexId{ i }).prop().pt;
        }
    }
    write_off(path, positions, chart_faces(mesh, vertex_has_uv));
}

} // namespace

void
test_exp_map()
{
    const auto data = read_off("sphere.off");
    assert(!data.vertices.empty());
    assert(!data.faces.empty());

    auto mesh = ExpMapMesh::new_in(data.faces);
    for (auto vertex : mesh.vertices()) {
        vertex.prop().pt = data.vertices[vertex.id.idx];
    }

    gpf::update_edge_lengths<3>(mesh);
    gpf::update_corner_angles(mesh);
    gpf::update_vertex_angle_sums(mesh);
    gpf::update_halfedge_signpost_angles(mesh);
    gpf::update_halfedge_vectors(mesh);

    assert(mesh.n_vertices() == data.vertices.size());
    assert(mesh.n_faces() == data.faces.size());
    for (auto vertex : mesh.vertices()) {
        assert(vertex.id.idx < data.vertices.size());
        vertex.prop().pt = data.vertices[vertex.id.idx];
    }

    const std::size_t n_vertices_before = mesh.n_vertices();
    const std::size_t n_vertices_capacity_before = mesh.n_vertices_capacity();
    const std::array<double, 3> center_pt{ -0.23, 0.26, 0.93 };
    const auto exp_map_result = gpf::exp_map(center_pt, mesh, 2.7);
    assert(exp_map_result.has_value());
    const auto& result = *exp_map_result;

    assert(result.center_vertex.valid());
    assert(result.center_vertex.idx == n_vertices_capacity_before);
    assert(result.center_vertex.idx < mesh.n_vertices_capacity());
    assert(mesh.vertex(result.center_vertex).data().valid());
    assert(mesh.n_vertices() == n_vertices_before + 1);
    assert(mesh.n_vertices_capacity() == n_vertices_capacity_before + 1);

    assert(!result.vertex_ids.empty());
    assert(!result.uvs.empty());
    assert(result.vertex_ids.size() == result.uvs.size());
    for (std::size_t i = 0; i < result.vertex_ids.size(); ++i) {
        const auto vertex_id = result.vertex_ids[i];
        assert(vertex_id.valid());
        assert(vertex_id.idx < mesh.n_vertices_capacity());
        assert(mesh.vertex(vertex_id).data().valid());
        assert(std::isfinite(result.uvs[i][0]));
        assert(std::isfinite(result.uvs[i][1]));
    }

    write_exp_map_off("exp_map_uv.off", mesh, result);
    write_exp_map_3d_off("exp_map_3d.off", mesh, result);

    const auto center_vertex = mesh.vertex(result.center_vertex);
    std::vector<gpf::VertexId> one_ring_vertices{ result.center_vertex };
    std::vector<gpf::HalfedgeId> one_ring_halfedges;
    std::vector<gpf::EdgeId> one_ring_edges;
    std::vector<gpf::FaceId> one_ring_faces;

    auto append_unique = [](auto& ids, const auto id) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    };
    for (const auto halfedge : center_vertex.outgoing_halfedges()) {
        const auto face_id = halfedge.face().id;
        if (face_id.valid()) {
            append_unique(one_ring_faces, face_id);
        }
        const auto twin_face_id = halfedge.twin().face().id;
        if (twin_face_id.valid()) {
            append_unique(one_ring_faces, twin_face_id);
        }
    }
    for (const auto face_id : one_ring_faces) {
        for (const auto halfedge : mesh.face(face_id).halfedges()) {
            append_unique(one_ring_vertices, halfedge.from().id);
            append_unique(one_ring_vertices, halfedge.to().id);
            append_unique(one_ring_halfedges, halfedge.id);
            append_unique(one_ring_edges, halfedge.edge().id);
        }
    }

    assert(one_ring_vertices.size() > 1);
    assert(!one_ring_halfedges.empty());
    assert(!one_ring_edges.empty());
    assert(!one_ring_faces.empty());
    assert(one_ring_halfedges.size() == one_ring_faces.size() * 3);

    auto contains = [](const auto& ids, const auto id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
    auto assert_halfedge_properties = [](const auto halfedge) {
        const double edge_len = halfedge.edge().prop().len;
        const auto& vector = halfedge.prop().vector;
        assert(std::isfinite(halfedge.prop().angle));
        assert(std::isfinite(halfedge.prop().signpost_angle));
        assert(std::isfinite(vector[0]));
        assert(std::isfinite(vector[1]));
        const double vector_len = std::hypot(vector[0], vector[1]);
        const double tolerance = 1e-12 * std::max(1.0, edge_len);
        assert(std::abs(vector_len - edge_len) <= tolerance);
    };

    for (const auto edge_id : one_ring_edges) {
        const double edge_len = mesh.edge(edge_id).prop().len;
        assert(std::isfinite(edge_len));
        assert(edge_len > 0.0);
    }
    for (const auto halfedge_id : one_ring_halfedges) {
        assert_halfedge_properties(mesh.halfedge(halfedge_id));
    }
    for (const auto vertex_id : one_ring_vertices) {
        const double angle_sum = mesh.vertex(vertex_id).prop().angle_sum;
        assert(std::isfinite(angle_sum));
        assert(angle_sum > 0.0);
    }

    const std::array<double, 2> zero_vector{};
    bool found_adjacent_out_of_ring_halfedge = false;
    for (const auto halfedge : mesh.halfedges()) {
        if (contains(one_ring_halfedges, halfedge.id)) {
            continue;
        }

        assert(halfedge.prop().angle == 0.0);
        assert(halfedge.prop().signpost_angle == 0.0);
        assert(halfedge.prop().vector == zero_vector);
        if (contains(one_ring_vertices, halfedge.from().id)) {
            found_adjacent_out_of_ring_halfedge = true;
        }
    }
    assert(found_adjacent_out_of_ring_halfedge);

    bool found_out_of_ring_edge = false;
    for (const auto edge : mesh.edges()) {
        if (contains(one_ring_edges, edge.id)) {
            continue;
        }

        assert(edge.prop().len == 0.0);
        found_out_of_ring_edge = true;
    }
    assert(found_out_of_ring_edge);

    for (const auto vertex : mesh.vertices()) {
        if (!contains(one_ring_vertices, vertex.id)) {
            assert(vertex.prop().angle_sum == 0.0);
        }
    }
}
