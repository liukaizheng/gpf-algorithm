#include <gpf/ids.hpp>
#include <gpf/manifold_mesh.hpp>
#include <gpf/project_polylines_on_mesh.hpp>
#include <gpf/triangulation.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ostream>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <unordered_map>
#include <variant>
#include <vector>

namespace {
struct VertexProp
{
    std::array<double, 2> pt;
};

struct VertexProp3d
{
    std::array<double, 3> pt;
};

using Mesh2d = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, gpf::Empty>;
using Mesh3d = gpf::ManifoldMesh<VertexProp3d, gpf::Empty, gpf::Empty, gpf::Empty>;

void
assert_path(const auto& mesh,
            const std::span<const gpf::HalfedgeId> path,
            const gpf::VertexId start,
            const gpf::VertexId end)
{
    assert(!path.empty());
    auto current = start;
    for (const auto hid : path) {
        assert(hid.valid());
        assert(mesh.he_from(hid) == current);
        current = mesh.he_to(hid);
    }
    assert(current == end);
}

template<std::size_t N>
void
assert_point_vertices(const auto& mesh,
                      const std::vector<std::array<double, N>>& points,
                      const std::vector<gpf::VertexId>& point_vertices)
{
    assert(point_vertices.size() == points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        assert(point_vertices[i].valid());
        assert(point_vertices[i].idx < mesh.n_vertices_capacity());
        const auto& pt = mesh.vertex_prop(point_vertices[i]).pt;
        for (std::size_t j = 0; j < N; ++j) {
            assert(std::abs(pt[j] - points[i][j]) < 1e-12);
        }
    }
}

void
assert_invalid_triangle_indices(const std::vector<std::array<double, 2>>& boundary_points,
                                const std::vector<std::size_t>& constraints,
                                const std::size_t max_triangle_index)
{
    const auto n_boundary = boundary_points.size();
    const auto boundary = std::views::iota(std::size_t{ 0 }, n_boundary) | std::ranges::to<std::vector>();
    auto mesh =
      Mesh2d::new_in(std::vector<std::vector<std::size_t>>{ boundary, { n_boundary, n_boundary + 1, n_boundary + 2 } });
    for (std::size_t i = 0; i < n_boundary; ++i) {
        mesh.vertex_prop(gpf::VertexId{ i }).pt = boundary_points[i];
    }
    mesh.vertex_prop(gpf::VertexId{ n_boundary }).pt = { 10.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ n_boundary + 1 }).pt = { 11.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ n_boundary + 2 }).pt = { 10.0, 1.0 };

    const gpf::FaceId fid{ 0 };
    const auto halfedges_before = mesh.face(fid).halfedges() | std::views::transform([](auto he) { return he.id; }) |
                                  std::ranges::to<std::vector>();
    const auto face_vertices = mesh.face(fid).halfedges() | std::views::transform([](auto he) { return he.to().id; }) |
                               std::ranges::to<std::vector>();
    std::vector<double> coordinates;
    std::vector<std::size_t> segments;
    for (std::size_t i = 0; i < face_vertices.size(); ++i) {
        coordinates.append_range(mesh.vertex_prop(face_vertices[i]).pt);
        segments.append_range(std::array{ i, (i + 1) % face_vertices.size() });
    }
    segments.append_range(constraints);
    const auto triangle_indices = gpf::triangulate_polygon(coordinates, segments, n_boundary, true);
    assert(!triangle_indices.empty());
    assert(std::ranges::find(triangle_indices, face_vertices.size()) != triangle_indices.end());
    assert(std::ranges::max(triangle_indices) == max_triangle_index);
    // These indices fit global mesh storage and the flattened coordinates, but not the local vertex list.
    assert(max_triangle_index < mesh.n_vertices_capacity());
    assert(max_triangle_index < coordinates.size());

    const auto cross_edge_vertices = constraints |
                                     std::views::transform([&](auto index) { return face_vertices[index]; }) |
                                     std::ranges::to<std::vector>();
    const auto n_vertices_before = mesh.n_vertices_capacity();
    const auto n_edges_before = mesh.n_edges_capacity();
    const auto n_faces_before = mesh.n_faces_capacity();
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map{ { fid, gpf::FaceId{ 7 } },
                                                                  { gpf::FaceId{ 1 }, gpf::FaceId{ 8 } } };
    const auto parents_before = face_parent_map;
    std::vector<gpf::VertexId> point_vertices;
    const auto result = gpf::detail::triangulate_on_face<2>(
      mesh, fid, {}, gpf::detail::FaceCoords<2>{}, {}, cross_edge_vertices, point_vertices, &face_parent_map);
    assert(!result.has_value());
    assert(result.error() == gpf::ProjectPolylinesOnMeshFailure::InvalidTriangleIndex);
    assert(mesh.n_vertices_capacity() == n_vertices_before);
    assert(mesh.n_edges_capacity() == n_edges_before);
    assert(mesh.n_faces_capacity() == n_faces_before);
    assert(mesh.n_faces() == 2);
    const auto halfedges_after = mesh.face(fid).halfedges() | std::views::transform([](auto he) { return he.id; }) |
                                 std::ranges::to<std::vector>();
    assert(halfedges_after == halfedges_before);
    for (std::size_t i = 0; i < halfedges_after.size(); ++i) {
        assert(mesh.he_to(halfedges_after[i]) == face_vertices[i]);
        assert(mesh.he_face(halfedges_after[i]) == fid);
    }
    assert(face_parent_map == parents_before);
    assert(point_vertices.empty());
}

void
write_obj(const std::string& name, const auto& mesh)
{
    std::ofstream out(name);
    for (const auto v : mesh.vertices()) {
        std::println(out, "v {} {} 0", v.prop().pt[0], v.prop().pt[1]);
    }
    for (const auto f : mesh.faces()) {
        std::print(out, "f");
        for (const auto he : f.halfedges()) {
            std::print(out, " {}", he.to().id.idx + 1);
        }
        std::println(out);
    }
    out.close();
}

std::array<Eigen::Vector3d, 3>
face_points(const Mesh3d& mesh, const gpf::FaceId fid)
{
    std::array<Eigen::Vector3d, 3> points;
    std::size_t idx = 0;
    for (const auto he : mesh.face(fid).halfedges()) {
        points[idx++] = Eigen::Vector3d::Map(mesh.vertex_prop(he.from().id).pt.data());
    }
    assert(idx == 3);
    return points;
}

std::array<double, 3>
barycentric_coordinates(const Mesh3d& mesh, const gpf::FaceId fid, const Eigen::Vector3d& pt)
{
    const auto points = face_points(mesh, fid);
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = points[1] - points[0];
    basis.col(1) = points[2] - points[0];
    const Eigen::Vector2d uv = (basis.transpose() * basis).ldlt().solve(basis.transpose() * (pt - points[0]));
    return { 1.0 - uv.x() - uv.y(), uv.x(), uv.y() };
}

Eigen::Vector3d
point_from_barycentric(const Mesh3d& mesh, const auto& point)
{
    const auto points = face_points(mesh, point.first);
    Eigen::Vector3d pt = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < points.size(); ++i) {
        pt += point.second[i] * points[i];
    }
    return pt;
}

void
assert_valid_barycentric(const auto& point)
{
    double sum = 0.0;
    for (const double coord : point.second) {
        assert(coord > -1e-9);
        assert(coord < 1.0 + 1e-9);
        sum += coord;
    }
    assert(std::abs(sum - 1.0) < 1e-9);
}

bool
is_close(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const double eps = 1e-9)
{
    return (a - b).norm() < eps;
}
} // namespace

void
test_project_polylines_on_mesh_2d_points()
{
    using Mesh = gpf::ManifoldMesh<VertexProp, gpf::Empty, gpf::Empty, gpf::Empty>;

    Mesh mesh = Mesh::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0 };

    std::vector<std::array<double, 2>> points{ { 0.75, 0.0 }, { 0.1, 0.8 } };
    const std::size_t N = 1000;

    std::mt19937 rng(42); // Seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    for (int i = 0; i < N; ++i) {
        const auto a = dist(rng);
        const auto b = dist(rng);
        const auto c = dist(rng);
        const auto sum = a + b + c;
        const auto t1 = a / sum;
        const auto t2 = b / sum;

        points.emplace_back(std::array<double, 2>{ t1, t2 });
    }
    const std::vector<std::vector<std::size_t>> polylines{ { 0, 1 } };

    const auto result = gpf::project_polylines_on_mesh<2>(points, polylines, mesh, 1e-3);
    assert(result.has_value());
    const auto& [point_vertices, paths] = *result;
    write_obj("project_mesh.obj", mesh);
    assert(paths.size() == 1);
    assert(!paths.front().empty());

    auto is_close = [](double a, double b) { return std::abs(a - b) < 1e-9; };
    for (std::size_t i = 0; i + 1 < paths.front().size(); ++i) {
        assert(mesh.he_to(paths.front()[i]) == mesh.he_from(paths.front()[i + 1]));
    }

    const auto first_he = mesh.halfedge(paths.front().front());
    const auto last_he = mesh.halfedge(paths.front().back());
    const auto pa = first_he.from().prop().pt;
    const auto pb = last_he.to().prop().pt;
    const bool forward = is_close(pa[0], 0.75) && is_close(pa[1], 0.0) && is_close(pb[0], 0.1) && is_close(pb[1], 0.8);
    const bool backward = is_close(pa[0], 0.1) && is_close(pa[1], 0.8) && is_close(pb[0], 0.75) && is_close(pb[1], 0.0);
    assert(forward || backward);

    assert(is_close(points[0][0], 0.75) && is_close(points[0][1], 0.0));
    assert(is_close(points[1][0], 0.1) && is_close(points[1][1], 0.8));
}

void
test_project_polylines_on_mesh_2d_success()
{
    auto mesh = Mesh2d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0 };
    std::vector<std::array<double, 2>> points{ { 0.0, 0.0 }, { 0.5, 0.0 }, { 0.25, 0.25 }, { 0.0, 1.0 } };
    const auto points_before = points;
    const std::vector<std::vector<std::size_t>> polylines{ { 0, 1, 2, 3 }, { 3, 2 }, { 1 } };
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> edge_parent_map;
    const auto result =
      gpf::project_polylines_on_mesh<2>(points, polylines, mesh, 1e-6, &face_parent_map, &edge_parent_map);
    assert(result.has_value());
    const auto& [point_vertices, paths] = *result;
    assert((point_vertices ==
            std::vector{ gpf::VertexId{ 0 }, gpf::VertexId{ 3 }, gpf::VertexId{ 4 }, gpf::VertexId{ 2 } }));
    assert_point_vertices(mesh, points, point_vertices);
    assert(points == points_before);
    assert(paths.size() == polylines.size());
    assert(paths[0].size() == 3);
    assert_path(mesh, paths[0], point_vertices[0], point_vertices[3]);
    assert(mesh.he_to(paths[0][0]) == point_vertices[1]);
    assert(mesh.he_to(paths[0][1]) == point_vertices[2]);
    assert_path(mesh, paths[1], point_vertices[3], point_vertices[2]);
    assert(paths[2].empty());
    assert(face_parent_map.size() == mesh.n_faces());
    assert(std::ranges::all_of(face_parent_map | std::views::values, [](auto fid) { return fid == gpf::FaceId{ 0 }; }));
    assert(edge_parent_map.size() == 2);
}

void
test_project_polylines_on_mesh_3d_success()
{
    auto mesh = Mesh3d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0, 1.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0, 1.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0, 1.0 };
    std::vector<std::array<double, 3>> points{
        { 0.0, 0.0, 2.0 }, { 0.5, 0.0, -1.0 }, { 0.25, 0.25, 3.0 }, { 0.0, 1.0, 0.0 }
    };
    const auto result = gpf::project_polylines_on_mesh<3>(points, { { 0, 1, 2, 3 }, { 3, 2 } }, mesh, 1e-6);
    assert(result.has_value());
    const auto& [point_vertices, paths] = *result;
    assert((point_vertices ==
            std::vector{ gpf::VertexId{ 0 }, gpf::VertexId{ 3 }, gpf::VertexId{ 4 }, gpf::VertexId{ 2 } }));
    assert_point_vertices(mesh, points, point_vertices);
    const std::vector<std::array<double, 3>> expected_points{
        { 0.0, 0.0, 1.0 }, { 0.5, 0.0, 1.0 }, { 0.25, 0.25, 1.0 }, { 0.0, 1.0, 1.0 }
    };
    assert(points == expected_points);
    assert(paths.size() == 2);
    assert(paths[0].size() == 3);
    assert_path(mesh, paths[0], point_vertices[0], point_vertices[3]);
    assert(mesh.he_to(paths[0][0]) == point_vertices[1]);
    assert(mesh.he_to(paths[0][1]) == point_vertices[2]);
    assert_path(mesh, paths[1], point_vertices[3], point_vertices[2]);
}

void
test_project_polylines_on_mesh_repeated_endpoints()
{
    auto mesh = Mesh2d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0 };
    std::vector<std::array<double, 2>> points{ { 0.0, 0.0 }, { 0.0, 0.0 }, { 1.0, 0.0 } };
    const auto result = gpf::project_polylines_on_mesh<2>(points, { { 0, 0, 1, 2, 2 }, { 0, 1, 0 } }, mesh, 1e-6);
    assert(result.has_value());
    const auto& [point_vertices, paths] = *result;
    assert((point_vertices == std::vector{ gpf::VertexId{ 0 }, gpf::VertexId{ 0 }, gpf::VertexId{ 1 } }));
    assert(paths.size() == 2);
    assert(paths[0].size() == 1);
    assert_path(mesh, paths[0], point_vertices[0], point_vertices[2]);
    assert(paths[1].empty());
    assert(mesh.n_vertices() == 3);
    assert(mesh.n_faces() == 1);
}

void
test_project_polylines_on_mesh_disconnected()
{
    auto mesh = Mesh3d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 }, { 3, 4, 5 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 3 }).pt = { 3.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 4 }).pt = { 4.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 5 }).pt = { 3.0, 1.0, 0.0 };
    std::vector<std::array<double, 3>> points{ { 0.5, 0.0, 2.0 }, { 3.25, 0.25, -1.0 } };
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> edge_parent_map;
    const auto result =
      gpf::project_polylines_on_mesh<3>(points, { { 0, 1 } }, mesh, 1e-6, &face_parent_map, &edge_parent_map);
    assert(result.has_value());
    const auto& [point_vertices, paths] = *result;
    assert_point_vertices(mesh, points, point_vertices);
    assert(paths.size() == 1);
    assert(paths[0].empty());
    // Point projection and its parent-map changes remain after the disconnected route fails.
    assert(mesh.n_vertices() == 8);
    assert(mesh.n_faces() == 5);
    assert((points[0] == std::array{ 0.5, 0.0, 0.0 }));
    assert((points[1] == std::array{ 3.25, 0.25, 0.0 }));
    assert(face_parent_map.size() == mesh.n_faces());
    assert(edge_parent_map.size() == 2);
}

void
test_project_polylines_on_mesh_crossing_constraints()
{
    const std::vector<std::array<double, 2>> positions{ { 99.0, 90.0 }, { 15.0, 95.0 }, { 28.0, 90.0 }, { 9.0, 20.0 },
                                                        { 75.0, 22.0 }, { 71.0, 34.0 }, { 96.0, 40.0 }, { 85.0, 90.0 },
                                                        { 26.0, 83.0 }, { 16.0, 62.0 }, { 16.0, 7.0 },  { 98.0, 6.0 } };
    const std::vector<std::array<std::size_t, 3>> faces{ { 3, 10, 5 }, { 9, 3, 5 },  { 3, 9, 1 },   { 8, 2, 1 },
                                                         { 9, 8, 1 },  { 8, 9, 5 },  { 1, 2, 7 },   { 2, 8, 7 },
                                                         { 4, 11, 6 }, { 10, 4, 5 }, { 10, 11, 4 }, { 5, 4, 6 },
                                                         { 6, 0, 7 },  { 5, 6, 7 },  { 11, 0, 6 },  { 1, 7, 0 },
                                                         { 8, 5, 7 } };
    auto mesh = Mesh2d::new_in(faces);
    for (auto vertex : mesh.vertices()) {
        vertex.prop().pt = positions[vertex.id.idx];
    }
    auto points = positions;
    const auto projected_vertices = gpf::detail::project_points_on_mesh<2>(points, mesh, 1e-6);
    assert(projected_vertices.has_value());
    assert_point_vertices(mesh, points, *projected_vertices);
    assert(mesh.n_vertices() == positions.size());

    // These ordered polylines encounter a locked constraint during geodesic shortening.
    const std::vector<std::vector<std::size_t>> polylines{ { 6, 3 }, { 5, 1 }, { 7, 9 }, { 2, 3 } };
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> edge_parent_map;
    const auto result =
      gpf::project_polylines_on_mesh<2>(points, polylines, mesh, 1e-6, &face_parent_map, &edge_parent_map);
    assert(result.has_value());
    const auto& [point_vertices, paths] = *result;
    assert_point_vertices(mesh, points, point_vertices);
    assert(point_vertices == *projected_vertices);
    assert(paths.size() == polylines.size());
    assert_path(mesh, paths[0], point_vertices[6], point_vertices[3]);
    assert(paths[1].empty());
    assert_path(mesh, paths[2], point_vertices[7], point_vertices[9]);
    assert(paths[3].empty());
    assert(points == positions);
    assert(mesh.n_vertices() == positions.size() + 4);
    assert(mesh.n_faces() == faces.size() + 8);
    assert(face_parent_map.size() == 12);
    assert(edge_parent_map.size() == 7);
}

void
test_project_polylines_on_mesh_initial_triangulation_failure()
{
    auto make_mesh = [] {
        auto mesh = Mesh2d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2, 3 } });
        // A self-crossing boundary makes initial point insertion return an unsupported intersection index.
        mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 2.0, 2.0 };
        mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 2.0 };
        mesh.vertex_prop(gpf::VertexId{ 3 }).pt = { 2.0, 0.0 };
        return mesh;
    };
    std::vector<std::array<double, 2>> points{ { 0.0, 0.0 } };
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map;
    std::unordered_map<gpf::EdgeId, gpf::EdgeId> edge_parent_map;
    auto point_mesh = make_mesh();
    const auto projected_vertices =
      gpf::detail::project_points_on_mesh<2>(points, point_mesh, 1e-6, &face_parent_map, &edge_parent_map);
    assert(!projected_vertices.has_value());
    assert(projected_vertices.error() == gpf::ProjectPolylinesOnMeshFailure::InvalidTriangleIndex);
    assert(point_mesh.n_vertices_capacity() == 4);
    assert(point_mesh.n_faces_capacity() == 1);
    assert(face_parent_map.empty());
    assert(edge_parent_map.empty());

    auto mesh = make_mesh();
    const auto result =
      gpf::project_polylines_on_mesh<2>(points, { { 0, 0 } }, mesh, 1e-6, &face_parent_map, &edge_parent_map);
    assert(!result.has_value());
    assert(result.error() == gpf::ProjectPolylinesOnMeshFailure::InvalidTriangleIndex);
    assert(mesh.n_vertices_capacity() == 4);
    assert(mesh.n_faces_capacity() == 1);
    assert(face_parent_map.empty());
    assert(edge_parent_map.empty());
}

void
test_resolve_polyline_path()
{
    auto mesh = Mesh2d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 }, { 2, 1, 3 }, { 4, 5, 6 } });
    const std::array<std::array<double, 2>, 7> points{
        { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 1.0 }, { 3.0, 0.0 }, { 4.0, 0.0 }, { 3.0, 1.0 } }
    };
    for (auto vertex : mesh.vertices()) {
        vertex.prop().pt = points[vertex.id.idx];
    }
    using Anchor = gpf::detail::TracePolyline<2, Mesh2d>::Anchor;
    const std::vector<gpf::VertexId> edge_point_vertices{ gpf::VertexId{ 1 } };
    const auto get_vertex_id = [&](const Anchor& anchor) {
        if (const auto* vid = std::get_if<gpf::VertexId>(&anchor)) {
            return *vid;
        }
        return edge_point_vertices[std::get<std::size_t>(anchor)];
    };

    const std::vector<Anchor> direct{ gpf::VertexId{ 0 }, std::size_t{ 0 } };
    const auto direct_result = gpf::detail::resolve_polyline_path<2>(mesh, direct, get_vertex_id);
    assert(direct_result.has_value());
    assert(*direct_result == std::vector{ mesh.he_from_vertices(gpf::VertexId{ 0 }, gpf::VertexId{ 1 }) });

    assert(!mesh.he_from_vertices(gpf::VertexId{ 0 }, gpf::VertexId{ 3 }).valid());
    const std::vector<Anchor> indirect{ gpf::VertexId{ 0 }, gpf::VertexId{ 3 } };
    const auto indirect_result = gpf::detail::resolve_polyline_path<2>(mesh, indirect, get_vertex_id);
    assert(indirect_result.has_value());
    assert(indirect_result->size() == 2);
    assert_path(mesh, *indirect_result, gpf::VertexId{ 0 }, gpf::VertexId{ 3 });

    const std::vector<Anchor> repeated{ gpf::VertexId{ 0 }, gpf::VertexId{ 0 }, std::size_t{ 0 }, gpf::VertexId{ 1 } };
    const auto repeated_result = gpf::detail::resolve_polyline_path<2>(mesh, repeated, get_vertex_id);
    assert(repeated_result.has_value());
    assert(*repeated_result == *direct_result);
    const std::vector<Anchor> zero_length{ gpf::VertexId{ 0 }, gpf::VertexId{ 0 } };
    const auto zero_result = gpf::detail::resolve_polyline_path<2>(mesh, zero_length, get_vertex_id);
    assert(zero_result.has_value());
    assert(zero_result->empty());

    const std::vector<Anchor> disconnected{ gpf::VertexId{ 0 }, std::size_t{ 0 }, gpf::VertexId{ 4 } };
    const auto disconnected_result = gpf::detail::resolve_polyline_path<2>(mesh, disconnected, get_vertex_id);
    assert(!disconnected_result.has_value());
    assert(disconnected_result.error() == gpf::ProjectPolylinesOnMeshFailure::PathNotFound);
}

void
test_triangulate_on_face_noop()
{
    auto mesh = Mesh2d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0 };
    const auto hid_before = mesh.face(gpf::FaceId{ 0 }).halfedge().id;
    std::vector<gpf::VertexId> point_vertices;
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map;
    const auto result = gpf::detail::triangulate_on_face<2>(
      mesh, gpf::FaceId{ 0 }, {}, gpf::detail::FaceCoords<2>{}, {}, {}, point_vertices, &face_parent_map);
    assert(result.has_value());
    assert(mesh.n_vertices_capacity() == 3);
    assert(mesh.n_edges_capacity() == 3);
    assert(mesh.n_faces_capacity() == 1);
    assert(mesh.face(gpf::FaceId{ 0 }).halfedge().id == hid_before);
    assert(point_vertices.empty());
    assert(face_parent_map.empty());
}

void
test_triangulate_on_face_success()
{
    auto mesh = Mesh2d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0 };
    const std::vector<std::array<double, 2>> points{ { 0.25, 0.25 } };
    std::vector<gpf::VertexId> point_vertices(points.size());
    const gpf::FaceId root_parent{ 7 };
    std::unordered_map<gpf::FaceId, gpf::FaceId> face_parent_map{ { gpf::FaceId{ 0 }, root_parent } };
    const auto result = gpf::detail::triangulate_on_face<2>(
      mesh, gpf::FaceId{ 0 }, points, gpf::detail::FaceCoords<2>{}, { 0 }, {}, point_vertices, &face_parent_map);
    assert(result.has_value());
    assert(point_vertices == std::vector{ gpf::VertexId{ 3 } });
    assert_point_vertices(mesh, points, point_vertices);
    assert(mesh.n_vertices() == 4);
    assert(mesh.n_faces() == 3);
    assert(face_parent_map.size() == 3);
    for (auto face : mesh.faces()) {
        assert(std::ranges::distance(face.halfedges()) == 3);
        assert(face_parent_map.at(face.id) == root_parent);
    }
}

void
test_triangulate_on_face_invalid_index_boundary()
{
    assert_invalid_triangle_indices({ { 0.0, 0.0 }, { 2.0, 0.0 }, { 2.0, 2.0 }, { 0.0, 2.0 } }, { 0, 2, 1, 3 }, 4);
}

void
test_triangulate_on_face_multiple_intersections()
{
    assert_invalid_triangle_indices(
      { { 0.0, 0.0 }, { 3.0, 0.0 }, { 4.0, 1.0 }, { 4.0, 4.0 }, { 1.0, 5.0 }, { -1.0, 2.0 } }, { 0, 3, 1, 4, 2, 5 }, 8);
}

void
test_prepare_projected_points_with_mbvh()
{
    Mesh3d mesh = Mesh3d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 }, { 2, 1, 3 } });
    mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { 0.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0, 0.0 };
    mesh.vertex_prop(gpf::VertexId{ 3 }).pt = { 1.0, 1.0, 0.0 };

    std::vector<std::array<double, 3>> points{
        { 0.25, 0.25, 2.0 },
        { -0.25, -0.25, 1.0 },
        { 0.4, -0.25, 1.0 },
        { 0.75, 0.75, -2.0 },
    };
    const auto [face_info_map, point_vertices, edge_to_points_map] =
      gpf::detail::prepare_projected_points<3>(points, mesh, 1e-6);

    assert(is_close(Eigen::Vector3d::Map(points[0].data()), Eigen::Vector3d{ 0.25, 0.25, 0.0 }));
    assert(is_close(Eigen::Vector3d::Map(points[1].data()), Eigen::Vector3d{ 0.0, 0.0, 0.0 }));
    assert(is_close(Eigen::Vector3d::Map(points[2].data()), Eigen::Vector3d{ 0.4, 0.0, 0.0 }));
    assert(is_close(Eigen::Vector3d::Map(points[3].data()), Eigen::Vector3d{ 0.75, 0.75, 0.0 }));

    assert(!point_vertices[0].valid());
    assert(point_vertices[1] == gpf::VertexId{ 0 });
    assert(!point_vertices[2].valid());
    assert(!point_vertices[3].valid());

    const auto edge_id = mesh.he_edge(mesh.he_from_vertices(gpf::VertexId{ 0 }, gpf::VertexId{ 1 }));
    assert(edge_to_points_map.at(edge_id) == std::vector<std::size_t>{ 2 });

    assert(face_info_map.at(gpf::FaceId{ 0 }).point_indices == std::vector<std::size_t>{ 0 });
    assert(face_info_map.at(gpf::FaceId{ 1 }).point_indices == std::vector<std::size_t>{ 3 });
    for (const auto& face_info : face_info_map | std::views::values) {
        assert(face_info.barycentric_coordinates.size() == face_info.point_indices.size());
        for (const auto& barycentric : face_info.barycentric_coordinates) {
            assert(std::abs(barycentric[0] + barycentric[1] + barycentric[2] - 1.0) < 1e-14);
        }
    }
}

void
test_walk_on_mesh_surface()
{
    auto make_single_triangle = [] {
        Mesh3d mesh = Mesh3d::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 }, { 2, 1, 3 }, { 0, 2, 3 } });
        mesh.vertex_prop(gpf::VertexId{ 0 }).pt = { -1.0, 0.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 1 }).pt = { 1.0, 0.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 2 }).pt = { 0.0, 1.0, 0.0 };
        mesh.vertex_prop(gpf::VertexId{ 3 }).pt = { 1e-6, 2.0, 0.0 };
        return mesh;
    };

    {
        auto mesh = make_single_triangle();
        const gpf::FaceId fid{ 0 };
        const std::array<double, 3> start_pt{ 0.0, 0.5, 0.0 };
        const std::array<double, 3> direction{ 0.0, 1.0, 0.0 };
        const std::array<double, 4> lengths{ 0.0, 0.1, 0.25, 0.4 };

        const auto result =
          gpf::walk_on_mesh_surface(mesh, fid, start_pt, direction, std::span<const double>{ lengths });
        assert(result.has_value());
        assert(result->size() == lengths.size() + 1);
        const auto& points = *result;
        assert(points[0].first == fid);
        const Eigen::Vector3d expected_start = Eigen::Vector3d::Map(start_pt.data());
        assert(is_close(point_from_barycentric(mesh, points[0]), expected_start, 1e-12));
        const auto expected_barycentric = barycentric_coordinates(mesh, fid, expected_start);
        for (std::size_t i = 0; i < expected_barycentric.size(); ++i) {
            assert(std::abs(points[0].second[i] - expected_barycentric[i]) < 1e-12);
        }
        for (const auto& point : points) {
            assert_valid_barycentric(point);
        }
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            const Eigen::Vector3d expected_pt = expected_start + Eigen::Vector3d::Map(direction.data()) * lengths[i];
            assert(is_close(point_from_barycentric(mesh, points[i + 1]), expected_pt, 1e-12));
        }
    }
    {
        std::vector<double> points{ 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
        const std::size_t N = 1000;

        std::mt19937 rng(42); // Seed for reproducibility
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            const auto a = dist(rng);
            const auto b = dist(rng);
            const auto c = dist(rng);
            const auto sum = a + b + c;
            const auto t1 = a / sum;
            const auto t2 = b / sum;

            points.append_range(std::array<double, 2>{ t1, t2 });
        }
        auto triangles = gpf::triangulate_points(points, true);
        assert(triangles.size() % 3 == 0);
        std::vector<std::array<std::size_t, 3>> triangle_faces;
        triangle_faces.reserve(triangles.size() / 3);
        for (std::size_t i = 0; i < triangles.size(); i += 3) {
            triangle_faces.push_back({ triangles[i], triangles[i + 1], triangles[i + 2] });
        }
        Mesh3d mesh = Mesh3d::new_in(triangle_faces);
        for (auto [v, i] : std::views::zip(mesh.vertices(), std::views::iota(std::size_t{ 0 }, mesh.n_vertices()))) {
            v.prop().pt = std::array{ points[2 * i], points[2 * i + 1], 0.0 };
        }

        write_obj("123.obj", mesh);

        const Eigen::Vector3d start_pt(0.75, 0.0005, 0.0);
        const Eigen::Vector3d end_pt(0.1, 0.8, 0.0);
        const Eigen::Vector3d diff = end_pt - start_pt;
        const double total = diff.norm();
        const Eigen::Vector3d direction = diff / total;
        const std::array<double, 5> lengths{ 0.0, total * 0.25, total * 0.5, total * 0.75, total };

        auto ret = gpf::walk_on_mesh_surface(mesh,
                                             gpf::FaceId{ 1325 },
                                             std::span<const double, 3>{ start_pt.data(), 3 },
                                             std::span<const double, 3>{ direction.data(), 3 },
                                             std::span<const double>{ lengths });
        assert(ret.has_value());
        assert(ret->size() == lengths.size() + 1);
        assert_valid_barycentric(ret->front());
        const auto walked_start = point_from_barycentric(mesh, ret->front());
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            const auto& point = (*ret)[i + 1];
            assert_valid_barycentric(point);
            const auto expected_pt = (walked_start + direction * lengths[i]).eval();
            assert(is_close(point_from_barycentric(mesh, point), expected_pt, 1e-7));
        }
    }
}
