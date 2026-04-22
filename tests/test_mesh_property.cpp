#include <array>
#include <cmath>

#include <gpf/mesh.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

namespace {
struct VertexProp
{
    std::array<double, 3> pt;
    double angle_sum = 0.0;
};

struct EdgeProp
{
    double square_len = 0.0;
    double len = 0.0;
};

struct HalfedgeProp
{
    double angle = 0.0;
    double signpost_angle = 0.0;
    std::array<double, 2> vector;
};
} // namespace

void
test_property_edge_length_updates()
{
    using Mesh = gpf::ManifoldMesh<VertexProp, HalfedgeProp, EdgeProp, gpf::Empty>;

    Mesh mesh = Mesh::new_in(std::vector<std::vector<std::size_t>>{ { 0, 1, 2 } });

    for (auto v : mesh.vertices()) {
        const auto id = v.id.idx;
        if (id == 0) {
            v.prop().pt = { 0.0, 0.0, 0.0 };
        } else if (id == 1) {
            v.prop().pt = { 1.0, 0.0, 0.0 };
        } else if (id == 2) {
            v.prop().pt = { 0.0, 2.0, 0.0 };
        } else {
            assert(false);
        }
    }

    gpf::update_edge_lengths_squared<3>(mesh);
    gpf::update_edge_lengths<3>(mesh);
    gpf::update_corner_angles(mesh);
    gpf::update_vertex_angle_sums(mesh);
    gpf::update_halfedge_signpost_angles(mesh);
    gpf::update_halfedge_vectors(mesh);

    for (auto e : mesh.edges()) {
        const auto [va, vb] = e.vertices();
        const auto pa = va.prop().pt;
        const auto pb = vb.prop().pt;

        const double dx = pa[0] - pb[0];
        const double dy = pa[1] - pb[1];
        const double dz = pa[2] - pb[2];
        const double expected = dx * dx + dy * dy + dz * dz;
        const double expected_len = std::sqrt(expected);

        assert(std::abs(e.data().square_len - expected) < 1e-12);
        assert(std::abs(e.data().len - expected_len) < 1e-12);
    }
}
