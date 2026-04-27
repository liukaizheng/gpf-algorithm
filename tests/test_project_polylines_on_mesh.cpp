#include <gpf/manifold_mesh.hpp>
#include <gpf/project_polylines_on_mesh.hpp>

#include <print>
#include <random>

namespace {
struct VertexProp
{
    std::array<double, 2> pt;
};
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

        points.emplace_back(std::array<double, 2>{t1, t2});
    }
    const std::vector<std::vector<std::size_t>> polylines{ { 0, 1 } };

    const auto [point_vertices, paths] = gpf::project_polylines_on_mesh<2>(points, polylines, mesh);
    write_obj("project_mesh.obj", mesh);
    assert(paths.size() == 1);
    assert(paths.front().size() == 1);

    const auto he = mesh.halfedge(paths.front().front());
    const auto pa = he.from().prop().pt;
    const auto pb = he.to().prop().pt;
    auto is_close = [](double a, double b) { return std::abs(a - b) < 1e-9; };
    const bool forward = is_close(pa[0], 0.25) && is_close(pa[1], 0.0) && is_close(pb[0], 0.75) && is_close(pb[1], 0.0);
    const bool backward =
      is_close(pa[0], 0.75) && is_close(pa[1], 0.0) && is_close(pb[0], 0.25) && is_close(pb[1], 0.0);
    assert(forward || backward);

    assert(is_close(points[0][0], 0.25) && is_close(points[0][1], 0.0));
    assert(is_close(points[1][0], 0.75) && is_close(points[1][1], 0.0));
}
