#include "read_off.hpp"
#include <array>
#include <gpf/detail.hpp>
#include <gpf/surface_mesh.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/mesh_upkeep.hpp>
#include <fstream>
#include <iomanip>
#include <queue>
#include <vector>

template <typename Mesh>
void write_off(const std::string& path, const std::vector<std::array<double, 3>>& vertices, const Mesh& mesh) {
    std::ofstream out(path);
    std::size_t n_triangles = 0;
    for (const auto& f : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto he : f.halfedges()) {
            face_vertices.push_back(he.from().id.idx);
        }
        if (face_vertices.size() >= 3) n_triangles += face_vertices.size() - 2;
    }
    out << "OFF\n";
    out << vertices.size() << ' ' << n_triangles << " 0\n";
    out << std::setprecision(17);
    for (const auto& v : vertices) {
        out << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    }
    for (const auto& f : mesh.faces()) {
        std::vector<std::size_t> face_vertices;
        for (const auto he : f.halfedges()) {
            face_vertices.push_back(he.from().id.idx);
        }
        for (std::size_t i = 1; i + 1 < face_vertices.size(); ++i) {
            out << "3 " << face_vertices[0] << ' ' << face_vertices[i] << ' ' << face_vertices[i + 1]
                << '\n';
        }
    }
}

void test_mesh_edge_collapse1() {
    using namespace gpf;
    auto data = read_off("interface.off");
    struct VertexProp {
        std::array<double, 3> pt;
    };
    struct EdgeProp {
        double len;
        bool need_update{false};
    };
    using Mesh = gpf::SurfaceMesh<VertexProp, Empty, EdgeProp>;
    auto mesh = Mesh::new_in(data.faces);
    for (auto v : mesh.vertices()) {
        v.prop().pt = data.vertices[v.id.idx];
    }
    gpf::update_edge_lengths<3>(mesh);
    gpf::collapse_short_edges(mesh, 0.02);
    write_off("edge_collapsed.off", data.vertices, mesh);
    const auto a = 2;
}
