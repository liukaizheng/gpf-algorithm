#include "read_off.hpp"
#include <algorithm>
#include <array>
#include <gpf/detail.hpp>
#include <gpf/ids.hpp>
#include <gpf/surface_mesh.hpp>
#include <gpf/mesh_property.hpp>
#include <gpf/mesh_upkeep.hpp>
#include <fstream>
#include <iomanip>
#include <queue>
#include <ranges>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>

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

template <typename Mesh>
void write_non_manifold_edges_obj(const std::string& path,
                                  const std::vector<std::array<double, 3>>& vertices,
                                  const Mesh& mesh) {
    std::ofstream out(path);
    out << std::setprecision(17);
    for (const auto& v : vertices) {
        out << "v " << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    }
    for (const auto& e : mesh.edges()) {
        std::size_t face_count = 0;
        for (const auto he : e.halfedges()) {
            if (he.face().id.valid()) {
                ++face_count;
            }
        }
        if (face_count > 2) {
            auto [va, vb] = e.vertices();
            out << "l " << (va.id.idx + 1) << ' ' << (vb.id.idx + 1) << '\n';
        }
    }
}

template <typename Mesh>
void collapse_degenerate_triangles(const std::vector<std::array<double, 3>>& vertices, Mesh &mesh, const double tol) {
    using namespace gpf;
    std::array<HalfedgeId, 3> tri_halfedges;
    std::array<double, 3> tri_edge_lengths;
    auto get_metric = [&mesh, &tri_halfedges, &tri_edge_lengths, tol](const gpf::FaceId fid) {
        auto face = mesh.face(fid);
        if (mesh.face_is_deleted(face.id)) {
            return std::make_pair(HalfedgeId{}, 0.0);
        }
        auto ha = face.halfedge();
        auto hb = ha.next();
        auto hc = hb.next();
        tri_halfedges[0] = ha.id;
        tri_halfedges[1] = hb.id;
        tri_halfedges[2] = hc.id;

        tri_edge_lengths[0] = ha.edge().prop().len;
        tri_edge_lengths[1] = hb.edge().prop().len;
        tri_edge_lengths[2] = hc.edge().prop().len;
        if (tri_edge_lengths[0] < tol || tri_edge_lengths[1] < tol || tri_edge_lengths[2] < tol) {
            return std::make_pair(HalfedgeId{}, 0.0);
        }
        std::size_t max_idx = 0;
        for (std::size_t i = 1; i < 3; ++i) {
            if (tri_edge_lengths[i] > tri_edge_lengths[max_idx]) {
                max_idx = i;
            }
        }
        return std::make_pair(tri_halfedges[max_idx], tri_edge_lengths[(max_idx + 1) % 3] + tri_edge_lengths[(max_idx + 2) % 3] - tri_edge_lengths[max_idx]);
    };

    std::queue<std::pair<gpf::HalfedgeId, double>> queue;
    for (auto face : mesh.faces()) {
        auto pair = get_metric(face.id);
        if (pair.first.valid() && pair.second < 2.0 * tol) {
            queue.push(std::move(pair));
        }
    }

    while (!queue.empty()) {
        auto old_pair = queue.front();
        queue.pop();
        auto curr_pair = get_metric(mesh.he_face(old_pair.first));
        if (old_pair != curr_pair) {
            continue;
        }

        auto he_ab = curr_pair.first;
        auto vb = mesh.he_to(he_ab);
        auto he_bc = mesh.he_next(he_ab);
        auto vc = mesh.he_to(he_bc);
        auto he_ca = mesh.he_next(he_bc);
        auto va = mesh.he_to(he_ca);

        auto pa = Eigen::Vector3d::Map(mesh.vertex_prop(va).pt.data());
        auto pb = Eigen::Vector3d::Map(mesh.vertex_prop(vb).pt.data());
        auto pc = Eigen::Vector3d::Map(mesh.vertex_prop(vc).pt.data());
        auto dir = (pb - pa).eval();
        dir /= mesh.edge_prop(mesh.he_edge(curr_pair.first)).len;

        auto d1 = std::abs((pc - pa).dot(dir));
        auto d2 = std::abs((pc - pb).dot(dir));
        auto t = d1 / (d2 + d1);
        std::array<double, 3> pd{};
        auto pd_ref = Eigen::Vector3d::Map(pd.data());
        pd_ref = pa * (1.0 - t) + pb * t;

        if ((pc - pd_ref).norm() > tol) {
            auto len = (pc - pd_ref).norm();
            continue;
        }

        gpf::VertexId target_vid{};
        gpf::EdgeId collpase_eid{};
        if (mesh.halfedge(he_ca).edge().prop().len < mesh.halfedge(he_bc).edge().prop().len) {
            target_vid = va;
            collpase_eid = mesh.he_edge(he_ca);
        } else {
            target_vid = vb;
            collpase_eid = mesh.he_edge(he_bc);
        }

        for (auto e : mesh.vertex(vc).edges()) {
            e.prop().need_update = true;
        }
        if (collpase_eid.idx == 16847) {
            write_off("before_collapse.off", vertices, mesh);
        }

        mesh.collapse_edge(collpase_eid, target_vid, vc);
        for (auto e : mesh.vertex(target_vid).edges()) {
            auto& ep = e.prop();
            if (ep.need_update) {
                update_edge_length<3>(e);
                ep.need_update = false;
            }
        }
        for (auto he : mesh.vertex(target_vid).incoming_halfedges()) {
            auto pair = get_metric(he.face().id);
            if (pair.first.valid() && pair.second < 2.0 * tol) {
                queue.push(std::move(pair));
            }
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

    const double tol = 0.02;
    gpf::collapse_short_edges(mesh, tol);
    write_off("edge_collapsed.off", data.vertices, mesh);
    // collapse_degenerate_triangles(data.vertices, mesh, tol);
    const auto n_queue_cap = static_cast<std::size_t>(mesh.n_edges() * 0.4);
    std::priority_queue<double> pq;
    for (const auto e : mesh.edges()) {
        const auto len = e.prop().len;
        if (pq.size() < n_queue_cap) {
            pq.emplace(len);
        } else if (len < pq.top()) {
            pq.pop();
            pq.emplace(len);
        }
    }
    const auto max_edge_len = std::max(5.0, pq.top());

    std::priority_queue<std::pair<double, EdgeId>> queue;
    for (const auto e : mesh.edges()) {
        auto edge_len = e.prop().len;
        if (edge_len > max_edge_len) {
            queue.emplace(edge_len, e.id);
        }
    }
    while(!queue.empty()) {
        auto [edge_len, eid] = queue.top();
        queue.pop();
        if (edge_len != mesh.edge_prop(eid).len) {
            continue;
        }

        auto [va, vb] = mesh.e_vertices(eid);
        const auto new_vid = mesh.split_edge(eid);
        Eigen::Vector3d::Map(mesh.vertex_prop(new_vid).pt.data()) = (Eigen::Vector3d::Map(mesh.vertex_prop(va).pt.data()) + Eigen::Vector3d::Map(mesh.vertex_prop(vb).pt.data())) * 0.5;

        auto new_v_he = mesh.vertex(new_vid).halfedge();
        const auto half_edge_len = edge_len * 0.5;
        new_v_he.edge().prop().len = half_edge_len;
        new_v_he.prev().edge().prop().len  = half_edge_len;

        for (auto he : mesh.edge(eid).halfedges()) {
            auto vc = he.next().to().id;
            mesh.split_face(he.face().id, new_vid, vc);
        }

        for (auto e : mesh.vertex(new_vid).edges()) {
            update_edge_length<3>(e);
            auto len = e.prop().len;
            if (len > max_edge_len) {
                queue.emplace(len, e.id);
            }
        }
    }


    write_non_manifold_edges_obj("non_manifold.obj", data.vertices, mesh);
    for (std::size_t i = data.vertices.size(); i < mesh.n_vertices_capacity(); i++) {
        data.vertices.emplace_back(mesh.vertex_prop(VertexId{i}).pt);
    }
    write_off("split.off", data.vertices, mesh);
    const auto a = 2;
}
