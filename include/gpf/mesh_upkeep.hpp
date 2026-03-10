#include <queue>
#include <unordered_set>
#include <algorithm>

#include <gpf/mesh.hpp>
#include <gpf/mesh_property.hpp>

namespace gpf {
template <typename Mesh>
void collapse_short_edges(Mesh& mesh, const double tol) {
    std::priority_queue<std::pair<double, EdgeId>, std::vector<std::pair<double, EdgeId>>, std::greater<std::pair<double, EdgeId>>> queue;
    for (auto edge : mesh.edges()) {
        if (edge.prop().len < tol) {
            queue.emplace(edge.prop().len, edge.id);
        }
    }

    std::unordered_set<VertexId> vb_oppo_vertices;
    while (!queue.empty()) {
        auto [len, eid] = queue.top();
        queue.pop();
        if (mesh.edge_is_deleted(eid) || mesh.edge_prop(eid).len != len) {
            continue;
        }
        const auto [va, vb] = mesh.e_vertices(eid);
        vb_oppo_vertices.clear();
        for (auto e : mesh.vertex(vb).edges()) {
            auto [v1, v2] = mesh.e_vertices(e.id);
            if (v1 == vb) {
                vb_oppo_vertices.emplace(v2);
            } else {
                assert(v2 == vb);
                vb_oppo_vertices.emplace(v1);
            }
        }

        if (std::ranges::any_of(mesh.vertex(va).edges(), [&mesh, &vb_oppo_vertices, va, eid](auto e) {
            auto [v1, v2] = mesh.e_vertices(e.id);
            const VertexId va_oppo = v1 == va ? v2 : v1;
            if (vb_oppo_vertices.contains(va_oppo)) {
                for (const auto he : mesh.edge(eid).halfedges()) {
                    if (he.next().to().id == va_oppo) {
                        return false;
                    }
                }
                return true;
            }
            return false;
        })) {
            continue;
        }

        for (auto e : mesh.vertex(vb).edges()) {
            e.prop().need_update = true;
        }

        mesh.collapse_edge(eid, va, vb);
        for (auto e : mesh.vertex(va).edges()) {
            auto& ep = e.prop();
            if (ep.need_update) {
                update_edge_length<3>(e);
                if (ep.len < len) {
                    queue.emplace(ep.len, e.id);
                }
                ep.need_update = false;
            }
        }
    }
}
}
