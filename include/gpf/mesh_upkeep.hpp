#include <array>
#include <limits>
#include <queue>

#include <gpf/mesh.hpp>
#include <gpf/mesh_property.hpp>

namespace gpf {
template <typename Mesh>
void collapse_short_edges(Mesh& mesh, const double tol) {
    std::queue<std::pair<EdgeId, double>> queue;
    for (auto edge : mesh.edges()) {
        if (edge.prop().len < tol) {
            queue.push({edge.id, edge.prop().len});
        }
    }
    while (!queue.empty()) {
        auto [eid, len] = queue.front();
        queue.pop();
        if (mesh.edge_is_deleted(eid) || mesh.edge_prop(eid).len != len) {
            continue;
        }
        const auto [va, vb] = mesh.e_vertices(eid);
        for (auto e : mesh.vertex(vb).edges()) {
            e.prop().need_update = true;
        }

        mesh.collapse_edge(eid, va, vb);
        for (auto e : mesh.vertex(va).edges()) {
            auto& ep = e.prop();
            if (ep.need_update) {
                update_edge_length<3>(e);
                if (ep.len < len) {
                    queue.push({e.id, ep.len});
                }
                ep.need_update = false;
            }
        }
    }
}
}
