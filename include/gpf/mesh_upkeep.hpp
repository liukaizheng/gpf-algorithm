#include <gpf/ids.hpp>
#include <queue>

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
    while (!queue.empty()) {
        auto [len, eid] = queue.top();
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
                    queue.emplace(ep.len, e.id);
                }
                ep.need_update = false;
            }
        }
    }
}
}
