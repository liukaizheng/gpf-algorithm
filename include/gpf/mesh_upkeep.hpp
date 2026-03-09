#include <gpf/mesh_property.hpp>
#include <queue>
#include <gpf/mesh.hpp>
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
        auto vb = mesh.edge(eid).halfedge().to();
        for (auto e : vb.edges()) {
            e.prop().need_update = true;
        }


        auto va = mesh.collapse_edge(eid);
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
