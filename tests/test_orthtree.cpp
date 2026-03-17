#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include <gpf/bbox.hpp>
#include <gpf/orthtree.hpp>

namespace {

using Bbox2 = gpf::BBox<2>;
using Bbox3 = gpf::BBox<3>;

using QuadTree = gpf::Orthtree<2>;
using OcTree = gpf::Orthtree<3>;

} // namespace

void
test_orthtree_quadtree()
{
    QuadTree tree;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    constexpr std::size_t N = 500;
    std::vector<Bbox2> boxes;
    std::vector<std::size_t> indices;
    boxes.reserve(N);
    indices.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        double w = dist(rng) * 0.05;
        double h = dist(rng) * 0.05;
        boxes.push_back(Bbox2({ x, y }, { x + w, y + h }));
        indices.push_back(i);
    }

    tree.insert_boxes(boxes, indices);
    tree.config.enlarge_ratio = 1.01;
    tree.config.max_depth = 16;
    tree.construct(true);

    assert(!tree.nodes.empty());
    assert(tree.root_node().is_internal());

    auto leaves = tree.all_leaf_nodes();
    assert(!leaves.empty());

    std::size_t total_in_leaves = 0;
    for (auto leaf_idx : leaves) {
        total_in_leaves += tree.node(leaf_idx).box_indices.size();
    }
    assert(total_in_leaves >= N);

    std::cout << "test_orthtree_quadtree: " << tree.nodes.size() << " nodes, " << leaves.size() << " leaves\n";
}

void
test_orthtree_octree()
{
    OcTree tree;

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    constexpr std::size_t N = 500;
    std::vector<Bbox3> boxes;
    std::vector<std::size_t> indices;
    boxes.reserve(N);
    indices.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        double z = dist(rng);
        double s = dist(rng) * 0.05;
        boxes.push_back(Bbox3({ x, y, z }, { x + s, y + s, z + s }));
        indices.push_back(i);
    }

    tree.insert_boxes(boxes, indices);
    tree.config.enlarge_ratio = 1.01;
    tree.config.max_depth = 16;
    tree.construct(true);

    assert(!tree.nodes.empty());
    assert(tree.root_node().is_internal());

    auto leaves = tree.all_leaf_nodes();
    assert(!leaves.empty());

    std::cout << "test_orthtree_octree: " << tree.nodes.size() << " nodes, " << leaves.size() << " leaves\n";
}

void
test_orthtree_traversal()
{
    QuadTree tree;

    constexpr std::size_t N = 100;
    std::vector<Bbox2> boxes;
    std::vector<std::size_t> indices;

    for (std::size_t i = 0; i < N; ++i) {
        double x = static_cast<double>(i);
        boxes.push_back(Bbox2({ x, 0.0 }, { x + 1.0, 1.0 }));
        indices.push_back(i);
    }

    tree.insert_boxes(boxes, indices);
    tree.config.max_depth = 16;
    tree.construct(false);

    QuadTree::BoxIntersectionTraversal trav(Bbox2({ 10.5, 0.0 }, { 20.5, 1.0 }));
    tree.traversal(trav);

    assert(!trav.result().empty());
    for (auto id : trav.result()) {
        assert(id >= 10 && id <= 21);
    }

    std::cout << "test_orthtree_traversal: found " << trav.result().size() << " intersections\n";
}
