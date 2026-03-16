#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gpf/orthtree.hpp>

namespace {

struct Point2
{
    double x, y;

    Point2() = default;
    Point2(double v)
      : x(v)
      , y(v)
    {
    }
    Point2(double x, double y)
      : x(x)
      , y(y)
    {
    }

    double operator[](std::size_t i) const { return i == 0 ? x : y; }
    double& operator[](std::size_t i) { return i == 0 ? x : y; }

    Point2 operator+(const Point2& o) const { return { x + o.x, y + o.y }; }
    Point2 operator-(const Point2& o) const { return { x - o.x, y - o.y }; }
    Point2 operator*(double s) const { return { x * s, y * s }; }

    double length() const { return std::sqrt(x * x + y * y); }
};

struct Bbox2
{
    Point2 lo, hi;

    Bbox2() = default;
    Bbox2(Point2 lo, Point2 hi)
      : lo(lo)
      , hi(hi)
    {
    }

    Point2& min_bound() { return lo; }
    const Point2& min_bound() const { return lo; }
    Point2& max_bound() { return hi; }
    const Point2& max_bound() const { return hi; }

    double min_coord(std::size_t i) const { return lo[i]; }
    double max_coord(std::size_t i) const { return hi[i]; }

    std::size_t longest_axis() const
    {
        double dx = hi.x - lo.x;
        double dy = hi.y - lo.y;
        return dx >= dy ? 0 : 1;
    }

    Bbox2& operator+=(const Bbox2& o)
    {
        lo.x = std::min(lo.x, o.lo.x);
        lo.y = std::min(lo.y, o.lo.y);
        hi.x = std::max(hi.x, o.hi.x);
        hi.y = std::max(hi.y, o.hi.y);
        return *this;
    }
};

struct Point3
{
    double x, y, z;

    Point3() = default;
    Point3(double v)
      : x(v)
      , y(v)
      , z(v)
    {
    }
    Point3(double x, double y, double z)
      : x(x)
      , y(y)
      , z(z)
    {
    }

    double operator[](std::size_t i) const { return i == 0 ? x : (i == 1 ? y : z); }
    double& operator[](std::size_t i) { return i == 0 ? x : (i == 1 ? y : z); }

    Point3 operator+(const Point3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Point3 operator-(const Point3& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Point3 operator*(double s) const { return { x * s, y * s, z * s }; }
};

struct Bbox3
{
    Point3 lo, hi;

    Bbox3() = default;
    Bbox3(Point3 lo, Point3 hi)
      : lo(lo)
      , hi(hi)
    {
    }

    Point3& min_bound() { return lo; }
    const Point3& min_bound() const { return lo; }
    Point3& max_bound() { return hi; }
    const Point3& max_bound() const { return hi; }

    double min_coord(std::size_t i) const { return lo[i]; }
    double max_coord(std::size_t i) const { return hi[i]; }

    std::size_t longest_axis() const
    {
        double dx = hi.x - lo.x;
        double dy = hi.y - lo.y;
        double dz = hi.z - lo.z;
        if (dx >= dy && dx >= dz)
            return 0;
        if (dy >= dz)
            return 1;
        return 2;
    }

    Bbox3& operator+=(const Bbox3& o)
    {
        lo.x = std::min(lo.x, o.lo.x);
        lo.y = std::min(lo.y, o.lo.y);
        lo.z = std::min(lo.z, o.lo.z);
        hi.x = std::max(hi.x, o.hi.x);
        hi.y = std::max(hi.y, o.hi.y);
        hi.z = std::max(hi.z, o.hi.z);
        return *this;
    }
};

struct SplitPred
{
    template<typename Tree, typename Node>
    bool operator()(const Tree&, const Node& nd) const
    {
        return nd.total_size > 10;
    }
};

struct DoIntersect2
{
    bool operator()(const Bbox2& a, const Bbox2& b) const
    {
        return a.lo.x <= b.hi.x && a.hi.x >= b.lo.x && a.lo.y <= b.hi.y && a.hi.y >= b.lo.y;
    }
};

struct CalcBbox2
{
    Bbox2 operator()(const Bbox2& b) const { return b; }
};

struct DoIntersect3
{
    bool operator()(const Bbox3& a, const Bbox3& b) const
    {
        return a.lo.x <= b.hi.x && a.hi.x >= b.lo.x && a.lo.y <= b.hi.y && a.hi.y >= b.lo.y && a.lo.z <= b.hi.z &&
               a.hi.z >= b.lo.z;
    }
};

struct CalcBbox3
{
    Bbox3 operator()(const Bbox3& b) const { return b; }
};

struct QuadTraits
{
    static constexpr std::size_t Dimension = 2;
    static constexpr std::size_t MaxDepth = 16;
    using NT = double;
    using BboxT = Bbox2;
    using PrimAttrT = std::size_t;
    using SplitPred = ::SplitPred;
    using DoIntersect = DoIntersect2;
    using CalcBbox = CalcBbox2;
};

struct OcTraits
{
    static constexpr std::size_t Dimension = 3;
    static constexpr std::size_t MaxDepth = 16;
    using NT = double;
    using BboxT = Bbox3;
    using PrimAttrT = std::size_t;
    using SplitPred = ::SplitPred;
    using DoIntersect = DoIntersect3;
    using CalcBbox = CalcBbox3;
};

} // namespace

void
test_orthtree_quadtree()
{
    gpf::Orthtree<QuadTraits> tree;

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
    tree.construct(true, 1.01);

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
    gpf::Orthtree<OcTraits> tree;

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
    tree.construct(true, 1.01);

    assert(!tree.nodes.empty());
    assert(tree.root_node().is_internal());

    auto leaves = tree.all_leaf_nodes();
    assert(!leaves.empty());

    std::cout << "test_orthtree_octree: " << tree.nodes.size() << " nodes, " << leaves.size() << " leaves\n";
}

void
test_orthtree_traversal()
{
    gpf::Orthtree<QuadTraits> tree;

    constexpr std::size_t N = 100;
    std::vector<Bbox2> boxes;
    std::vector<std::size_t> indices;

    for (std::size_t i = 0; i < N; ++i) {
        double x = static_cast<double>(i);
        boxes.push_back(Bbox2({ x, 0.0 }, { x + 1.0, 1.0 }));
        indices.push_back(i);
    }

    tree.insert_boxes(boxes, indices);
    tree.construct(false, 1.2);

    gpf::BoxIntersectionTraversal<QuadTraits> trav(Bbox2({ 10.5, 0.0 }, { 20.5, 1.0 }));
    tree.traversal(trav);

    assert(!trav.result().empty());
    for (auto id : trav.result()) {
        assert(id >= 10 && id <= 21);
    }

    std::cout << "test_orthtree_traversal: found " << trav.result().size() << " intersections\n";
}
