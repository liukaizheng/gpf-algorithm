#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <deque>
#include <mutex>
#include <numeric>
#include <queue>
#include <type_traits>
#include <vector>

namespace gpf {

constexpr std::size_t kOrthtreeInvalidIndex = static_cast<std::size_t>(-1);

inline bool
is_valid_index(std::size_t idx) noexcept
{
    return idx != kOrthtreeInvalidIndex;
}

namespace orthtree_detail {

template<typename T>
concept HasMaxDepth = requires {
    { T::MaxDepth } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept HasStoreBoxesInInternalNodes = requires {
    { T::StoreBoxesInInternalNodes } -> std::convertible_to<bool>;
};

template<typename T>
concept HasNT = requires { typename T::NT; };

template<typename T>
concept HasPrimAttrT = requires { typename T::PrimAttrT; };

template<typename T>
concept HasNodeAttrT = requires { typename T::NodeAttrT; };

template<typename T>
concept HasShapeRefinePred = requires { typename T::ShapeRefinePred; };

struct NoShapeRefinePred
{
    template<typename Tree, typename Node>
    bool operator()(const Tree&, const Node&, auto&) const
    {
        return false;
    }
};

template<typename T>
struct deduce_nt
{
    using type = double;
};
template<HasNT T>
struct deduce_nt<T>
{
    using type = typename T::NT;
};
template<typename T>
using deduce_nt_t = typename deduce_nt<T>::type;

template<typename T>
struct deduce_prim_attr
{
    using type = void;
};
template<HasPrimAttrT T>
struct deduce_prim_attr<T>
{
    using type = typename T::PrimAttrT;
};
template<typename T>
using deduce_prim_attr_t = typename deduce_prim_attr<T>::type;

template<typename T>
struct deduce_node_attr
{
    using type = void;
};
template<HasNodeAttrT T>
struct deduce_node_attr<T>
{
    using type = typename T::NodeAttrT;
};
template<typename T>
using deduce_node_attr_t = typename deduce_node_attr<T>::type;

template<typename T>
struct deduce_shape_refine_pred
{
    using type = NoShapeRefinePred;
};
template<HasShapeRefinePred T>
struct deduce_shape_refine_pred<T>
{
    using type = typename T::ShapeRefinePred;
};
template<typename T>
using deduce_shape_refine_pred_t = typename deduce_shape_refine_pred<T>::type;

} // namespace orthtree_detail

template<typename UserTraits>
struct OrthtreeTraits
{
    static constexpr std::size_t Dimension = UserTraits::Dimension;
    using BboxT = typename UserTraits::BboxT;
    using SplitPred = typename UserTraits::SplitPred;
    using DoIntersect = typename UserTraits::DoIntersect;
    using CalcBbox = typename UserTraits::CalcBbox;

    static constexpr std::size_t MaxDepth = [] {
        if constexpr (orthtree_detail::HasMaxDepth<UserTraits>)
            return UserTraits::MaxDepth;
        else
            return std::size_t(32);
    }();

    static constexpr bool StoreBoxesInInternalNodes = [] {
        if constexpr (orthtree_detail::HasStoreBoxesInInternalNodes<UserTraits>)
            return UserTraits::StoreBoxesInInternalNodes;
        else
            return false;
    }();

    using NT = orthtree_detail::deduce_nt_t<UserTraits>;
    using PrimAttrT = orthtree_detail::deduce_prim_attr_t<UserTraits>;
    using NodeAttrT = orthtree_detail::deduce_node_attr_t<UserTraits>;
    using ShapeRefinePred = orthtree_detail::deduce_shape_refine_pred_t<UserTraits>;
    struct TreeBboxNoAttr : public BboxT
    {
        [[nodiscard]] const BboxT& bbox() const { return *static_cast<const BboxT*>(this); }
        [[nodiscard]] BboxT& bbox() { return *static_cast<BboxT*>(this); }
    };

    struct TreeBboxAttr : public BboxT
    {
        [[nodiscard]] const BboxT& bbox() const { return *static_cast<const BboxT*>(this); }
        [[nodiscard]] BboxT& bbox() { return *static_cast<BboxT*>(this); }

        [[nodiscard]] PrimAttrT attr() const { return attr_; }
        [[nodiscard]] PrimAttrT& attr() { return attr_; }

      private:
        PrimAttrT attr_;
    };

    using TreeBboxT = std::conditional_t<std::is_void_v<PrimAttrT>, TreeBboxNoAttr, TreeBboxAttr>;
};

template<typename Traits>
class OrthtreeNodeBase
{
  public:
    static constexpr std::size_t MaxDepth = Traits::MaxDepth;
    static constexpr std::size_t Dimension = Traits::Dimension;
    static constexpr std::size_t Degree = (1u << Dimension);

    static_assert(MaxDepth <= 64);

    using BboxT = typename Traits::BboxT;
    using TreeBboxT = typename Traits::TreeBboxT;

    OrthtreeNodeBase()
    {
        parent = kOrthtreeInvalidIndex;
        children_start = kOrthtreeInvalidIndex;
        n_children = 0;
        depth = 0;
        total_size = 0;
        std::fill(child_map.begin(), child_map.end(), kOrthtreeInvalidIndex);
    }

    void shallow_copy_from(const OrthtreeNodeBase& rhs)
    {
        parent = rhs.parent;
        children_start = rhs.children_start;
        n_children = rhs.n_children;
        depth = rhs.depth;
        loose_box = rhs.loose_box;
        tight_box = rhs.tight_box;
        total_size = 0;
        child_map = rhs.child_map;
    }

    void clear_box_indices() { box_indices = std::vector<std::size_t>(); }

    [[nodiscard]] bool is_root() const noexcept { return depth == 0; }
    [[nodiscard]] bool is_internal() const noexcept { return is_valid_index(children_start); }
    [[nodiscard]] bool is_leaf() const noexcept { return !is_valid_index(children_start); }

    [[nodiscard]] std::size_t child(std::size_t index) const
    {
        assert(is_valid_index(children_start));
        assert(index < n_children);
        return children_start + index;
    }

    // Topology
    std::size_t parent;
    std::size_t children_start;
    std::size_t n_children;
    std::size_t depth;

    // Geometry
    BboxT loose_box;
    BboxT tight_box;

    std::vector<std::size_t> box_indices;
    std::size_t total_size;

    // Maps child index in the full Degree-sized space to the actual compacted child index
    std::array<std::size_t, Degree> child_map;
};

template<typename Traits, typename NodeAttrT>
class OrthtreeNode : public OrthtreeNodeBase<Traits>
{
  public:
    [[nodiscard]] NodeAttrT& attribute() { return attribute_; }
    [[nodiscard]] const NodeAttrT& attribute() const { return attribute_; }

  private:
    NodeAttrT attribute_;
};

template<typename Traits>
class OrthtreeNode<Traits, void> : public OrthtreeNodeBase<Traits>
{};

template<typename UserTraits>
class Orthtree
{
  public:
    using Traits = OrthtreeTraits<UserTraits>;

    static constexpr std::size_t MaxDepth = Traits::MaxDepth;
    static constexpr std::size_t Dimension = Traits::Dimension;
    static constexpr std::size_t Degree = (1u << Dimension);
    static constexpr bool StoreBoxesInInternalNodes = Traits::StoreBoxesInInternalNodes;

    using NT = typename Traits::NT;
    using BboxT = typename Traits::BboxT;
    using TreeBboxT = typename Traits::TreeBboxT;
    using TreePoint = std::remove_cvref_t<decltype(std::declval<TreeBboxT>().min_bound())>;

    using CalcBbox = typename Traits::CalcBbox;
    using DoIntersect = typename Traits::DoIntersect;
    using SplitPred = typename Traits::SplitPred;
    using ShapeRefinePred = typename Traits::ShapeRefinePred;

    using NodeAttrT = typename Traits::NodeAttrT;
    using Node = OrthtreeNode<Traits, NodeAttrT>;

    Orthtree() = default;

    Orthtree(const Orthtree& rhs) { shallow_copy(rhs); }

    Orthtree& operator=(const Orthtree& rhs)
    {
        shallow_copy(rhs);
        return *this;
    }

    void shallow_copy(const Orthtree& rhs)
    {
        nodes.clear();
        nodes.resize(rhs.nodes.size());
        for (std::size_t i = 0; i < rhs.nodes.size(); ++i) {
            nodes[i].shallow_copy_from(rhs.nodes[i]);
        }
        bbox = rhs.bbox;
        split_pred = rhs.split_pred;
        shape_refine_pred = rhs.shape_refine_pred;
        do_intersect = rhs.do_intersect;
        calc_bbox = rhs.calc_bbox;
        enlarge_ratio = rhs.enlarge_ratio;
    }

    template<typename Primitives, typename Attributes>
    void insert_primitives(const Primitives& primitives, const Attributes& attributes)
    {
        assert(primitives.size() == attributes.size());
        clear();
        boxes.reserve(static_cast<std::size_t>(primitives.size() * 1.2));
        boxes.resize(primitives.size());
        for (std::size_t i = 0; i < primitives.size(); ++i) {
            boxes[i].bbox() = calc_bbox(primitives[i]);
            boxes[i].attr() = attributes[i];
        }
    }

    template<typename Bboxes, typename Attributes>
    void insert_boxes(const Bboxes& bboxes, const Attributes& attributes)
    {
        assert(bboxes.size() == attributes.size());
        clear();
        boxes.reserve(static_cast<std::size_t>(bboxes.size() * 1.2));
        boxes.resize(bboxes.size());
        for (std::size_t i = 0; i < bboxes.size(); ++i) {
            boxes[i].bbox() = bboxes[i];
            boxes[i].attr() = attributes[i];
        }
    }

    void construct(bool compact_box = false, NT enlarge = NT(1.2), NT adaptive_thres = NT(0.1))
    {
        enlarge_ratio = enlarge;
        adaptive_threshold = adaptive_thres;

        bbox = calc_bbox_from_boxes(boxes.begin(), boxes.end());

        nodes.clear();
        nodes.emplace_back();
        root_node().tight_box = bbox;

        TreePoint bbox_center = (bbox.max_bound() + bbox.min_bound()) * NT(0.5);
        TreePoint side_length = bbox.max_bound() - bbox.min_bound();
        assert(enlarge >= NT(1.0));
        if (!compact_box) {
            std::size_t longest = bbox.longest_axis();
            side_length = TreePoint(side_length[longest]);
        }
        side_length = side_length * enlarge;

        bbox.min_bound() = bbox_center - side_length * NT(0.5);
        bbox.max_bound() = bbox_center + side_length * NT(0.5);

        root_node().loose_box = bbox;
        root_node().box_indices.resize(boxes.size());
        root_node().total_size = boxes.size();
        std::iota(root_node().box_indices.begin(), root_node().box_indices.end(), std::size_t(0));

        std::deque<std::size_t> nodes_to_split;
        nodes_to_split.push_back(kRootIdx);

        while (!nodes_to_split.empty()) {
            std::size_t cur_idx = nodes_to_split.front();
            nodes_to_split.pop_front();
            Node& cur = node(cur_idx);
            if (cur.depth < MaxDepth && split_pred(*this, cur)) {
                if (split(cur_idx)) {
                    for (std::size_t i = 0; i < node(cur_idx).n_children; ++i) {
                        nodes_to_split.push_back(node(cur_idx).child(i));
                    }
                }
            }
        }
    }

    void shape_refine()
    {
        std::queue<std::size_t> nodes_to_split;
        for (std::size_t nidx = 0; nidx < nodes.size(); ++nidx) {
            if (node(nidx).is_leaf()) {
                nodes_to_split.push(nidx);
            }
        }

        while (!nodes_to_split.empty()) {
            std::size_t cur_idx = nodes_to_split.front();
            nodes_to_split.pop();

            std::array<bool, Dimension> partitionable;
            if (node(cur_idx).depth < MaxDepth && shape_refine_pred(*this, node(cur_idx), partitionable)) {
                TreePoint center = node_center(node(cur_idx));

                std::array<BboxT, Degree> child_boxes;
                calc_box_for_children(node(cur_idx), center, child_boxes);

                bool need_moving = std::find(partitionable.begin(), partitionable.end(), false) != partitionable.end();

                if (need_moving) {
                    std::array<std::size_t, Degree> collapse_dest;
                    calc_collapse_destination(partitionable, collapse_dest);
                    for (std::size_t i = 0; i < Degree; ++i) {
                        std::size_t dest = collapse_dest[i];
                        if (dest == i)
                            continue;
                        child_boxes[dest] += child_boxes[i];
                    }
                    unsigned int n_ch = 1u << static_cast<unsigned int>(std::count_if(
                                          partitionable.begin(), partitionable.end(), [](bool b) { return b; }));

                    std::size_t children_idx = new_children(n_ch);
                    node(cur_idx).children_start = children_idx;
                    node(cur_idx).n_children = n_ch;
                    node(cur_idx).child_map = collapse_dest;

                    std::array<std::size_t, Degree> sorted_dest = collapse_dest;
                    std::sort(sorted_dest.begin(), sorted_dest.end());
                    auto end_it = std::unique(sorted_dest.begin(), sorted_dest.end());
                    (void)end_it;
                    for (std::size_t i = 0; i < n_ch; ++i) {
                        Node& ch = node(node(cur_idx).child(i));
                        ch.depth = node(cur_idx).depth + 1;
                        ch.parent = cur_idx;
                        ch.loose_box = child_boxes[sorted_dest[i]];
                        ch.tight_box = ch.loose_box;
                        ch.total_size = 0;
                        for (std::size_t& dest : node(cur_idx).child_map)
                            if (dest == sorted_dest[i])
                                dest = i;
                    }
                } else {
                    std::size_t children_idx = new_children(Degree);
                    node(cur_idx).children_start = children_idx;
                    node(cur_idx).n_children = Degree;
                    std::iota(node(cur_idx).child_map.begin(), node(cur_idx).child_map.end(), std::size_t(0));
                    for (std::size_t i = 0; i < Degree; ++i) {
                        Node& ch = node(node(cur_idx).child(i));
                        ch.depth = node(cur_idx).depth + 1;
                        ch.parent = cur_idx;
                        ch.loose_box = child_boxes[i];
                        ch.tight_box = ch.loose_box;
                        ch.total_size = 0;
                    }
                }
                for (std::size_t i = 0; i < node(cur_idx).n_children; ++i) {
                    nodes_to_split.push(node(cur_idx).child(i));
                }
            }
        }
    }

    void clear()
    {
        nodes.clear();
        bbox = BboxT();
        boxes.clear();
    }

    void clear_boxes()
    {
        for (auto& nd : nodes) {
            nd.clear_box_indices();
        }
        boxes.clear();
        boxes.shrink_to_fit();
    }

    [[nodiscard]] const BboxT& box() const { return bbox; }
    [[nodiscard]] const TreeBboxT& tree_box(std::size_t id) const { return boxes[id]; }
    [[nodiscard]] TreeBboxT& tree_box(std::size_t id) { return boxes[id]; }

    [[nodiscard]] std::size_t root_node_idx() const { return kRootIdx; }
    [[nodiscard]] Node& root_node() { return node(kRootIdx); }
    [[nodiscard]] const Node& root_node() const { return node(kRootIdx); }

    [[nodiscard]] Node& node(std::size_t idx) { return nodes[idx]; }
    [[nodiscard]] const Node& node(std::size_t idx) const { return nodes[idx]; }

    [[nodiscard]] std::size_t size() const { return boxes.size(); }

    [[nodiscard]] TreePoint node_center(const Node& nd) const
    {
        return (nd.tight_box.min_bound() + nd.tight_box.max_bound()) * NT(0.5);
    }

    [[nodiscard]] std::vector<std::size_t> all_leaf_nodes() const
    {
        std::vector<std::size_t> res;
        res.reserve(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            if (node(i).is_leaf()) {
                res.push_back(i);
            }
        }
        return res;
    }

    template<typename TraversalTrait>
    void traversal(TraversalTrait& traits) const
    {
        traversal_node(root_node(), traits);
    }

  private:
    static constexpr std::size_t kRootIdx = 0;

    [[nodiscard]] std::size_t new_children(std::size_t count)
    {
        std::size_t first_idx = nodes.size();
        nodes.resize(nodes.size() + count);
        return first_idx;
    }

    bool split(std::size_t node_idx)
    {
        TreePoint center = node_center(node(node_idx));

        std::array<std::vector<std::size_t>, Degree> assign_res;
        std::array<std::size_t, Dimension> lower{}, higher{};
        assign_boxes(node(node_idx), center, assign_res, lower, higher);

        std::size_t total = node(node_idx).total_size;
        std::array<bool, Dimension> partitionable;
        for (std::size_t i = 0; i < Dimension; ++i) {
            partitionable[i] =
              (static_cast<NT>(lower[i] + higher[i] - total) / static_cast<NT>(total) < adaptive_threshold) &&
              (lower[i] > 0 && higher[i] > 0);
        }

        if (std::find(partitionable.begin(), partitionable.end(), true) == partitionable.end()) {
            return false;
        }

        std::array<BboxT, Degree> child_boxes;
        calc_box_for_children(node(node_idx), center, child_boxes);

        bool need_moving = std::find(partitionable.begin(), partitionable.end(), false) != partitionable.end();
        if (need_moving) {
            std::array<std::size_t, Degree> collapse_dest;
            calc_collapse_destination(partitionable, collapse_dest);

            for (std::size_t i = 0; i < Degree; ++i) {
                std::size_t dest = collapse_dest[i];
                if (dest == i)
                    continue;
                auto& dst = assign_res[dest];
                auto& src = assign_res[i];
                std::vector<std::size_t> tmp = std::move(dst);
                dst.reserve(tmp.size() + src.size());
                merge_unique(tmp.begin(),
                             tmp.end(),
                             src.begin(),
                             src.end(),
                             std::back_inserter(dst),
                             std::less<std::size_t>(),
                             std::equal_to<std::size_t>());
                src.clear();
                src.shrink_to_fit();
                child_boxes[dest] += child_boxes[i];
            }

            unsigned int n_ch = 1u << static_cast<unsigned int>(
                                  std::count_if(partitionable.begin(), partitionable.end(), [](bool b) { return b; }));

            std::size_t children_idx = new_children(n_ch);
            node(node_idx).children_start = children_idx;
            node(node_idx).n_children = n_ch;
            node(node_idx).child_map = collapse_dest;

            std::array<std::size_t, Degree> sorted_dest = collapse_dest;
            std::sort(sorted_dest.begin(), sorted_dest.end());
            auto end_it = std::unique(sorted_dest.begin(), sorted_dest.end());
            (void)end_it;
            for (std::size_t i = 0; i < n_ch; ++i) {
                Node& ch = node(node(node_idx).child(i));
                ch.depth = node(node_idx).depth + 1;
                ch.parent = node_idx;
                ch.loose_box = child_boxes[sorted_dest[i]];
                ch.box_indices = assign_res[sorted_dest[i]];
                ch.total_size = assign_res[sorted_dest[i]].size();
                for (std::size_t& dest : node(node_idx).child_map)
                    if (dest == sorted_dest[i])
                        dest = i;
            }
        } else {
            std::size_t children_idx = new_children(Degree);
            node(node_idx).children_start = children_idx;
            node(node_idx).n_children = Degree;
            std::iota(node(node_idx).child_map.begin(), node(node_idx).child_map.end(), std::size_t(0));
            for (std::size_t i = 0; i < Degree; ++i) {
                Node& ch = node(node(node_idx).child(i));
                ch.depth = node(node_idx).depth + 1;
                ch.parent = node_idx;
                ch.loose_box = child_boxes[i];
                ch.box_indices = assign_res[i];
                ch.total_size = assign_res[i].size();
            }
        }

        calc_tight_box_for_children(node(node_idx));

        if constexpr (!StoreBoxesInInternalNodes) {
            node(node_idx).box_indices = std::vector<std::size_t>();
        }

        return true;
    }

    void collapse(std::size_t node_idx)
    {
        assert(node(node_idx).is_internal());
        Node& nd = node(node_idx);
        std::size_t ch = nd.children_start;

        if constexpr (!StoreBoxesInInternalNodes) {
            auto& box_idx = nd.box_indices;
            box_idx.clear();
            box_idx.reserve(nd.total_size);
            for (std::size_t i = 0; i < nd.n_children; ++i) {
                auto& ch_boxes = node(ch + i).box_indices;
                box_idx.insert(box_idx.end(), ch_boxes.begin(), ch_boxes.end());
                ch_boxes = std::vector<std::size_t>();
            }
        }

        nd.children_start = kOrthtreeInvalidIndex;
    }

    void assign_boxes(const Node& nd,
                      const TreePoint& center,
                      std::array<std::vector<std::size_t>, Degree>& assign_res,
                      std::array<std::size_t, Dimension>& lower,
                      std::array<std::size_t, Dimension>& higher)
    {
        for (auto& v : assign_res) {
            v.clear();
            v.reserve(nd.total_size / Degree * 2);
        }
        std::fill(lower.begin(), lower.end(), std::size_t(0));
        std::fill(higher.begin(), higher.end(), std::size_t(0));

        for (std::size_t box_idx : nd.box_indices) {
            auto [lo, hi] = compare_box_with_center(boxes[box_idx], center);

            for (std::size_t i = 0; i < Dimension; ++i) {
                lower[i] += lo[i];
                higher[i] += hi[i];
            }

            std::size_t rf = lo.to_ulong();
            std::size_t rs = hi.to_ulong();
            for (std::size_t i = 0; i < Degree; ++i) {
                if ((((i ^ rf) | (~(i ^ rs))) & (Degree - 1)) == Degree - 1) {
                    assign_res[i].push_back(box_idx);
                }
            }
        }
    }

    [[nodiscard]] static std::pair<std::bitset<Dimension>, std::bitset<Dimension>> compare_box_with_center(
      const TreeBboxT& box,
      const TreePoint& center)
    {
        std::pair<std::bitset<Dimension>, std::bitset<Dimension>> res;
        for (std::size_t i = 0; i < Dimension; ++i) {
            res.first[i] = box.min_coord(i) < center[i];
            res.second[i] = box.max_coord(i) >= center[i];
        }
        return res;
    }

    void calc_collapse_destination(const std::array<bool, Dimension>& partitionable,
                                   std::array<std::size_t, Degree>& destination)
    {
        std::iota(destination.begin(), destination.end(), std::size_t(0));
        for (int i = static_cast<int>(Dimension) - 1; i >= 0; --i) {
            if (partitionable[i])
                continue;
            std::size_t offset = std::size_t(1) << std::size_t(i);
            for (std::size_t j = 0; j < Degree; ++j) {
                if (destination[j] & offset)
                    destination[j] -= offset;
            }
        }
    }

    void calc_box_for_children(const Node& nd, const TreePoint& center, std::array<BboxT, Degree>& child_boxes)
    {
        const auto& minb = nd.loose_box.min_bound();
        const auto& maxb = nd.loose_box.max_bound();

        if constexpr (Dimension == 2) {
            child_boxes[0] = BboxT(minb, center);
            child_boxes[1] = BboxT(TreePoint(center[0], minb[1]), TreePoint(maxb[0], center[1]));
            child_boxes[2] = BboxT(TreePoint(minb[0], center[1]), TreePoint(center[0], maxb[1]));
            child_boxes[3] = BboxT(center, maxb);
        } else if constexpr (Dimension == 3) {
            child_boxes[0] = BboxT(minb, center);
            child_boxes[1] = BboxT(TreePoint(center[0], minb[1], minb[2]), TreePoint(maxb[0], center[1], center[2]));
            child_boxes[2] = BboxT(TreePoint(minb[0], center[1], minb[2]), TreePoint(center[0], maxb[1], center[2]));
            child_boxes[3] = BboxT(TreePoint(center[0], center[1], minb[2]), TreePoint(maxb[0], maxb[1], center[2]));
            child_boxes[4] = BboxT(TreePoint(minb[0], minb[1], center[2]), TreePoint(center[0], center[1], maxb[2]));
            child_boxes[5] = BboxT(TreePoint(center[0], minb[1], center[2]), TreePoint(maxb[0], center[1], maxb[2]));
            child_boxes[6] = BboxT(TreePoint(minb[0], center[1], center[2]), TreePoint(center[0], maxb[1], maxb[2]));
            child_boxes[7] = BboxT(center, maxb);
        } else {
            for (std::size_t i = 0; i < Degree; ++i) {
                TreePoint child_min, child_max;
                for (std::size_t d = 0; d < Dimension; ++d) {
                    if (i & (std::size_t(1) << d)) {
                        child_min[d] = center[d];
                        child_max[d] = maxb[d];
                    } else {
                        child_min[d] = minb[d];
                        child_max[d] = center[d];
                    }
                }
                child_boxes[i] = BboxT(child_min, child_max);
            }
        }
    }

    void calc_tight_box_for_children(Node& nd)
    {
        for (std::size_t chi = 0; chi < nd.n_children; ++chi) {
            Node& ch = node(nd.child(chi));
            ch.tight_box = ch.loose_box;
        }
    }

    template<typename TraversalTrait>
    bool traversal_node(const Node& nd, TraversalTrait& traits) const
    {
        bool go_next = true;
        if (traits.do_inter(nd.tight_box)) {
            if (nd.is_internal()) {
                for (std::size_t i = 0; i < nd.n_children; ++i) {
                    if (go_next) {
                        go_next = traversal_node(node(nd.child(i)), traits);
                    } else {
                        break;
                    }
                }
            } else {
                for (std::size_t box_idx : nd.box_indices) {
                    if (go_next) {
                        go_next = traits.intersection(boxes[box_idx]);
                    } else {
                        break;
                    }
                }
            }
        }
        return go_next;
    }

    template<typename Iter, typename OutIter, typename LessPred, typename EqualPred>
    static void merge_unique(Iter b1, Iter e1, Iter b2, Iter e2, OutIter o, LessPred lp, EqualPred ep)
    {
        using ValueT = std::remove_cvref_t<decltype(*b1)>;
        ValueT last{};
        if (b1 == e1 || b2 == e2) {
            if (b1 != e1) {
                *o = *b1;
                last = *b1;
                ++o;
                ++b1;
            } else if (b2 != e2) {
                *o = *b2;
                last = *b2;
                ++o;
                ++b2;
            } else {
                return;
            }
        } else {
            if (lp(*b1, *b2)) {
                *o = *b1;
                last = *b1;
                ++o;
                ++b1;
            } else if (lp(*b2, *b1)) {
                *o = *b2;
                last = *b2;
                ++o;
                ++b2;
            } else {
                *o = *b1;
                last = *b1;
                ++o;
                ++b1;
                ++b2;
            }
        }
        while (b1 != e1 && b2 != e2) {
            if (lp(*b1, *b2)) {
                if (!ep(*b1, last)) {
                    *o = *b1;
                    last = *b1;
                    ++o;
                }
                ++b1;
            } else if (lp(*b2, *b1)) {
                if (!ep(*b2, last)) {
                    *o = *b2;
                    last = *b2;
                    ++o;
                }
                ++b2;
            } else {
                if (!ep(*b1, last)) {
                    *o = *b1;
                    last = *b1;
                    ++o;
                }
                ++b1;
                ++b2;
            }
        }
        while (b1 != e1) {
            *o = *b1;
            ++o;
            ++b1;
        }
        while (b2 != e2) {
            *o = *b2;
            ++o;
            ++b2;
        }
    }

    template<typename Iter>
    static BboxT calc_bbox_from_boxes(Iter begin, Iter end)
    {
        BboxT result = begin->bbox();
        for (auto it = std::next(begin); it != end; ++it) {
            result += it->bbox();
        }
        return result;
    }

  public:
    std::vector<Node> nodes;
    std::vector<TreeBboxT> boxes;
    BboxT bbox;

    SplitPred split_pred;
    ShapeRefinePred shape_refine_pred;
    DoIntersect do_intersect;
    CalcBbox calc_bbox;

    NT enlarge_ratio = NT(1.5);
    NT adaptive_threshold = NT(0.1);
};

template<typename Traits>
class BoxIntersectionTraversal
{
  public:
    using DeducedTraits = OrthtreeTraits<Traits>;
    using NT = typename DeducedTraits::NT;
    using BboxT = typename DeducedTraits::BboxT;
    using TreeBboxT = typename DeducedTraits::TreeBboxT;
    using CalcBbox = typename DeducedTraits::CalcBbox;
    using DoIntersect = typename DeducedTraits::DoIntersect;

    template<typename QPrimT>
    explicit BoxIntersectionTraversal(const QPrimT& query)
      : box_of_query(CalcBbox()(query))
    {
    }

    bool intersection(const TreeBboxT& leaf_bbox)
    {
        if (DoIntersect()(box_of_query, leaf_bbox.bbox())) {
            intersected_ids.push_back(leaf_bbox.attr());
        }
        return true;
    }

    [[nodiscard]] bool do_inter(const BboxT& b) const { return DoIntersect()(b, box_of_query); }

    [[nodiscard]] const std::vector<std::size_t>& result() const { return intersected_ids; }

  private:
    BboxT box_of_query;
    std::vector<std::size_t> intersected_ids;
};

} // namespace gpf
