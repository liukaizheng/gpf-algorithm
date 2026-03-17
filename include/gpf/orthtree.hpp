#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <deque>
#include <numeric>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include <gpf/bbox.hpp>

namespace gpf {

constexpr std::size_t kOrthtreeInvalidIndex = static_cast<std::size_t>(-1);

inline bool
is_valid_index(std::size_t idx) noexcept
{
    return idx != kOrthtreeInvalidIndex;
}

struct OrthtreeConfig
{
    double enlarge_ratio = 1.01;
    double adaptive_threshold = 0.2;
    std::size_t max_depth = 16;
    std::size_t max_leaf_size = 100;
};

template<std::size_t Dim>
class OrthtreeNodeBase
{
  public:
    static constexpr std::size_t Dimension = Dim;
    static constexpr std::size_t Degree = (1u << Dim);

    using BboxT = BBox<Dim>;

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

template<std::size_t Dim>
class Orthtree
{
  public:
    static constexpr std::size_t Dimension = Dim;
    static constexpr std::size_t Degree = (1u << Dim);

    using BboxT = BBox<Dim>;
    using TreePoint = std::array<double, Dim>;
    using EigenVec = Eigen::Vector<double, static_cast<int>(Dim)>;

    using Node = OrthtreeNodeBase<Dim>;

    struct TreeBboxT : public BboxT
    {
        [[nodiscard]] const BboxT& bbox() const { return *static_cast<const BboxT*>(this); }
        [[nodiscard]] BboxT& bbox() { return *static_cast<BboxT*>(this); }

        [[nodiscard]] std::size_t attr() const { return attr_; }
        [[nodiscard]] std::size_t& attr() { return attr_; }

      private:
        std::size_t attr_;
    };

    class BoxIntersectionTraversal
    {
      public:
        template<typename QPrimT>
        explicit BoxIntersectionTraversal(const QPrimT& query)
          : box_of_query(query)
        {
        }

        bool intersection(const TreeBboxT& leaf_bbox)
        {
            if (box_of_query.intersects(leaf_bbox.bbox())) {
                intersected_ids.push_back(leaf_bbox.attr());
            }
            return true;
        }

        [[nodiscard]] bool do_inter(const BboxT& b) const { return b.intersects(box_of_query); }

        [[nodiscard]] const std::vector<std::size_t>& result() const { return intersected_ids; }

      private:
        BboxT box_of_query;
        std::vector<std::size_t> intersected_ids;
    };

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
        config = rhs.config;
    }

    template<typename Bboxes, typename Attributes>
    void insert_boxes(const Bboxes& bboxes, const Attributes& attributes)
    {
        assert(bboxes.size() == attributes.size());
        clear();
        boxes.resize(bboxes.size());
        for (std::size_t i = 0; i < bboxes.size(); ++i) {
            boxes[i].bbox() = bboxes[i];
            boxes[i].attr() = attributes[i];
        }
    }

    void construct(bool compact_box = false)
    {
        bbox = calc_bbox_from_boxes(boxes.begin(), boxes.end());

        nodes.clear();
        nodes.emplace_back();
        root_node().tight_box = bbox;

        TreePoint bbox_center;
        EigenVec::Map(bbox_center.data()) =
          (EigenVec::Map(bbox.min_bound().data()) + EigenVec::Map(bbox.max_bound().data())) * 0.5;
        TreePoint side_length;
        EigenVec::Map(side_length.data()) =
          EigenVec::Map(bbox.max_bound().data()) - EigenVec::Map(bbox.min_bound().data());
        if (!compact_box) {
            std::size_t longest = bbox.longest_axis();
            double longest_val = side_length[longest];
            side_length.fill(longest_val);
        }
        EigenVec::Map(side_length.data()) *= config.enlarge_ratio;

        EigenVec::Map(bbox.min_bound().data()) =
          EigenVec::Map(bbox_center.data()) - EigenVec::Map(side_length.data()) * 0.5;
        EigenVec::Map(bbox.max_bound().data()) =
          EigenVec::Map(bbox_center.data()) + EigenVec::Map(side_length.data()) * 0.5;

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
            if (cur.depth < config.max_depth && cur.total_size > config.max_leaf_size) {
                if (split(cur_idx)) {
                    for (std::size_t i = 0; i < node(cur_idx).n_children; ++i) {
                        nodes_to_split.push_back(node(cur_idx).child(i));
                    }
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
        TreePoint result;
        EigenVec::Map(result.data()) =
          (EigenVec::Map(nd.tight_box.min_bound().data()) + EigenVec::Map(nd.tight_box.max_bound().data())) * 0.5;
        return result;
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
            partitionable[i] = (static_cast<double>(lower[i] + higher[i] - total) / static_cast<double>(total) <
                                config.adaptive_threshold) &&
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
                std::vector<std::size_t> tmp;
                tmp.swap(dst);
                dst.reserve(tmp.size() + src.size());
                std::ranges::set_union(tmp, src, std::back_inserter(dst));
                src.clear();
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

        node(node_idx).box_indices = std::vector<std::size_t>();

        return true;
    }

    void collapse(std::size_t node_idx)
    {
        assert(node(node_idx).is_internal());
        Node& nd = node(node_idx);
        std::size_t ch = nd.children_start;

        auto& box_idx = nd.box_indices;
        box_idx.clear();
        box_idx.reserve(nd.total_size);
        for (std::size_t i = 0; i < nd.n_children; ++i) {
            auto& ch_boxes = node(ch + i).box_indices;
            box_idx.insert(box_idx.end(), ch_boxes.begin(), ch_boxes.end());
            ch_boxes = std::vector<std::size_t>();
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
            child_boxes[1] = BboxT(TreePoint{ center[0], minb[1] }, TreePoint{ maxb[0], center[1] });
            child_boxes[2] = BboxT(TreePoint{ minb[0], center[1] }, TreePoint{ center[0], maxb[1] });
            child_boxes[3] = BboxT(center, maxb);
        } else if constexpr (Dimension == 3) {
            child_boxes[0] = BboxT(minb, center);
            child_boxes[1] =
              BboxT(TreePoint{ center[0], minb[1], minb[2] }, TreePoint{ maxb[0], center[1], center[2] });
            child_boxes[2] =
              BboxT(TreePoint{ minb[0], center[1], minb[2] }, TreePoint{ center[0], maxb[1], center[2] });
            child_boxes[3] =
              BboxT(TreePoint{ center[0], center[1], minb[2] }, TreePoint{ maxb[0], maxb[1], center[2] });
            child_boxes[4] =
              BboxT(TreePoint{ minb[0], minb[1], center[2] }, TreePoint{ center[0], center[1], maxb[2] });
            child_boxes[5] =
              BboxT(TreePoint{ center[0], minb[1], center[2] }, TreePoint{ maxb[0], center[1], maxb[2] });
            child_boxes[6] =
              BboxT(TreePoint{ minb[0], center[1], center[2] }, TreePoint{ center[0], maxb[1], maxb[2] });
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

    OrthtreeConfig config;
};

} // namespace gpf
