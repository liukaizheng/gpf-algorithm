#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

namespace gpf {

template<std::size_t N>
struct BBox
{
    std::array<double, N> lo{};
    std::array<double, N> hi{};

    BBox() = default;

    BBox(const std::array<double, N>& lo, const std::array<double, N>& hi)
      : lo(lo)
      , hi(hi)
    {
    }

    std::array<double, N>& min_bound() { return lo; }
    const std::array<double, N>& min_bound() const { return lo; }
    std::array<double, N>& max_bound() { return hi; }
    const std::array<double, N>& max_bound() const { return hi; }

    double min_coord(std::size_t i) const { return lo[i]; }
    double max_coord(std::size_t i) const { return hi[i]; }

    std::size_t longest_axis() const
    {
        std::size_t best = 0;
        double best_len = hi[0] - lo[0];
        for (std::size_t i = 1; i < N; ++i) {
            double len = hi[i] - lo[i];
            if (len > best_len) {
                best = i;
                best_len = len;
            }
        }
        return best;
    }

    BBox& operator+=(const BBox& o)
    {
        for (std::size_t i = 0; i < N; ++i) {
            lo[i] = std::min(lo[i], o.lo[i]);
            hi[i] = std::max(hi[i], o.hi[i]);
        }
        return *this;
    }

    [[nodiscard]] bool intersects(const BBox& o) const noexcept
    {
        for (std::size_t i = 0; i < N; ++i) {
            if (lo[i] > o.hi[i] || hi[i] < o.lo[i])
                return false;
        }
        return true;
    }
};

using BBox2 = BBox<2>;
using BBox3 = BBox<3>;

} // namespace gpf
