#include <cassert>
#include <vector>

#include "gpf/triangulation.hpp"

void test_triangulation_construction() {
  // Test: Create a Triangulation with some 2D points
  std::vector<double> points = {
      0.0, 0.0,  // point 0
      1.0, 0.0,  // point 1
      0.5, 1.0,  // point 2
  };

  gpf::triangulation::Triangulation tri(points);

  // Verify points span is correctly assigned
  assert(tri.points.size() == 6);
  assert(tri.points.data() == points.data());

  // Verify mesh is initially empty
  assert(tri.mesh.n_vertices() == 0);
  assert(tri.mesh.n_faces() == 0);
  assert(tri.mesh.n_edges() == 0);

  // Verify sorted_vertices is initially empty
  assert(tri.sorted_vertices.empty());
}

void test_triangulate_points_simple() {
  // Test: Triangulate 3 points (simple triangle)
  std::vector<double> points = {
      0.0, 0.0,  // point 0
      1.0, 0.0,  // point 1
      0.5, 1.0,  // point 2
  };

  auto triangles = gpf::triangulation::triangulate_points(points, true);

  // Should produce exactly 1 triangle with 3 indices
  assert(triangles.size() == 3);

  // Verify all indices are valid (0, 1, or 2)
  for (std::size_t idx : triangles) {
    assert(idx < 3);
  }
}

void test_triangulate_points_square() {
  // Test: Triangulate 4 points (square)
  std::vector<double> points = {
      0.0, 0.0,  // point 0
      1.0, 0.0,  // point 1
      1.0, 1.0,  // point 2
      0.0, 1.0,  // point 3
  };

  auto triangles = gpf::triangulation::triangulate_points(points, true);

  // Should produce 2 triangles with 6 indices
  assert(triangles.size() == 6);

  // Verify all indices are valid (0-3)
  for (std::size_t idx : triangles) {
    assert(idx < 4);
  }
}

void test_triangulate_points_pentagon() {
  // Test: Triangulate 5 points (regular pentagon-ish)
  std::vector<double> points = {
      0.5, 0.0,  // point 0
      1.0, 0.4,  // point 1
      0.8, 1.0,  // point 2
      0.2, 1.0,  // point 3
      0.0, 0.4,  // point 4
  };

  auto triangles = gpf::triangulation::triangulate_points(points, true);

  // Should produce 3 triangles with 9 indices
  assert(triangles.size() == 9);

  // Verify all indices are valid (0-4)
  for (std::size_t idx : triangles) {
    assert(idx < 5);
  }
}

void test_alternate_axes() {
  // Test: Verify alternate_axes produces a valid ordering
  std::vector<double> points = {
      0.0, 0.0,
      1.0, 0.0,
      0.5, 1.0,
      0.0, 1.0,
      1.0, 1.0,
  };

  std::vector<gpf::VertexId> indices;
  for (std::size_t i = 0; i < 5; ++i) {
    indices.emplace_back(gpf::VertexId{i});
  }

  gpf::triangulation::alternate_axes(points.data(), std::span<gpf::VertexId>(indices), true);

  // Verify all original indices are present
  std::vector<bool> found(5, false);
  for (const auto& vid : indices) {
    assert(vid.idx < 5);
    found[vid.idx] = true;
  }
  for (bool b : found) {
    assert(b);
  }
}

void test_unique_indices() {
  // Test: unique_indices deduplication
  std::vector<std::size_t> indices = {0, 1, 2, 1, 3, 0, 2};
  auto [new_indices, mapping] = gpf::triangulation::unique_indices(indices);

  // Should have 4 unique values
  assert(mapping.size() == 4);

  // Verify mapping is correct
  for (std::size_t i = 0; i < indices.size(); ++i) {
    assert(mapping[new_indices[i]] == indices[i]);
  }
}
