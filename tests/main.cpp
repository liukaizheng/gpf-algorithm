#include <cstdlib>
#include <iostream>

void
test_encoded_array_iteration();
void
test_exp_map();
// void
// test_exp_map_planar_projection_is_isometric();
// void
// test_exp_map_existing_center_radius_and_components();
// void
// test_exp_map_cylinder_matches_unrolled_patch();
void
test_triangulate_bug1();
void
test_triangulate_bug2();
void
test_triangulate_points_simple();
void
test_triangulate_points_square();
void
test_triangulate_points_pentagon();
void
test_triangulate_10000_random_points();
void
test_cdt_with_intersections();
void
test_property_edge_length_updates();
void
test_project_polylines_on_mesh_2d_points();
void
test_project_polylines_on_mesh_2d_success();
void
test_project_polylines_on_mesh_3d_success();
void
test_project_polylines_on_mesh_repeated_endpoints();
void
test_project_polylines_on_mesh_disconnected();
void
test_project_polylines_on_mesh_crossing_constraints();
void
test_project_polylines_on_mesh_initial_triangulation_failure();
void
test_resolve_polyline_path();
void
test_triangulate_on_face_noop();
void
test_triangulate_on_face_success();
void
test_triangulate_on_face_invalid_index_boundary();
void
test_triangulate_on_face_multiple_intersections();
void
test_prepare_projected_points_with_mbvh();
void
test_walk_on_mesh_surface();
void
test_mesh_edge_collapse1();
void
test_orthtree_quadtree();
void
test_orthtree_octree();
void
test_orthtree_traversal();
void
test_mesh_flood_fill_surround_single_face();
void
test_mesh_flood_fill_surround_two_faces();
void
test_collapse_short_edges_2d_skips_flip();
void
test_collapse_short_edges_2d_collapses();
void
test_collapse_short_edges_2d_swaps_direction();
void
test_collapse_1000_points();
void
test_collapse_on_triangle();
void
test_build_bvh();
void
test_degenerate();
int
main()
{
    test_encoded_array_iteration();
    test_project_polylines_on_mesh_2d_success();
    test_project_polylines_on_mesh_3d_success();
    test_project_polylines_on_mesh_repeated_endpoints();
    test_project_polylines_on_mesh_disconnected();
    test_project_polylines_on_mesh_crossing_constraints();
    test_project_polylines_on_mesh_initial_triangulation_failure();
    test_resolve_polyline_path();
    test_triangulate_on_face_noop();
    test_triangulate_on_face_success();
    test_triangulate_on_face_invalid_index_boundary();
    test_triangulate_on_face_multiple_intersections();
    test_project_polylines_on_mesh_2d_points();
    test_prepare_projected_points_with_mbvh();
    test_walk_on_mesh_surface();
    test_exp_map();
    // test_exp_map_planar_projection_is_isometric();
    // test_exp_map_existing_center_radius_and_components();
    // test_exp_map_cylinder_matches_unrolled_patch();
    // test_build_bvh();
    test_cdt_with_intersections();
    return 0;
    test_degenerate();
    test_collapse_on_triangle();
    test_collapse_1000_points();
    test_collapse_short_edges_2d_skips_flip();
    test_collapse_short_edges_2d_collapses();
    test_collapse_short_edges_2d_swaps_direction();
    test_mesh_edge_collapse1();
    test_triangulate_bug2();
    test_triangulate_bug1();
    test_triangulate_10000_random_points();
    test_triangulate_points_simple();
    test_triangulate_points_square();
    test_triangulate_points_pentagon();
    test_property_edge_length_updates();
    test_orthtree_quadtree();
    test_orthtree_octree();
    test_orthtree_traversal();
    test_mesh_flood_fill_surround_single_face();
    test_mesh_flood_fill_surround_two_faces();

    std::cout << "gpf_algorithm_tests: OK\n";
    return EXIT_SUCCESS;
}
