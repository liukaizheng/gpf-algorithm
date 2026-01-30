#include <cstdlib>
#include <iostream>

void
test_triangulation_construction();
void
test_triangulate_points_simple();
void
test_triangulate_points_square();
void
test_triangulate_points_pentagon();
void
test_alternate_axes();
void
test_unique_indices();
void
test_triangulate_10000_random_points();

int
main()
{
    test_triangulation_construction();
    test_triangulate_points_simple();
    test_triangulate_points_square();
    test_triangulate_points_pentagon();
    test_alternate_axes();
    test_unique_indices();
    test_triangulate_10000_random_points();

    std::cout << "gpf_triangulation_tests: OK\n";
    return EXIT_SUCCESS;
}
