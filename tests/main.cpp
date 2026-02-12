#include <cstdlib>
#include <iostream>

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
void test_cdt_with_intersections();

int
main()
{
    test_triangulate_bug2();
    test_triangulate_bug1();
    test_cdt_with_intersections();
    test_triangulate_10000_random_points();
    test_triangulate_points_simple();
    test_triangulate_points_square();
    test_triangulate_points_pentagon();

    std::cout << "gpf_triangulation_tests: OK\n";
    return EXIT_SUCCESS;
}
