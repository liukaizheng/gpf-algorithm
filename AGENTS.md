# AGENTS.md — gpf-algorithm

## Build

```bash
# Configure (first time or after CMakeLists.txt changes)
cmake -B build

# Build everything
cmake --build build

# Build only tests
cmake --build build --target gpf_algorithm_tests

# Run tests (must run from build/ directory — some tests load data files)
cd build && ./gpf_algorithm_tests

# Run cmake with debug
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

There is no single-test runner. All tests are compiled into one binary.
Test functions are called sequentially from `tests/main.cpp`.
To isolate a test, comment out other calls in `main()` temporarily.

## Project Structure

Header-only C++ library (`INTERFACE` target). No `.cpp` files in `include/`.

```
include/gpf/       — Public headers (the library)
tests/             — Test files, each test_*.cpp has free functions called from main.cpp
cmake/             — CPM.cmake dependency manager
```

## Language & Standard

- **C++23** (`CMAKE_CXX_STANDARD 23`)
- Compiler: system C++ compiler (Apple Clang on macOS)
- Build system: CMake 3.24+, Makefiles

## Dependencies

- **Eigen 5.0.1** — Linear algebra (vectors, maps)
- **CGAL** — Computational geometry (found via system `find_package`)
- **gpf::mesh** — Mesh data structure (CPM from `gh:liukaizheng/mesh`)
- **predicates** — Geometric predicates (CPM from `gh:liukaizheng/predicates`)

## Code Style

Enforced by `.clang-format` (Mozilla-derived). Key rules:

### Formatting
- **Indent**: 4 spaces, no tabs
- **Column limit**: 120
- **Braces**: Mozilla style (Allman for functions/classes/structs, K&R for control flow)
- **Return type**: on its own line for top-level function definitions
- **Constructor initializers**: break before comma, 2-space indent
- **Continuation indent**: 2 spaces
- **Namespace**: no indentation inside namespaces
- **Pointer/Reference**: left-aligned (`int* p`, `int& r`)

### Naming Conventions
- **Types/Structs/Classes**: `PascalCase` (`OrthtreeNode`, `BBox`, `TreeBboxT`)
- **Template params**: `PascalCase` (`SplitPredT`, `DoIntersectT`)
- **Functions/Methods**: `snake_case` (`min_bound`, `longest_axis`, `calc_bbox_from_boxes`)
- **Variables/Members**: `snake_case` (`bbox_center`, `side_length`, `enlarge_ratio`)
- **Constants**: `k` prefix + `PascalCase` (`kOrthtreeInvalidIndex`, `kRootIdx`)
- **Type aliases**: `PascalCase` with `T` suffix when disambiguating (`BboxT`, `NodeAttrT`, `PrimAttrT`)
- **Concept names**: `PascalCase` (`HasMaxDepth`, `HasPositionProperty`)

### Includes
- `#pragma once` for header guards
- Groups: std headers, then third-party (`<Eigen/...>`, `<CGAL/...>`), then project (`<gpf/...>`)
- Group ordering preserved (not re-sorted across groups)

### Point/Vector Arithmetic
- Points and vectors are `std::array<double, N>` — never custom Point classes
- All arithmetic delegated to Eigen via `Eigen::Map`:
```cpp
using EigenVec = Eigen::Vector<double, static_cast<int>(Dimension)>;
// a = b + c
EigenVec::Map(a.data()) = EigenVec::Map(b.data()) + EigenVec::Map(c.data());
// a *= scalar
EigenVec::Map(a.data()) *= s;
```
- Brace-init for constructing `std::array` values: `TreePoint{x, y, z}` not `TreePoint(x, y, z)`

### Type Patterns
- `[[nodiscard]]` on all query methods
- `noexcept` on trivial predicates
- `static constexpr` for compile-time constants
- `std::conditional_t` for compile-time type selection
- `if constexpr` for dimension-specific optimizations
- Direct template parameters with defaults over traits structs:
```cpp
gpf::Orthtree<Dim, SplitPredT, DoIntersectT, CalcBboxT, PrimAttrT, MaxDepth, ...>
```

### Error Handling
- `assert()` for invariant checks (preconditions, array bounds)
- No exceptions in library code
- `std::runtime_error` only in test I/O helpers

### Templates
- Template declaration on its own line, separate from function/class
- Prefer concepts over SFINAE
- Deduced types via `decltype` + `std::remove_cvref_t`

### Comments
- Minimal — code should be self-documenting
- No function-level doc comments
- End-of-namespace comments: `} // namespace gpf`
- Only add comments for genuinely non-obvious logic

### Testing
- Plain `assert()` — no test framework
- Free functions named `test_<module>_<scenario>()`
- Declared in `tests/main.cpp`, defined in `tests/test_*.cpp`
- Test data types (traits, functors) defined in anonymous namespace within test files

## Common Patterns

### Orthtree Usage
Direct template parameters — no traits struct needed:
```cpp
gpf::Orthtree<3, MySplitPred, MyDoIntersect> tree;
// Defaults: CalcBbox=IdentityCalcBbox, PrimAttrT=std::size_t, MaxDepth=32
// BoxIntersectionTraversal is a nested class:
decltype(tree)::BoxIntersectionTraversal trav(query_bbox);
tree.traversal(trav);
```

### Mesh Property Access
Vertex/edge properties via `.prop()`, positions via `.prop().pt` (a `std::array<double, 3>`).
