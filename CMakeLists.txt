cmake_minimum_required(VERSION 3.24)

project(multi-label-partition VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CPM.cmake)

# CPMAddPackage("gh:liukaizheng/mesh#main")
CPMAddPackage(
    NAME mesh
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mesh
)
# CPMAddPackage(
#     NAME predicates
#     SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../predicates
# )
CPMAddPackage("gh:liukaizheng/predicates#main")

add_library(triangulation INTERFACE)
add_library(gpf::triangulation ALIAS triangulation)

target_compile_features(triangulation INTERFACE cxx_std_23)
target_link_libraries(triangulation INTERFACE
    gpf::mesh
    predicates
)
target_include_directories(triangulation INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(GPF_TRIANGULATION_IS_TOP_LEVEL OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(GPF_TRIANGULATION_IS_TOP_LEVEL ON)
endif()

option(GPF_TRIANGULATION_BUILD_TESTS "Build triangulation tests" ${GPF_TRIANGULATION_IS_TOP_LEVEL})

if(GPF_TRIANGULATION_BUILD_TESTS)
  enable_testing()
  add_executable(
    gpf_triangulation_tests
    tests/main.cpp
    tests/test_triangulation.cpp
  )
  target_link_libraries(gpf_triangulation_tests PRIVATE gpf::triangulation)
  add_test(NAME gpf_triangulation_tests COMMAND gpf_triangulation_tests)
endif()
