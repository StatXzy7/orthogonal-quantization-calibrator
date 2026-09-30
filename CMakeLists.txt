cmake_minimum_required(VERSION 3.16)
project(orthogonal_quant_calibrator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Eigen3 3.3 QUIET NO_MODULE)

if(TARGET Eigen3::Eigen)
    set(EIGEN_TARGET Eigen3::Eigen)
else()
    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Dense
        PATH_SUFFIXES eigen3
        PATHS /usr/include /usr/local/include)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 headers not found (only Eigen is required)")
    endif()
    message(STATUS "Using Eigen headers at: ${EIGEN3_INCLUDE_DIR}")
    add_library(eigen_headers INTERFACE)
    target_include_directories(eigen_headers INTERFACE ${EIGEN3_INCLUDE_DIR})
    set(EIGEN_TARGET eigen_headers)
endif()

add_library(oqc STATIC
    src/calibrator.cpp
    src/quantize.cpp
    src/packing.cpp
    src/archive.cpp
)
target_include_directories(oqc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(oqc PUBLIC ${EIGEN_TARGET})

enable_testing()
add_subdirectory(tests)
