cmake_minimum_required(VERSION 3.16)
project(orthogonal_quantization_calibrator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

option(CALIB_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

add_library(calibrator
    src/calibration.cpp
    src/quantize.cpp
    src/pack.cpp
    src/io.cpp
)
target_include_directories(calibrator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(calibrator PUBLIC Eigen3::Eigen)
target_compile_options(calibrator PRIVATE -Wall -Wextra -Wpedantic)
if(CALIB_WARNINGS_AS_ERRORS)
    target_compile_options(calibrator PRIVATE -Werror)
endif()

enable_testing()
add_subdirectory(tests)

add_executable(calibrator_demo apps/demo.cpp)
target_link_libraries(calibrator_demo PRIVATE calibrator)
target_compile_options(calibrator_demo PRIVATE -Wall -Wextra -Wpedantic)
