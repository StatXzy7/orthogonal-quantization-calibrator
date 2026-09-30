set(CALIB_TESTS
    test_rotation
    test_quantize
    test_pack
    test_io
    test_validation
    test_end_to_end
)

foreach(test_name ${CALIB_TESTS})
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE calibrator)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Meta-test: every real test executable must register a positive number of
# TEST_CASE entries and pass. A zero-test build is treated as failure.
string(REPLACE ";" "|" TEST_BINARIES_FLAT "${CALIB_TESTS}")
add_test(
    NAME meta_no_zero_tests
    COMMAND ${CMAKE_COMMAND}
        -DTEST_BINARIES=${TEST_BINARIES_FLAT}
        -DWORKDIR=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/meta_check.cmake
)
