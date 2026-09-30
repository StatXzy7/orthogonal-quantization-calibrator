add_executable(oqc_tests test_calibrator.cpp)
target_link_libraries(oqc_tests PRIVATE oqc)
target_include_directories(oqc_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/test_calibrator.cpp" _test_source)
string(REGEX MATCHALL "TEST_CASE\\([A-Za-z0-9_]+" _matches "${_test_source}")

set(_case_names "")
foreach(_match IN LISTS _matches)
    string(REGEX REPLACE "TEST_CASE\\(" "" _name "${_match}")
    list(APPEND _case_names "${_name}")
    add_test(NAME "oqc.${_name}" COMMAND oqc_tests "${_name}")
endforeach()

add_test(NAME oqc.all COMMAND oqc_tests)
message(STATUS "Registered ${_case_names} test case(s) with CTest")
