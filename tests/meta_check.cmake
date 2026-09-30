set(total 0)
string(REPLACE "|" ";" binaries "${TEST_BINARIES}")
foreach(bin ${binaries})
    execute_process(
        COMMAND ${WORKDIR}/${bin}
        WORKING_DIRECTORY ${WORKDIR}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE out
    )
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "test binary ${bin} failed:\n${out}")
    endif()
    string(REGEX MATCH "\\[==========\\] ([0-9]+) tests registered" m "${out}")
    if(NOT CMAKE_MATCH_1)
        message(FATAL_ERROR "could not read registration count from ${bin}")
    endif()
    if(CMAKE_MATCH_1 LESS 1)
        message(FATAL_ERROR "${bin} registered zero tests - vacuous test suite")
    endif()
    math(EXPR total "${total} + ${CMAKE_MATCH_1}")
endforeach()
if(total LESS 15)
    message(FATAL_ERROR "expected at least 15 registered test cases, found ${total}")
endif()
message(STATUS "meta check: ${total} non-vacuous test cases across all suites")
