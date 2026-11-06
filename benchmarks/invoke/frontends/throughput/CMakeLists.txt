# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Cross-frontend invoke callback throughput harness. common/ holds the host
# code (a copy of bench_invoke_throughput's, templated on a kernel launcher)
# and its Python bridge; cpp/, flydsl/ and triton/ are frontends that supply
# only the kernel. Executables and scripts land at the mirrored build paths; the Python
# modules land in the build tree's python/.
set(_common ${CMAKE_CURRENT_SOURCE_DIR}/common)

# C++ frontend: bench_invoke_throughput's kernel, built like it.
add_invoke_benchmark(harness_invoke_throughput_cpp cpp/harness_invoke_throughput.hip.cpp)
target_include_directories(harness_invoke_throughput_cpp PRIVATE ${_common})
set_target_properties(harness_invoke_throughput_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpp)

if(KIWI_PYTHON_FRONTENDS)
    include(KiwiPybind)
    kiwi_add_pybind_module(kiwi_invoke_throughput_bench_ext
        common/kiwi_invoke_throughput_bench_ext.hip.cpp
        OUTPUT_DIRECTORY "${KIWI_PYTHON_BINARY_DIR}")
    target_include_directories(kiwi_invoke_throughput_bench_ext PRIVATE
        ${BENCH_INCLUDE_DIR} ${_common})
    configure_file(common/kiwi_invoke_throughput_bench.py
        "${KIWI_PYTHON_BINARY_DIR}/kiwi_invoke_throughput_bench.py" COPYONLY)
endif()
if(KIWI_BUILD_FLYDSL)
    configure_file(flydsl/bench_invoke_flydsl_throughput.py
        flydsl/bench_invoke_flydsl_throughput.py COPYONLY)
endif()
if(KIWI_BUILD_TRITON)
    configure_file(triton/bench_invoke_triton_throughput.py
        triton/bench_invoke_triton_throughput.py COPYONLY)
endif()

# Short runs of each frontend through the harness's host code.
if(KIWI_BUILD_TESTS)
    set(_smoke -m 100000 -e 4 -t 2)
    add_test(NAME test_cpp_invoke_throughput_bench
        COMMAND harness_invoke_throughput_cpp ${_smoke})
    set(_tests test_cpp_invoke_throughput_bench)
    if(KIWI_BUILD_FLYDSL)
        add_test(NAME test_flydsl_invoke_throughput_bench
            COMMAND ${KIWI_FLYDSL_PYTHON} flydsl/bench_invoke_flydsl_throughput.py ${_smoke}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        list(APPEND _tests test_flydsl_invoke_throughput_bench)
    endif()
    if(KIWI_BUILD_TRITON)
        add_test(NAME test_triton_invoke_throughput_bench
            COMMAND ${KIWI_TRITON_PYTHON} triton/bench_invoke_triton_throughput.py ${_smoke}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        list(APPEND _tests test_triton_invoke_throughput_bench)
    endif()
    set_tests_properties(${_tests} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${KIWI_PYTHON_BINARY_DIR}:$ENV{PYTHONPATH}"
        RUN_SERIAL TRUE TIMEOUT 300)
endif()
