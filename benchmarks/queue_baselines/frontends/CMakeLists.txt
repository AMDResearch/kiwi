# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Cross-frontend kiwi-spsc queue-baseline harness. common/ holds the host code
# (a copy of bench_queue_baselines', templated on a kernel launcher) and its
# Python bridge; cpp/, triton/, ... are frontends that supply only the
# kernels. It uses only the KIWI adapter, so it does not need the third-party
# queues (KIWI_BUILD_QUEUE_BASELINES). Executables and scripts land at the
# mirrored build paths; the Python modules land in the build tree's python/.
set(_includes ${BENCH_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/common)

# C++ frontend: bench_queue_baselines' kiwi-spsc kernels, built like it.
set(_source cpp/harness_queue_baselines.hip.cpp)
set_source_files_properties(${_source} PROPERTIES LANGUAGE HIP)
add_executable(harness_queue_baselines_cpp ${_source})
target_link_libraries(harness_queue_baselines_cpp PRIVATE kiwi_invoke CLI11::CLI11 pthread)
target_include_directories(harness_queue_baselines_cpp PRIVATE ${_includes})
set_hip_compile_options(harness_queue_baselines_cpp)
set_common_definitions(harness_queue_baselines_cpp)
set_target_properties(harness_queue_baselines_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpp)

if(KIWI_PYTHON_FRONTENDS)
    include(KiwiPybind)
    kiwi_add_pybind_module(kiwi_queue_bench_ext common/kiwi_queue_bench_ext.hip.cpp
        OUTPUT_DIRECTORY "${KIWI_PYTHON_BINARY_DIR}")
    target_include_directories(kiwi_queue_bench_ext PRIVATE ${_includes})
    configure_file(common/kiwi_queue_bench.py
        "${KIWI_PYTHON_BINARY_DIR}/kiwi_queue_bench.py" COPYONLY)
endif()
if(KIWI_BUILD_TRITON)
    configure_file(triton/bench_triton_queue_baselines.py
        triton/bench_triton_queue_baselines.py COPYONLY)
endif()

# Short runs of each frontend through the harness's host code.
if(KIWI_BUILD_TESTS)
    set(_smoke --mode latency --iters 1000 --warmup 100)
    add_test(NAME test_kiwi_cpp_spsc_baseline COMMAND harness_queue_baselines_cpp ${_smoke})
    set(_tests test_kiwi_cpp_spsc_baseline)
    if(KIWI_BUILD_TRITON)
        add_test(NAME test_kiwi_triton_spsc_baseline
            COMMAND ${KIWI_TRITON_PYTHON} triton/bench_triton_queue_baselines.py ${_smoke}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(test_kiwi_triton_spsc_baseline PROPERTIES SKIP_RETURN_CODE 77)
        list(APPEND _tests test_kiwi_triton_spsc_baseline)
    endif()
    set_tests_properties(${_tests} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${KIWI_PYTHON_BINARY_DIR}:$ENV{PYTHONPATH}"
        RUN_SERIAL TRUE TIMEOUT 120)
endif()
