# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Cross-frontend KIWI single-PUT readiness harness. common/ holds the host
# code (a copy of bench_kiwi_readiness's, templated on a kernel launcher) and
# its Python bridge; cpp/, flydsl/ and triton/ are frontends that supply only
# the protocol kernel. Executables and scripts land at the mirrored build
# paths; the Python modules land in the build tree's python/.
set(_includes ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)  # readiness_bench_common.hip.hpp, bench_affinity.hpp

# C++ frontend: bench_kiwi_readiness's kernel, built like it.
add_executable(harness_kiwi_readiness_cpp cpp/harness_kiwi_readiness.hip.cpp)
set_source_files_properties(cpp/harness_kiwi_readiness.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(harness_kiwi_readiness_cpp PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpp)
target_link_libraries(harness_kiwi_readiness_cpp PRIVATE kiwi_p2p CLI11::CLI11)
target_include_directories(harness_kiwi_readiness_cpp PRIVATE ${_includes})

if(KIWI_PYTHON_FRONTENDS)
    include(KiwiPybind)
    kiwi_add_pybind_module(kiwi_readiness_bench_ext common/kiwi_readiness_bench_ext.hip.cpp
        OUTPUT_DIRECTORY "${KIWI_PYTHON_BINARY_DIR}")
    target_link_libraries(kiwi_readiness_bench_ext PRIVATE kiwi_p2p)
    target_include_directories(kiwi_readiness_bench_ext PRIVATE
        ${PROJECT_SOURCE_DIR}/benchmarks/include ${_includes})
    configure_file(common/kiwi_readiness_bench.py
        "${KIWI_PYTHON_BINARY_DIR}/kiwi_readiness_bench.py" COPYONLY)
endif()
if(KIWI_BUILD_FLYDSL)
    configure_file(flydsl/bench_kiwi_flydsl_readiness.py
        flydsl/bench_kiwi_flydsl_readiness.py COPYONLY)
endif()
if(KIWI_BUILD_TRITON)
    configure_file(triton/bench_kiwi_triton_readiness.py
        triton/bench_kiwi_triton_readiness.py COPYONLY)
endif()

# Short two-rank runs of each frontend through the harness's host code.
if(KIWI_BUILD_TESTS)
    set(_mpi ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2)
    set(_smoke --depth 4 --warmup 100 --iters 1000)
    add_test(NAME test_cpp_kiwi_readiness_bench
        COMMAND ${_mpi} $<TARGET_FILE:harness_kiwi_readiness_cpp> ${_smoke})
    set(_tests test_cpp_kiwi_readiness_bench)
    if(KIWI_BUILD_FLYDSL)
        add_test(NAME test_flydsl_kiwi_readiness_bench
            COMMAND ${_mpi} ${KIWI_FLYDSL_PYTHON} flydsl/bench_kiwi_flydsl_readiness.py ${_smoke}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        list(APPEND _tests test_flydsl_kiwi_readiness_bench)
    endif()
    if(KIWI_BUILD_TRITON)
        add_test(NAME test_triton_kiwi_readiness_bench
            COMMAND ${_mpi} ${KIWI_TRITON_PYTHON} triton/bench_kiwi_triton_readiness.py ${_smoke}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        list(APPEND _tests test_triton_kiwi_readiness_bench)
    endif()
    set_tests_properties(${_tests} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${KIWI_PYTHON_BINARY_DIR}:$ENV{PYTHONPATH}"
        RUN_SERIAL TRUE TIMEOUT 300)
endif()
