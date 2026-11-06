# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Python (pybind11 + PyTorch) extension for kiwi-ep
#
# Build with: cmake -DKIWI_BUILD_PYTHON=ON ...
# The resulting kiwi_ep_ext.so can be imported from Python.

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Locate PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_result
)
if(NOT _torch_result EQUAL 0)
    message(FATAL_ERROR "PyTorch not found. Is it installed?")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# PyTorch include and library paths
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import torch; import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import torch; import torch.utils.cpp_extension; print(';'.join(torch.utils.cpp_extension.library_paths()))"
    OUTPUT_VARIABLE TORCH_LIBRARY_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Build as a shared library with HIP compilation (not pybind11_add_module,
# which doesn't know about HIP source properties).
add_library(kiwi_ep_ext MODULE kiwi_ep_bindings.hip.cpp)

set_source_files_properties(kiwi_ep_bindings.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(kiwi_ep_ext PROPERTIES
    PREFIX ""                             # No "lib" prefix
    SUFFIX ".so"                          # Standard Python extension suffix
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)

target_include_directories(kiwi_ep_ext PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

target_link_directories(kiwi_ep_ext PRIVATE ${TORCH_LIBRARY_DIRS})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_python.so'))"
    OUTPUT_VARIABLE TORCH_PYTHON_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_link_libraries(kiwi_ep_ext PRIVATE
    kiwi_ep
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    pybind11::headers
    Python3::Python
)

# Optional side-by-side tests against the official ROCm/MORI Python operator.
# The script exits 77 when PyTorch/MORI/hardware support is unavailable, and
# CTest marks those cases as skipped. Semantic mismatches in a built kiwi_ep_ext
# remain hard failures.
if(KIWI_BUILD_TESTS)
    find_package(MPI REQUIRED)
    set(_kiwi_ep_python_test_env
        "PYTHONPATH=$<TARGET_FILE_DIR:kiwi_ep_ext>:$ENV{PYTHONPATH}")

    add_test(NAME test_kiwi_ep_cudagraph_ws2
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_kiwi_ep_cudagraph.py)
    set_tests_properties(test_kiwi_ep_cudagraph_ws2 PROPERTIES
        ENVIRONMENT "${_kiwi_ep_python_test_env}"
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE
        PROCESSORS 2
        TIMEOUT 300)

    set(_official_mori_test_env
        "PYTHONPATH=$<TARGET_FILE_DIR:kiwi_ep_ext>:$ENV{PYTHONPATH}"
        "MORI_SHMEM_MODE=vmm_heap"
        "MORI_SHMEM_HEAP_SIZE=512M"
        "MORI_SHMEM_VMM_CHUNK_SIZE=64M")
    set(_official_mori_master_port 29537)

    function(add_official_mori_compat_test name ranks)
        set(_port ${_official_mori_master_port})
        math(EXPR _next_port "${_official_mori_master_port} + 1")
        set(_official_mori_master_port ${_next_port} PARENT_SCOPE)
        add_test(NAME ${name}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_official_mori_compat.py
                    ${ARGN})
        set_tests_properties(${name} PROPERTIES
            ENVIRONMENT "${_official_mori_test_env};MASTER_PORT=${_port}"
            SKIP_RETURN_CODE 77
            RUN_SERIAL TRUE
            PROCESSORS ${ranks}
            TIMEOUT 300)
    endfunction()

    # Topology matrix for the synchronous MORI API.  IntraNode covers the
    # local-only world_size=1 case and single-node EP with gpu_per_node=world_size.
    foreach(_ws IN ITEMS 1 2 4 8)
        add_official_mori_compat_test(
            test_official_mori_compat_intranode_ws${_ws} ${_ws}
            --kernel-type IntraNode
            --api-mode sync
            --gpu-per-node ${_ws}
            --dispatch-dtype float32)
    endforeach()

    # InterNodeV1 coverage keeps gpu_per_node on auto (MPI local size), so a
    # single-node run tests realistic local topology and a multi-node run uses
    # each node's actual visible rank count instead of the old hard-coded 1.
    foreach(_ws IN ITEMS 2 4 8)
        add_official_mori_compat_test(
            test_official_mori_compat_internode_v1_ws${_ws} ${_ws}
            --kernel-type InterNodeV1
            --api-mode sync
            --dispatch-dtype float32)
    endforeach()

    # Synchronous scale-bearing coverage catches the vLLM/SGLang normal-mode
    # call shape for quantized dispatch, not just the AsyncLL split path below.
    add_official_mori_compat_test(
        test_official_mori_compat_intranode_fp8_scale_sync_ws8 8
        --kernel-type IntraNode
        --api-mode sync
        --gpu-per-node 8
        --dispatch-dtype fp8
        --hidden-dim 128
        --scale-dim 1
        --scale-dtype float32
        --max-token-type-size 2
        --no-combine-output-weights)

    add_official_mori_compat_test(
        test_official_mori_compat_intranode_fp4_scale_sync_ws8 8
        --kernel-type IntraNode
        --api-mode sync
        --gpu-per-node 8
        --dispatch-dtype fp4
        --hidden-dim 128
        --scale-dim 4
        --scale-dtype e8m0
        --max-token-type-size 2
        --no-combine-output-weights)

    add_official_mori_compat_test(
        test_official_mori_compat_internode_v1_fp8_scale_sync_ws8 8
        --kernel-type InterNodeV1
        --api-mode sync
        --dispatch-dtype fp8
        --hidden-dim 128
        --scale-dim 1
        --scale-dtype float32
        --max-token-type-size 2
        --no-combine-output-weights)

    add_official_mori_compat_test(
        test_official_mori_compat_internode_v1_fp4_scale_sync_ws8 8
        --kernel-type InterNodeV1
        --api-mode sync
        --dispatch-dtype fp4
        --hidden-dim 128
        --scale-dim 4
        --scale-dtype e8m0
        --max-token-type-size 2
        --no-combine-output-weights)

    add_official_mori_compat_test(
        test_official_mori_compat_internode_v1ll_ws8 8
        --kernel-type InterNodeV1LL
        --api-mode sync
        --dispatch-dtype float32
        --max-num-inp-token-per-rank 256
        --num-tokens 128)

    add_official_mori_compat_test(
        test_official_mori_compat_internode_v1ll_fp8_scale_sync_ws8 8
        --kernel-type InterNodeV1LL
        --api-mode sync
        --dispatch-dtype fp8
        --hidden-dim 128
        --scale-dim 1
        --scale-dtype float32
        --max-token-type-size 2
        --max-num-inp-token-per-rank 256
        --num-tokens 128
        --no-combine-output-weights)

    add_official_mori_compat_test(
        test_official_mori_compat_bf16_multisource_ws4 4
        --kernel-type InterNodeV1
        --api-mode sync
        --dispatch-dtype bfloat16
        --hidden-dim 64
        --require-multi-source-bf16)

    # SGLang low-latency uses AsyncLL split dispatch/combine calls.
    add_official_mori_compat_test(
        test_official_mori_compat_asyncll_split_ws8 8
        --kernel-type AsyncLL
        --api-mode split
        --dispatch-dtype bfloat16
        --hidden-dim 64
        --no-combine-output-weights)

    # Quantized dispatch coverage: nonzero scale tensors must be returned with
    # MORI-compatible shape/dtype/values, followed by a vLLM-like combine smoke.
    add_official_mori_compat_test(
        test_official_mori_compat_fp8_scale_ws8 8
        --kernel-type AsyncLL
        --api-mode split
        --dispatch-dtype fp8
        --hidden-dim 128
        --scale-dim 1
        --scale-dtype float32
        --max-token-type-size 2
        --no-combine-output-weights)

    add_official_mori_compat_test(
        test_official_mori_compat_fp4_scale_ws8 8
        --kernel-type AsyncLL
        --api-mode split
        --dispatch-dtype fp4
        --hidden-dim 128
        --scale-dim 4
        --scale-dtype e8m0
        --max-token-type-size 2
        --no-combine-output-weights)

    # Empty-batch edges are separate from the 1..256 sweeps so current failures
    # cannot stop the sweep before it reaches the positive token-count range.
    add_official_mori_compat_test(
        test_official_mori_compat_zero_tokens_intranode_ws2 2
        --kernel-type IntraNode
        --api-mode sync
        --gpu-per-node 2
        --dispatch-dtype float32
        --max-num-inp-token-per-rank 256
        --num-tokens 0)

    add_official_mori_compat_test(
        test_official_mori_compat_zero_tokens_fp8_scale_intranode_ws2 2
        --kernel-type IntraNode
        --api-mode sync
        --gpu-per-node 2
        --dispatch-dtype fp8
        --hidden-dim 128
        --scale-dim 1
        --scale-dtype float32
        --max-token-type-size 2
        --max-num-inp-token-per-rank 256
        --num-tokens 0
        --no-combine-output-weights)

    # Token-count sweep coverage catches every positive token count from 1
    # through 256 and the max-token capacity boundary in representative no-scale
    # and scale-bearing synchronous call shapes.
    add_official_mori_compat_test(
        test_official_mori_compat_token_sweep_1_256_intranode_ws2 2
        --kernel-type IntraNode
        --api-mode sync
        --gpu-per-node 2
        --dispatch-dtype float32
        --max-num-inp-token-per-rank 256
        --num-tokens-list 1-256
        --stop-on-first-token-error)

    add_official_mori_compat_test(
        test_official_mori_compat_token_sweep_1_256_fp8_scale_intranode_ws2 2
        --kernel-type IntraNode
        --api-mode sync
        --gpu-per-node 2
        --dispatch-dtype fp8
        --hidden-dim 128
        --scale-dim 1
        --scale-dtype float32
        --max-token-type-size 2
        --max-num-inp-token-per-rank 256
        --num-tokens-list 1-256
        --no-combine-output-weights
        --stop-on-first-token-error)

    add_official_mori_compat_test(
        test_official_mori_compat_token_sweep_1_256_fp4_scale_intranode_ws2 2
        --kernel-type IntraNode
        --api-mode sync
        --gpu-per-node 2
        --dispatch-dtype fp4
        --hidden-dim 128
        --scale-dim 4
        --scale-dtype e8m0
        --max-token-type-size 2
        --max-num-inp-token-per-rank 256
        --num-tokens-list 1-256
        --no-combine-output-weights
        --stop-on-first-token-error)
endif()
