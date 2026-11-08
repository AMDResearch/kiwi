# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# queue tests

# Basic tests (CPU-only)
add_hip_test(test_queue_basic test_basic.hip.cpp)

# Concurrent tests (GPU-CPU interaction)
add_hip_test(test_queue_concurrent test_concurrent.hip.cpp)

# Cached host poll-state tests (GPU-CPU interaction)
add_hip_test(test_queue_host_poll_state test_host_poll_state.hip.cpp)

# Hardware assumption tests (CRITICAL)
add_hip_test(test_queue_hardware test_hardware_assumptions.hip.cpp)

# Wavefront coalesced write atomicity test
add_hip_test(test_queue_wavefront_atomicity test_wavefront_atomicity.hip.cpp)

# Device-to-host chunk format and push paths, on both D2H queues
add_hip_test(test_queue_chunk test_chunk_queue.hip.cpp)

# Lane-masked device-to-host pushes (push_lane_masked), on both D2H queues
add_hip_test(test_queue_push_lane_masked test_push_lane_masked.hip.cpp)

# GPU cache latency characterization (standalone, no gtest)
add_hip_standalone_test(test_queue_cache_latency test_cache_latency.hip.cpp)

# Fine-grained device memory: CPU write → GPU read
add_hip_test(test_queue_fine_grained_device_mem test_fine_grained_device_mem.hip.cpp)

# Host→Device queue concurrent tests
add_hip_test(test_host_to_device_concurrent test_host_to_device_concurrent.hip.cpp)

# MPMC Host→Device queue concurrent tests
add_hip_test(test_queue_mpmc_h2d_concurrent test_mpmc_h2d_concurrent.hip.cpp)

# CPU→GPU write visibility latency diagnostic
add_hip_test(test_fg_devmem_latency test_fg_devmem_latency.hip.cpp)

# Push-path ISA (gfx942, compile only; the kernels include header-only invoke):
# no scratch, ring stores are global_store_dwordx4 with no flat accesses,
# wave-uniform broadcasts use v_readfirstlane_b32, and no push compares two
# 64-bit VGPR pairs (a handle left in VGPRs).
if(KIWI_BUILD_INVOKE)
    set(KIWI_PUSH_ISA ${CMAKE_CURRENT_BINARY_DIR}/isa_push_paths.s)
    file(GLOB_RECURSE KIWI_PUSH_ISA_HEADERS CONFIGURE_DEPENDS
         ${PROJECT_SOURCE_DIR}/include/kiwi/*.hpp)
    add_custom_command(
        OUTPUT ${KIWI_PUSH_ISA}
        COMMAND ${CMAKE_HIP_COMPILER} -x hip --rocm-path=${ROCM_PATH}
                --offload-arch=gfx942 --cuda-device-only -S -O3 -std=c++17 -DNDEBUG
                -I${PROJECT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/isa_push_paths.hip.cpp -o ${KIWI_PUSH_ISA}
        DEPENDS isa_push_paths.hip.cpp ${KIWI_PUSH_ISA_HEADERS}
        VERBATIM)
    add_custom_target(queue_push_isa ALL DEPENDS ${KIWI_PUSH_ISA})
    add_test(NAME test_queue_push_isa
             COMMAND ${CMAKE_COMMAND} -DASM=${KIWI_PUSH_ISA}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/check_push_isa.cmake)
endif()
