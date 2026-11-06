# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Shared benchmark headers
set(BENCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Align loops to 32 bytes in the queue and invoke benchmarks. A one-wave push
# loop is short enough that its rate depends on where it lands relative to the
# instruction-fetch boundary (7.9-8.75 Mmsg/s for the same loop shifted by a
# few no-ops on MI300X), so unrelated code changes would move single-endpoint
# results. Benchmarks only; library users keep the compiler's default.
set(BENCH_HIP_LOOP_ALIGN $<$<COMPILE_LANGUAGE:HIP>:-falign-loops=32>)

# Add subdirectories for organized benchmarks
add_subdirectory(queue)
if(KIWI_BUILD_INVOKE)
    add_subdirectory(invoke)
    # Cross-frontend kiwi-spsc harness (C++, Triton, ...); see
    # queue_baselines/README.md.
    add_subdirectory(queue_baselines/frontends)
endif()
add_subdirectory(hardware)
# Python helpers shared by the cross-frontend harnesses' frontends.
if(KIWI_PYTHON_FRONTENDS)
    add_subdirectory(frontends)
endif()
if(KIWI_BUILD_QUEUE_BASELINES)
    add_subdirectory(queue_baselines)
endif()
