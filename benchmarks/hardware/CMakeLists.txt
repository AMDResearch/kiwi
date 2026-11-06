# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Hardware-level benchmarks (no library dependencies)

set_source_files_properties(pcie_latency.hip.cpp PROPERTIES LANGUAGE HIP)

add_executable(bench_pcie_latency pcie_latency.hip.cpp)
target_include_directories(bench_pcie_latency PRIVATE ${BENCH_INCLUDE_DIR})
set_hip_compile_options(bench_pcie_latency)

add_executable(bench_host_mapped_grid_poll bench_host_mapped_grid_poll.hip.cpp)
set_source_files_properties(bench_host_mapped_grid_poll.hip.cpp PROPERTIES LANGUAGE HIP)
set_hip_compile_options(bench_host_mapped_grid_poll)
