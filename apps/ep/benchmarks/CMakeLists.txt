# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# EP performance benchmarks. The MORI-style graph benchmark lives in
# apps/ep/benchmarks/python/bench_kiwi_ep.py; native standalone microbenchmarks
# live here. Native dispatch/combine correctness coverage remains under apps/ep/tests.

add_executable(bench_ep_collectives bench_ep_collectives.hip.cpp)
set_source_files_properties(bench_ep_collectives.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_ep_collectives PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_ep_collectives PRIVATE kiwi_ep CLI11::CLI11)

add_executable(bench_ep_hidden_copy bench_ep_hidden_copy.hip.cpp)
set_source_files_properties(bench_ep_hidden_copy.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_ep_hidden_copy PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
find_library(HSA_RUNTIME64_LIBRARY hsa-runtime64 REQUIRED
    PATHS /opt/rocm/lib /opt/rocm/lib64
)
target_link_libraries(bench_ep_hidden_copy PRIVATE CLI11::CLI11 ${IBVERBS_LIBRARY} ${HSA_RUNTIME64_LIBRARY})

find_package(MPI REQUIRED)

add_executable(bench_ep_copy bench_ep_copy.hip.cpp)
set_source_files_properties(bench_ep_copy.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_ep_copy PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_ep_copy PRIVATE CLI11::CLI11 MPI::MPI_CXX)
