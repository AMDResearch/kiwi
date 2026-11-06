# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Benchmarks for kiwi-p2p

find_package(Threads REQUIRED)
find_package(MPI REQUIRED)

add_executable(bench_latency bench_latency.hip.cpp)
set_source_files_properties(bench_latency.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_latency PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_latency PRIVATE kiwi_p2p CLI11::CLI11)

add_executable(bench_bandwidth bench_bandwidth.hip.cpp)
set_source_files_properties(bench_bandwidth.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_bandwidth PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_bandwidth PRIVATE kiwi_p2p CLI11::CLI11)

add_executable(bench_loadstore_latency bench_loadstore_latency.hip.cpp)
set_source_files_properties(bench_loadstore_latency.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_loadstore_latency PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_loadstore_latency PRIVATE CLI11::CLI11)

add_executable(bench_lci_loopback bench_lci_loopback.hip.cpp)
set_source_files_properties(bench_lci_loopback.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_lci_loopback PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_lci_loopback PRIVATE kiwi_p2p CLI11::CLI11)

add_executable(bench_lci_put_latency bench_lci_put_latency.hip.cpp)
set_source_files_properties(bench_lci_put_latency.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_lci_put_latency PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_lci_put_latency PRIVATE kiwi_p2p CLI11::CLI11)

add_executable(bench_lci_put_bandwidth bench_lci_put_bandwidth.hip.cpp)
set_source_files_properties(bench_lci_put_bandwidth.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_lci_put_bandwidth PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_lci_put_bandwidth PRIVATE kiwi_p2p CLI11::CLI11)

add_executable(bench_kiwi_readiness bench_kiwi_readiness.hip.cpp)
set_source_files_properties(bench_kiwi_readiness.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_kiwi_readiness PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_kiwi_readiness PRIVATE kiwi_p2p CLI11::CLI11)

add_executable(bench_lci_readiness bench_lci_readiness.hip.cpp)
set_source_files_properties(bench_lci_readiness.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_lci_readiness PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(bench_lci_readiness PRIVATE kiwi_p2p CLI11::CLI11)

add_executable(bench_ibverbs_readiness bench_ibverbs_readiness.hip.cpp)
set_source_files_properties(bench_ibverbs_readiness.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(bench_ibverbs_readiness PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
find_package(hsa-runtime64 CONFIG REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
target_link_libraries(bench_ibverbs_readiness PRIVATE
    CLI11::CLI11 MPI::MPI_CXX hsa-runtime64::hsa-runtime64 ${IBVERBS_LIBRARY})

# Cross-frontend harness (C++, FlyDSL, ...) for bench_kiwi_readiness; see
# apps/README.md.
add_subdirectory(frontends/readiness)
