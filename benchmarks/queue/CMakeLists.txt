# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# queue benchmarks

# Benchmark helper function
function(add_queue_benchmark bench_name source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)

    add_executable(${bench_name} ${source})
    target_link_libraries(${bench_name} PRIVATE kiwi_queue CLI11::CLI11)
    target_include_directories(${bench_name} PRIVATE ${BENCH_INCLUDE_DIR})
    target_compile_options(${bench_name} PRIVATE ${BENCH_HIP_LOOP_ALIGN})
    set_hip_compile_options(${bench_name})
    set_common_definitions(${bench_name})
endfunction()

# Benchmarks
add_queue_benchmark(bench_queue_overhead bench_overhead.hip.cpp)
add_queue_benchmark(bench_queue_throughput bench_throughput.hip.cpp)
add_queue_benchmark(bench_queue_latency bench_latency.hip.cpp)
add_queue_benchmark(bench_host_to_device_latency bench_host_to_device_latency.hip.cpp)
add_queue_benchmark(bench_mpmc_h2d_latency bench_mpmc_h2d_latency.hip.cpp)
add_queue_benchmark(bench_mpmc_h2d_throughput bench_mpmc_h2d_throughput.hip.cpp)
