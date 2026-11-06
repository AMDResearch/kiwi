# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# invoke benchmarks

# Benchmark helper function
function(add_invoke_benchmark bench_name source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)

    add_executable(${bench_name} ${source})
    target_link_libraries(${bench_name} PRIVATE kiwi_invoke CLI11::CLI11)
    target_include_directories(${bench_name} PRIVATE ${BENCH_INCLUDE_DIR})
    target_compile_options(${bench_name} PRIVATE ${BENCH_HIP_LOOP_ALIGN})
    set_hip_compile_options(${bench_name})
    set_common_definitions(${bench_name})
endfunction()

# Benchmarks
add_invoke_benchmark(bench_invoke_overhead bench_overhead.hip.cpp)
add_invoke_benchmark(bench_invoke_throughput bench_throughput.hip.cpp)
add_invoke_benchmark(bench_invoke_latency bench_latency.hip.cpp)

# Cross-frontend harnesses (C++, FlyDSL, ...) for the latency and throughput
# benchmarks; see benchmarks/README.md.
add_subdirectory(frontends/latency)
add_subdirectory(frontends/throughput)
