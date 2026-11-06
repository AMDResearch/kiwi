// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file harness_invoke_throughput.hip.cpp
 * @brief C++ frontend of the cross-frontend invoke callback throughput harness
 *
 * The kernel and launch of the standalone bench_invoke_throughput
 * (benchmarks/invoke/bench_throughput.hip.cpp), run through the harness's
 * host code (common/invoke_throughput_bench.hpp) like the Python frontends.
 * Same options and output as the standalone benchmark.
 */

#include <invoke_throughput_bench.hpp>
#include <hip/hip_runtime.h>
#include <CLI/CLI.hpp>
#include <iostream>

using namespace kiwi::invoke;

// Producer kernel: one wavefront owns one endpoint. Launch blocks are capped
// at 256 by packing additional endpoint wavefronts into each block.
__global__ void throughput_invoke_kernel(
    DeviceContext<>* ctx,
    uint8_t callback_id,
    int num_msgs_per_endpoint,
    int num_endpoints,
    int waves_per_block
) {
    int wave_in_block = threadIdx.x / 64;
    int endpoint = blockIdx.x * waves_per_block + wave_in_block;
    if (endpoint >= num_endpoints) return;

    auto handle = ctx->get_device_handle(endpoint);

    for (int i = 0; i < num_msgs_per_endpoint; ++i) {
        uint32_t value = static_cast<uint32_t>(endpoint * num_msgs_per_endpoint + i);

        while (!handle.invoke(callback_id, value)) {
            // Retry on full
        }
    }
}

// Launches throughput_invoke_kernel for the harness's host code.
struct HipThroughputLauncher {
    void prepare() {
        require_no_scratch(throughput_invoke_kernel, "throughput_invoke_kernel");
    }

    void launch(const kiwi_bench::ThroughputLaunch& l) {
        throughput_invoke_kernel<<<l.launch_blocks, 64 * l.waves_per_block>>>(
            l.ctx, l.callback_id, l.num_msgs_per_endpoint,
            l.num_endpoints, l.waves_per_block);
    }
};

int main(int argc, char** argv) {
    CLI::App app{"invoke Throughput Benchmark"};

    kiwi_bench::ThroughputParams params;

    // Configure CLI options
    app.add_option("-c,--capacity", params.queue_capacity,
                   "Queue capacity (64-byte lines)")
        ->default_val(1024)
        ->check(CLI::PositiveNumber);

    app.add_option("-m,--messages", params.num_messages,
                   "Total number of messages")
        ->default_val(1000000)
        ->check(CLI::PositiveNumber);

    app.add_option("-e,--endpoints", params.num_endpoints,
                   "Number of endpoints (parallel queues)")
        ->default_val(1)
        ->check(CLI::PositiveNumber);

    app.add_option("-t,--cpu-threads", params.num_cpu_threads,
                   "Number of CPU consumer threads")
        ->default_val(1)
        ->check(CLI::PositiveNumber);
    app.add_option("--poll-mode", params.poll_mode,
                   "Consumer polling: direct or host-poll-state")
        ->default_val("host-poll-state")
        ->check(CLI::IsMember(kiwi_bench::throughput_poll_modes()));

    app.add_flag("-v,--verbose", params.verbose,
                 "Verbose output");

    // Parse command line
    CLI11_PARSE(app, argc, argv);

    if (params.num_endpoints > kiwi_bench::kMaxThroughputEndpoints) {
        std::cerr << "endpoint count must not exceed " << kiwi_bench::kMaxThroughputEndpoints
                  << "\n";
        return 1;
    }

    HipThroughputLauncher launcher;
    const int processed = kiwi_bench::run_throughput_benchmark(
        params, launcher, "invoke Throughput Benchmark (C++ kernel)");

    // As the Python driver: every message must have been processed.
    const int expected = params.num_messages / params.num_endpoints * params.num_endpoints;
    if (processed != expected) {
        std::cerr << "processed " << processed << " of " << expected << " messages\n";
        return 1;
    }
    return 0;
}
