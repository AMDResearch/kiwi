// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file harness_invoke_latency.hip.cpp
 * @brief C++ frontend of the cross-frontend invoke round-trip latency harness
 *
 * The kernel and launch of the standalone bench_invoke_latency
 * (benchmarks/invoke/bench_latency.hip.cpp), run through the harness's host
 * code (common/invoke_latency_bench.hpp) like the Python frontends. Same
 * options and output as the standalone benchmark.
 */

#include <invoke_latency_bench.hpp>
#include <hip/hip_runtime.h>
#include <CLI/CLI.hpp>

using namespace kiwi::invoke;

// DeviceContext kernel: loads DeviceHandle from device-allocated DeviceContext
// at kernel start, then all queue state is in registers for the hot loop.
template<typename PayloadT>
__global__ void ping_pong_latency_handle_kernel(
    DeviceContext<>* ctx,               // device pointer
    uint8_t ping_cb,
    volatile uint64_t* ack_flag,
    int num_iterations,
    uint64_t* latencies,
    int warmup_iterations,
    bool latency_stats
) {
    auto handle = ctx->get_device_handle(0);

    uint64_t prev_ack = 0;

    // Measure full cycle time: end-of-previous → end-of-current iteration.
    // This captures loop overhead so the per-iteration average matches the
    // total-time / iterations figure reported by the CPU wall clock.
    uint64_t t_prev;
    if (latency_stats && __lane_id() == 0)
        t_prev = wall_clock64();

    for (int i = 0; i < num_iterations; ++i) {
        PayloadT msg{};
        msg.value = static_cast<uint32_t>(prev_ack);
        while (!handle.invoke(ping_cb, msg)) {
            // Retry on full queue
        }
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

        // Wait for CPU acknowledgment (all lanes wait in lockstep).
        // Use volatile dereference directly — __builtin_nontemporal_load strips
        // volatile from the pointer (its parameter is const T*), which lets the
        // compiler eliminate this poll loop entirely.  A plain volatile load is
        // correct here: both hipHostMallocCoherent and hipDeviceMallocUncached
        // memory is already system-coherent/uncached from the GPU side.
        uint64_t current_ack;
        do {
            current_ack = *ack_flag;
        } while (current_ack == prev_ack);
        prev_ack = current_ack;

        if (latency_stats && __lane_id() == 0) {
            uint64_t t_now = wall_clock64();
            if (i >= warmup_iterations)
                latencies[i - warmup_iterations] = t_now - t_prev;
            t_prev = t_now;
        }
    }
}

// Launches ping_pong_latency_handle_kernel for the harness's host code.
struct HipLatencyLauncher {
    template<typename PayloadT>
    void prepare() {
        require_no_scratch(ping_pong_latency_handle_kernel<PayloadT>,
                           "ping_pong_latency_handle_kernel");
    }

    template<typename PayloadT>
    void launch(const kiwi_bench::LatencyLaunch& l) {
        ping_pong_latency_handle_kernel<PayloadT><<<1, 64, 0, l.stream>>>(
            l.ctx, l.ping_cb, l.ack_flag, l.num_iterations, l.latencies,
            l.warmup_iterations, l.latency_stats);
    }
};

int main(int argc, char** argv) {
    CLI::App app{"invoke Round-Trip Latency Benchmark"};

    kiwi_bench::LatencyParams params;

    // Configure CLI options
    app.add_option("-c,--capacity", params.queue_capacity,
                   "Queue capacity (64-byte lines)")
        ->default_val(1024)
        ->check(CLI::PositiveNumber);

    app.add_option("-n,--iterations", params.num_iterations,
                   "Number of iterations")
        ->default_val(1000000)
        ->check(CLI::PositiveNumber);

    app.add_option("-w,--warmup", params.warmup_iterations,
                   "Warmup iterations for GPU clock ramp (separate kernel)")
        ->default_val(2000000);

    app.add_option("-p,--payload-size", params.payload_size,
                   "Payload size in bytes")
        ->default_val(4)
        ->check(CLI::IsMember(kiwi_bench::latency_payload_sizes()));

    app.add_flag("-s,--stats", params.latency_stats,
                 "Collect and report detailed latency statistics (percentiles)");

    app.add_flag("-v,--verbose", params.verbose,
                 "Verbose output");

    app.add_flag("--device-ack", params.device_ack,
                 "Use fine-grained device memory for ack_flag (GPU VRAM, CPU-writable)");

    // Parse command line
    CLI11_PARSE(app, argc, argv);

    HipLatencyLauncher launcher;
    kiwi_bench::run_latency_benchmark(params, launcher,
                                      "invoke Round-Trip Latency Benchmark (C++ kernel)");

    return 0;
}
