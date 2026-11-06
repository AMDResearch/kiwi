// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_mpmc_h2d_latency.hip.cpp
 * @brief Measure round-trip CPU-GPU-CPU latency for MPMC Host-to-Device queue
 *
 * Measures full round-trip latency: CPU push -> GPU pop -> GPU ack -> CPU sees ack.
 * Timed from CPU (producer) side using ping-pong pattern.
 *
 * Uses MPMCH2DQueue (multi-producer multi-consumer), but ping-pong is inherently
 * single-producer single-consumer to get clean per-message latency numbers.
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <hip/hip_runtime.h>
#include <bench_common.hpp>
#include <CLI/CLI.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstring>

using namespace kiwi::queue;

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_iterations = 1000000;
    int warmup_iterations = 2000000;
    bool latency_stats = false;
    bool verbose = false;
};

// Ping-pong kernel: GPU pops one message then writes ack.
// CPU measures the round-trip time (push + wait-for-ack).
// Launch with <<<1, 64>>>.
__global__ void mpmc_h2d_ping_pong_kernel(
    MPMCH2DQueue::DeviceHandle queue,
    volatile uint64_t* ack_flag,
    int num_iterations
) {
    uint8_t buf[56] = {};
    for (int i = 0; i < num_iterations; ++i) {
        // Pop message (spin until data arrives)
        while (!queue.pop(buf)) {}

        // Write ack so CPU can see we received the message
        if (__lane_id() == 0) {
            *ack_flag = i + 1;
        }
        // Flush store to host memory
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    queue.flush_consumer_index();
}

void print_stats(const std::vector<uint64_t>& latencies, const char* title) {
    if (latencies.empty()) return;

    std::vector<uint64_t> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    uint64_t min_lat = sorted.front();
    uint64_t max_lat = sorted.back();
    uint64_t p50 = sorted[n / 2];
    uint64_t p95 = sorted[n * 95 / 100];
    uint64_t p99 = sorted[n * 99 / 100];

    double avg = 0;
    for (auto lat : sorted) avg += lat;
    avg /= n;

    std::cout << title << " (n=" << n << "):\n";
    std::cout << "  Min:    " << std::setw(8) << min_lat << " ns\n";
    std::cout << "  p50:    " << std::setw(8) << p50 << " ns\n";
    std::cout << "  p95:    " << std::setw(8) << p95 << " ns\n";
    std::cout << "  p99:    " << std::setw(8) << p99 << " ns\n";
    std::cout << "  Max:    " << std::setw(8) << max_lat << " ns\n";
    std::cout << "  Avg:    " << std::setw(8) << std::fixed << std::setprecision(1) << avg << " ns\n";

    std::cout << "\n  p50:    " << std::setw(8) << std::fixed << std::setprecision(3) << p50 / 1000.0 << " us\n";
    std::cout << "  p99:    " << std::setw(8) << p99 / 1000.0 << " us\n";
    std::cout << "  Avg:    " << std::setw(8) << avg / 1000.0 << " us\n";
}

void benchmark_ping_pong_latency(const ExperimentParams& params) {
    std::cout << "\n=== MPMC H2D Round-Trip Latency (CPU->GPU->CPU ping-pong) ===\n";

    // notify_interval=1 for minimum latency in ping-pong (flush every pop)
    auto queue = std::make_unique<MPMCH2DQueue>(params.queue_capacity, /*notify_interval=*/1);

    int warmup = params.warmup_iterations;
    int measured = params.num_iterations;

    // Ack flag: GPU writes here after each pop (pinned host mem)
    volatile uint64_t* ack_flag;
    KIWI_HIP_CHECK(hipHostMalloc((void**)&ack_flag, sizeof(uint64_t), hipHostMallocCoherent));
    *ack_flag = 0;

    hipStream_t stream;
    KIWI_HIP_CHECK(hipStreamCreate(&stream));

    // Warmup kernel — ramps GPU clock and warms the path
    {
        auto warmup_queue = std::make_unique<MPMCH2DQueue>(
            params.queue_capacity, /*notify_interval=*/1);
        volatile uint64_t* warmup_ack;
        KIWI_HIP_CHECK(hipHostMalloc((void**)&warmup_ack, sizeof(uint64_t), hipHostMallocCoherent));
        *warmup_ack = 0;

        mpmc_h2d_ping_pong_kernel<<<1, 64, 0, stream>>>(
            warmup_queue->device_handle(), warmup_ack, warmup
        );
        uint8_t buf[56] = {};
        for (int i = 0; i < warmup; ++i) {
            uint32_t val = static_cast<uint32_t>(i);
            std::memcpy(buf, &val, sizeof(val));
            while (!warmup_queue->push(buf)) {}
            while (*warmup_ack < static_cast<uint64_t>(i + 1)) {}
        }
        KIWI_HIP_CHECK(hipStreamSynchronize(stream));
        KIWI_HIP_CHECK(hipHostFree((void*)warmup_ack));
    }

    // Measured phase — CPU times each push+wait_for_ack round-trip
    mpmc_h2d_ping_pong_kernel<<<1, 64, 0, stream>>>(
        queue->device_handle(), ack_flag, measured
    );

    std::vector<uint64_t> latencies(measured);
    uint8_t buf[56] = {};

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < measured; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();

        // Push message
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        while (!queue->push(buf)) {}

        // Wait for GPU ack
        while (*ack_flag < static_cast<uint64_t>(i + 1)) {}

        auto t_end = std::chrono::high_resolution_clock::now();
        latencies[i] = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
    }

    KIWI_HIP_CHECK(hipStreamSynchronize(stream));

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    std::cout << "Total time:   " << std::fixed << std::setprecision(3)
              << total_time_ms << " ms\n"
              << "Throughput:   " << std::setprecision(0)
              << (measured / (total_time_ms / 1000.0)) << " ops/sec\n"
              << "Avg RT:       " << std::setprecision(2)
              << (total_time_ms * 1e6 / measured) << " ns\n\n";

    if (params.latency_stats) {
        print_stats(latencies, "Round-trip latency (CPU->GPU->CPU)");
    }

    KIWI_HIP_CHECK(hipHostFree((void*)ack_flag));
    KIWI_HIP_CHECK(hipStreamDestroy(stream));
}

int main(int argc, char** argv) {
    CLI::App app{"MPMCH2DQueue Round-Trip Latency Benchmark"};

    ExperimentParams params;

    app.add_option("-c,--capacity", params.queue_capacity, "Queue capacity")
        ->default_val(1024)->check(CLI::PositiveNumber);
    app.add_option("-n,--iterations", params.num_iterations, "Measured iterations")
        ->default_val(1000000)->check(CLI::PositiveNumber);
    app.add_option("-w,--warmup", params.warmup_iterations,
                   "Warmup iterations (separate kernel)")
        ->default_val(2000000);
    app.add_flag("-s,--stats", params.latency_stats,
                 "Detailed latency percentiles");
    app.add_flag("-v,--verbose", params.verbose, "Verbose output");

    CLI11_PARSE(app, argc, argv);

    std::cout << "MPMCH2DQueue Round-Trip Latency Benchmark\n";
    std::cout << "==========================================\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Iterations: " << params.num_iterations
              << " (" << params.warmup_iterations << " warmup)\n";
    std::cout << "Latency stats: " << (params.latency_stats ? "enabled" : "disabled") << "\n\n";

    check_numa_affinity();
    benchmark_ping_pong_latency(params);

    return 0;
}
