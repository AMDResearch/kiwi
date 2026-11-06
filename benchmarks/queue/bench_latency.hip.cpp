// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_latency.hip.cpp
 * @brief Measure round-trip GPU-CPU-GPU latency using ping-pong pattern
 *
 * Measures full round-trip latency: GPU push -> CPU pop -> CPU ack -> GPU receives ack.
 * Reports percentile statistics for latency distribution.
 *
 * Uses the SPSC (default) or MPSC (--mpsc) device-to-host queue with one
 * producer wavefront and messages of 8 to 512 bytes (--bytes).
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <hip/hip_runtime.h>
#include <bench_common.hpp>
#include "queue_bench.hip.hpp"
#include <CLI/CLI.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cstring>

using namespace kiwi::queue;

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_iterations = 1000000;
    bool mpsc = false;
    int message_bytes = 56;
    bool latency_stats = false;
    bool verbose = false;
};

// Templated ping-pong kernel — works with any DeviceHandle type
template<typename DeviceHandle, size_t Bytes>
__global__ void ping_pong_latency_kernel(
    DeviceHandle queue,
    volatile uint64_t* ack_flag,
    int num_iterations,
    uint64_t* latencies,
    int warmup_iterations,
    bool latency_stats
) {
    queue_bench::Message<Bytes> msg{};

    uint64_t t_prev;
    if (latency_stats)
        t_prev = wall_clock64();

    for (int i = 0; i < num_iterations; ++i) {
        msg.value = static_cast<uint32_t>(i);
        while (!queue.push_reg(msg)) {}
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

        uint64_t expected_ack = i + 1;
        while (*ack_flag < expected_ack) {}

        if (latency_stats) {
            uint64_t t_now = wall_clock64();
            if (__lane_id() == 0 && i >= warmup_iterations)
                latencies[i - warmup_iterations] = t_now - t_prev;
            t_prev = t_now;
        }
    }
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

// Templated benchmark — works with any queue type
template<typename QueueType, size_t Bytes>
void benchmark_ping_pong_latency(const ExperimentParams& params) {
    const auto kernel = ping_pong_latency_kernel<typename QueueType::DeviceHandle, Bytes>;
    require_no_scratch(kernel, "ping_pong_latency_kernel");

    int wallClkRate;
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    auto queue = std::make_unique<QueueType>(params.queue_capacity);
    int measured = params.num_iterations;

    volatile uint64_t* ack_flag;
    KIWI_HIP_CHECK(hipHostMalloc((void**)&ack_flag, sizeof(uint64_t), hipHostMallocCoherent));
    *ack_flag = 0;

    uint64_t* d_latencies;
    KIWI_HIP_CHECK(hipMalloc(&d_latencies, measured * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_latencies, 0, measured * sizeof(uint64_t)));

    hipStream_t stream;
    KIWI_HIP_CHECK(hipStreamCreate(&stream));

    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<1, 64, 0, stream>>>(
        queue->device_handle(), ack_flag, measured, d_latencies, 0, params.latency_stats
    );

    for (int i = 0; i < measured; ++i) {
        alignas(8) uint8_t buf[kMaxMessageBytes];
        while (!queue->pop(buf)) {}

        uint32_t value = 0;
        std::memcpy(&value, buf, sizeof(value));
        if (value != static_cast<uint32_t>(i)) {
            std::cerr << "ERROR: Expected " << i << ", got " << value << "\n";
            break;
        }

        std::atomic_thread_fence(std::memory_order_release);
        *ack_flag = i + 1;
    }

    KIWI_HIP_CHECK(hipStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << total_time_ms << " ms\n"
              << "Throughput: " << (measured / (total_time_ms / 1000.0)) << " ops/sec\n"
              << "Average round-trip latency: " << (total_time_ms * 1e6 / measured) << " ns\n\n";

    if (params.latency_stats) {
        std::vector<uint64_t> latencies(measured);
        KIWI_HIP_CHECK(hipMemcpy(latencies.data(), d_latencies, measured * sizeof(uint64_t),
                hipMemcpyDeviceToHost));

        for (auto& t : latencies) {
            t = static_cast<uint64_t>(t * ticks_to_ns);
        }

        std::vector<uint64_t> valid;
        for (auto t : latencies) {
            if (t > 0) valid.push_back(t);
        }
        print_stats(valid, "Round-trip latency (GPU->CPU->GPU)");
    }

    KIWI_HIP_CHECK(hipFree(d_latencies));
    KIWI_HIP_CHECK(hipHostFree((void*)ack_flag));
    KIWI_HIP_CHECK(hipStreamDestroy(stream));
}

int main(int argc, char** argv) {
    CLI::App app{"queue Round-Trip Latency Benchmark"};

    ExperimentParams params;

    app.add_option("-c,--capacity", params.queue_capacity, "Queue capacity (64-byte lines)")
        ->default_val(1024)->check(CLI::PositiveNumber);
    app.add_option("-n,--iterations", params.num_iterations, "Measured iterations")
        ->default_val(1000000)->check(CLI::PositiveNumber);
    app.add_flag("--mpsc", params.mpsc, "Use the MPSC queue");
    app.add_option("--bytes", params.message_bytes, "Message bytes")
        ->default_val(56)->check(CLI::IsMember(queue_bench::sizes()));
    app.add_flag("-s,--stats", params.latency_stats, "Detailed latency percentiles");
    app.add_flag("-v,--verbose", params.verbose, "Verbose output");

    CLI11_PARSE(app, argc, argv);

    check_numa_affinity();
    gpu_clock_warmup();

    std::cout << "\n" << (params.mpsc ? "MPSC" : "SPSC") << " Round-Trip Latency Benchmark\n";
    std::cout << "====================================\n";
    std::cout << "Queue capacity: " << params.queue_capacity << " lines\n";
    std::cout << "Message bytes: " << params.message_bytes << "\n";
    std::cout << "Iterations: " << params.num_iterations << "\n";
    std::cout << "Latency stats: " << (params.latency_stats ? "enabled" : "disabled") << "\n";
    std::cout << "Target: < 2 us average latency\n\n";

    queue_bench::dispatch(params.message_bytes, [&](auto bytes) {
        constexpr size_t B = decltype(bytes)::value;
        if (params.mpsc)
            benchmark_ping_pong_latency<MPSCD2HQueue, B>(params);
        else
            benchmark_ping_pong_latency<SPSCD2HQueue, B>(params);
    });

    return 0;
}
