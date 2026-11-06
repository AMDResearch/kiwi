// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_overhead.hip.cpp
 * @brief Measure GPU invoke operation overhead with invoke
 *
 * Measures the overhead of callback invocations using GPU realtime clock.
 * Reports per-operation timing with percentile statistics.
 */

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/error.hpp>
#include <bench_common.hpp>
#include <hip/hip_runtime.h>
#include <CLI/CLI.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>

using namespace kiwi::invoke;

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_messages = 100000;
    int warmup_iterations = -1;  // -1 means auto-calculate
    bool verbose = false;

    int get_warmup() const {
        if (warmup_iterations >= 0) {
            return warmup_iterations;
        }
        return std::min(1000, num_messages / 10);
    }
};

// GPU wall clock - returns raw ticks (converted to nanoseconds on host side)
__device__ __forceinline__ uint64_t gpu_clock_ticks() {
    return wall_clock64();
}

// Kernel that measures individual invoke operation overhead
// Launch with <<<1, 64>>>. Full wavefront cooperative push.
// Stores timing data in device memory for later analysis.
__global__ void timed_invoke_kernel(DeviceContext<>* ctx,
                                    uint8_t callback_id,
                                    int num_msgs,
                                    uint64_t* invoke_times,  // Array of invoke durations (in clock ticks)
                                    int warmup_msgs) {
    auto handle = ctx->get_device_handle(0);

    for (int i = 0; i < num_msgs; ++i) {
        uint64_t t0 = gpu_clock_ticks();

        // Invoke with sequence number as payload
        while (!handle.invoke(callback_id, static_cast<uint32_t>(i))) {
            // Busy wait for space - backpressure from CPU
        }

        uint64_t t1 = gpu_clock_ticks();

        // Record timing after warmup (lane 0 only)
        if (__lane_id() == 0 && i >= warmup_msgs) {
            invoke_times[i - warmup_msgs] = t1 - t0;
        }
    }
}

// Simple invoke kernel without timing (for pre-filling or throughput tests)
// Launch with <<<1, 64>>>. Full wavefront cooperative push.
__global__ void simple_invoke_kernel(DeviceContext<>* ctx,
                                     uint8_t callback_id,
                                     int num_msgs,
                                     volatile int* done_flag) {
    auto handle = ctx->get_device_handle(0);

    for (int i = 0; i < num_msgs; ++i) {
        while (!handle.invoke(callback_id, static_cast<uint32_t>(i))) {
            // Busy wait
        }
    }
    if (__lane_id() == 0 && done_flag) {
        __threadfence_system();
        *done_flag = 1;
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
}

// Measure GPU invoke overhead with active CPU consumer
void benchmark_invoke_overhead(const ExperimentParams& params) {
    std::cout << "\n=== GPU Invoke Overhead (Active Consumer) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Messages: " << params.num_messages << "\n\n";

    // Query wall clock rate for GPU tick → nanosecond conversion
    int wallClkRate;  // in kHz
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    Context ctx(1, params.queue_capacity);

    int warmup_msgs = params.get_warmup();
    int measured_msgs = params.num_messages - warmup_msgs;

    // Register callback
    std::atomic<uint32_t> expected{0};
    std::function<void(uint32_t)> callback = [&](uint32_t value) {
        if (value != expected.load()) {
            std::cerr << "ERROR: Expected " << expected.load() << " got " << value << "\n";
        }
        expected++;
    };
    uint8_t callback_id = ctx.register_callback<uint32_t>(callback);

    // Allocate device memory for timing results
    uint64_t* d_invoke_times;
    KIWI_HIP_CHECK(hipMalloc(&d_invoke_times, measured_msgs * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_invoke_times, 0, measured_msgs * sizeof(uint64_t)));

    // Start GPU kernel
    hipStream_t stream;
    KIWI_HIP_CHECK(hipStreamCreate(&stream));
    require_no_scratch(timed_invoke_kernel, "timed_invoke_kernel");
    timed_invoke_kernel<<<1, 64, 0, stream>>>(ctx.device_context(), callback_id, params.num_messages,
                                               d_invoke_times, warmup_msgs);

    // CPU consumer - process all callbacks to prevent queue from filling
    auto start = std::chrono::high_resolution_clock::now();

    while (expected.load() < static_cast<uint32_t>(params.num_messages)) {
        ctx.progress(0);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Wait for kernel
    KIWI_HIP_CHECK(hipStreamSynchronize(stream));

    // Copy timing results back
    std::vector<uint64_t> invoke_times(measured_msgs);
    KIWI_HIP_CHECK(hipMemcpy(invoke_times.data(), d_invoke_times,
                             measured_msgs * sizeof(uint64_t), hipMemcpyDeviceToHost));

    // Convert raw GPU ticks to nanoseconds
    for (auto& t : invoke_times) {
        t = static_cast<uint64_t>(t * ticks_to_ns);
    }

    // Calculate throughput
    auto total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double throughput = (double)params.num_messages / total_time_us * 1e6;

    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << throughput / 1e6 << " M msgs/sec\n";
    std::cout << "Total time: " << total_time_us / 1000.0 << " ms\n\n";

    // Filter out any zero entries (shouldn't happen but be safe)
    std::vector<uint64_t> valid_times;
    valid_times.reserve(measured_msgs);
    for (auto t : invoke_times) {
        if (t > 0) valid_times.push_back(t);
    }

    print_stats(valid_times, "GPU invoke() overhead");

    // Cleanup
    KIWI_HIP_CHECK(hipFree(d_invoke_times));
    KIWI_HIP_CHECK(hipStreamDestroy(stream));
}

// Measure GPU invoke overhead with empty queue (no backpressure)
void benchmark_invoke_no_backpressure(const ExperimentParams& params) {
    std::cout << "\n=== GPU Invoke Overhead (Empty Queue, No Backpressure) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n\n";

    // Query wall clock rate for GPU tick → nanosecond conversion
    int wallClkRate;  // in kHz
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    Context ctx(1, params.queue_capacity);

    // One-line messages: only capacity-1 fit before blocking
    int num_msgs = params.queue_capacity - 1;
    int warmup_msgs = std::min(100, num_msgs / 10);
    int measured_msgs = num_msgs - warmup_msgs;

    // Register dummy callback
    std::function<void(uint32_t)> callback = [](uint32_t) {};
    uint8_t callback_id = ctx.register_callback<uint32_t>(callback);

    // Allocate device memory for timing results
    uint64_t* d_invoke_times;
    KIWI_HIP_CHECK(hipMalloc(&d_invoke_times, measured_msgs * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_invoke_times, 0, measured_msgs * sizeof(uint64_t)));

    // Run kernel - no consumer needed since queue won't fill
    require_no_scratch(timed_invoke_kernel, "timed_invoke_kernel");
    timed_invoke_kernel<<<1, 64>>>(ctx.device_context(), callback_id, num_msgs,
                                    d_invoke_times, warmup_msgs);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Copy timing results back
    std::vector<uint64_t> invoke_times(measured_msgs);
    KIWI_HIP_CHECK(hipMemcpy(invoke_times.data(), d_invoke_times,
                             measured_msgs * sizeof(uint64_t), hipMemcpyDeviceToHost));

    // Convert raw GPU ticks to nanoseconds
    for (auto& t : invoke_times) {
        t = static_cast<uint64_t>(t * ticks_to_ns);
    }

    // Filter out zeros
    std::vector<uint64_t> valid_times;
    for (auto t : invoke_times) {
        if (t > 0) valid_times.push_back(t);
    }

    print_stats(valid_times, "GPU invoke() overhead (no backpressure)");

    // Cleanup
    KIWI_HIP_CHECK(hipFree(d_invoke_times));
}

// Measure inter-arrival time on CPU side
void benchmark_inter_arrival(const ExperimentParams& params) {
    std::cout << "\n=== Inter-arrival Time (CPU side) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Messages: " << params.num_messages << "\n\n";

    Context ctx(1, params.queue_capacity);

    // Register callback that records arrival time
    std::vector<std::chrono::high_resolution_clock::time_point> arrival_times;
    arrival_times.reserve(params.num_messages);
    std::atomic<uint32_t> expected{0};

    std::function<void(uint32_t)> callback = [&](uint32_t value) {
        arrival_times.push_back(std::chrono::high_resolution_clock::now());
        if (value != expected.load()) {
            std::cerr << "ERROR: Expected " << expected.load() << " got " << value << "\n";
        }
        expected++;
    };
    uint8_t callback_id = ctx.register_callback<uint32_t>(callback);

    // Launch simple invoke kernel
    require_no_scratch(simple_invoke_kernel, "simple_invoke_kernel");
    simple_invoke_kernel<<<1, 64>>>(ctx.device_context(), callback_id, params.num_messages, nullptr);

    // CPU consumer - process callbacks
    while (expected.load() < static_cast<uint32_t>(params.num_messages)) {
        ctx.progress(0);
    }

    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Report callback-to-callback times (inter-arrival latency)
    int warmup = params.get_warmup();
    std::vector<uint64_t> inter_arrival;
    inter_arrival.reserve(params.num_messages - warmup - 1);

    for (int i = warmup + 1; i < params.num_messages; ++i) {
        auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
            arrival_times[i] - arrival_times[i-1]).count();
        if (dt > 0) inter_arrival.push_back(dt);
    }

    print_stats(inter_arrival, "Inter-arrival time (CPU side)");
}

int main(int argc, char** argv) {
    CLI::App app{"invoke Operation Overhead Benchmark"};

    ExperimentParams params;

    // Configure CLI options
    app.add_option("-c,--capacity", params.queue_capacity,
                   "Queue capacity (64-byte lines)")
        ->default_val(1024)
        ->check(CLI::PositiveNumber);

    app.add_option("-m,--messages", params.num_messages,
                   "Number of messages")
        ->default_val(100000)
        ->check(CLI::PositiveNumber);

    app.add_option("-w,--warmup", params.warmup_iterations,
                   "Warmup iterations (default: auto = 10% of messages, max 1000)")
        ->default_val(-1);

    app.add_flag("-v,--verbose", params.verbose,
                 "Verbose output");

    // Parse command line
    CLI11_PARSE(app, argc, argv);

    // Print configuration
    std::cout << "invoke Operation Overhead Benchmark\n";
    std::cout << "====================================\n";

    // First: invoke overhead with no backpressure (empty queue)
    benchmark_invoke_no_backpressure(params);

    // Second: invoke overhead with active consumer (steady state)
    benchmark_invoke_overhead(params);

    // Third: inter-arrival time on CPU
    benchmark_inter_arrival(params);

    return 0;
}
