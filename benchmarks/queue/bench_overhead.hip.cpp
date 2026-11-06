// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_overhead.hip.cpp
 * @brief Measure GPU push operation overhead
 *
 * Measures the overhead of queue push operations using GPU realtime clock.
 * Reports per-operation timing with percentile statistics.
 *
 * Uses the SPSC (default) or MPSC (--mpsc) device-to-host queue with one
 * producer wavefront and messages of 8 to 512 bytes (--bytes).
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <bench_common.hpp>
#include "queue_bench.hip.hpp"
#include <hip/hip_runtime.h>
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
    int num_messages = 100000;
    int warmup_iterations = -1;
    bool mpsc = false;
    int message_bytes = 56;
    bool verbose = false;

    int get_warmup() const {
        if (warmup_iterations >= 0) return warmup_iterations;
        return std::min(1000, num_messages / 10);
    }
};

__device__ __forceinline__ uint64_t gpu_clock_ticks() {
    return wall_clock64();
}

// Templated timed push kernel — works with any DeviceHandle type
template<typename DeviceHandle, size_t Bytes>
__global__ void timed_push_kernel(DeviceHandle queue, int num_msgs,
                                   uint64_t* push_times, int warmup_msgs) {
    queue_bench::Message<Bytes> msg{};
    for (int i = 0; i < num_msgs; ++i) {
        uint64_t t0 = gpu_clock_ticks();

        msg.value = static_cast<uint32_t>(i);
        while (!queue.push_reg(msg)) {}

        uint64_t t1 = gpu_clock_ticks();

        if (__lane_id() == 0 && i >= warmup_msgs) {
            push_times[i - warmup_msgs] = t1 - t0;
        }
    }
}

// Templated simple push kernel (no timing)
template<typename DeviceHandle, size_t Bytes>
__global__ void simple_push_kernel(DeviceHandle queue, int num_msgs,
                                    volatile int* done_flag) {
    queue_bench::Message<Bytes> msg{};
    for (int i = 0; i < num_msgs; ++i) {
        msg.value = static_cast<uint32_t>(i);
        while (!queue.push_reg(msg)) {}
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

// Measure GPU push overhead with active CPU consumer
template<typename QueueType, size_t Bytes>
void benchmark_push_overhead(const ExperimentParams& params) {
    std::cout << "\n=== GPU Push Overhead (Active Consumer) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Messages: " << params.num_messages << "\n\n";

    int wallClkRate;
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    auto queue = std::make_unique<QueueType>(params.queue_capacity);

    int warmup_msgs = params.get_warmup();
    int measured_msgs = params.num_messages - warmup_msgs;

    uint64_t* d_push_times;
    KIWI_HIP_CHECK(hipMalloc(&d_push_times, measured_msgs * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_push_times, 0, measured_msgs * sizeof(uint64_t)));

    hipStream_t stream;
    KIWI_HIP_CHECK(hipStreamCreate(&stream));
    timed_push_kernel<typename QueueType::DeviceHandle, Bytes><<<1, 64, 0, stream>>>(queue->device_handle(), params.num_messages,
                                             d_push_times, warmup_msgs);

    uint32_t expected = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (expected < static_cast<uint32_t>(params.num_messages)) {
        alignas(8) uint8_t buf[kMaxMessageBytes];
        if (queue->pop(buf)) {
            uint32_t value = 0;
            std::memcpy(&value, buf, sizeof(value));
            if (value != expected) {
                std::cerr << "ERROR: Expected " << expected << " got " << value << "\n";
                break;
            }
            expected++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    KIWI_HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<uint64_t> push_times(measured_msgs);
    KIWI_HIP_CHECK(hipMemcpy(push_times.data(), d_push_times,
                             measured_msgs * sizeof(uint64_t), hipMemcpyDeviceToHost));

    for (auto& t : push_times) {
        t = static_cast<uint64_t>(t * ticks_to_ns);
    }

    auto total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double throughput = (double)params.num_messages / total_time_us * 1e6;

    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << throughput / 1e6 << " M msgs/sec\n";
    std::cout << "Total time: " << total_time_us / 1000.0 << " ms\n\n";

    std::vector<uint64_t> valid_times;
    valid_times.reserve(measured_msgs);
    for (auto t : push_times) {
        if (t > 0) valid_times.push_back(t);
    }

    print_stats(valid_times, "GPU push() overhead");

    KIWI_HIP_CHECK(hipFree(d_push_times));
    KIWI_HIP_CHECK(hipStreamDestroy(stream));
}

// Measure GPU push overhead with empty queue (no backpressure)
template<typename QueueType, size_t Bytes>
void benchmark_push_no_backpressure(const ExperimentParams& params) {
    std::cout << "\n=== GPU Push Overhead (Empty Queue, No Backpressure) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n\n";

    int wallClkRate;
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    auto queue = std::make_unique<QueueType>(params.queue_capacity);

    // Messages that fit without a consumer.
    int num_msgs = static_cast<int>(params.queue_capacity / message_lines(Bytes)) - 1;
    int warmup_msgs = std::min(100, num_msgs / 10);
    int measured_msgs = num_msgs - warmup_msgs;

    uint64_t* d_push_times;
    KIWI_HIP_CHECK(hipMalloc(&d_push_times, measured_msgs * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_push_times, 0, measured_msgs * sizeof(uint64_t)));

    timed_push_kernel<typename QueueType::DeviceHandle, Bytes><<<1, 64>>>(queue->device_handle(), num_msgs,
                                  d_push_times, warmup_msgs);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> push_times(measured_msgs);
    KIWI_HIP_CHECK(hipMemcpy(push_times.data(), d_push_times,
                             measured_msgs * sizeof(uint64_t), hipMemcpyDeviceToHost));

    for (auto& t : push_times) {
        t = static_cast<uint64_t>(t * ticks_to_ns);
    }

    std::vector<uint64_t> valid_times;
    for (auto t : push_times) {
        if (t > 0) valid_times.push_back(t);
    }

    print_stats(valid_times, "GPU push() overhead (no backpressure)");

    KIWI_HIP_CHECK(hipFree(d_push_times));
}

// Measure inter-arrival time on CPU side
template<typename QueueType, size_t Bytes>
void benchmark_inter_arrival(const ExperimentParams& params) {
    std::cout << "\n=== Inter-arrival Time (CPU side) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Messages: " << params.num_messages << "\n\n";

    auto queue = std::make_unique<QueueType>(params.queue_capacity);

    simple_push_kernel<typename QueueType::DeviceHandle, Bytes><<<1, 64>>>(queue->device_handle(), params.num_messages, nullptr);

    std::vector<std::chrono::high_resolution_clock::time_point> pop_times;
    pop_times.reserve(params.num_messages);

    uint32_t expected = 0;
    while (expected < static_cast<uint32_t>(params.num_messages)) {
        alignas(8) uint8_t buf[kMaxMessageBytes];
        if (queue->pop(buf)) {
            pop_times.push_back(std::chrono::high_resolution_clock::now());
            uint32_t value = 0;
            std::memcpy(&value, buf, sizeof(value));
            if (value != expected) {
                std::cerr << "ERROR: Expected " << expected << " got " << value << "\n";
                break;
            }
            expected++;
        }
    }

    KIWI_HIP_CHECK(hipDeviceSynchronize());

    int warmup = params.get_warmup();
    std::vector<uint64_t> inter_arrival;
    inter_arrival.reserve(params.num_messages - warmup - 1);

    for (int i = warmup + 1; i < params.num_messages; ++i) {
        auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
            pop_times[i] - pop_times[i-1]).count();
        if (dt > 0) inter_arrival.push_back(dt);
    }

    print_stats(inter_arrival, "Inter-arrival time (CPU side)");
}

template<typename QueueType, size_t Bytes>
void run_all(const ExperimentParams& params) {
    using DeviceHandle = typename QueueType::DeviceHandle;
    require_no_scratch(timed_push_kernel<DeviceHandle, Bytes>, "timed_push_kernel");
    require_no_scratch(simple_push_kernel<DeviceHandle, Bytes>, "simple_push_kernel");
    benchmark_push_no_backpressure<QueueType, Bytes>(params);
    benchmark_push_overhead<QueueType, Bytes>(params);
    benchmark_inter_arrival<QueueType, Bytes>(params);
}

int main(int argc, char** argv) {
    CLI::App app{"queue Operation Overhead Benchmark"};

    ExperimentParams params;

    app.add_option("-c,--capacity", params.queue_capacity, "Queue capacity (64-byte lines)")
        ->default_val(1024)->check(CLI::PositiveNumber);
    app.add_option("-m,--messages", params.num_messages, "Number of messages")
        ->default_val(100000)->check(CLI::PositiveNumber);
    app.add_option("-w,--warmup", params.warmup_iterations,
                   "Warmup iterations (default: auto = 10% of messages, max 1000)")
        ->default_val(-1);
    app.add_flag("--mpsc", params.mpsc, "Use the MPSC queue");
    app.add_option("--bytes", params.message_bytes, "Message bytes")
        ->default_val(56)->check(CLI::IsMember(queue_bench::sizes()));
    app.add_flag("-v,--verbose", params.verbose, "Verbose output");

    CLI11_PARSE(app, argc, argv);

    gpu_clock_warmup();

    std::cout << "\n" << (params.mpsc ? "MPSC" : "SPSC") << " Operation Overhead Benchmark\n";
    std::cout << "====================================\n";

    std::cout << "Message bytes: " << params.message_bytes << "\n";
    queue_bench::dispatch(params.message_bytes, [&](auto bytes) {
        if (params.mpsc)
            run_all<MPSCD2HQueue, decltype(bytes)::value>(params);
        else
            run_all<SPSCD2HQueue, decltype(bytes)::value>(params);
    });

    return 0;
}
