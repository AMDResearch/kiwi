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
 * producer wavefront and messages of 8 to 512 bytes (--bytes), pushed
 * collectively or lane-masked (--push, --lanes; see queue_bench.hip.hpp). One
 * timing is one step (one push call); a mixed step is also timed in its two
 * halves.
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
using queue_bench::Push;

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_messages = 100000;
    int warmup_iterations = -1;
    bool mpsc = false;
    int message_bytes = 56;
    Push push = Push::Collective;
    int lanes = 64;
    bool verbose = false;

    int get_warmup(int steps) const {
        if (warmup_iterations >= 0) return warmup_iterations;
        return std::min(1000, steps / 10);
    }
    int step_messages() const { return queue_bench::step_messages(push, lanes); }
};

__device__ __forceinline__ uint64_t gpu_clock_ticks() {
    return wall_clock64();
}

// Timed push kernel: one step of pattern P per timing. For mixed steps,
// first_times gets the collective half.
template<typename DeviceHandle, Push P, size_t Bytes>
__global__ void timed_push_kernel(DeviceHandle queue, int num_steps, int lanes,
                                  uint64_t* push_times, uint64_t* first_times, int warmup_steps) {
    if constexpr (P == Push::Collective) {
        // The collective loop as before lane-masked pushes, so its code (and
        // results) stay comparable.
        queue_bench::Message<Bytes> msg{};
        for (int i = 0; i < num_steps; ++i) {
            uint64_t t0 = gpu_clock_ticks();

            msg.value = static_cast<uint32_t>(i);
            while (!queue.push_reg(msg)) {}

            uint64_t t1 = gpu_clock_ticks();

            if (__lane_id() == 0 && i >= warmup_steps) {
                push_times[i - warmup_steps] = t1 - t0;
            }
        }
        return;
    }
    const int msgs = queue_bench::step_messages(P, lanes);
    for (int i = 0; i < num_steps; ++i) {
        uint64_t t_mid = 0;
        uint64_t t0 = gpu_clock_ticks();

        queue_bench::push_step<P, Bytes>(queue, static_cast<uint32_t>(i * msgs), lanes, &t_mid);

        uint64_t t1 = gpu_clock_ticks();

        if (__lane_id() == 0 && i >= warmup_steps) {
            push_times[i - warmup_steps] = t1 - t0;
            if (P == Push::Mixed) first_times[i - warmup_steps] = t_mid - t0;
        }
    }
}

// Simple push kernel (no timing)
template<typename DeviceHandle, Push P, size_t Bytes>
__global__ void simple_push_kernel(DeviceHandle queue, int num_steps, int lanes,
                                   volatile int* done_flag) {
    if constexpr (P == Push::Collective) {
        // The collective loop as before lane-masked pushes.
        queue_bench::Message<Bytes> msg{};
        for (int i = 0; i < num_steps; ++i) {
            msg.value = static_cast<uint32_t>(i);
            while (!queue.push_reg(msg)) {}
        }
    } else {
        const int msgs = queue_bench::step_messages(P, lanes);
        for (int i = 0; i < num_steps; ++i) {
            queue_bench::push_step<P, Bytes>(queue, static_cast<uint32_t>(i * msgs), lanes);
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

// Prints a mixed step's two halves: the collective push (first_times, in
// ticks) and the lane-masked push (the rest of the step). push_times are the
// steps in ns; unset (zero) steps are skipped.
void print_mixed_halves(const std::vector<uint64_t>& push_times, const uint64_t* d_first_times,
                        double ticks_to_ns) {
    std::vector<uint64_t> first(push_times.size());
    KIWI_HIP_CHECK(hipMemcpy(first.data(), d_first_times, first.size() * sizeof(uint64_t),
                             hipMemcpyDeviceToHost));
    std::vector<uint64_t> collective, lane_masked;
    for (size_t i = 0; i < push_times.size(); ++i) {
        if (push_times[i] == 0) continue;
        const auto t = static_cast<uint64_t>(first[i] * ticks_to_ns);
        collective.push_back(t);
        lane_masked.push_back(push_times[i] - t);
    }
    std::cout << "\n";
    print_stats(collective, "  mixed: collective push");
    std::cout << "\n";
    print_stats(lane_masked, "  mixed: lane-masked push");
}

// Measure GPU push overhead with active CPU consumer
template<typename QueueType, Push P, size_t Bytes>
void benchmark_push_overhead(const ExperimentParams& params) {
    const int msgs = params.step_messages();
    const int steps = params.num_messages / msgs;
    const uint64_t total = static_cast<uint64_t>(steps) * msgs;

    std::cout << "\n=== GPU Push Overhead (Active Consumer) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Messages: " << total << "\n\n";

    int wallClkRate;
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    auto queue = std::make_unique<QueueType>(params.queue_capacity);

    int warmup_steps = params.get_warmup(steps);
    int measured_steps = steps - warmup_steps;

    uint64_t *d_push_times, *d_first_times;
    KIWI_HIP_CHECK(hipMalloc(&d_push_times, measured_steps * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_push_times, 0, measured_steps * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMalloc(&d_first_times, measured_steps * sizeof(uint64_t)));

    hipStream_t stream;
    KIWI_HIP_CHECK(hipStreamCreate(&stream));
    timed_push_kernel<typename QueueType::DeviceHandle, P, Bytes><<<1, 64, 0, stream>>>(
        queue->device_handle(), steps, params.lanes, d_push_times, d_first_times, warmup_steps);

    uint32_t expected = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (expected < static_cast<uint32_t>(total)) {
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

    std::vector<uint64_t> push_times(measured_steps);
    KIWI_HIP_CHECK(hipMemcpy(push_times.data(), d_push_times,
                             measured_steps * sizeof(uint64_t), hipMemcpyDeviceToHost));

    for (auto& t : push_times) {
        t = static_cast<uint64_t>(t * ticks_to_ns);
    }

    auto total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double throughput = (double)total / total_time_us * 1e6;

    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << throughput / 1e6 << " M msgs/sec\n";
    std::cout << "Total time: " << total_time_us / 1000.0 << " ms\n\n";

    std::vector<uint64_t> valid_times;
    valid_times.reserve(measured_steps);
    for (auto t : push_times) {
        if (t > 0) valid_times.push_back(t);
    }

    print_stats(valid_times, "GPU push() overhead");
    if (P == Push::Mixed) print_mixed_halves(push_times, d_first_times, ticks_to_ns);

    KIWI_HIP_CHECK(hipFree(d_push_times));
    KIWI_HIP_CHECK(hipFree(d_first_times));
    KIWI_HIP_CHECK(hipStreamDestroy(stream));
}

// Measure GPU push overhead with empty queue (no backpressure)
template<typename QueueType, Push P, size_t Bytes>
void benchmark_push_no_backpressure(const ExperimentParams& params) {
    std::cout << "\n=== GPU Push Overhead (Empty Queue, No Backpressure) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n\n";

    int wallClkRate;
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    auto queue = std::make_unique<QueueType>(params.queue_capacity);

    // Steps whose messages fit without a consumer.
    int num_steps = (static_cast<int>(params.queue_capacity / message_lines(Bytes)) - 1) /
                    params.step_messages();
    int warmup_steps = std::min(100, num_steps / 10);
    int measured_steps = num_steps - warmup_steps;
    if (measured_steps <= 0) {
        std::cerr << "ERROR: the ring holds no measured step; raise --capacity\n";
        exit(1);
    }

    uint64_t *d_push_times, *d_first_times;
    KIWI_HIP_CHECK(hipMalloc(&d_push_times, measured_steps * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_push_times, 0, measured_steps * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMalloc(&d_first_times, measured_steps * sizeof(uint64_t)));

    timed_push_kernel<typename QueueType::DeviceHandle, P, Bytes><<<1, 64>>>(
        queue->device_handle(), num_steps, params.lanes, d_push_times, d_first_times, warmup_steps);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> push_times(measured_steps);
    KIWI_HIP_CHECK(hipMemcpy(push_times.data(), d_push_times,
                             measured_steps * sizeof(uint64_t), hipMemcpyDeviceToHost));

    for (auto& t : push_times) {
        t = static_cast<uint64_t>(t * ticks_to_ns);
    }

    std::vector<uint64_t> valid_times;
    for (auto t : push_times) {
        if (t > 0) valid_times.push_back(t);
    }

    print_stats(valid_times, "GPU push() overhead (no backpressure)");
    if (P == Push::Mixed) print_mixed_halves(push_times, d_first_times, ticks_to_ns);

    KIWI_HIP_CHECK(hipFree(d_push_times));
    KIWI_HIP_CHECK(hipFree(d_first_times));
}

// Measure inter-arrival time on CPU side
template<typename QueueType, Push P, size_t Bytes>
void benchmark_inter_arrival(const ExperimentParams& params) {
    const int msgs = params.step_messages();
    const int steps = params.num_messages / msgs;
    const uint64_t total = static_cast<uint64_t>(steps) * msgs;

    std::cout << "\n=== Inter-arrival Time (CPU side) ===\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Messages: " << total << "\n\n";

    auto queue = std::make_unique<QueueType>(params.queue_capacity);

    simple_push_kernel<typename QueueType::DeviceHandle, P, Bytes><<<1, 64>>>(
        queue->device_handle(), steps, params.lanes, nullptr);

    std::vector<std::chrono::high_resolution_clock::time_point> pop_times;
    pop_times.reserve(total);

    uint32_t expected = 0;
    while (expected < static_cast<uint32_t>(total)) {
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

    int warmup = params.get_warmup(static_cast<int>(total));
    std::vector<uint64_t> inter_arrival;
    inter_arrival.reserve(total - warmup - 1);

    for (size_t i = warmup + 1; i < pop_times.size(); ++i) {
        auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
            pop_times[i] - pop_times[i-1]).count();
        if (dt > 0) inter_arrival.push_back(dt);
    }

    print_stats(inter_arrival, "Inter-arrival time (CPU side)");
}

template<typename QueueType, Push P, size_t Bytes>
void run_all(const ExperimentParams& params) {
    using DeviceHandle = typename QueueType::DeviceHandle;
    require_no_scratch(timed_push_kernel<DeviceHandle, P, Bytes>, "timed_push_kernel");
    require_no_scratch(simple_push_kernel<DeviceHandle, P, Bytes>, "simple_push_kernel");
    benchmark_push_no_backpressure<QueueType, P, Bytes>(params);
    benchmark_push_overhead<QueueType, P, Bytes>(params);
    benchmark_inter_arrival<QueueType, P, Bytes>(params);
}

int main(int argc, char** argv) {
    CLI::App app{"queue Operation Overhead Benchmark"};

    ExperimentParams params;

    app.add_option("-c,--capacity", params.queue_capacity, "Queue capacity (64-byte lines)")
        ->default_val(1024)->check(CLI::PositiveNumber);
    app.add_option("-m,--messages", params.num_messages, "Number of messages")
        ->default_val(100000)->check(CLI::PositiveNumber);
    app.add_option("-w,--warmup", params.warmup_iterations,
                   "Warmup steps (default: auto = 10% of steps, max 1000)")
        ->default_val(-1);
    app.add_flag("--mpsc", params.mpsc, "Use the MPSC queue");
    app.add_option("--bytes", params.message_bytes, "Message bytes")
        ->default_val(56)->check(CLI::IsMember(queue_bench::sizes()));
    app.add_option("--push", params.push, "collective, lane-masked or mixed")
        ->transform(CLI::CheckedTransformer(queue_bench::push_names()));
    app.add_option("--lanes", params.lanes, "Active lanes per lane-masked push call")
        ->default_val(64)->check(CLI::Range(1, 64));
    app.add_flag("-v,--verbose", params.verbose, "Verbose output");

    CLI11_PARSE(app, argc, argv);
    if (!queue_bench::lane_masked_call_fits(params.push, params.lanes, params.message_bytes,
                                            params.queue_capacity)) {
        return 1;
    }

    gpu_clock_warmup();

    std::cout << "\n" << (params.mpsc ? "MPSC" : "SPSC") << " Operation Overhead Benchmark\n";
    std::cout << "====================================\n";

    std::cout << "Message bytes: " << params.message_bytes << "\n";
    std::cout << "Push: " << queue_bench::push_name(params.push);
    if (params.push != Push::Collective) std::cout << ", " << params.lanes << " lanes";
    std::cout << "\n";
    queue_bench::dispatch(params.message_bytes, [&](auto bytes) {
        queue_bench::dispatch_push(params.push, [&](auto push) {
            constexpr Push P = decltype(push)::value;
            constexpr size_t B = decltype(bytes)::value;
            if (params.mpsc)
                run_all<MPSCD2HQueue, P, B>(params);
            else
                run_all<SPSCD2HQueue, P, B>(params);
        });
    });

    return 0;
}
