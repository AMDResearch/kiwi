// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_latency.hip.cpp
 * @brief Measure round-trip GPU-CPU-GPU latency using ping-pong pattern with invoke
 *
 * Measures full round-trip latency: GPU invoke → CPU callback → CPU ack → GPU receives ack.
 * Reports percentile statistics for latency distribution.
 */

#include <kiwi/invoke/invoke.hpp>
#include <hip/hip_runtime.h>
#include <bench_common.hpp>
#include <CLI/CLI.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>

using namespace kiwi::invoke;

// Fixed-size payload template. First 4 bytes carry the validation value;
// remaining bytes are zero padding to bulk up the message. Up to 32 bytes
// fit in one 64-byte ring line; each further 32 bytes take one more line.
template<size_t N>
struct __attribute__((packed)) FixedPayload {
    static_assert(N >= 4, "Minimum payload size is 4 bytes (need room for uint32_t value)");
    uint32_t value;
    uint8_t  padding[N - 4];
};

template<>
struct __attribute__((packed)) FixedPayload<4> {
    uint32_t value;
};

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_iterations = 1000000;
    // GPU needs ~3s to ramp from deep sleep (131 MHz) to full clock (2100 MHz).
    // At ~2 us/iteration steady state, 2M iterations ≈ 4s — enough headroom.
    // Warmup runs in a separate kernel, so this doesn't reduce measured iterations.
    int warmup_iterations = 2000000;
    int payload_size = 4;
    bool latency_stats = false;
    bool verbose = false;
    bool device_ack = false;     // Use fine-grained device memory for ack_flag
};

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

template<typename PayloadT>
void benchmark_ping_pong_latency(const ExperimentParams& params) {
    require_no_scratch(ping_pong_latency_handle_kernel<PayloadT>,
                       "ping_pong_latency_handle_kernel");
    // Query wall clock rate for GPU tick → nanosecond conversion
    int wallClkRate;  // in kHz
    KIWI_HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    // Create invoke context with single endpoint
    Context ctx(1, params.queue_capacity);

    // Calculate warmup iterations
    int warmup = params.warmup_iterations;
    int measured = params.num_iterations;

    // Allocate ack flag (CPU→GPU communication)
    volatile uint64_t* ack_flag;
    if (params.device_ack) {
        // Fine-grained device memory: resides in GPU VRAM but CPU-accessible
        // CPU writes via fine-grained coherency path;
        // combine with --uncached-ack to bypass GPU L2 on reads
        KIWI_HIP_CHECK(hipExtMallocWithFlags((void**)&ack_flag, sizeof(uint64_t),
                                             //  hipDeviceMallocFinegrained));
                                              hipDeviceMallocUncached));
        KIWI_HIP_CHECK(hipMemset((void*)ack_flag, 0, sizeof(uint64_t)));
    } else {
        KIWI_HIP_CHECK(hipHostMalloc((void**)&ack_flag, sizeof(uint64_t), hipHostMallocCoherent));
        *ack_flag = 0;  // Initialize
    }

    // Allocate device memory for latency results
    uint64_t* d_latencies;
    KIWI_HIP_CHECK(hipMalloc(&d_latencies, measured * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemset(d_latencies, 0, measured * sizeof(uint64_t)));

    // Register ping callback — writes ack directly from callback handler
    uint32_t expected_val = 0;
    std::function<void(PayloadT)> ping_func = [&](PayloadT msg) {
        if (msg.value != expected_val) {
            fprintf(stderr, "MISMATCH: callback got %u, expected %u\n", msg.value, expected_val);
            abort();
        }
        expected_val++;
        // Delay ack write to expose race condition if GPU poll loop was eliminated
        *ack_flag = msg.value + 1;
    };
    uint8_t ping_cb = ctx.register_callback<PayloadT>(ping_func);

    hipStream_t stream;
    KIWI_HIP_CHECK(hipStreamCreate(&stream));

    // Warmup kernel — ramps GPU from deep sleep (131 MHz → 2100 MHz).
    // Clock stays up 60+ seconds after kernel completion within same process,
    // so the gap before the measured kernel is safe.
    {
        Context warmup_ctx(1, params.queue_capacity);
        volatile uint64_t* warmup_ack;
        KIWI_HIP_CHECK(hipHostMalloc((void**)&warmup_ack, sizeof(uint64_t), hipHostMallocCoherent));
        *warmup_ack = 0;

        uint32_t warmup_expected = 0;
        std::function<void(PayloadT)> warmup_func = [&](PayloadT msg) {
            warmup_expected++;
            *warmup_ack = msg.value + 1;
        };
        uint8_t warmup_cb = warmup_ctx.register_callback<PayloadT>(warmup_func);

        ping_pong_latency_handle_kernel<PayloadT><<<1, 64, 0, stream>>>(
            warmup_ctx.device_context(), warmup_cb, warmup_ack, warmup, d_latencies, warmup, false
        );
        for (int i = 0; i < warmup; ++i) {
            while (!warmup_ctx.progress(0, 1)) {}
        }
        KIWI_HIP_CHECK(hipStreamSynchronize(stream));
        KIWI_HIP_CHECK(hipHostFree((void*)warmup_ack));
    }

    // Measured kernel — fresh queue state
    auto start = std::chrono::high_resolution_clock::now();
    ping_pong_latency_handle_kernel<PayloadT><<<1, 64, 0, stream>>>(
        ctx.device_context(), ping_cb, ack_flag, measured, d_latencies, 0,
        params.latency_stats
    );

    for (int i = 0; i < measured; ++i) {
        while (!ctx.progress(0, 1)) {}
        if (*ack_flag != static_cast<uint64_t>(i + 1)) {
            fprintf(stderr, "ack_flag mismatch: got %lu, expected %lu (i=%d)\n",
                    (unsigned long)*ack_flag, (unsigned long)(i + 1), i);
            abort();
        }
    }

    KIWI_HIP_CHECK(hipStreamSynchronize(stream));

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Total time: " << total_time_ms << " ms\n"
              << "Throughput: " << (measured / (total_time_ms / 1000.0)) << " ops/sec\n"
              << "Average round-trip latency: " << (total_time_ms * 1e6 / measured) << " ns\n\n";

    if (params.latency_stats) {
        // Copy results to host and analyze
        std::vector<uint64_t> latencies(measured);
        KIWI_HIP_CHECK(hipMemcpy(latencies.data(), d_latencies, measured * sizeof(uint64_t),
                hipMemcpyDeviceToHost));

        // Convert raw GPU ticks to nanoseconds
        for (auto& t : latencies) {
            t = static_cast<uint64_t>(t * ticks_to_ns);
        }

        // Filter zeros and print statistics
        std::vector<uint64_t> valid;
        for (auto t : latencies) {
            if (t > 0) valid.push_back(t);
        }
        print_stats(valid, "Round-trip latency (GPU→CPU→GPU)");
    }

    // Cleanup
    KIWI_HIP_CHECK(hipFree(d_latencies));
    if (params.device_ack) {
        KIWI_HIP_CHECK(hipFree((void*)ack_flag));
    } else {
        KIWI_HIP_CHECK(hipHostFree((void*)ack_flag));
    }
    KIWI_HIP_CHECK(hipStreamDestroy(stream));
}

// Dispatch to the correct template instantiation based on runtime payload size.
void run_benchmark(const ExperimentParams& params) {
    switch (params.payload_size) {
        case 4:   benchmark_ping_pong_latency<FixedPayload<4>>(params);   break;
        case 8:   benchmark_ping_pong_latency<FixedPayload<8>>(params);   break;
        case 16:  benchmark_ping_pong_latency<FixedPayload<16>>(params);  break;
        case 32:  benchmark_ping_pong_latency<FixedPayload<32>>(params);  break;
        case 52:  benchmark_ping_pong_latency<FixedPayload<52>>(params);  break;
        case 112: benchmark_ping_pong_latency<FixedPayload<112>>(params); break;
        case 224: benchmark_ping_pong_latency<FixedPayload<224>>(params); break;
        case 512: benchmark_ping_pong_latency<FixedPayload<512>>(params); break;
        default:
            fprintf(stderr, "Unsupported payload size: %d. Use one of: 4, 8, 16, 32, 52, 112, 224, 512\n",
                    params.payload_size);
            exit(1);
    }
}

int main(int argc, char** argv) {
    CLI::App app{"invoke Round-Trip Latency Benchmark"};

    ExperimentParams params;

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
        ->check(CLI::IsMember({4, 8, 16, 32, 52, 112, 224, 512}));

    app.add_flag("-s,--stats", params.latency_stats,
                 "Collect and report detailed latency statistics (percentiles)");

    app.add_flag("-v,--verbose", params.verbose,
                 "Verbose output");

    app.add_flag("--device-ack", params.device_ack,
                 "Use fine-grained device memory for ack_flag (GPU VRAM, CPU-writable)");

    // Parse command line
    CLI11_PARSE(app, argc, argv);

    // Print configuration
    std::cout << "invoke Round-Trip Latency Benchmark\n";
    std::cout << "====================================\n";
    std::cout << "Queue capacity: " << params.queue_capacity << "\n";
    std::cout << "Iterations: " << params.num_iterations << " ("
              << params.warmup_iterations << " warmup)\n";
    std::cout << "Latency stats: " << (params.latency_stats ? "enabled" : "disabled") << "\n";
    std::cout << "Device ack mode: " << (params.device_ack ? "YES (fine-grained VRAM)" : "no (host coherent)") << "\n";
    std::cout << "Payload size: " << params.payload_size << " bytes ("
              << kiwi::queue::message_lines(params.payload_size) << " ring lines)\n";
    std::cout << "Target: < 2 us average latency\n\n";

    check_numa_affinity();
    run_benchmark(params);

    return 0;
}
