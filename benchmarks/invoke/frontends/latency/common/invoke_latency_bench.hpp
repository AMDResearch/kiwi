// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the cross-frontend invoke round-trip latency harness.
//
// GPU invoke -> CPU callback -> CPU ack -> GPU receives ack, one wavefront.
// A copy of the host code of the standalone bench_invoke_latency
// (benchmarks/invoke/bench_latency.hip.cpp) with the kernel launch taken from
// a launcher, so every frontend of the harness runs the same host code: the
// C++ kernel (cpp/) and the Python kernels (flydsl/, ...) through
// kiwi_invoke_latency_bench_ext.
//
//   struct Launcher {
//     // Before anything else, once per run (e.g. a scratch check).
//     template<typename PayloadT> void prepare();
//     // Launch the ping-pong kernel asynchronously on `l.stream`.
//     template<typename PayloadT> void launch(const LatencyLaunch& l);
//   };

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/error.hpp>
#include <bench_common.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kiwi_bench {

// Runs `f` when the scope exits, including by an exception from a launcher.
template<typename F>
struct ScopeExit {
    F f;
    ~ScopeExit() { f(); }
};
template<typename F>
ScopeExit<F> on_scope_exit(F f) { return {std::move(f)}; }

// Payload sizes the benchmark is instantiated for (FixedPayload<N>).
using LatencyPayloadSizes = std::integer_sequence<int, 4, 8, 16, 32, 52, 112, 224, 512>;

template<int... N>
std::vector<int> to_vector(std::integer_sequence<int, N...>) { return {N...}; }

inline std::vector<int> latency_payload_sizes() { return to_vector(LatencyPayloadSizes{}); }

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

struct LatencyParams {
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

// Arguments of one ping-pong kernel launch.
struct LatencyLaunch {
    kiwi::invoke::DeviceContext<>* ctx;  // device pointer
    uint8_t ping_cb;
    volatile uint64_t* ack_flag;
    int num_iterations;
    uint64_t* latencies;
    int warmup_iterations;
    bool latency_stats;
    hipStream_t stream;
};

inline void print_latency_stats(const std::vector<uint64_t>& latencies, const char* title) {
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

template<typename PayloadT, typename Launcher>
void benchmark_ping_pong_latency(const LatencyParams& params, Launcher& launcher) {
    using kiwi::invoke::Context;

    launcher.template prepare<PayloadT>();
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
    volatile uint64_t* ack_flag = nullptr;
    const auto free_ack_flag = on_scope_exit([&] {
        if (!ack_flag) return;
        if (params.device_ack) {
            KIWI_HIP_CHECK(hipFree((void*)ack_flag));
        } else {
            KIWI_HIP_CHECK(hipHostFree((void*)ack_flag));
        }
    });
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
    uint64_t* d_latencies = nullptr;
    const auto free_latencies = on_scope_exit([&] {
        if (d_latencies) KIWI_HIP_CHECK(hipFree(d_latencies));
    });
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

    hipStream_t stream = nullptr;
    const auto destroy_stream = on_scope_exit([&] {
        if (stream) KIWI_HIP_CHECK(hipStreamDestroy(stream));
    });
    KIWI_HIP_CHECK(hipStreamCreate(&stream));

    // Warmup kernel — ramps GPU from deep sleep (131 MHz → 2100 MHz).
    // Clock stays up 60+ seconds after kernel completion within same process,
    // so the gap before the measured kernel is safe.
    {
        Context warmup_ctx(1, params.queue_capacity);
        volatile uint64_t* warmup_ack = nullptr;
        const auto free_warmup_ack = on_scope_exit([&] {
            if (warmup_ack) KIWI_HIP_CHECK(hipHostFree((void*)warmup_ack));
        });
        KIWI_HIP_CHECK(hipHostMalloc((void**)&warmup_ack, sizeof(uint64_t), hipHostMallocCoherent));
        *warmup_ack = 0;

        uint32_t warmup_expected = 0;
        std::function<void(PayloadT)> warmup_func = [&](PayloadT msg) {
            warmup_expected++;
            *warmup_ack = msg.value + 1;
        };
        uint8_t warmup_cb = warmup_ctx.register_callback<PayloadT>(warmup_func);

        launcher.template launch<PayloadT>(LatencyLaunch{
            warmup_ctx.device_context(), warmup_cb, warmup_ack, warmup, d_latencies, warmup,
            false, stream});
        for (int i = 0; i < warmup; ++i) {
            while (!warmup_ctx.progress(0, 1)) {}
        }
        KIWI_HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Measured kernel — fresh queue state
    auto start = std::chrono::high_resolution_clock::now();
    launcher.template launch<PayloadT>(LatencyLaunch{
        ctx.device_context(), ping_cb, ack_flag, measured, d_latencies, 0,
        params.latency_stats, stream});

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
        print_latency_stats(valid, "Round-trip latency (GPU→CPU→GPU)");
    }
}

inline bool is_latency_payload_size(int payload_size) {
    const std::vector<int> sizes = latency_payload_sizes();
    return std::find(sizes.begin(), sizes.end(), payload_size) != sizes.end();
}

// Runs benchmark_ping_pong_latency<FixedPayload<N>> for N == payload_size.
template<typename Launcher, int... N>
bool dispatch_latency(const LatencyParams& params, Launcher& launcher,
                      std::integer_sequence<int, N...>) {
    return ((params.payload_size == N
                 ? (benchmark_ping_pong_latency<FixedPayload<N>>(params, launcher), true)
                 : false) || ...);
}

// Prints the configuration and runs the benchmark for params.payload_size.
template<typename Launcher>
void run_latency_benchmark(const LatencyParams& params, Launcher& launcher,
                           const char* title = "invoke Round-Trip Latency Benchmark") {
    std::cout << title << "\n";
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

    // Dispatch to the correct template instantiation based on runtime payload size.
    if (!dispatch_latency(params, launcher, LatencyPayloadSizes{}))
        throw std::invalid_argument("unsupported payload size");
}

}  // namespace kiwi_bench
