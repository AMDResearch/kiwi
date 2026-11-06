// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_cache_latency.hip.cpp
 * @brief Measure GPU cache hit latency using pointer chasing.
 *
 * Pointer chasing forces a serial dependency chain: each load depends on the
 * previous result, so the compiler cannot hide latency via ILP. We use a
 * random permutation so the compiler cannot predict the access pattern.
 *
 * Two variants tested:
 *   1. Scalar path (s_load) — compiler promotes uniform address to SGPRs
 *   2. Vector path (global_load) — forced via inline asm, goes through vL1D
 *
 * Timing: host-side std::chrono around kernel launch + hipDeviceSynchronize.
 * Each kernel run targets ~1 second of wall time for accuracy.
 */

#include <hip/hip_runtime.h>
#include <kiwi/queue/error.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <random>

using Clock = std::chrono::high_resolution_clock;

// Build a random single-cycle permutation of [0, n).
static void build_random_cycle(uint64_t* chain, int n) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(order.begin(), order.end(), rng);
    for (int i = 0; i < n; ++i) {
        chain[order[i]] = order[(i + 1) % n];
    }
}

// ---------------------------------------------------------------------------
// Variant 1: scalar path (s_load_dwordx2)
// The compiler sees a uniform address chain from lane 0 and uses SMEM.
// ---------------------------------------------------------------------------
__global__ void kernel_scalar(
    const uint64_t* __restrict__ chain,
    int64_t iters,
    uint64_t* sink
) {
    if (__lane_id() != 0) return;

    uint64_t idx = 0;
    for (int64_t i = 0; i < iters; ++i)
        idx = chain[idx];

    *sink = idx;
}

// ---------------------------------------------------------------------------
// Variant 2: vector path (global_load_dwordx2)
// Force the index into VGPRs so the compiler must use VMEM loads through vL1D.
// ---------------------------------------------------------------------------
__global__ void kernel_vector(
    const uint64_t* __restrict__ chain,
    int64_t iters,
    uint64_t* sink
) {
    if (__lane_id() != 0) return;

    uint64_t idx = 0;
    asm volatile("" : "+v"(idx));

    for (int64_t i = 0; i < iters; ++i) {
        idx = chain[idx];
        asm volatile("" : "+v"(idx));
    }

    *sink = idx;
}

// ---------------------------------------------------------------------------

using KernelFn = void(*)(const uint64_t*, int64_t, uint64_t*);

static void run(const char* label, int chain_len, KernelFn kernel) {
    std::vector<uint64_t> h_chain(chain_len);
    build_random_cycle(h_chain.data(), chain_len);

    uint64_t* d_chain;
    KIWI_HIP_CHECK(hipMalloc(&d_chain, chain_len * sizeof(uint64_t)));
    KIWI_HIP_CHECK(hipMemcpy(d_chain, h_chain.data(), chain_len * sizeof(uint64_t), hipMemcpyHostToDevice));

    uint64_t* d_sink;
    KIWI_HIP_CHECK(hipMalloc(&d_sink, sizeof(uint64_t)));

    // Calibration: run a short burst to estimate per-load time, then scale to ~1s.
    int64_t calibration_iters = 1000000;

    // Warmup run (also primes caches)
    kernel<<<1, 64>>>(d_chain, calibration_iters, d_sink);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Calibration run
    auto cal_start = Clock::now();
    kernel<<<1, 64>>>(d_chain, calibration_iters, d_sink);
    KIWI_HIP_CHECK(hipDeviceSynchronize());
    auto cal_end = Clock::now();

    double cal_sec = std::chrono::duration<double>(cal_end - cal_start).count();
    // Target ~1 second
    int64_t target_iters = static_cast<int64_t>(calibration_iters / cal_sec);

    // Measured run
    auto t0 = Clock::now();
    kernel<<<1, 64>>>(d_chain, target_iters, d_sink);
    KIWI_HIP_CHECK(hipDeviceSynchronize());
    auto t1 = Clock::now();

    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double per_load_ns = (elapsed_sec * 1e9) / target_iters;

    std::cout << std::left << std::setw(36) << label
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(7) << per_load_ns << " ns"
              << "  (" << std::setprecision(3) << per_load_ns / 1000.0 << " us)"
              << "  ~" << std::setprecision(0) << per_load_ns * 2.1 << " cyc"
              << "  [" << std::setprecision(2) << elapsed_sec << "s]\n";

    KIWI_HIP_CHECK(hipFree(d_chain));
    KIWI_HIP_CHECK(hipFree(d_sink));
}

int main() {
    std::cout << "=== GPU Cache Latency — pointer chasing ===\n";
    std::cout << "Timer: host-side chrono (each run ~1s)\n";

    struct { const char* name; int n; } sizes[] = {
        {"128 B",   16},
        {"1 KB",   128},
        {"8 KB",  1024},
        {"16 KB", 2048},
        {"32 KB", 4096},
        {"64 KB",  8192},
        {"256 KB", 32768},
        {"4 MB", 524288},
        {"16 MB", 2097152},
        {"64 MB", 8388608},
        {"256 MB", 33554432},
    };

    std::cout << "\n--- Scalar path (s_load → scalar data cache) ---\n";
    for (auto& s : sizes) {
        char label[64];
        snprintf(label, sizeof(label), "  %s", s.name);
        run(label, s.n, kernel_scalar);
    }

    std::cout << "\n--- Vector path (global_load → vL1D) ---\n";
    for (auto& s : sizes) {
        char label[64];
        snprintf(label, sizeof(label), "  %s", s.name);
        run(label, s.n, kernel_vector);
    }

    return 0;
}
