// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_fg_devmem_latency.hip.cpp
 * @brief Measure CPU→GPU write visibility latency through fine-grained device memory
 *        vs host coherent memory. Diagnostic for SPSCH2DQueue ping-pong latency.
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <xmmintrin.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU kernel: poll a volatile flag, measure latency from start of poll to flag match.
// Runs N iterations. Each iteration:
//   1. GPU writes ack (signals CPU to write next flag value)
//   2. GPU polls flag until it matches expected value
//   3. GPU records latency
__global__ void poll_latency_kernel(
    volatile uint64_t* flag,
    volatile uint64_t* ack,
    int num_iterations,
    uint64_t* latencies
) {
    for (int i = 0; i < num_iterations; ++i) {
        uint64_t expected = static_cast<uint64_t>(i + 1);

        // Signal CPU to write next value
        *ack = expected;
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

        // Measure poll latency
        uint64_t t0 = wall_clock64();
        while (*flag != expected) {}
        uint64_t t1 = wall_clock64();

        latencies[i] = t1 - t0;
    }
}

static void print_latency_stats(const std::vector<uint64_t>& raw_ticks, double ticks_to_ns,
                                 const char* label) {
    std::vector<uint64_t> ns_vals;
    for (auto t : raw_ticks) {
        ns_vals.push_back(static_cast<uint64_t>(t * ticks_to_ns));
    }
    std::sort(ns_vals.begin(), ns_vals.end());
    size_t n = ns_vals.size();

    std::cout << label << " (n=" << n << "):\n"
              << "  Min:  " << std::setw(10) << ns_vals.front() << " ns\n"
              << "  p50:  " << std::setw(10) << ns_vals[n/2] << " ns\n"
              << "  p95:  " << std::setw(10) << ns_vals[n*95/100] << " ns\n"
              << "  p99:  " << std::setw(10) << ns_vals[n*99/100] << " ns\n"
              << "  Max:  " << std::setw(10) << ns_vals.back() << " ns\n"
              << "  p50:  " << std::setw(10) << std::fixed << std::setprecision(3)
              << ns_vals[n/2] / 1000.0 << " us\n\n";
}

// Test: CPU writes flag in fine-grained DEVICE memory, GPU polls it
TEST(FgDevmemLatency, DeviceMemPingPong) {
    int wallClkRate;
    HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    constexpr int ITERS = 1000;

    uint64_t* fg_flag = nullptr;
    hipError_t err = hipExtMallocWithFlags(
        (void**)&fg_flag, sizeof(uint64_t), hipDeviceMallocFinegrained);
    if (err != hipSuccess) {
        GTEST_SKIP() << "Fine-grained device memory not supported";
    }

    volatile uint64_t* ack;
    HIP_CHECK(hipHostMalloc((void**)&ack, sizeof(uint64_t), hipHostMallocCoherent));

    uint64_t* d_latencies;
    HIP_CHECK(hipMalloc(&d_latencies, ITERS * sizeof(uint64_t)));

    *fg_flag = 0;
    *ack = 0;

    poll_latency_kernel<<<1, 1>>>(fg_flag, ack, ITERS, d_latencies);

    // CPU: wait for ack, then write flag
    for (int i = 0; i < ITERS; ++i) {
        uint64_t expected_ack = static_cast<uint64_t>(i + 1);
        while (*ack < expected_ack) {}

        // Write flag value (CPU → fine-grained device memory)
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *fg_flag = expected_ack;
        _mm_sfence();
    }

    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> latencies(ITERS);
    HIP_CHECK(hipMemcpy(latencies.data(), d_latencies, ITERS * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    std::cout << "\n=== CPU → Fine-Grained Device Memory Write Visibility ===\n";
    print_latency_stats(latencies, ticks_to_ns, "Device mem poll latency");

    HIP_CHECK(hipFree(d_latencies));
    HIP_CHECK(hipHostFree((void*)ack));
    HIP_CHECK(hipFree(fg_flag));
}

// Test: CPU writes flag in HOST coherent memory, GPU polls it (baseline comparison)
TEST(FgDevmemLatency, HostCoherentPingPong) {
    int wallClkRate;
    HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    constexpr int ITERS = 1000;

    volatile uint64_t* host_flag;
    HIP_CHECK(hipHostMalloc((void**)&host_flag, sizeof(uint64_t), hipHostMallocCoherent));

    volatile uint64_t* ack;
    HIP_CHECK(hipHostMalloc((void**)&ack, sizeof(uint64_t), hipHostMallocCoherent));

    uint64_t* d_latencies;
    HIP_CHECK(hipMalloc(&d_latencies, ITERS * sizeof(uint64_t)));

    *host_flag = 0;
    *ack = 0;

    poll_latency_kernel<<<1, 1>>>(host_flag, ack, ITERS, d_latencies);

    for (int i = 0; i < ITERS; ++i) {
        uint64_t expected_ack = static_cast<uint64_t>(i + 1);
        while (*ack < expected_ack) {}

        std::atomic_thread_fence(std::memory_order_release);
        *host_flag = expected_ack;
    }

    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> latencies(ITERS);
    HIP_CHECK(hipMemcpy(latencies.data(), d_latencies, ITERS * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    std::cout << "\n=== CPU → Host Coherent Memory Write Visibility ===\n";
    print_latency_stats(latencies, ticks_to_ns, "Host mem poll latency");

    HIP_CHECK(hipFree(d_latencies));
    HIP_CHECK(hipHostFree((void*)ack));
    HIP_CHECK(hipHostFree((void*)host_flag));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n========================================\n";
    std::cout << "CPU→GPU WRITE VISIBILITY LATENCY TEST\n";
    std::cout << "Comparing fine-grained device mem vs host coherent mem\n";
    std::cout << "========================================\n";

    return RUN_ALL_TESTS();
}
