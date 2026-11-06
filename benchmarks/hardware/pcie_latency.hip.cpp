// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file pcie_latency.hip.cpp
 * @brief Bare-minimum PCIe/xGMI ping-pong latency benchmark.
 *
 * Compile:  hipcc -O3 -o pcie_latency pcie_latency.hip.cpp --offload-arch=gfx942
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <bench_common.hpp>

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %d: %s at %s:%d\n", \
                e, hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void pingpong_kernel(
    volatile uint64_t* gpu_to_cpu,
    volatile uint64_t* cpu_to_gpu,
    int n
) {
    for (int i = 0; i < n; i++) {
        *gpu_to_cpu = i + 1;
        uint64_t expected = i + 1;
        while (*cpu_to_gpu < expected) {}
    }
}

static void cpu_pingpong(volatile uint64_t* g2c, volatile uint64_t* c2g, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t expected = i + 1;
        while (*g2c < expected) {}
        *c2g = expected;
    }
}

int main(int argc, char** argv) {
    int n = 100000;
    int warmup = 2000000;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) warmup = atoi(argv[2]);

    check_numa_affinity();

    volatile uint64_t *g2c, *c2g;
    HIP_CHECK(hipHostMalloc((void**)&g2c, sizeof(uint64_t), hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc((void**)&c2g, sizeof(uint64_t), hipHostMallocCoherent));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Separate warmup kernel — ramps GPU from deep sleep (131 MHz) to full clock (2100 MHz).
    // Clock stays up for 60+ seconds after kernel completion (within same process),
    // so the gap between warmup and measurement kernel is safe.
    *g2c = 0;
    *c2g = 0;
    pingpong_kernel<<<1, 64, 0, stream>>>(g2c, c2g, warmup);
    cpu_pingpong(g2c, c2g, warmup);
    HIP_CHECK(hipStreamSynchronize(stream));

    // Measured kernel
    *g2c = 0;
    *c2g = 0;
    pingpong_kernel<<<1, 64, 0, stream>>>(g2c, c2g, n);

    auto t0 = std::chrono::steady_clock::now();
    cpu_pingpong(g2c, c2g, n);
    auto t1 = std::chrono::steady_clock::now();

    HIP_CHECK(hipStreamSynchronize(stream));

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("PCIe ping-pong: %d iterations (after %d warmup), %.1f ms total, %.2f us avg round-trip\n",
           n, warmup, total_us / 1000, total_us / n);

    HIP_CHECK(hipHostFree((void*)g2c));
    HIP_CHECK(hipHostFree((void*)c2g));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
