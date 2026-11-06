// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_fine_grained_device_mem.hip.cpp
 * @brief Test whether CPU can write to fine-grained GPU device memory
 *        and GPU kernels can observe those writes.
 *
 * On MI300X, hipExtMallocWithFlags with hipDeviceMallocFinegrained allocates
 * device memory with fine-grained coherence. The CPU can directly write to
 * this memory and the GPU should observe the writes without explicit copies.
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Test 1: Basic CPU write → GPU read
// CPU writes a known pattern to fine-grained device memory,
// then GPU kernel reads it and stores the result.
// ============================================================================

__global__ void read_single_value_kernel(const uint64_t* src, uint64_t* dst) {
    *dst = *src;
}

TEST(FineGrainedDeviceMem, CPUWriteGPURead) {
    uint64_t* fg_dev_ptr = nullptr;
    hipError_t err = hipExtMallocWithFlags(
        (void**)&fg_dev_ptr, sizeof(uint64_t), hipDeviceMallocFinegrained);

    if (err != hipSuccess) {
        GTEST_SKIP() << "hipExtMallocWithFlags(hipDeviceMallocFinegrained) not supported: "
                     << hipGetErrorString(err);
    }

    // Regular device memory for GPU to store its read result
    uint64_t* result_dev;
    HIP_CHECK(hipMalloc(&result_dev, sizeof(uint64_t)));

    constexpr uint64_t MAGIC = 0xDEADBEEFCAFEBABEULL;

    // CPU directly writes to fine-grained device memory
    *fg_dev_ptr = MAGIC;

    // GPU kernel reads from the same address
    read_single_value_kernel<<<1, 1>>>(fg_dev_ptr, result_dev);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy result back to host
    uint64_t result_host = 0;
    HIP_CHECK(hipMemcpy(&result_host, result_dev, sizeof(uint64_t), hipMemcpyDeviceToHost));

    EXPECT_EQ(result_host, MAGIC)
        << "GPU did not observe CPU write to fine-grained device memory";

    HIP_CHECK(hipFree(result_dev));
    HIP_CHECK(hipFree(fg_dev_ptr));
}

// ============================================================================
// Test 2: Multiple values - CPU writes array, GPU reads all elements
// ============================================================================

__global__ void read_array_kernel(const uint64_t* src, uint64_t* dst, int count) {
    int tid = threadIdx.x;
    if (tid < count) {
        dst[tid] = src[tid];
    }
}

TEST(FineGrainedDeviceMem, CPUWriteArrayGPURead) {
    constexpr int COUNT = 64;

    uint64_t* fg_dev_ptr = nullptr;
    hipError_t err = hipExtMallocWithFlags(
        (void**)&fg_dev_ptr, COUNT * sizeof(uint64_t), hipDeviceMallocFinegrained);

    if (err != hipSuccess) {
        GTEST_SKIP() << "hipExtMallocWithFlags(hipDeviceMallocFinegrained) not supported: "
                     << hipGetErrorString(err);
    }

    uint64_t* result_dev;
    HIP_CHECK(hipMalloc(&result_dev, COUNT * sizeof(uint64_t)));

    // CPU writes distinct values to each slot
    for (int i = 0; i < COUNT; ++i) {
        fg_dev_ptr[i] = 0xA000000000000000ULL | (uint64_t)i;
    }

    // GPU reads all values
    read_array_kernel<<<1, COUNT>>>(fg_dev_ptr, result_dev, COUNT);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint64_t> results(COUNT);
    HIP_CHECK(hipMemcpy(results.data(), result_dev, COUNT * sizeof(uint64_t), hipMemcpyDeviceToHost));

    for (int i = 0; i < COUNT; ++i) {
        EXPECT_EQ(results[i], 0xA000000000000000ULL | (uint64_t)i)
            << "Mismatch at index " << i;
    }

    HIP_CHECK(hipFree(result_dev));
    HIP_CHECK(hipFree(fg_dev_ptr));
}

// ============================================================================
// Test 3: CPU writes WHILE GPU kernel is polling (concurrent visibility)
// GPU kernel spins on a flag, CPU sets it. Tests live coherence.
// ============================================================================

__global__ void poll_for_flag_kernel(volatile uint64_t* flag,
                                     uint64_t expected,
                                     uint64_t* payload_out,
                                     volatile uint64_t* payload_addr,
                                     int* success) {
    // Spin until we see the flag value
    uint64_t deadline = wall_clock64() + 1000000000ULL; // ~1 second at ~1GHz
    while (*flag != expected) {
        if (wall_clock64() > deadline) {
            *success = 0;
            return;
        }
    }

    // Flag seen! Read the payload too.
    *payload_out = *payload_addr;
    *success = 1;
}

TEST(FineGrainedDeviceMem, ConcurrentCPUWriteGPUPoll) {
    // Allocate two fine-grained device memory locations: flag and payload
    uint64_t* fg_flag = nullptr;
    uint64_t* fg_payload = nullptr;
    hipError_t err;

    err = hipExtMallocWithFlags((void**)&fg_flag, sizeof(uint64_t), hipDeviceMallocFinegrained);
    if (err != hipSuccess) {
        GTEST_SKIP() << "hipExtMallocWithFlags(hipDeviceMallocFinegrained) not supported: "
                     << hipGetErrorString(err);
    }
    err = hipExtMallocWithFlags((void**)&fg_payload, sizeof(uint64_t), hipDeviceMallocFinegrained);
    if (err != hipSuccess) {
        HIP_CHECK(hipFree(fg_flag));
        GTEST_SKIP() << "hipExtMallocWithFlags(hipDeviceMallocFinegrained) not supported";
    }

    // Result storage on device
    uint64_t* result_payload_dev;
    int* result_success_dev;
    HIP_CHECK(hipMalloc(&result_payload_dev, sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&result_success_dev, sizeof(int)));

    // Initialize: flag=0, payload=0
    *fg_flag = 0;
    *fg_payload = 0;

    constexpr uint64_t FLAG_VALUE = 0x1234567890ABCDEFULL;
    constexpr uint64_t PAYLOAD_VALUE = 0xFEDCBA0987654321ULL;

    // Launch GPU kernel that polls the flag
    poll_for_flag_kernel<<<1, 1>>>(fg_flag, FLAG_VALUE,
                                    result_payload_dev, fg_payload,
                                    result_success_dev);

    // Small delay to let the kernel start polling
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // CPU writes payload first, then flag (ordering matters)
    *fg_payload = PAYLOAD_VALUE;
    std::atomic_thread_fence(std::memory_order_release);
    *fg_flag = FLAG_VALUE;

    HIP_CHECK(hipDeviceSynchronize());

    // Check results
    int success_host = 0;
    uint64_t payload_host = 0;
    HIP_CHECK(hipMemcpy(&success_host, result_success_dev, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&payload_host, result_payload_dev, sizeof(uint64_t), hipMemcpyDeviceToHost));

    EXPECT_EQ(success_host, 1) << "GPU kernel timed out waiting for CPU flag write";
    if (success_host) {
        EXPECT_EQ(payload_host, PAYLOAD_VALUE)
            << "GPU saw the flag but read wrong payload value";
    }

    HIP_CHECK(hipFree(result_success_dev));
    HIP_CHECK(hipFree(result_payload_dev));
    HIP_CHECK(hipFree(fg_payload));
    HIP_CHECK(hipFree(fg_flag));
}

// ============================================================================
// Test 4: Read latency from fine-grained device memory (GPU reading CPU writes)
// Compare with hipHostMalloc coherent memory for reference.
// ============================================================================

__global__ void measure_read_latency_kernel(volatile uint64_t* addr, int iterations,
                                             long long* latencies) {
    for (int i = 0; i < iterations; ++i) {
        long long start = wall_clock64();
        __builtin_amdgcn_s_waitcnt(0xC07F);
        uint64_t val = *addr;
        __builtin_amdgcn_s_waitcnt(0xF70);
        long long end = wall_clock64();
        __builtin_amdgcn_s_waitcnt(0xC07F);
        latencies[i] = end - start;
        (void)val;
    }
}

TEST(FineGrainedDeviceMem, ReadLatencyComparison) {
    constexpr int ITERATIONS = 100;

    // Fine-grained device memory
    uint64_t* fg_dev_ptr = nullptr;
    hipError_t err = hipExtMallocWithFlags(
        (void**)&fg_dev_ptr, sizeof(uint64_t), hipDeviceMallocFinegrained);
    if (err != hipSuccess) {
        GTEST_SKIP() << "hipExtMallocWithFlags(hipDeviceMallocFinegrained) not supported: "
                     << hipGetErrorString(err);
    }

    // Host coherent memory (for comparison)
    uint64_t* host_coherent_ptr;
    HIP_CHECK(hipHostMalloc(&host_coherent_ptr, sizeof(uint64_t), hipHostMallocCoherent));

    // Regular device memory (for comparison)
    uint64_t* regular_dev_ptr;
    HIP_CHECK(hipMalloc(&regular_dev_ptr, sizeof(uint64_t)));
    uint64_t init_val = 0xAABBCCDDEEFF0011ULL;
    HIP_CHECK(hipMemcpy(regular_dev_ptr, &init_val, sizeof(uint64_t), hipMemcpyHostToDevice));

    // Latency buffers
    long long* lat_fg_dev;
    long long* lat_host_dev;
    long long* lat_reg_dev;
    HIP_CHECK(hipMalloc(&lat_fg_dev, ITERATIONS * sizeof(long long)));
    HIP_CHECK(hipMalloc(&lat_host_dev, ITERATIONS * sizeof(long long)));
    HIP_CHECK(hipMalloc(&lat_reg_dev, ITERATIONS * sizeof(long long)));

    // Write values via CPU
    *fg_dev_ptr = init_val;
    *host_coherent_ptr = init_val;

    // Measure reads from GPU
    measure_read_latency_kernel<<<1, 1>>>(fg_dev_ptr, ITERATIONS, lat_fg_dev);
    HIP_CHECK(hipDeviceSynchronize());

    measure_read_latency_kernel<<<1, 1>>>(host_coherent_ptr, ITERATIONS, lat_host_dev);
    HIP_CHECK(hipDeviceSynchronize());

    measure_read_latency_kernel<<<1, 1>>>(regular_dev_ptr, ITERATIONS, lat_reg_dev);
    HIP_CHECK(hipDeviceSynchronize());

    // Copy results
    std::vector<long long> lat_fg(ITERATIONS), lat_host(ITERATIONS), lat_reg(ITERATIONS);
    HIP_CHECK(hipMemcpy(lat_fg.data(), lat_fg_dev, ITERATIONS * sizeof(long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(lat_host.data(), lat_host_dev, ITERATIONS * sizeof(long long), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(lat_reg.data(), lat_reg_dev, ITERATIONS * sizeof(long long), hipMemcpyDeviceToHost));

    std::sort(lat_fg.begin(), lat_fg.end());
    std::sort(lat_host.begin(), lat_host.end());
    std::sort(lat_reg.begin(), lat_reg.end());

    int wallClkRate;  // kHz
    HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    auto to_ns = [&](long long ticks) {
        return static_cast<long long>(ticks * ticks_to_ns);
    };

    std::cout << "GPU read latency comparison (ns, p50):\n"
              << "  Fine-grained device mem: " << to_ns(lat_fg[ITERATIONS / 2])   << " ns\n"
              << "  Host coherent mem:       " << to_ns(lat_host[ITERATIONS / 2]) << " ns\n"
              << "  Regular device mem:      " << to_ns(lat_reg[ITERATIONS / 2])  << " ns\n";

    // Fine-grained device mem should be readable (threshold: 1ms)
    EXPECT_LT(to_ns(lat_fg[ITERATIONS / 2]), 1'000'000) << "Fine-grained device mem read latency unreasonable";

    HIP_CHECK(hipFree(lat_reg_dev));
    HIP_CHECK(hipFree(lat_host_dev));
    HIP_CHECK(hipFree(lat_fg_dev));
    HIP_CHECK(hipFree(regular_dev_ptr));
    HIP_CHECK(hipHostFree(host_coherent_ptr));
    HIP_CHECK(hipFree(fg_dev_ptr));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n========================================\n";
    std::cout << "FINE-GRAINED DEVICE MEMORY TESTS\n";
    std::cout << "Testing CPU write → GPU read on device memory\n";
    std::cout << "========================================\n\n";

    return RUN_ALL_TESTS();
}
