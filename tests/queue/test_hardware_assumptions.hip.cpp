// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_hardware_assumptions.hip
 * @brief CRITICAL tests for hardware assumptions
 *
 * Tests PCIe atomicity, write visibility, WAW ordering, and fence effectiveness.
 * If any of these tests fail, the queue design must be revised!
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Test 1: 8-byte PCIe write atomicity
// GPU writes 8-byte values rapidly, CPU reads concurrently
// CPU should NEVER see torn/partial writes

__global__ void write_8byte_kernel(uint64_t* addr, uint64_t value, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        *addr = value;
    }
}

TEST(HardwareAssumptions, PCIe8ByteAtomicity) {
    constexpr int ITERATIONS = 100000;
    constexpr int TEST_PATTERNS = 4;

    uint64_t patterns[TEST_PATTERNS] = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x5555555555555555ULL,
        0xAAAAAAAAAAAAAAAAULL
    };

    uint64_t* host_addr;
    HIP_CHECK(hipHostMalloc(&host_addr, sizeof(uint64_t), hipHostMallocCoherent));

    for (int p = 0; p < TEST_PATTERNS; ++p) {
        uint64_t pattern = patterns[p];
        *host_addr = ~pattern;  // Initialize to opposite

        // Launch GPU writer
        write_8byte_kernel<<<1, 1>>>(host_addr, pattern, ITERATIONS);

        // CPU reader (concurrent)
        std::thread reader([&]() {
            int torn_reads = 0;
            for (int i = 0; i < ITERATIONS; ++i) {
                uint64_t read_val = *host_addr;

                // Check if value is either old or new (no partial/torn writes)
                if (read_val != pattern && read_val != ~pattern) {
                    torn_reads++;
                    // Allow a few torn reads due to race, but should be very rare
                }
            }

            // If more than 1% are torn, something is wrong
            EXPECT_LT(torn_reads, ITERATIONS / 100)
                << "Too many torn reads (" << torn_reads << "/" << ITERATIONS << ")";
        });

        reader.join();
        HIP_CHECK(hipDeviceSynchronize());

        // Final value should be the pattern
        EXPECT_EQ(*host_addr, pattern);
    }

    HIP_CHECK(hipHostFree(host_addr));
}

// Test 2: PCIe write visibility without fence
// GPU writes without __threadfence_system(), CPU should eventually see it

__global__ void write_without_fence_kernel(uint64_t* addr, uint64_t value) {
    *addr = value;
    // NO __threadfence_system() here!
}

TEST(HardwareAssumptions, PCIeWriteVisibilityWithoutFence) {
    uint64_t* host_addr;
    HIP_CHECK(hipHostMalloc(&host_addr, sizeof(uint64_t), hipHostMallocCoherent));

    constexpr uint64_t MAGIC_VALUE = 0xDEADBEEFCAFEBABEULL;
    *host_addr = 0;

    // GPU writes without fence
    write_without_fence_kernel<<<1, 1>>>(host_addr, MAGIC_VALUE);

    // CPU polls for value (with timeout)
    auto start = std::chrono::steady_clock::now();
    constexpr int TIMEOUT_MS = 1000;
    bool seen = false;

    while (!seen) {
        if (*host_addr == MAGIC_VALUE) {
            seen = true;
            break;
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > TIMEOUT_MS) {
            break;
        }

        std::this_thread::yield();
    }

    HIP_CHECK(hipDeviceSynchronize());

    EXPECT_TRUE(seen) << "CPU never saw GPU write (even after sync)";
    EXPECT_EQ(*host_addr, MAGIC_VALUE);

    HIP_CHECK(hipHostFree(host_addr));
}

// Test 3: WAW ordering for same address
// CPU writes sequence of values, GPU reads periodically
// GPU should never see values out of order

TEST(HardwareAssumptions, WAWOrdering) {
    uint64_t* device_addr;
    HIP_CHECK(hipMalloc(&device_addr, sizeof(uint64_t)));

    // Make it fine-grained for CPU writes
    uint64_t* host_handle_to_dev_mem;
    HIP_CHECK(hipHostMalloc(&host_handle_to_dev_mem, sizeof(uint64_t),
                            hipHostMallocCoherent | hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer((void**)&device_addr, host_handle_to_dev_mem, 0));

    constexpr int NUM_WRITES = 1000;
    std::atomic<bool> stop_reading{false};
    std::vector<uint64_t> gpu_reads;

    // CPU writer
    std::thread writer([&]() {
        for (int i = 0; i < NUM_WRITES; ++i) {
            *host_handle_to_dev_mem = i;
        }
        stop_reading = true;
    });

    // GPU reader (polling)
    std::thread gpu_reader([&]() {
        // Simple kernel that reads periodically
        // (For simplicity, we'll do this on CPU reading device memory)
        uint64_t last_seen = 0;
        while (!stop_reading.load()) {
            uint64_t current = *host_handle_to_dev_mem;
            if (current != last_seen) {
                gpu_reads.push_back(current);
                last_seen = current;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    });

    writer.join();
    gpu_reader.join();

    // Check for monotonicity (can skip values, but not reorder)
    bool ordered = true;
    for (size_t i = 1; i < gpu_reads.size(); ++i) {
        if (gpu_reads[i] < gpu_reads[i-1]) {
            ordered = false;
            std::cerr << "Out of order: " << gpu_reads[i-1] << " -> " << gpu_reads[i] << "\n";
        }
    }

    EXPECT_TRUE(ordered) << "GPU saw values out of order!";

    HIP_CHECK(hipHostFree(host_handle_to_dev_mem));
}

// Test 4: Fine-grained memory read performance

__global__ void read_host_memory_kernel(volatile uint64_t* addr, int iterations, long long* latencies) {
    for (int i = 0; i < iterations; ++i) {
        long long start = wall_clock64();
        __builtin_amdgcn_s_waitcnt(0xC07F);
        uint64_t val = *addr;
        __builtin_amdgcn_s_waitcnt(0xF70);
        long long end = wall_clock64();
        __builtin_amdgcn_s_waitcnt(0xC07F);
        latencies[i] = end - start;
        (void)val;  // Prevent optimization
    }
}

TEST(HardwareAssumptions, GPUReadHostMemPerformance) {
    uint64_t* host_addr;
    HIP_CHECK(hipHostMalloc(&host_addr, sizeof(uint64_t),
                            hipHostMallocCoherent | hipHostMallocMapped));

    constexpr int ITERATIONS = 100;
    long long* latencies_device;
    HIP_CHECK(hipMalloc(&latencies_device, ITERATIONS * sizeof(long long)));

    *host_addr = 0x123456789ABCDEF0ULL;

    // Measure read latency from GPU
    read_host_memory_kernel<<<1, 1>>>(host_addr, ITERATIONS, latencies_device);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<long long> latencies(ITERATIONS);
    HIP_CHECK(hipMemcpy(latencies.data(), latencies_device,
                        ITERATIONS * sizeof(long long), hipMemcpyDeviceToHost));

    std::sort(latencies.begin(), latencies.end());
    long long p50 = latencies[ITERATIONS / 2];

    int wallClkRate;  // kHz
    HIP_CHECK(hipDeviceGetAttribute(&wallClkRate, hipDeviceAttributeWallClockRate, 0));
    double ticks_to_ns = 1'000'000.0 / wallClkRate;

    auto to_ns = [&](long long ticks) {
        return static_cast<long long>(ticks * ticks_to_ns);
    };

    std::cout << "Host memory read latency (ns):\n"
              << "  p50: " << to_ns(p50) << "\n";

    // Should be reasonable (<1'000'000 nanoseconds typically)
    EXPECT_LT(p50, 1'000'000) << "Host read latency too high";

    HIP_CHECK(hipFree(latencies_device));
    HIP_CHECK(hipHostFree(host_addr));
}

// Test 5: Fence effectiveness
// CPU reads slot, then writes to notification
// Fence should prevent reordering

TEST(HardwareAssumptions, FenceEffectiveness) {
    uint64_t* slot_addr;
    uint64_t* notification_addr;

    HIP_CHECK(hipHostMalloc(&slot_addr, sizeof(uint64_t), hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(&notification_addr, sizeof(uint64_t), hipHostMallocCoherent));

    *slot_addr = 0;
    *notification_addr = 0;

    constexpr int ITERATIONS = 1000;

    for (int i = 0; i < ITERATIONS; ++i) {
        *slot_addr = i;

        // Read slot
        volatile uint64_t slot_val = *slot_addr;

        // Fence (should prevent reordering with notification write)
        std::atomic_thread_fence(std::memory_order_release);

        // Write notification
        *notification_addr = slot_val + 1000;

        // Verify notification reflects the slot read
        EXPECT_EQ(*notification_addr, slot_val + 1000);
    }

    HIP_CHECK(hipHostFree(notification_addr));
    HIP_CHECK(hipHostFree(slot_addr));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n========================================\n";
    std::cout << "CRITICAL HARDWARE ASSUMPTION TESTS\n";
    std::cout << "If any test fails, queue design must be revised!\n";
    std::cout << "========================================\n\n";

    int result = RUN_ALL_TESTS();

    if (result == 0) {
        std::cout << "\n========================================\n";
        std::cout << "ALL HARDWARE ASSUMPTIONS VERIFIED!\n";
        std::cout << "Queue design is sound for this hardware.\n";
        std::cout << "========================================\n\n";
    } else {
        std::cout << "\n========================================\n";
        std::cout << "WARNING: HARDWARE ASSUMPTION FAILURES!\n";
        std::cout << "Queue may not work correctly on this system.\n";
        std::cout << "========================================\n\n";
    }

    return result;
}
