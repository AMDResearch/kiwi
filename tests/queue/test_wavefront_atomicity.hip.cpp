// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_wavefront_atomicity.hip.cpp
 * @brief Test whether wavefront coalesced writes provide multi-byte atomicity
 *
 * On AMD MI300X, threads in a wavefront (64 threads) execute in SIMD lockstep.
 * When threads write to consecutive addresses in extended-scope host memory,
 * the memory controller coalesces them into larger transactions. This test
 * checks whether those coalesced writes appear atomic from the CPU's
 * perspective.
 *
 * The queue requires the 8-thread x 8-byte (64-byte) case to have no tears.
 * Larger writes are reported as diagnostics and are not part of the contract.
 *
 * Test methodology (trigger-based):
 * - GPU: A wavefront writes generation counters to N consecutive 8-byte slots,
 *   then waits for CPU acknowledgment before the next generation.
 * - CPU: Spins on slot[0]. When it changes, immediately reads all other slots
 *   and checks if they all match. Then increments the ack counter to let the
 *   GPU proceed.
 * - A "torn write" is detected when the CPU sees mixed generations.
 * - If coalesced writes are truly atomic, the CPU can never observe a mixed
 *   state, because atomicity means the entire update is visible instantaneously.
 *
 * Slot value encoding: [generation (56 bits) | thread_id (8 bits)]
 */

#include <gtest/gtest.h>
#include <kiwi/queue/detail/host_memory.hpp>
#include <kiwi/queue/error.hpp>
#include <hip/hip_runtime.h>
#include <thread>
#include <cstring>
#include <cstdio>
#include <cinttypes>

// ---------------------------------------------------------------------------
// GPU Kernels
// ---------------------------------------------------------------------------

// Parallel write: each thread in the wavefront writes to its own 8-byte slot.
// After each generation, thread 0 spins waiting for the CPU to acknowledge.
// Due to wavefront lockstep, all active threads effectively wait.
__global__ void wavefront_coalesced_write(uint64_t* slots,
                                          volatile uint64_t* ack,
                                          int num_active,
                                          int num_generations) {
    int tid = threadIdx.x;
    if (tid >= num_active) return;

    for (int gen = 1; gen <= num_generations; gen++) {
        // All active threads store simultaneously (wavefront SIMD).
        // Encodes: upper 56 bits = generation, lower 8 bits = thread id.
        slots[tid] = ((uint64_t)gen << 8) | (uint64_t)tid;

        // No __threadfence_system() is needed. Extended-scope host memory is
        // mapped uncached for GPU access on both MI300X and MI355X.
        //
        // slots is intentionally NOT __restrict__. Adding it lets the compiler
        // prove slots and ack do not alias, which can sink slot writes outside
        // the loop and deadlock the test.

        // Thread 0 waits for CPU ack. In SIMD lockstep, the wavefront
        // cannot advance past this branch until thread 0 exits the loop,
        // so all active threads effectively wait.
        if (tid == 0) {
            while (*ack < (uint64_t)gen) {}
        }
    }
}

// Control: single thread writes to all slots sequentially.
// Each slot write is a separate memory transaction -- definitely NOT atomic
// across slots.  If the test methodology is sound, this should produce tears.
// After each generation, waits for CPU ack before proceeding.
__global__ void sequential_write(uint64_t* slots,
                                 volatile uint64_t* ack,
                                 int num_slots,
                                 int num_generations) {
    for (int gen = 1; gen <= num_generations; gen++) {
        for (int i = 0; i < num_slots; i++) {
            slots[i] = ((uint64_t)gen << 8) | (uint64_t)i;
        }
        // Wait for CPU to acknowledge this generation.
        while (*ack < (uint64_t)gen) {}
    }
}

// Single-lane publish-last write: payload words first, marker word last.
// Volatile is intentional here: this test is about whether the CPU can observe
// the final marker before earlier stores from the same GPU lane are visible, so
// keep the compiler from reordering/removing the store sequence.
__global__ void single_lane_publish_last_write(volatile uint64_t* slots,
                                               volatile uint64_t* ack,
                                               int num_generations) {
    for (int gen = 1; gen <= num_generations; gen++) {
        // Payload words.
        for (int i = 0; i < 7; i++) {
            slots[i] = ((uint64_t)gen << 8) | (uint64_t)i;
        }

        // Publish marker, written last.
        slots[7] = ((uint64_t)gen << 8) | 7ULL;

        // Wait for CPU to acknowledge this generation.
        while (*ack < (uint64_t)gen) {}
    }
}

// ---------------------------------------------------------------------------
// CPU reader
// ---------------------------------------------------------------------------

struct TestResult {
    int64_t total_reads  = 0;
    int64_t torn_reads   = 0;
    double  tear_pct     = 0.0;
};

// Trigger-based reader: spins on slot[0] waiting for a new generation.
// When slot[0] changes, immediately reads all other slots and checks
// consistency.  Then increments the ack counter to let the GPU proceed.
// Runs for exactly `num_generations` iterations.

struct OrderingResult {
    int64_t total_reads = 0;
    int64_t marker_before_payload = 0;
};

// Publish-order reader: waits for the last word/marker for each generation,
// then checks whether all earlier payload words are already visible.
static OrderingResult cpu_publish_order_reader(volatile uint64_t* slots,
                                               volatile uint64_t* ack,
                                               int num_generations) {
    OrderingResult r;

    for (int gen = 1; gen <= num_generations; gen++) {
        while ((slots[7] >> 8) < (uint64_t)gen) {}

        bool marker_before_payload = false;
        for (int i = 0; i < 7; i++) {
            if ((slots[i] >> 8) != (uint64_t)gen) {
                marker_before_payload = true;
                break;
            }
        }

        r.total_reads++;
        if (marker_before_payload) r.marker_before_payload++;

        *ack = gen;
    }

    return r;
}

static TestResult cpu_reader(volatile uint64_t* slots, int num_slots,
                             volatile uint64_t* ack, int num_generations) {
    TestResult r;

    for (int gen = 1; gen <= num_generations; gen++) {
        // Spin on slot[0] until we see this generation.
        while ((slots[0] >> 8) < (uint64_t)gen) {}

        // slot[0] has been updated.  Immediately read all other slots.
        uint64_t expected_gen = (uint64_t)gen;
        bool torn = false;

        for (int i = 1; i < num_slots; ++i) {
            if ((slots[i] >> 8) != expected_gen) { torn = true; break; }
        }

        r.total_reads++;
        if (torn) r.torn_reads++;

        // Acknowledge: tell GPU we've read this generation.
        *ack = gen;
    }

    r.tear_pct = r.total_reads > 0
        ? static_cast<double>(r.torn_reads) / r.total_reads * 100.0
        : 0.0;
    return r;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static constexpr int NUM_GENERATIONS     = 1000000;

static TestResult run_coalesced_test(int num_active_threads) {
    auto* slots =
        kiwi::queue::detail::host_malloc_uc<uint64_t>(num_active_threads);
    memset(slots, 0, num_active_threads * sizeof(uint64_t));

    uint64_t* ack;
    KIWI_HIP_CHECK(hipHostMalloc(&ack, sizeof(uint64_t), hipHostMallocCoherent));
    *ack = 0;

    TestResult result;

    std::thread reader([&]() {
        result = cpu_reader(reinterpret_cast<volatile uint64_t*>(slots),
                            num_active_threads,
                            reinterpret_cast<volatile uint64_t*>(ack),
                            NUM_GENERATIONS);
    });

    // Launch one full wavefront (64 threads); only num_active_threads participate.
    wavefront_coalesced_write<<<1, 64>>>(
        slots, ack, num_active_threads, NUM_GENERATIONS);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    reader.join();

    kiwi::queue::detail::host_free_uc(slots);
    KIWI_HIP_CHECK(hipHostFree(ack));
    return result;
}


static OrderingResult run_single_lane_publish_order_test() {
    constexpr int NUM_WORDS = 8;

    auto* slots = kiwi::queue::detail::host_malloc_uc<uint64_t>(NUM_WORDS);
    memset(slots, 0, NUM_WORDS * sizeof(uint64_t));

    uint64_t* ack;
    KIWI_HIP_CHECK(hipHostMalloc(&ack, sizeof(uint64_t), hipHostMallocCoherent));
    *ack = 0;

    OrderingResult result;

    std::thread reader([&]() {
        result = cpu_publish_order_reader(
            reinterpret_cast<volatile uint64_t*>(slots),
            reinterpret_cast<volatile uint64_t*>(ack),
            NUM_GENERATIONS);
    });

    single_lane_publish_last_write<<<1, 1>>>(
        reinterpret_cast<volatile uint64_t*>(slots),
        reinterpret_cast<volatile uint64_t*>(ack),
        NUM_GENERATIONS);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    reader.join();

    kiwi::queue::detail::host_free_uc(slots);
    KIWI_HIP_CHECK(hipHostFree(ack));
    return result;
}

static TestResult run_sequential_test(int num_slots) {
    auto* slots = kiwi::queue::detail::host_malloc_uc<uint64_t>(num_slots);
    memset(slots, 0, num_slots * sizeof(uint64_t));

    uint64_t* ack;
    KIWI_HIP_CHECK(hipHostMalloc(&ack, sizeof(uint64_t), hipHostMallocCoherent));
    *ack = 0;

    TestResult result;

    std::thread reader([&]() {
        result = cpu_reader(reinterpret_cast<volatile uint64_t*>(slots),
                            num_slots,
                            reinterpret_cast<volatile uint64_t*>(ack),
                            NUM_GENERATIONS);
    });

    sequential_write<<<1, 1>>>(
        slots, ack, num_slots, NUM_GENERATIONS);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    reader.join();

    kiwi::queue::detail::host_free_uc(slots);
    KIWI_HIP_CHECK(hipHostFree(ack));
    return result;
}

static void print_result(const char* label, int num_bytes,
                          const TestResult& r) {
    printf("  %-40s %4dB | reads: %12" PRId64 " | torn: %8" PRId64 " | "
           "tear rate: %7.3f%%\n",
           label, num_bytes, r.total_reads, r.torn_reads,
           r.tear_pct);
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class WavefrontAtomicity : public ::testing::Test {
protected:
    void SetUp() override {
        int count;
        KIWI_HIP_CHECK(hipGetDeviceCount(&count));
        ASSERT_GT(count, 0) << "No GPU devices found";
    }
};

// ---------------------------------------------------------------------------
// Tests -- each checks tear rate against a threshold
// ---------------------------------------------------------------------------

// 8 threads x 8 bytes = 64 bytes -- should be well within atomic boundary
TEST_F(WavefrontAtomicity, CoalescedWrite_8x8_64B) {
    printf("\n--- Coalesced wavefront write: 8 threads x 8B = 64 bytes ---\n");
    auto r = run_coalesced_test(8);
    print_result("Coalesced 8x8B", 64, r);
    EXPECT_EQ(r.torn_reads, (int64_t)0)
        << "64B coalesced write is NOT atomic: " << r.torn_reads << " tears detected";
}

// 16 threads x 8 bytes = 128 bytes -- beyond proven atomic boundary
TEST_F(WavefrontAtomicity, CoalescedWrite_16x8_128B) {
    printf("\n--- Coalesced wavefront write: 16 threads x 8B = 128 bytes ---\n");
    auto r = run_coalesced_test(16);
    print_result("Coalesced 16x8B", 128, r);
    // Larger writes are outside the queue contract; report for reference.
    printf("  -> Tear rate %.3f%% (beyond 64B atomic boundary)\n",
           r.tear_pct);
}

// 32 threads x 8 bytes = 256 bytes -- beyond hypothesized boundary?
TEST_F(WavefrontAtomicity, CoalescedWrite_32x8_256B) {
    printf("\n--- Coalesced wavefront write: 32 threads x 8B = 256 bytes ---\n");
    auto r = run_coalesced_test(32);
    print_result("Coalesced 32x8B", 256, r);
    // No hard tear rate assertion here -- we're exploring the boundary.
    // Just report results and let the user compare with 128B.
    printf("  -> Tear rate %.3f%% (compare with 128B to find boundary)\n",
           r.tear_pct);
}

// 64 threads x 8 bytes = 512 bytes -- full wavefront
TEST_F(WavefrontAtomicity, CoalescedWrite_64x8_512B) {
    printf("\n--- Coalesced wavefront write: 64 threads x 8B = 512 bytes ---\n");
    auto r = run_coalesced_test(64);
    print_result("Coalesced 64x8B", 512, r);
    printf("  -> Tear rate %.3f%% (compare with 128B to find boundary)\n",
           r.tear_pct);
}

// Single-lane publish-order check: if one lane writes payload first and a
// marker/sequence word last, the host should not observe the marker generation
// before the payload generation.  This does not make the whole 64B slot atomic;
// it only validates the publish-last ordering pattern.
TEST_F(WavefrontAtomicity, SingleLanePublishLastObservedInOrder) {
    printf("\n--- Single-lane publish-last ordering: payload then marker ---\n");
    auto r = run_single_lane_publish_order_test();
    printf("  Publish-order reads: %12" PRId64 " | marker-before-payload: %8"
           PRId64 "\n",
           r.total_reads, r.marker_before_payload);
    EXPECT_EQ(r.marker_before_payload, (int64_t)0)
        << "Host observed the publish marker before earlier payload words from "
           "the same GPU lane became visible";
}

// Single-lane 64B slot write: one GPU lane writes the 8 x 8B words that
// make up one queue wave-slot sequentially.  This is intentionally NOT a
// wavefront-coalesced write.  The CPU triggers on word 0 changing and then
// immediately samples words 1..7; observing mixed generations proves the 64B
// slot update is not atomic when produced by one lane as multiple stores.
TEST_F(WavefrontAtomicity, SingleLaneSlotWrite_8x8_64B) {
    printf("\n--- Single-lane sequential slot write: 8 x 8B = 64 bytes ---\n");
    printf("    (Expect tears: one lane cannot atomically publish a 64B slot)\n");
    auto r = run_sequential_test(8);
    print_result("Single-lane 8x8B slot", 64, r);
    EXPECT_GT(r.torn_reads, (int64_t)0)
        << "Single-lane 64B slot writes produced 0 tears -- either the "
           "methodology missed the race or this hardware/compiler emitted an "
           "unexpected atomic 64B publication";
    printf("  -> 64B slot atomic as single-lane stores: %s\n",
           r.torn_reads == 0 ? "MAYBE" : "NO");
}

// Positive control: sequential writes should tear (validates methodology)
TEST_F(WavefrontAtomicity, SequentialWriteBaseline) {
    printf("\n--- Control: 1 thread writing 16 slots sequentially ---\n");
    printf("    (Expect tears to validate test methodology can detect them)\n");
    auto r = run_sequential_test(16);
    print_result("Sequential 1x16", 128, r);
    EXPECT_GT(r.torn_reads, (int64_t)0)
        << "Sequential writes produced 0 tears -- methodology may be unsound";
    printf("  -> Tears detected: %s\n",
           r.torn_reads > 0 ? "YES (methodology validated)"
                            : "NO (writes too fast to catch?)");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    printf("\n========================================\n");
    printf("WAVEFRONT COALESCED WRITE ATOMICITY TEST\n");
    printf("========================================\n");
    printf("Testing whether wavefront SIMD writes to extended-scope host memory\n");
    printf("appear atomic from the CPU at various sizes.\n");
    printf("Required result: 8 threads x 8B = 64B has no tears.\n");
    printf("Larger writes are diagnostic only.\n\n");
    printf("Pass criteria: coalesced 64B torn reads == 0\n");
    printf("               single-lane publish-last marker-before-payload == 0\n");
    printf("               single-lane 64B slot torn reads > 0\n");
    printf("               sequential baseline torn reads > 0\n\n");

    int result = RUN_ALL_TESTS();

    if (result == 0) {
        printf("\n========================================\n");
        printf("ALL TESTS PASSED\n");
        printf("========================================\n\n");
    }

    return result;
}
