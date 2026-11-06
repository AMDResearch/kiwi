// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_mpmc_h2d_concurrent.hip.cpp
 * @brief Concurrent CPU-GPU tests for MPMCH2DQueue
 *
 * Tests multi-producer multi-consumer host-to-device scenarios:
 *   - SPSC: single CPU thread push, single wavefront pop
 *   - MPSC: multiple CPU threads push, single wavefront pop
 *   - SPMC: single CPU thread push, multiple wavefronts pop
 *   - MPMC: multiple CPU threads push, multiple wavefronts pop
 *   - Queue full with retry
 *   - Wrap-around with small queue
 *   - Stress test with high volume
 *
 * All consumer kernels are launched with 64 threads per block (one wavefront).
 * Multiple consumers use multiple blocks (<<<N, 64>>>).
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <set>
#include <algorithm>
#include <atomic>

using namespace kiwi::queue;

// ---------------------------------------------------------------------------
// Consumer kernels (GPU pops)
// ---------------------------------------------------------------------------

// Single-wavefront consumer: pops num_msgs messages, stores values in results array.
// Launch with <<<1, 64>>> (full wavefront).
__global__ void mpmc_h2d_consumer_kernel(MPMCH2DQueue::DeviceHandle queue,
                                          int num_msgs,
                                          uint32_t* results, int* pop_count) {
    if (__lane_id() == 0) {
        *pop_count = 0;
    }

    uint8_t buf[56] = {};
    for (int i = 0; i < num_msgs; ++i) {
        // Spin until data is available
        while (!queue.pop(buf)) {
            // Queue empty, retry
        }
        if (__lane_id() == 0) {
            uint32_t val;
            __builtin_memcpy(&val, buf, sizeof(val));
            results[i] = val;
            (*pop_count)++;
        }
    }
    queue.flush_consumer_index();
}

// Multi-wavefront consumer: each block pops msgs_per_block messages.
// Results are stored per-block: results[blockIdx.x * msgs_per_block + i].
// Launch with <<<num_blocks, 64>>>.
__global__ void mpmc_h2d_multi_consumer_kernel(MPMCH2DQueue::DeviceHandle queue,
                                                int msgs_per_block,
                                                uint32_t* results,
                                                int* block_pop_counts) {
    int block_id = blockIdx.x;
    if (__lane_id() == 0) {
        block_pop_counts[block_id] = 0;
    }

    uint8_t buf[56] = {};
    for (int i = 0; i < msgs_per_block; ++i) {
        while (!queue.pop(buf)) {
            // Queue empty, retry
        }
        if (__lane_id() == 0) {
            uint32_t val;
            __builtin_memcpy(&val, buf, sizeof(val));
            results[block_id * msgs_per_block + i] = val;
            block_pop_counts[block_id]++;
        }
    }
    queue.flush_consumer_index();
}

// Try-pop kernel: pops up to max_attempts messages, stops on first empty.
// Used for QueueFull/QueueEmpty tests where we don't want to block.
__global__ void mpmc_h2d_try_pop_kernel(MPMCH2DQueue::DeviceHandle queue,
                                         uint32_t* results, int* pop_count,
                                         int max_attempts) {
    if (__lane_id() == 0) {
        *pop_count = 0;
    }

    uint8_t buf[56] = {};
    for (int i = 0; i < max_attempts; ++i) {
        if (queue.pop(buf)) {
            if (__lane_id() == 0) {
                uint32_t val;
                __builtin_memcpy(&val, buf, sizeof(val));
                results[*pop_count] = val;
                (*pop_count)++;
            }
        } else {
            break;
        }
    }
    queue.flush_consumer_index();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Push messages from a CPU thread into the queue until `remaining` reaches 0.
/// Values encoded as: (thread_id << 16) | msg_index_within_thread.
static void producer_thread_fn(MPMCH2DQueue* queue,
                                int thread_id,
                                int num_msgs,
                                std::atomic<int>& remaining,
                                int timeout_ms = 5000) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    uint8_t buf[56] = {};
    for (int i = 0; i < num_msgs; ++i) {
        uint32_t val = (static_cast<uint32_t>(thread_id) << 16) |
                        static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        while (!queue->push(buf)) {
            if (std::chrono::steady_clock::now() > deadline) {
                return;  // Timeout
            }
            std::this_thread::yield();
        }
        remaining.fetch_sub(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// 1. Basic SPSC: 1 CPU thread pushes 100 messages, 1 wavefront pops.
///    Verify all received in order (single producer guarantees ordering).
TEST(MpmcH2dConcurrentTest, SingleProducerSingleConsumer) {
    constexpr int NUM_MSGS = 100;
    auto queue = std::make_unique<MPMCH2DQueue>(128);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, NUM_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Launch consumer kernel first (it will poll for data)
    mpmc_h2d_consumer_kernel<<<1, 64>>>(queue->device_handle(), NUM_MSGS,
                                         results_dev, pop_count_dev);

    // CPU pushes messages
    uint8_t buf[56] = {};
    for (int i = 0; i < NUM_MSGS; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        while (!queue->push(buf)) {
            // Retry on full
        }
    }

    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Verify
    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, NUM_MSGS);

    std::vector<uint32_t> results(NUM_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, NUM_MSGS * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));
    for (int i = 0; i < NUM_MSGS; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i))
            << "Message " << i << " out of order";
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

/// 2. MPSC: 4 CPU threads push 50 each (200 total). 1 wavefront pops.
///    Verify: all 200 messages received, per-thread ordering preserved.
TEST(MpmcH2dConcurrentTest, MultiProducerSingleConsumer) {
    constexpr int NUM_THREADS = 4;
    constexpr int MSGS_PER_THREAD = 50;
    constexpr int TOTAL_MSGS = NUM_THREADS * MSGS_PER_THREAD;
    auto queue = std::make_unique<MPMCH2DQueue>(256);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, TOTAL_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Launch consumer kernel first
    mpmc_h2d_consumer_kernel<<<1, 64>>>(queue->device_handle(), TOTAL_MSGS,
                                         results_dev, pop_count_dev);

    // Launch CPU producer threads
    std::atomic<int> remaining(TOTAL_MSGS);
    std::vector<std::thread> producers;
    for (int t = 0; t < NUM_THREADS; ++t) {
        producers.emplace_back([&, t]() {
            producer_thread_fn(queue.get(), t, MSGS_PER_THREAD, remaining);
        });
    }

    for (auto& t : producers) t.join();
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Verify pop count
    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_EQ(pop_count, TOTAL_MSGS)
        << "Expected " << TOTAL_MSGS << " messages, got " << pop_count;

    // Copy results
    std::vector<uint32_t> results(TOTAL_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, TOTAL_MSGS * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));

    // Verify no duplicates
    std::set<uint32_t> value_set(results.begin(), results.end());
    EXPECT_EQ(value_set.size(), TOTAL_MSGS) << "Duplicate messages detected";

    // Verify per-thread ordering: for each thread, msg indices must be in order
    for (int thr = 0; thr < NUM_THREADS; ++thr) {
        std::vector<uint16_t> thread_msgs;
        for (uint32_t v : results) {
            uint16_t thread_id = static_cast<uint16_t>(v >> 16);
            uint16_t msg_idx = static_cast<uint16_t>(v & 0xFFFF);
            if (thread_id == thr) {
                thread_msgs.push_back(msg_idx);
            }
        }
        ASSERT_EQ(thread_msgs.size(), MSGS_PER_THREAD)
            << "Thread " << thr << " message count mismatch";
        for (int i = 0; i < MSGS_PER_THREAD; ++i) {
            EXPECT_EQ(thread_msgs[i], static_cast<uint16_t>(i))
                << "Thread " << thr << " message " << i << " out of order";
        }
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

/// 3. SPMC: 1 CPU thread pushes 200 messages. 4 wavefronts pop concurrently.
///    Verify: all 200 values present, no duplicates.
TEST(MpmcH2dConcurrentTest, SingleProducerMultiConsumer) {
    constexpr int NUM_MSGS = 200;
    constexpr int NUM_BLOCKS = 4;
    constexpr int MSGS_PER_BLOCK = NUM_MSGS / NUM_BLOCKS;
    auto queue = std::make_unique<MPMCH2DQueue>(256);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_counts_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, NUM_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_counts_dev, NUM_BLOCKS * sizeof(int)));

    // Launch multi-wavefront consumer
    mpmc_h2d_multi_consumer_kernel<<<NUM_BLOCKS, 64>>>(
        queue->device_handle(), MSGS_PER_BLOCK, results_dev, pop_counts_dev);

    // CPU pushes all messages
    uint8_t buf[56] = {};
    for (int i = 0; i < NUM_MSGS; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        while (!queue->push(buf)) {
            std::this_thread::yield();
        }
    }

    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Copy results
    std::vector<uint32_t> results(NUM_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, NUM_MSGS * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));

    // Verify total pop count
    std::vector<int> pop_counts(NUM_BLOCKS);
    KIWI_HIP_CHECK(hipMemcpy(pop_counts.data(), pop_counts_dev, NUM_BLOCKS * sizeof(int),
                              hipMemcpyDeviceToHost));
    int total_popped = 0;
    for (int c : pop_counts) total_popped += c;
    ASSERT_EQ(total_popped, NUM_MSGS)
        << "Expected " << NUM_MSGS << " messages, got " << total_popped;

    // Sort and verify: all values 0..199, no duplicates
    std::sort(results.begin(), results.end());
    for (int i = 0; i < NUM_MSGS; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i))
            << "Missing or duplicate message at index " << i;
    }

    KIWI_HIP_CHECK(hipFree(pop_counts_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

/// 4. MPMC: 4 CPU threads push 50 each (200 total). 4 wavefronts pop.
///    Verify: all 200 unique values received, no duplicates, no lost messages.
TEST(MpmcH2dConcurrentTest, MultiProducerMultiConsumer) {
    constexpr int NUM_THREADS = 4;
    constexpr int MSGS_PER_THREAD = 50;
    constexpr int TOTAL_MSGS = NUM_THREADS * MSGS_PER_THREAD;
    constexpr int NUM_BLOCKS = 4;
    constexpr int MSGS_PER_BLOCK = TOTAL_MSGS / NUM_BLOCKS;
    auto queue = std::make_unique<MPMCH2DQueue>(256);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_counts_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, TOTAL_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_counts_dev, NUM_BLOCKS * sizeof(int)));

    // Launch multi-wavefront consumer first
    mpmc_h2d_multi_consumer_kernel<<<NUM_BLOCKS, 64>>>(
        queue->device_handle(), MSGS_PER_BLOCK, results_dev, pop_counts_dev);

    // Launch CPU producer threads
    std::atomic<int> remaining(TOTAL_MSGS);
    std::vector<std::thread> producers;
    for (int t = 0; t < NUM_THREADS; ++t) {
        producers.emplace_back([&, t]() {
            producer_thread_fn(queue.get(), t, MSGS_PER_THREAD, remaining);
        });
    }

    for (auto& t : producers) t.join();
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Copy results
    std::vector<uint32_t> results(TOTAL_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, TOTAL_MSGS * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));

    // Verify all unique values present
    std::set<uint32_t> value_set(results.begin(), results.end());
    EXPECT_EQ(value_set.size(), TOTAL_MSGS) << "Duplicate messages detected";

    // Verify all expected values exist
    for (int thr = 0; thr < NUM_THREADS; ++thr) {
        for (int i = 0; i < MSGS_PER_THREAD; ++i) {
            uint32_t expected = (static_cast<uint32_t>(thr) << 16) |
                                 static_cast<uint32_t>(i);
            EXPECT_TRUE(value_set.count(expected))
                << "Missing message: thread=" << thr << " index=" << i;
        }
    }

    KIWI_HIP_CHECK(hipFree(pop_counts_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

/// 5. QueueFull: small queue (16 slots), push until full, verify.
TEST(MpmcH2dConcurrentTest, QueueFull) {
    constexpr int CAPACITY = 16;
    auto queue = std::make_unique<MPMCH2DQueue>(CAPACITY);

    // CPU pushes until full
    uint8_t buf[56] = {};
    int push_count = 0;
    for (int i = 0; i < CAPACITY + 10; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        if (queue->push(buf)) {
            push_count++;
        } else {
            break;
        }
    }

    // Should have pushed exactly capacity messages
    EXPECT_EQ(push_count, CAPACITY);

    // GPU drains all messages
    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, CAPACITY * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    mpmc_h2d_try_pop_kernel<<<1, 64>>>(queue->device_handle(), results_dev,
                                        pop_count_dev, CAPACITY + 10);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, CAPACITY);

    // Verify values
    std::vector<uint32_t> results(CAPACITY);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, CAPACITY * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));
    std::sort(results.begin(), results.end());
    for (int i = 0; i < CAPACITY; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i));
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

/// 6. WrapAround: tiny queue (8 slots), 200 messages from 2 CPU threads.
///    Consumer runs concurrently, verifying all messages arrive.
TEST(MpmcH2dConcurrentTest, WrapAround) {
    constexpr int CAPACITY = 8;
    constexpr int NUM_THREADS = 2;
    constexpr int MSGS_PER_THREAD = 100;
    constexpr int TOTAL_MSGS = NUM_THREADS * MSGS_PER_THREAD;
    auto queue = std::make_unique<MPMCH2DQueue>(CAPACITY, /*notify_interval=*/2);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, TOTAL_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Launch consumer kernel first (polls for data)
    mpmc_h2d_consumer_kernel<<<1, 64>>>(queue->device_handle(), TOTAL_MSGS,
                                         results_dev, pop_count_dev);

    // Launch CPU producer threads
    std::atomic<int> remaining(TOTAL_MSGS);
    std::vector<std::thread> producers;
    for (int t = 0; t < NUM_THREADS; ++t) {
        producers.emplace_back([&, t]() {
            producer_thread_fn(queue.get(), t, MSGS_PER_THREAD, remaining, 10000);
        });
    }

    for (auto& t : producers) t.join();
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Verify all messages received
    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_EQ(pop_count, TOTAL_MSGS)
        << "Expected " << TOTAL_MSGS << " messages, got " << pop_count;

    // Copy results and verify all unique values present
    std::vector<uint32_t> results(TOTAL_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, TOTAL_MSGS * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));

    std::set<uint32_t> value_set(results.begin(), results.end());
    EXPECT_EQ(value_set.size(), TOTAL_MSGS) << "Duplicate or lost messages";

    // Verify per-thread ordering preserved (within single consumer)
    for (int thr = 0; thr < NUM_THREADS; ++thr) {
        std::vector<uint16_t> thread_msgs;
        for (uint32_t v : results) {
            if (static_cast<uint16_t>(v >> 16) == thr) {
                thread_msgs.push_back(static_cast<uint16_t>(v & 0xFFFF));
            }
        }
        ASSERT_EQ(thread_msgs.size(), MSGS_PER_THREAD)
            << "Thread " << thr << " message count mismatch";
        for (int i = 0; i < MSGS_PER_THREAD; ++i) {
            EXPECT_EQ(thread_msgs[i], static_cast<uint16_t>(i))
                << "Thread " << thr << " message " << i << " out of order";
        }
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

/// 7. StressTest: 8 CPU threads x 50000 messages = 400000 total.
///    4 GPU wavefronts pop. Queue capacity 1024.
///    Forces heavy contention on both CPU CAS and GPU atomicAdd.
TEST(MpmcH2dConcurrentTest, StressTest) {
    constexpr int NUM_THREADS = 8;
    constexpr int MSGS_PER_THREAD = 50000;
    constexpr int TOTAL_MSGS = NUM_THREADS * MSGS_PER_THREAD;
    constexpr int NUM_BLOCKS = 4;
    constexpr int MSGS_PER_BLOCK = TOTAL_MSGS / NUM_BLOCKS;
    constexpr int CAPACITY = 1024;
    auto queue = std::make_unique<MPMCH2DQueue>(CAPACITY);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_counts_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, TOTAL_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_counts_dev, NUM_BLOCKS * sizeof(int)));

    // Launch multi-wavefront consumer
    mpmc_h2d_multi_consumer_kernel<<<NUM_BLOCKS, 64>>>(
        queue->device_handle(), MSGS_PER_BLOCK, results_dev, pop_counts_dev);

    // Launch CPU producer threads
    std::atomic<int> remaining(TOTAL_MSGS);
    std::vector<std::thread> producers;
    for (int t = 0; t < NUM_THREADS; ++t) {
        producers.emplace_back([&, t]() {
            producer_thread_fn(queue.get(), t, MSGS_PER_THREAD, remaining, 60000);
        });
    }

    for (auto& t : producers) t.join();
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Copy results
    std::vector<uint32_t> results(TOTAL_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, TOTAL_MSGS * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));

    // Verify total count
    std::vector<int> pop_counts(NUM_BLOCKS);
    KIWI_HIP_CHECK(hipMemcpy(pop_counts.data(), pop_counts_dev, NUM_BLOCKS * sizeof(int),
                              hipMemcpyDeviceToHost));
    int total_popped = 0;
    for (int c : pop_counts) total_popped += c;
    ASSERT_EQ(total_popped, TOTAL_MSGS)
        << "Expected " << TOTAL_MSGS << " messages, got " << total_popped;

    // Verify all unique values, no duplicates
    std::set<uint32_t> value_set(results.begin(), results.end());
    EXPECT_EQ(value_set.size(), TOTAL_MSGS)
        << "Duplicate messages detected: received " << TOTAL_MSGS
        << " total but only " << value_set.size() << " unique";

    // Verify every expected value is present
    int missing_count = 0;
    for (int thr = 0; thr < NUM_THREADS; ++thr) {
        for (int i = 0; i < MSGS_PER_THREAD; ++i) {
            uint32_t expected = (static_cast<uint32_t>(thr) << 16) |
                                 static_cast<uint32_t>(i);
            if (!value_set.count(expected)) {
                missing_count++;
                if (missing_count <= 10) {
                    ADD_FAILURE() << "Missing message: thread=" << thr
                                  << " index=" << i
                                  << " (value=0x" << std::hex << expected
                                  << std::dec << ")";
                }
            }
        }
    }
    if (missing_count > 10) {
        ADD_FAILURE() << "... and " << (missing_count - 10)
                      << " more missing messages (total missing: "
                      << missing_count << ")";
    }

    // Verify per-thread ordering within each GPU block's view
    for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
        int blk_start = blk * MSGS_PER_BLOCK;
        int blk_end = blk_start + pop_counts[blk];
        for (int thr = 0; thr < NUM_THREADS; ++thr) {
            int32_t last_idx = -1;
            for (int r = blk_start; r < blk_end; ++r) {
                if (static_cast<uint16_t>(results[r] >> 16) == thr) {
                    int32_t idx = static_cast<int32_t>(results[r] & 0xFFFF);
                    EXPECT_GT(idx, last_idx)
                        << "Block " << blk << " thread " << thr
                        << ": message " << idx << " arrived after " << last_idx;
                    last_idx = idx;
                }
            }
        }
    }

    KIWI_HIP_CHECK(hipFree(pop_counts_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

/// 8. QueueEmpty: verify pop returns false on empty queue.
TEST(MpmcH2dConcurrentTest, QueueEmpty) {
    auto queue = std::make_unique<MPMCH2DQueue>(16);

    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Queue is empty, pop should fail
    mpmc_h2d_try_pop_kernel<<<1, 64>>>(queue->device_handle(), results_dev,
                                        pop_count_dev, 1);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, 0);

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
