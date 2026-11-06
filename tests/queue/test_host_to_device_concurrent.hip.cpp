// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_host_to_device_concurrent.hip.cpp
 * @brief Concurrent CPU-GPU tests for SPSCH2DQueue
 *
 * Tests CPU push → GPU pop operations, queue full/empty conditions,
 * and in-order delivery using full-wavefront (64 threads) cooperative pop.
 *
 * Key difference from test_concurrent.hip.cpp: the GPU kernel is launched
 * FIRST (polling for data), then CPU pushes. This is because the GPU is
 * the consumer in SPSCH2DQueue.
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>

using namespace kiwi::queue;

// Consumer kernel: full wavefront cooperative pop of N messages
// Launch with <<<1, 64>>> (full wavefront).
// Stores received values in results array. Lane 0 writes results.
__global__ void consumer_kernel(SPSCH2DQueue::DeviceHandle queue, int num_msgs,
                                 uint32_t* results, int* pop_count) {
    if (__lane_id() == 0) {
        *pop_count = 0;
    }

    uint8_t buf[56] = {};
    for (int i = 0; i < num_msgs; ++i) {
        while (!queue.pop(buf)) {
            // Retry on empty
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

// Try-pop kernel: pops up to max_attempts messages, stops on first empty.
// Used for QueueFull/QueueEmpty tests where we don't want to block.
__global__ void try_pop_kernel(SPSCH2DQueue::DeviceHandle queue,
                                uint32_t* results, int* pop_count, int max_attempts) {
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

// Test basic single message push/pop
TEST(HostToDeviceConcurrentTest, SingleMessage) {
    auto queue = std::make_unique<SPSCH2DQueue>(16);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // CPU pushes one message
    uint8_t buf[56] = {};
    uint32_t value = 42;
    std::memcpy(buf, &value, sizeof(value));
    EXPECT_TRUE(queue->push(buf));

    // GPU pops it
    try_pop_kernel<<<1, 64>>>(queue->device_handle(), results_dev, pop_count_dev, 1);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Verify
    int pop_count = 0;
    uint32_t result = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    KIWI_HIP_CHECK(hipMemcpy(&result, results_dev, sizeof(uint32_t), hipMemcpyDeviceToHost));

    EXPECT_EQ(pop_count, 1);
    EXPECT_EQ(result, 42);

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

// Test multiple messages in order
TEST(HostToDeviceConcurrentTest, InOrderDelivery) {
    constexpr int NUM_MSGS = 100;
    auto queue = std::make_unique<SPSCH2DQueue>(128);

    // Allocate device memory for results
    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, NUM_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Launch consumer kernel first (it will poll for data)
    consumer_kernel<<<1, 64>>>(queue->device_handle(), NUM_MSGS, results_dev, pop_count_dev);

    // CPU pushes messages (consumer is already running on GPU)
    uint8_t buf[56] = {};
    for (int i = 0; i < NUM_MSGS; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        while (!queue->push(buf)) {
            // Retry on full
        }
    }

    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Verify order
    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, NUM_MSGS);

    std::vector<uint32_t> results(NUM_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, NUM_MSGS * sizeof(uint32_t), hipMemcpyDeviceToHost));

    for (int i = 0; i < NUM_MSGS; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i)) << "Message " << i << " out of order";
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

// Test queue full behavior
TEST(HostToDeviceConcurrentTest, QueueFull) {
    constexpr int CAPACITY = 16;
    auto queue = std::make_unique<SPSCH2DQueue>(CAPACITY);

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

    try_pop_kernel<<<1, 64>>>(queue->device_handle(), results_dev, pop_count_dev, CAPACITY + 10);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, CAPACITY);

    // Verify values
    std::vector<uint32_t> results(CAPACITY);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, CAPACITY * sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < CAPACITY; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i));
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

// Test queue empty behavior
TEST(HostToDeviceConcurrentTest, QueueEmpty) {
    auto queue = std::make_unique<SPSCH2DQueue>(16);

    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Queue is empty, pop should fail
    try_pop_kernel<<<1, 64>>>(queue->device_handle(), results_dev, pop_count_dev, 1);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, 0);

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

// Test wrap-around with small queue
TEST(HostToDeviceConcurrentTest, WrapAround) {
    constexpr int CAPACITY = 8;
    constexpr int TOTAL_MSGS = 100;
    // Small notify_interval to prevent deadlock: GPU must notify CPU
    // frequently so CPU sees space to push more items.
    auto queue = std::make_unique<SPSCH2DQueue>(CAPACITY, /*notify_interval=*/2);

    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, TOTAL_MSGS * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Launch consumer kernel first (polls for data)
    consumer_kernel<<<1, 64>>>(queue->device_handle(), TOTAL_MSGS, results_dev, pop_count_dev);

    // CPU pushes messages (blocking retry on full)
    uint8_t buf[56] = {};
    for (int i = 0; i < TOTAL_MSGS; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        while (!queue->push(buf)) {
            // Queue full, retry — GPU will pop and notify
        }
    }

    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Verify all messages received in order
    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, TOTAL_MSGS);

    std::vector<uint32_t> results(TOTAL_MSGS);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, TOTAL_MSGS * sizeof(uint32_t), hipMemcpyDeviceToHost));

    for (int i = 0; i < TOTAL_MSGS; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i)) << "Message " << i << " out of order";
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

// Test epoch increments correctly after wrapping
TEST(HostToDeviceConcurrentTest, EpochIncrement) {
    constexpr int CAPACITY = 4;
    auto queue = std::make_unique<SPSCH2DQueue>(CAPACITY);

    uint32_t* results_dev;
    int* pop_count_dev;
    KIWI_HIP_CHECK(hipMalloc(&results_dev, CAPACITY * sizeof(uint32_t)));
    KIWI_HIP_CHECK(hipMalloc(&pop_count_dev, sizeof(int)));

    // Batch 1: push CAPACITY messages, GPU pops them
    uint8_t buf[56] = {};
    for (int i = 0; i < CAPACITY; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        std::memcpy(buf, &val, sizeof(val));
        EXPECT_TRUE(queue->push(buf));
    }

    try_pop_kernel<<<1, 64>>>(queue->device_handle(), results_dev, pop_count_dev, CAPACITY);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    int pop_count = 0;
    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, CAPACITY);

    std::vector<uint32_t> results(CAPACITY);
    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, CAPACITY * sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < CAPACITY; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i));
    }

    // Batch 2: push CAPACITY more messages (should use epoch 1 now)
    for (int i = 0; i < CAPACITY; ++i) {
        uint32_t val = static_cast<uint32_t>(i + 100);
        std::memcpy(buf, &val, sizeof(val));
        EXPECT_TRUE(queue->push(buf));
    }

    try_pop_kernel<<<1, 64>>>(queue->device_handle(), results_dev, pop_count_dev, CAPACITY);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    KIWI_HIP_CHECK(hipMemcpy(&pop_count, pop_count_dev, sizeof(int), hipMemcpyDeviceToHost));
    EXPECT_EQ(pop_count, CAPACITY);

    KIWI_HIP_CHECK(hipMemcpy(results.data(), results_dev, CAPACITY * sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < CAPACITY; ++i) {
        EXPECT_EQ(results[i], static_cast<uint32_t>(i + 100));
    }

    KIWI_HIP_CHECK(hipFree(pop_count_dev));
    KIWI_HIP_CHECK(hipFree(results_dev));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
