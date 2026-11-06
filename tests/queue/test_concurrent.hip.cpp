// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_concurrent.hip
 * @brief Concurrent GPU-CPU tests for queue
 *
 * Tests wave-level push/pop operations, queue full/empty conditions,
 * and in-order delivery using full-wavefront (64 threads) cooperative push.
 * Each message is one 4-byte value (one ring line).
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

// Producer kernel: full wavefront cooperative push of N messages
// Launch with <<<1, 64>>> (full wavefront).
__global__ void producer_kernel(SPSCD2HQueue::DeviceHandle queue, int num_msgs) {
    for (int i = 0; i < num_msgs; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        while (!queue.push_reg(val)) {
            // Retry on full
        }
    }
}

// Fill queue kernel: full wavefront cooperative push, lane 0 tracks push_count
__global__ void fill_queue_kernel(SPSCD2HQueue::DeviceHandle queue, int capacity,
                                   int* push_count) {
    if (__lane_id() == 0) {
        *push_count = 0;
    }

    for (int i = 0; i < capacity + 10; ++i) {
        uint32_t val = static_cast<uint32_t>(i);
        if (queue.push_reg(val)) {
            if (__lane_id() == 0) {
                (*push_count)++;
            }
        } else {
            // Queue full, stop
            break;
        }
    }
}

// Test basic single message push/pop
TEST(ConcurrentTest, SingleMessage) {
    auto queue = std::make_unique<SPSCD2HQueue>(16);

    // Push one message from GPU (full wavefront)
    producer_kernel<<<1, 64>>>(queue->device_handle(), 1);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Pop on CPU
    uint8_t buf[kMaxMessageBytes] = {};
    EXPECT_TRUE(queue->pop(buf));
    uint32_t value = 0;
    std::memcpy(&value, buf, sizeof(value));
    EXPECT_EQ(value, 0);

    // Queue should be empty now
    EXPECT_FALSE(queue->pop(buf));
}

// Test multiple messages in order
TEST(ConcurrentTest, InOrderDelivery) {
    constexpr int NUM_MSGS = 100;
    auto queue = std::make_unique<SPSCD2HQueue>(128);

    // Push messages from GPU (full wavefront)
    producer_kernel<<<1, 64>>>(queue->device_handle(), NUM_MSGS);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Pop and verify order
    std::vector<uint32_t> received;
    uint8_t buf[kMaxMessageBytes];
    while (queue->pop(buf)) {
        uint32_t value = 0;
        std::memcpy(&value, buf, sizeof(value));
        received.push_back(value);
    }

    EXPECT_EQ(received.size(), NUM_MSGS);
    for (int i = 0; i < NUM_MSGS; ++i) {
        EXPECT_EQ(received[i], i) << "Message " << i << " out of order";
    }
}

// Test queue full behavior
TEST(ConcurrentTest, QueueFull) {
    constexpr int CAPACITY = 16;
    auto queue = std::make_unique<SPSCD2HQueue>(CAPACITY);

    int* push_count_device;
    KIWI_HIP_CHECK(hipMalloc(&push_count_device, sizeof(int)));

    // Try to overfill queue (full wavefront)
    fill_queue_kernel<<<1, 64>>>(queue->device_handle(), CAPACITY, push_count_device);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    int push_count;
    KIWI_HIP_CHECK(hipMemcpy(&push_count, push_count_device, sizeof(int), hipMemcpyDeviceToHost));

    // Should have pushed exactly capacity messages
    EXPECT_EQ(push_count, CAPACITY);

    // Pop all messages
    uint8_t buf[kMaxMessageBytes];
    int pop_count = 0;
    while (queue->pop(buf)) {
        pop_count++;
    }
    EXPECT_EQ(pop_count, CAPACITY);

    KIWI_HIP_CHECK(hipFree(push_count_device));
}

// Test queue empty behavior
TEST(ConcurrentTest, QueueEmpty) {
    auto queue = std::make_unique<SPSCD2HQueue>(16);

    // Queue is empty, pop should fail
    uint8_t buf[kMaxMessageBytes];
    EXPECT_FALSE(queue->pop(buf));
}

// Test wrap-around with small queue
TEST(ConcurrentTest, WrapAround) {
    constexpr int CAPACITY = 8;
    constexpr int TOTAL_MSGS = 100;
    auto queue = std::make_unique<SPSCD2HQueue>(CAPACITY);

    // Launch producer in background (full wavefront)
    std::thread producer([&]() {
        producer_kernel<<<1, 64>>>(queue->device_handle(), TOTAL_MSGS);
        KIWI_HIP_CHECK(hipDeviceSynchronize());
    });

    // Consumer pops as messages arrive
    std::vector<uint32_t> received;
    uint8_t buf[kMaxMessageBytes];

    while (received.size() < TOTAL_MSGS) {
        if (queue->pop(buf)) {
            uint32_t value = 0;
            std::memcpy(&value, buf, sizeof(value));
            received.push_back(value);
        } else {
            // Wait a bit for more messages
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    producer.join();

    // Verify all messages received in order
    EXPECT_EQ(received.size(), TOTAL_MSGS);
    for (int i = 0; i < TOTAL_MSGS; ++i) {
        EXPECT_EQ(received[i], i);
    }
}

// Test epoch increments correctly after wrapping
TEST(ConcurrentTest, EpochIncrement) {
    constexpr int CAPACITY = 4;
    auto queue = std::make_unique<SPSCD2HQueue>(CAPACITY);

    // Push and pop CAPACITY messages (one full cycle, full wavefront)
    producer_kernel<<<1, 64>>>(queue->device_handle(), CAPACITY);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    for (int i = 0; i < CAPACITY; ++i) {
        uint8_t buf[kMaxMessageBytes] = {};
        EXPECT_TRUE(queue->pop(buf));
        uint32_t value = 0;
        std::memcpy(&value, buf, sizeof(value));
        EXPECT_EQ(value, i);
    }

    // Push another batch (should have epoch 1 now)
    producer_kernel<<<1, 64>>>(queue->device_handle(), CAPACITY);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Pop and verify
    for (int i = 0; i < CAPACITY; ++i) {
        uint8_t buf[kMaxMessageBytes] = {};
        EXPECT_TRUE(queue->pop(buf));
        uint32_t value = 0;
        std::memcpy(&value, buf, sizeof(value));
        EXPECT_EQ(value, i);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
