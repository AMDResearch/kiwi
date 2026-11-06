// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_host_poll_state.hip.cpp
 * @brief Tests for cached host polling of SPSC device-to-host queues
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

using namespace kiwi::queue;

static_assert(
    sizeof(void*) != 8 || sizeof(SPSCD2HQueue::HostPollState) == 24,
    "HostPollState must remain 24 bytes on 64-bit hosts");

__global__ void push_one(
    SPSCD2HQueue::DeviceHandle queue,
    uint32_t value) {
    while (!queue.push_reg(value)) {}
}

// Two messages: one line, then two lines (48 bytes).
__global__ void push_two(
    SPSCD2HQueue::DeviceHandle queue,
    uint32_t first_value) {
    while (!queue.push_reg(first_value)) {}
    uint32_t second[12] = {first_value + 1};
    while (!queue.push_reg(second)) {}
}

static void expect_pop_value(SPSCD2HQueue& queue, uint32_t expected) {
    uint8_t payload[kMaxMessageBytes] = {};
    ASSERT_TRUE(queue.pop(payload));

    uint32_t actual = 0;
    std::memcpy(&actual, payload, sizeof(actual));
    EXPECT_EQ(actual, expected);
}

TEST(HostPollStateTest, InitiallyEmpty) {
    SPSCD2HQueue queue(4);
    auto state = queue.host_poll_state();

    EXPECT_FALSE(state.ready());
}

TEST(HostPollStateTest, ReadyAfterGpuPushAndRefreshAfterPop) {
    SPSCD2HQueue queue(4);
    auto state = queue.host_poll_state();

    EXPECT_FALSE(state.ready());

    push_one<<<1, 64>>>(queue.device_handle(), 42);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    EXPECT_TRUE(state.ready());
    expect_pop_value(queue, 42);

    state.refresh();
    EXPECT_FALSE(state.ready());
}

TEST(HostPollStateTest, RefreshTracksEachMessage) {
    SPSCD2HQueue queue(4);
    auto state = queue.host_poll_state();

    push_two<<<1, 64>>>(queue.device_handle(), 100);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    EXPECT_TRUE(state.ready());
    expect_pop_value(queue, 100);

    state.refresh();
    EXPECT_TRUE(state.ready());
    expect_pop_value(queue, 101);

    state.refresh();
    EXPECT_FALSE(state.ready());
}

TEST(HostPollStateTest, TracksRingAndEpochWrap) {
    constexpr size_t kCapacity = 4;  // three lines per cycle: wraps at every offset
    constexpr uint32_t kEpochCycles = 400;  // 300 laps: the 8-bit epoch wraps

    SPSCD2HQueue queue(kCapacity);
    auto state = queue.host_poll_state();

    for (uint32_t cycle = 0; cycle < kEpochCycles; ++cycle) {
        const uint32_t first_value = cycle * 2;
        push_two<<<1, 64>>>(queue.device_handle(), first_value);
        KIWI_HIP_CHECK(hipDeviceSynchronize());

        EXPECT_TRUE(state.ready());
        expect_pop_value(queue, first_value);

        state.refresh();
        EXPECT_TRUE(state.ready());
        expect_pop_value(queue, first_value + 1);

        state.refresh();
        EXPECT_FALSE(state.ready());
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
