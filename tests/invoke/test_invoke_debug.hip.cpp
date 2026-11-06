// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/error.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

using DeviceContext = kiwi::invoke::DeviceContext<>;

// Simple kernel to test single uint32_t argument
// Full wavefront: all 64 threads participate in invoke
__global__ void test_uint32_kernel(
    DeviceContext* ctx,
    uint8_t callback_id,
    uint32_t value
) {
    auto handle = ctx->get_device_handle(0);
    if (__lane_id() == 0) {
        printf("GPU: About to invoke with value=%u\n", value);
    }
    bool success = handle.invoke(callback_id, value);
    if (__lane_id() == 0) {
        printf("GPU: invoke returned %s\n", success ? "true" : "false");
    }
}

// Test with two arguments
// Full wavefront: all 64 threads participate in invoke
__global__ void test_two_args_kernel(
    DeviceContext* ctx,
    uint8_t callback_id,
    uint32_t id,
    float value
) {
    auto handle = ctx->get_device_handle(0);
    if (__lane_id() == 0) {
        printf("GPU: About to invoke with id=%u, value=%f\n", id, value);
    }
    bool success = handle.invoke(callback_id, id, value);
    if (__lane_id() == 0) {
        printf("GPU: invoke returned %s\n", success ? "true" : "false");
    }
}

TEST(KiwiInvokeDebugTest, SingleUint32) {
    std::cout << "\n=== Testing single uint32_t argument ===" << std::endl;

    kiwi::invoke::Context ctx(1, 1024);

    uint32_t received = 0;
    std::function<void(uint32_t)> func = [&](uint32_t val) {
        std::cout << "CPU callback: received value=" << val << std::endl;
        received = val;
    };

    uint8_t callback_id = ctx.register_callback<uint32_t>(func);
    std::cout << "Registered callback with ID: " << static_cast<int>(callback_id) << std::endl;

    uint32_t test_value = 42;
    std::cout << "Launching kernel with value=" << test_value << std::endl;

    test_uint32_kernel<<<1, 64>>>(ctx.device_context(), callback_id, test_value);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Calling progress()..." << std::endl;
    size_t executed = ctx.progress(0);

    std::cout << "Executed " << executed << " callbacks" << std::endl;
    std::cout << "Received value: " << received << std::endl;

    EXPECT_EQ(executed, 1);
    EXPECT_EQ(received, test_value);
}

TEST(KiwiInvokeDebugTest, TwoArguments) {
    std::cout << "\n=== Testing two arguments (uint32_t, float) ===" << std::endl;

    kiwi::invoke::Context ctx(1, 1024);

    uint32_t received_id = 0;
    float received_value = 0.0f;

    std::function<void(uint32_t, float)> func = [&](uint32_t id, float val) {
        std::cout << "CPU callback: received id=" << id << ", value=" << val << std::endl;
        received_id = id;
        received_value = val;
    };

    uint8_t callback_id = ctx.register_callback<uint32_t, float>(func);
    std::cout << "Registered callback with ID: " << static_cast<int>(callback_id) << std::endl;

    uint32_t test_id = 123;
    float test_value = 3.14f;
    std::cout << "Launching kernel with id=" << test_id << ", value=" << test_value << std::endl;

    test_two_args_kernel<<<1, 64>>>(ctx.device_context(), callback_id, test_id, test_value);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    std::cout << "Calling progress()..." << std::endl;
    size_t executed = ctx.progress(0);

    std::cout << "Executed " << executed << " callbacks" << std::endl;
    std::cout << "Received id: " << received_id << ", value: " << received_value << std::endl;

    EXPECT_EQ(executed, 1);
    EXPECT_EQ(received_id, test_id);
    EXPECT_FLOAT_EQ(received_value, test_value);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
