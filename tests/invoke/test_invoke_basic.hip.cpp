// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/invoke/internal/serializer.hip.hpp>
#include <kiwi/invoke/internal/callback_wrapper.hpp>
#include <kiwi/invoke/internal/callback_registry.hpp>
#include <gtest/gtest.h>
#include <cstring>

using namespace kiwi::invoke::internal;

// Test argument serializer - no arguments
TEST(SerializerTest, NoArguments) {
    using Serializer = ArgumentSerializer<>;

    EXPECT_EQ(Serializer::total_bytes, 0);

    uint8_t buffer[1];
    Serializer::serialize(buffer);

    auto result = Serializer::deserialize(buffer);
    EXPECT_EQ(std::tuple_size_v<decltype(result)>, 0);
}

// Test argument serializer - single uint32_t
TEST(SerializerTest, SingleUint32) {
    using Serializer = ArgumentSerializer<uint32_t>;

    EXPECT_EQ(Serializer::total_bytes, 4);

    uint8_t buffer[4];
    Serializer::serialize(buffer, 0x12345678);

    auto result = Serializer::deserialize(buffer);
    EXPECT_EQ(std::get<0>(result), 0x12345678);
}

// Test argument serializer - two arguments (uint32_t, float)
TEST(SerializerTest, TwoArguments) {
    using Serializer = ArgumentSerializer<uint32_t, float>;

    EXPECT_EQ(Serializer::total_bytes, 8);

    uint8_t buffer[8];
    Serializer::serialize(buffer, 42u, 3.14f);

    auto result = Serializer::deserialize(buffer);
    EXPECT_EQ(std::get<0>(result), 42u);
    EXPECT_FLOAT_EQ(std::get<1>(result), 3.14f);
}

// Test argument serializer - three arguments
TEST(SerializerTest, ThreeArguments) {
    using Serializer = ArgumentSerializer<uint32_t, uint32_t, uint64_t>;

    EXPECT_EQ(Serializer::total_bytes, 16);

    uint8_t buffer[16];
    Serializer::serialize(buffer, 111u, 222u, 333ull);

    auto result = Serializer::deserialize(buffer);
    EXPECT_EQ(std::get<0>(result), 111u);
    EXPECT_EQ(std::get<1>(result), 222u);
    EXPECT_EQ(std::get<2>(result), 333ull);
}

// Test argument serializer - large struct
TEST(SerializerTest, LargeStruct) {
    struct LargePayload {
        uint64_t timestamp;
        uint32_t data[11];  // Total: 8 + 44 = 52 bytes
    } __attribute__((packed));

    using Serializer = ArgumentSerializer<LargePayload>;

    size_t expected_size = sizeof(LargePayload);
    EXPECT_EQ(Serializer::total_bytes, expected_size);

    LargePayload payload;
    payload.timestamp = 0xDEADBEEFCAFEBABE;
    for (int i = 0; i < 11; ++i) {
        payload.data[i] = i * 100;
    }

    uint8_t buffer[64];  // Use actual size
    Serializer::serialize(buffer, payload);

    auto result = Serializer::deserialize(buffer);
    const auto& restored = std::get<0>(result);

    EXPECT_EQ(restored.timestamp, 0xDEADBEEFCAFEBABE);
    for (int i = 0; i < 11; ++i) {
        EXPECT_EQ(restored.data[i], i * 100);
    }
}

// Test callback wrapper - no arguments
TEST(CallbackWrapperTest, NoArguments) {
    int counter = 0;
    std::function<void()> func = [&]() { counter++; };
    auto wrapper = CallbackWrapper::create<>(func);

    EXPECT_EQ(wrapper.expected_arg_bytes(), 0);

    uint8_t dummy[1];
    wrapper.execute(dummy, 0);
    EXPECT_EQ(counter, 1);

    wrapper.execute(dummy, 0);
    EXPECT_EQ(counter, 2);
}

// Test callback wrapper - with arguments
TEST(CallbackWrapperTest, WithArguments) {
    uint32_t received_id = 0;
    float received_value = 0.0f;

    std::function<void(uint32_t, float)> func = [&](uint32_t id, float value) {
        received_id = id;
        received_value = value;
    };
    auto wrapper = CallbackWrapper::create<uint32_t, float>(func);

    EXPECT_EQ(wrapper.expected_arg_bytes(), 8);

    // Serialize arguments
    uint8_t buffer[8];
    ArgumentSerializer<uint32_t, float>::serialize(buffer, 42u, 3.14f);

    wrapper.execute(buffer, 8);

    EXPECT_EQ(received_id, 42u);
    EXPECT_FLOAT_EQ(received_value, 3.14f);
}

// Test callback wrapper - size mismatch error
TEST(CallbackWrapperTest, SizeMismatch) {
    std::function<void(uint32_t)> func = [](uint32_t) {};
    auto wrapper = CallbackWrapper::create<uint32_t>(func);

    EXPECT_EQ(wrapper.expected_arg_bytes(), 4);

    uint8_t buffer[8];
    EXPECT_THROW(wrapper.execute(buffer, 8), std::runtime_error);
}

// Test callback registry - basic registration
TEST(CallbackRegistryTest, BasicRegistration) {
    CallbackRegistry registry;

    int counter = 0;
    std::function<void()> func1 = [&]() { counter++; };
    uint8_t id1 = registry.register_callback<>(func1);

    EXPECT_EQ(id1, 0);
    EXPECT_EQ(registry.size(), 1);
    EXPECT_EQ(registry.expected_arg_bytes(id1), 0);

    uint32_t value = 0;
    std::function<void(uint32_t)> func2 = [&](uint32_t v) { value = v; };
    uint8_t id2 = registry.register_callback<uint32_t>(func2);

    EXPECT_EQ(id2, 1);
    EXPECT_EQ(registry.size(), 2);
    EXPECT_EQ(registry.expected_arg_bytes(id2), 4);
}

// Test callback registry - execution
TEST(CallbackRegistryTest, Execution) {
    CallbackRegistry registry;

    int result = 0;
    std::function<void(uint32_t, uint32_t)> func = [&](uint32_t a, uint32_t b) { result = a + b; };
    uint8_t id = registry.register_callback<uint32_t, uint32_t>(func);

    // Serialize arguments
    uint8_t buffer[8];
    ArgumentSerializer<uint32_t, uint32_t>::serialize(buffer, 10u, 20u);

    // Execute via wrapper
    const auto& wrapper = registry.get(id);
    wrapper.execute(buffer, 8);

    EXPECT_EQ(result, 30);
}

// Test callback registry - invalid ID
TEST(CallbackRegistryTest, InvalidId) {
    CallbackRegistry registry;

    EXPECT_THROW(registry.get(0), std::out_of_range);
    EXPECT_THROW(registry.expected_arg_bytes(0), std::out_of_range);

    std::function<void()> func = []() {};
    registry.register_callback<>(func);

    EXPECT_THROW(registry.get(1), std::out_of_range);
}

// Test callback registry - full (256 callbacks)
TEST(CallbackRegistryTest, RegistryFull) {
    CallbackRegistry registry;

    std::function<void()> func = []() {};

    // Register 256 callbacks
    for (int i = 0; i < 256; ++i) {
        uint8_t id = registry.register_callback<>(func);
        EXPECT_EQ(id, i);
    }

    EXPECT_EQ(registry.size(), 256);

    // Try to register 257th callback
    EXPECT_THROW(
        registry.register_callback<>(func),
        std::runtime_error
    );
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
