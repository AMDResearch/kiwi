// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/error.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>

using DeviceContext = kiwi::invoke::DeviceContext<>;

// Simple signal kernel - no arguments
// Full wavefront: all 64 threads participate in invoke
__global__ void signal_kernel(
    DeviceContext* ctx,
    uint8_t callback_id
) {
    auto handle = ctx->get_device_handle(0);
    handle.invoke(callback_id);
}

// Repeated signal kernel - invoke N times from a single kernel launch
// DeviceHandle state (producer_index) persists across iterations within one launch
__global__ void repeat_signal_kernel(
    DeviceContext* ctx,
    uint8_t callback_id,
    int count
) {
    auto handle = ctx->get_device_handle(0);
    for (int i = 0; i < count; ++i) {
        while (!handle.invoke(callback_id)) {
            // Retry on full queue
        }
    }
}

// Sensor data kernel - 2 arguments
// Full wavefront: all 64 threads participate in invoke
__global__ void sensor_kernel(
    DeviceContext* ctx,
    uint8_t callback_id,
    uint32_t num_sensors
) {
    auto handle = ctx->get_device_handle(0);
    for (uint32_t id = 0; id < num_sensors; ++id) {
        float value = id * 1.5f;
        handle.invoke(callback_id, id, value);
    }
}

// Multi-endpoint kernel - explicit routing via DeviceContext
// Full wavefront: all 64 threads participate in invoke
// Note: handles are cached in local vars so producer_index persists across iterations
__global__ void multi_endpoint_kernel(
    DeviceContext* ctx,
    uint8_t callback_id,
    size_t num_endpoints,
    uint32_t num_blocks
) {
    // Cache handles so producer_index state persists across pushes
    KIWI_QUEUE_ASSERT(num_endpoints <= 4);
    DeviceContext::DeviceHandle handles[4];
    for (size_t i = 0; i < num_endpoints; ++i) {
        handles[i] = ctx->get_device_handle(i);
    }

    for (uint32_t block_id = 0; block_id < num_blocks; ++block_id) {
        size_t endpoint_id = block_id % num_endpoints;
        handles[endpoint_id].invoke(callback_id, block_id);
    }
}

// Error reporting kernel - 3 arguments
// Full wavefront: all 64 threads participate in invoke
__global__ void error_kernel(
    DeviceContext* ctx,
    uint8_t callback_id,
    uint32_t num_errors
) {
    auto handle = ctx->get_device_handle(0);
    for (uint32_t i = 0; i < num_errors; ++i) {
        uint32_t error_code = 404;
        uint32_t thread_id = i;
        uint64_t timestamp = 1234567890ULL + i;
        handle.invoke(callback_id, error_code, thread_id, timestamp);
    }
}

// Complex data kernel - large struct
struct LargePayload {
    uint64_t timestamp;
    uint32_t data[13];  // Total: 8 + 52 = 60 bytes
};

struct PartialMessagePayload {
    uint8_t bytes[64];
};

// Full wavefront: all 64 threads participate in invoke
__global__ void complex_kernel(
    DeviceContext* ctx,
    uint8_t callback_id,
    uint32_t num_payloads
) {
    auto handle = ctx->get_device_handle(0);
    for (uint32_t tid = 0; tid < num_payloads; ++tid) {
        LargePayload lp;
        lp.timestamp = 0xDEADBEEFCAFEBABE;
        for (int i = 0; i < 13; ++i) {
            lp.data[i] = i * 100 + tid;
        }
        handle.invoke(callback_id, lp);
    }
}

// Write one two-line invoke message in two steps: the chunks of line 0
// (lanes 0-3, including chunk 0 with the length and callback id) first, the
// chunks of line 1 (lanes 4-7) in a later launch. In between, the message is
// ready() but not whole.
__global__ void publish_message_lines(
    kiwi::queue::SPSCD2HQueue::DeviceHandle queue,
    uint8_t callback_id,
    PartialMessagePayload payload,
    bool second_line
) {
    const uint64_t word = kiwi::queue::detail::select_lane_word(payload);
    const unsigned lane = __lane_id();
    if ((lane >= 4) == second_line) {
        kiwi::queue::detail::write_message(
            queue.chunks, queue.log2_capacity, queue.producer_index, word,
            sizeof(payload), callback_id, kiwi::queue::message_lines(sizeof(payload)));
    }
}

// Multiple callbacks kernel
// Full wavefront: all 64 threads participate in invoke
__global__ void multi_callback_kernel(
    DeviceContext* ctx,
    uint8_t cb1,
    uint8_t cb2,
    uint8_t cb3
) {
    auto handle = ctx->get_device_handle(0);
    handle.invoke(cb1);
    handle.invoke(cb2, static_cast<uint32_t>(42));
    handle.invoke(cb3, static_cast<uint32_t>(10), static_cast<float>(3.14f));
}

// Test: Simple signal (no arguments)
TEST(KiwiInvokeConcurrentTest, SimpleSignal) {
    kiwi::invoke::Context ctx(1, 1024);

    std::atomic<int> counter{0};
    std::function<void()> func = [&]() { counter++; };
    uint8_t callback_id = ctx.register_callback(func);

    signal_kernel<<<1, 64>>>(ctx.device_context(), callback_id);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    size_t executed = ctx.progress(0);
    EXPECT_EQ(executed, 1);
    EXPECT_EQ(counter.load(), 1);
}

// Test: Sensor data (2 arguments)
TEST(KiwiInvokeConcurrentTest, SensorData) {
    kiwi::invoke::Context ctx(1, 1024);

    std::vector<float> results(32, 0.0f);
    std::function<void(uint32_t, float)> func = [&](uint32_t id, float value) {
        if (id < 32) {
            results[id] = value;
        }
    };
    uint8_t callback_id = ctx.register_callback<uint32_t, float>(func);

    sensor_kernel<<<1, 64>>>(ctx.device_context(), callback_id, 32);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    size_t executed = ctx.progress(0);
    EXPECT_EQ(executed, 32);

    // Verify results
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(results[i], i * 1.5f);
    }
}

// Test: Multi-endpoint with explicit routing
TEST(KiwiInvokeConcurrentTest, MultiEndpoint) {
    const size_t num_endpoints = 4;
    kiwi::invoke::Context ctx(num_endpoints, 1024);

    std::vector<std::atomic<int>> received(8);
    for (auto& r : received) {
        r = 0;
    }

    std::function<void(uint32_t)> func = [&](uint32_t block_id) {
        if (block_id < 8) {
            received[block_id]++;
        }
    };
    uint8_t callback_id = ctx.register_callback<uint32_t>(func);

    multi_endpoint_kernel<<<1, 64>>>(ctx.device_context(), callback_id, num_endpoints, 8);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Process all endpoints
    size_t total = 0;
    for (size_t i = 0; i < num_endpoints; ++i) {
        total += ctx.progress(i);
    }

    EXPECT_EQ(total, 8);

    // Verify each block sent message
    for (int i = 0; i < 8; ++i) {
        EXPECT_EQ(received[i].load(), 1) << "Block " << i;
    }
}

// Test: Error reporting (3 arguments, 16 bytes)
TEST(KiwiInvokeConcurrentTest, ErrorReporting) {
    kiwi::invoke::Context ctx(1, 1024);

    struct ErrorReport {
        uint32_t error_code;
        uint32_t thread_id;
        uint64_t timestamp;
    };

    std::vector<ErrorReport> reports;
    std::function<void(uint32_t, uint32_t, uint64_t)> func =
        [&](uint32_t error_code, uint32_t thread_id, uint64_t timestamp) {
            reports.push_back({error_code, thread_id, timestamp});
        };
    uint8_t callback_id = ctx.register_callback<uint32_t, uint32_t, uint64_t>(func);

    error_kernel<<<1, 64>>>(ctx.device_context(), callback_id, 8);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    size_t executed = ctx.progress(0);
    EXPECT_EQ(executed, 8);
    EXPECT_EQ(reports.size(), 8);

    // Verify reports
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(reports[i].error_code, 404u);
        EXPECT_EQ(reports[i].thread_id, i);
        EXPECT_EQ(reports[i].timestamp, 1234567890ULL + i);
    }
}

// Test: Complex data (60-byte struct)
TEST(KiwiInvokeConcurrentTest, ComplexData) {
    kiwi::invoke::Context ctx(1, 1024);

    std::vector<LargePayload> payloads;
    std::function<void(LargePayload)> func =
        [&](LargePayload lp) {
            payloads.push_back(lp);
        };
    uint8_t callback_id = ctx.register_callback<LargePayload>(func);

    complex_kernel<<<1, 64>>>(ctx.device_context(), callback_id, 4);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    size_t executed = ctx.progress(0);
    EXPECT_EQ(executed, 4);
    EXPECT_EQ(payloads.size(), 4);

    // Verify payloads
    for (size_t tid = 0; tid < 4; ++tid) {
        EXPECT_EQ(payloads[tid].timestamp, 0xDEADBEEFCAFEBABE);
        for (int i = 0; i < 13; ++i) {
            EXPECT_EQ(payloads[tid].data[i], i * 100 + tid);
        }
    }
}

TEST(KiwiInvokeConcurrentTest, ProgressSkipsMessageWithLateChunk) {
    constexpr size_t endpoint_id = 0;
    kiwi::invoke::Context ctx(1, 4);

    PartialMessagePayload sent{};
    for (size_t i = 0; i < sizeof(sent.bytes); ++i) {
        sent.bytes[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    static_assert(kiwi::queue::message_lines(sizeof(sent)) == 2);

    PartialMessagePayload received{};
    int callback_count = 0;
    std::function<void(PartialMessagePayload)> callback =
        [&](PartialMessagePayload payload) {
            received = payload;
            ++callback_count;
        };
    const uint8_t callback_id =
        ctx.register_callback<PartialMessagePayload>(callback);

    auto* queue = ctx.raw_queue(endpoint_id);
    auto poll_state = queue->host_poll_state();
    EXPECT_FALSE(poll_state.ready());

    const auto handle = queue->device_handle();
    publish_message_lines<<<1, 64>>>(handle, callback_id, sent, false);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Chunk 0 is visible, the second line is not: progress() returns 0
    // without consuming the message or running the callback.
    ASSERT_TRUE(poll_state.ready());
    EXPECT_EQ(ctx.progress(endpoint_id), 0u);
    EXPECT_EQ(callback_count, 0);
    poll_state.refresh();
    EXPECT_TRUE(poll_state.ready());

    publish_message_lines<<<1, 64>>>(handle, callback_id, sent, true);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    EXPECT_EQ(ctx.progress(endpoint_id), 1u);
    poll_state.refresh();
    EXPECT_FALSE(poll_state.ready());

    EXPECT_EQ(callback_count, 1);
    for (size_t i = 0; i < sizeof(sent.bytes); ++i) {
        EXPECT_EQ(received.bytes[i], sent.bytes[i]) << "byte " << i;
    }
}

// Test: Multiple callbacks interleaved
TEST(KiwiInvokeConcurrentTest, MultipleCallbacks) {
    kiwi::invoke::Context ctx(1, 1024);

    std::atomic<int> count1{0};
    std::atomic<int> count2{0};
    std::atomic<int> count3{0};

    std::function<void()> func1 = [&]() { count1++; };
    std::function<void(uint32_t)> func2 = [&](uint32_t v) {
        count2++;
        EXPECT_EQ(v, 42u);
    };
    std::function<void(uint32_t, float)> func3 = [&](uint32_t v, float f) {
        count3++;
        EXPECT_EQ(v, 10u);
        EXPECT_FLOAT_EQ(f, 3.14f);
    };

    uint8_t cb1 = ctx.register_callback(func1);
    uint8_t cb2 = ctx.register_callback<uint32_t>(func2);
    uint8_t cb3 = ctx.register_callback<uint32_t, float>(func3);

    multi_callback_kernel<<<1, 64>>>(ctx.device_context(), cb1, cb2, cb3);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    size_t executed = ctx.progress(0);
    EXPECT_EQ(executed, 3);
    EXPECT_EQ(count1.load(), 1);
    EXPECT_EQ(count2.load(), 1);
    EXPECT_EQ(count3.load(), 1);
}

// Test: Bounded progress (max_callbacks parameter)
TEST(KiwiInvokeConcurrentTest, BoundedProgress) {
    kiwi::invoke::Context ctx(1, 1024);

    std::atomic<int> counter{0};
    std::function<void()> func = [&]() { counter++; };
    uint8_t callback_id = ctx.register_callback(func);

    // Send 10 signals from a single kernel (DeviceHandle state persists within launch)
    repeat_signal_kernel<<<1, 64>>>(ctx.device_context(), callback_id, 10);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Process 3 callbacks
    size_t executed = ctx.progress(0, 3);
    EXPECT_EQ(executed, 3);
    EXPECT_EQ(counter.load(), 3);

    // Process 5 more
    executed = ctx.progress(0, 5);
    EXPECT_EQ(executed, 5);
    EXPECT_EQ(counter.load(), 8);

    // Process remaining (should be 2)
    executed = ctx.progress(0);
    EXPECT_EQ(executed, 2);
    EXPECT_EQ(counter.load(), 10);
}

// Test: Queue full handling
// Note: This is hard to test reliably because we'd need to fill the queue
// For now, we just verify that the system handles a large number of messages
TEST(KiwiInvokeConcurrentTest, HighThroughput) {
    kiwi::invoke::Context ctx(1, 1024);

    std::atomic<int> counter{0};
    std::function<void()> func = [&]() { counter++; };
    uint8_t callback_id = ctx.register_callback(func);

    // Send many signals from a single kernel (DeviceHandle state persists within launch)
    repeat_signal_kernel<<<1, 64>>>(ctx.device_context(), callback_id, 100);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Process all
    size_t executed = ctx.progress(0);
    EXPECT_EQ(executed, 100);
    EXPECT_EQ(counter.load(), 100);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
