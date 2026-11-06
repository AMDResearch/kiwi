// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/error.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>

using DeviceContext = kiwi::invoke::DeviceContext<>;

// GPU kernel demonstrating multi-endpoint callback usage
// Full wavefront (64 threads) cooperate on each wave-level invoke
__global__ void sensor_monitoring_kernel(
    DeviceContext* ctx,
    uint8_t signal_cb,
    uint8_t sensor_cb,
    uint8_t alert_cb,
    size_t num_endpoints,
    uint32_t num_blocks,
    uint32_t threads_per_block
) {
    // Cache handles so producer_index state persists across iterations
    KIWI_QUEUE_ASSERT(num_endpoints <= 4);
    DeviceContext::DeviceHandle handles[4];
    for (size_t i = 0; i < num_endpoints; ++i) {
        handles[i] = ctx->get_device_handle(i);
    }

    // Loop through all "virtual" blocks and threads
    for (uint32_t bid = 0; bid < num_blocks; ++bid) {
        size_t endpoint_id = bid % num_endpoints;

        for (uint32_t tid = 0; tid < threads_per_block; ++tid) {
            // Simulate sensor reading
            float sensor_value = bid * 10.0f + tid * 1.5f;

            // Invoke sensor callback with block ID, thread ID, and sensor value
            handles[endpoint_id].invoke(sensor_cb,
                        static_cast<uint32_t>(bid),
                        static_cast<uint32_t>(tid),
                        sensor_value);

            // If sensor value exceeds threshold, send alert
            if (sensor_value > 50.0f) {
                uint32_t alert_code = 1001;  // High temperature alert
                handles[endpoint_id].invoke(alert_cb, alert_code,
                            static_cast<uint32_t>(bid),
                            static_cast<uint32_t>(tid));
            }

            // Send completion signal
            handles[endpoint_id].invoke(signal_cb);
        }
    }
}

int main() {
    std::cout << "=== KIWI Invoke Multi-Endpoint Example ===" << std::endl;
    std::cout << std::endl;

    // Configuration
    const size_t num_endpoints = 4;
    const size_t queue_capacity = 1024;
    const size_t num_blocks = 8;
    const size_t threads_per_block = 32;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Endpoints: " << num_endpoints << std::endl;
    std::cout << "  Queue capacity per endpoint: " << queue_capacity << std::endl;
    std::cout << "  GPU blocks: " << num_blocks << std::endl;
    std::cout << "  Threads per block: " << threads_per_block << std::endl;
    std::cout << std::endl;

    // Create context with 4 endpoints
    kiwi::invoke::Context ctx(num_endpoints, queue_capacity);

    // Register callbacks (global across all endpoints)
    std::atomic<int> signal_count{0};
    std::function<void()> signal_func = [&]() { signal_count++; };
    uint8_t signal_cb = ctx.register_callback(signal_func);

    std::vector<std::vector<float>> sensor_values(num_blocks,
                                                   std::vector<float>(threads_per_block));
    std::function<void(uint32_t, uint32_t, float)> sensor_func =
        [&](uint32_t block_id, uint32_t thread_id, float value) {
            if (block_id < num_blocks && thread_id < threads_per_block) {
                sensor_values[block_id][thread_id] = value;
            }
        };
    uint8_t sensor_cb = ctx.register_callback<uint32_t, uint32_t, float>(sensor_func);

    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> alerts;
    std::function<void(uint32_t, uint32_t, uint32_t)> alert_func =
        [&](uint32_t alert_code, uint32_t block_id, uint32_t thread_id) {
            alerts.push_back(std::make_tuple(alert_code, block_id, thread_id));
        };
    uint8_t alert_cb = ctx.register_callback<uint32_t, uint32_t, uint32_t>(alert_func);

    std::cout << "Registered callbacks:" << std::endl;
    std::cout << "  Signal callback ID: " << static_cast<int>(signal_cb) << std::endl;
    std::cout << "  Sensor callback ID: " << static_cast<int>(sensor_cb) << std::endl;
    std::cout << "  Alert callback ID: " << static_cast<int>(alert_cb) << std::endl;
    std::cout << std::endl;

    // Launch GPU kernel — DeviceContext already has handle array
    std::cout << "Launching GPU kernel..." << std::endl;
    sensor_monitoring_kernel<<<1, 64>>>(
        ctx.device_context(),
        signal_cb,
        sensor_cb,
        alert_cb,
        num_endpoints,
        num_blocks,
        threads_per_block
    );

    // CPU processing loop - poll all endpoints
    std::cout << "Processing callbacks..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    size_t expected_signals = num_blocks * threads_per_block;
    while (signal_count.load() < static_cast<int>(expected_signals)) {
        size_t total_handled = 0;
        for (size_t i = 0; i < ctx.num_endpoints(); ++i) {
            size_t handled = ctx.progress(i);
            total_handled += handled;
        }

        if (total_handled == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    // Synchronize GPU
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    // Final progress to catch any remaining callbacks
    for (size_t i = 0; i < ctx.num_endpoints(); ++i) {
        ctx.progress(i);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time
    );

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Received " << signal_count.load() << " signals" << std::endl;
    std::cout << "Received " << alerts.size() << " alerts" << std::endl;
    std::cout << "Processing time: " << duration.count() << " us" << std::endl;
    std::cout << std::endl;

    // Display some sensor values
    std::cout << "Sample sensor values (first 4 blocks, first 8 threads):" << std::endl;
    for (size_t b = 0; b < 4 && b < num_blocks; ++b) {
        std::cout << "  Block " << b << ": ";
        for (size_t t = 0; t < 8 && t < threads_per_block; ++t) {
            std::cout << std::fixed << std::setprecision(1)
                      << sensor_values[b][t] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Display alerts
    if (!alerts.empty()) {
        std::cout << "Alerts (first 10):" << std::endl;
        for (size_t i = 0; i < std::min(alerts.size(), size_t(10)); ++i) {
            auto [code, block, thread] = alerts[i];
            std::cout << "  Alert " << code << " from block " << block
                      << ", thread " << thread << std::endl;
        }
        std::cout << std::endl;
    }

    // Verify all sensors reported
    int missing_count = 0;
    for (size_t b = 0; b < num_blocks; ++b) {
        for (size_t t = 0; t < threads_per_block; ++t) {
            float expected = b * 10.0f + t * 1.5f;
            if (std::abs(sensor_values[b][t] - expected) > 0.01f) {
                missing_count++;
            }
        }
    }

    if (missing_count == 0) {
        std::cout << "All sensor values received correctly!" << std::endl;
    } else {
        std::cout << "Missing or incorrect values: " << missing_count << std::endl;
    }

    return 0;
}
