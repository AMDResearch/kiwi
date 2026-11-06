// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file simple_example.hip
 * @brief Simple example demonstrating queue usage
 *
 * Shows basic producer-consumer pattern with GPU pushing and CPU popping.
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cstring>

using namespace kiwi::queue;

// GPU producer kernel: full wavefront cooperative push
// Launch with <<<1, 64>>> (full wavefront).
__global__ void producer_kernel(SPSCD2HQueue::DeviceHandle queue, int num_messages) {
    if (__lane_id() == 0) {
        printf("GPU: Starting to push %d messages...\n", num_messages);
    }

    for (int i = 0; i < num_messages; ++i) {
        uint32_t message = static_cast<uint32_t>(i * 100);

        // Push the 4-byte message from registers (retry if the queue is
        // full) -- full wavefront cooperates
        while (!queue.push_reg(message)) {
            // Queue full, retry
        }

        // Optionally print progress (only lane 0 prints)
        if (__lane_id() == 0 && i % 100 == 0 && i > 0) {
            printf("GPU: Pushed %d messages\n", i);
        }
    }

    if (__lane_id() == 0) {
        printf("GPU: Finished pushing all messages\n");
    }
}

int main(int argc, char** argv) {
    int num_messages = 1000;
    int queue_capacity = 256;

    if (argc > 1) {
        num_messages = std::atoi(argv[1]);
    }
    if (argc > 2) {
        queue_capacity = std::atoi(argv[2]);
    }

    std::cout << "=== queue Simple Example ===\n\n";

    // 1. Create queue with default configuration
    std::cout << "Creating queue...\n";
    std::cout << "  Capacity: " << queue_capacity << " lines (64 B each)\n";

    try {
        auto queue = std::make_unique<SPSCD2HQueue>(queue_capacity);
        std::cout << "  Queue created successfully\n\n";

        // 2. Launch GPU producer kernel
        std::cout << "Launching GPU producer kernel...\n";
        std::cout << "  Messages to send: " << num_messages << "\n\n";

        producer_kernel<<<1, 64>>>(queue->device_handle(), num_messages);

        // 3. CPU consumer loop
        std::cout << "CPU: Starting consumer loop...\n";

        int received_count = 0;
        uint8_t buf[kMaxMessageBytes];

        while (received_count < num_messages) {
            if (queue->pop(buf)) {
                // Extract and verify value
                uint32_t value = 0;
                std::memcpy(&value, buf, sizeof(value));
                if (static_cast<int>(value) != received_count * 100) {
                    std::cerr << "ERROR: Expected " << received_count * 100
                              << " but got " << value << "\n";
                }

                received_count++;

                // Print progress
                if (received_count % 100 == 0) {
                    std::cout << "CPU: Received " << received_count << " messages\n";
                }
            }
            // If queue is empty, CPU can do other work here
        }

        std::cout << "\nCPU: Finished receiving all messages\n";

        // 4. Wait for GPU to finish
        KIWI_HIP_CHECK(hipDeviceSynchronize());
        std::cout << "GPU: Kernel finished\n";

        // 5. Verify final state
        std::cout << "\n=== Verification ===\n";
        std::cout << "  Messages sent (GPU):     " << num_messages << "\n";
        std::cout << "  Messages received (CPU): " << received_count << "\n";

        if (received_count == num_messages) {
            std::cout << "  All messages delivered successfully\n";
        } else {
            std::cout << "  Message count mismatch!\n";
            return 1;
        }

        // Queue should be empty now
        if (queue->pop(buf)) {
            std::cout << "  Queue not empty (unexpected)\n";
            return 1;
        } else {
            std::cout << "  Queue is empty (as expected)\n";
        }

        std::cout << "\n=== Example completed successfully! ===\n";

    } catch (const queue_error& e) {
        std::cerr << "Queue error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
