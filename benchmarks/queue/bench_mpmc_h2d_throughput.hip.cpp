// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_mpmc_h2d_throughput.hip.cpp
 * @brief Measure message throughput for MPMC Host-to-Device queue
 *
 * Multiple CPU threads push, multiple GPU wavefronts pop.
 * Measures total msgs/sec and per-thread/per-block breakdown.
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <bench_common.hpp>
#include <hip/hip_runtime.h>
#include <CLI/CLI.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <pthread.h>
#include <cstring>

using namespace kiwi::queue;

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_messages = 1000000;
    int num_cpu_threads = 1;
    int num_gpu_blocks = 1;
    bool pin_cpu = false;
    bool verbose = false;
};

static void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

// GPU consumer kernel: each block pops msgs_per_block messages.
// Launch with <<<num_blocks, 64>>>.
__global__ void mpmc_h2d_consumer_kernel(MPMCH2DQueue::DeviceHandle queue,
                                          int msgs_per_block,
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
            block_pop_counts[block_id]++;
        }
    }
    queue.flush_consumer_index();
}

// CPU producer: pushes msgs_per_thread messages as fast as possible
static void producer_thread_fn(MPMCH2DQueue* queue, int tid,
                                int msgs_per_thread, bool pin_cpu,
                                std::atomic<int>& thread_pushed) {
    if (pin_cpu) pin_thread_to_core(tid);

    uint8_t buf[56] = {};
    int pushed = 0;
    for (int i = 0; i < msgs_per_thread; ++i) {
        uint32_t val = static_cast<uint32_t>(tid * msgs_per_thread + i);
        std::memcpy(buf, &val, sizeof(val));
        while (!queue->push(buf)) {
            __builtin_ia32_pause();
        }
        pushed++;
    }
    thread_pushed.store(pushed);
}

void benchmark_throughput(const ExperimentParams& params) {
    int msgs_per_thread = params.num_messages / params.num_cpu_threads;
    int msgs_per_block = params.num_messages / params.num_gpu_blocks;
    int total_messages = msgs_per_thread * params.num_cpu_threads;
    // Ensure GPU consumes the same total
    int total_gpu = msgs_per_block * params.num_gpu_blocks;
    if (total_gpu != total_messages) {
        // Adjust msgs_per_block to match
        msgs_per_block = total_messages / params.num_gpu_blocks;
        total_gpu = msgs_per_block * params.num_gpu_blocks;
        // Distribute remainder to block 0 if needed — for simplicity, require divisibility
        if (total_gpu != total_messages) {
            std::cerr << "Warning: total messages (" << total_messages
                      << ") not evenly divisible by GPU blocks (" << params.num_gpu_blocks
                      << "). Using " << total_gpu << " messages.\n";
            total_messages = total_gpu;
            msgs_per_thread = total_messages / params.num_cpu_threads;
        }
    }

    auto queue = std::make_unique<MPMCH2DQueue>(params.queue_capacity);

    // Allocate device memory for per-block pop counts
    int* block_pop_counts_dev;
    KIWI_HIP_CHECK(hipMalloc(&block_pop_counts_dev, params.num_gpu_blocks * sizeof(int)));
    KIWI_HIP_CHECK(hipMemset(block_pop_counts_dev, 0, params.num_gpu_blocks * sizeof(int)));

    // Launch GPU consumer FIRST (it polls for data)
    auto t_start = std::chrono::high_resolution_clock::now();

    mpmc_h2d_consumer_kernel<<<params.num_gpu_blocks, 64>>>(
        queue->device_handle(), msgs_per_block, block_pop_counts_dev);

    // Launch CPU producer threads
    std::vector<std::atomic<int>> per_thread(params.num_cpu_threads);
    for (auto& c : per_thread) c.store(0);

    std::vector<std::thread> producers;
    for (int t = 0; t < params.num_cpu_threads; ++t) {
        producers.emplace_back(producer_thread_fn, queue.get(), t,
                               msgs_per_thread, params.pin_cpu,
                               std::ref(per_thread[t]));
    }

    // Wait for all producers and GPU consumer to finish
    for (auto& t : producers) t.join();
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    auto t_end = std::chrono::high_resolution_clock::now();
    double duration_s = std::chrono::duration<double>(t_end - t_start).count();

    // Read per-block pop counts
    std::vector<int> block_pop_counts(params.num_gpu_blocks);
    KIWI_HIP_CHECK(hipMemcpy(block_pop_counts.data(), block_pop_counts_dev,
                              params.num_gpu_blocks * sizeof(int), hipMemcpyDeviceToHost));

    // Print results
    double throughput_mps = total_messages / duration_s / 1e6;
    double bandwidth_mbps = (total_messages * 56.0) / duration_s / (1024 * 1024);

    std::cout << "\nTotal messages:    " << total_messages << "\n";
    std::cout << "Duration:          " << std::fixed << std::setprecision(3)
              << duration_s << " s\n";
    std::cout << "Throughput:        " << std::setprecision(2)
              << throughput_mps << " M msgs/s\n";
    std::cout << "Payload bandwidth: " << std::setprecision(2)
              << bandwidth_mbps << " MB/s\n";

    if (params.verbose) {
        std::cout << "\nPer-producer (CPU) breakdown:\n";
        for (int t = 0; t < params.num_cpu_threads; ++t) {
            std::cout << "  Thread " << t << ": " << per_thread[t].load() << " msgs pushed\n";
        }
        std::cout << "\nPer-consumer (GPU) breakdown:\n";
        int total_gpu_popped = 0;
        for (int b = 0; b < params.num_gpu_blocks; ++b) {
            std::cout << "  Block " << b << ":  " << block_pop_counts[b] << " msgs popped\n";
            total_gpu_popped += block_pop_counts[b];
        }
        std::cout << "  Total GPU popped: " << total_gpu_popped << "\n";
    }

    if (throughput_mps >= 1.0)
        std::cout << "\n[PASS] throughput >= 1 M msgs/s\n";
    else
        std::cout << "\n[FAIL] throughput < 1 M msgs/s\n";

    KIWI_HIP_CHECK(hipFree(block_pop_counts_dev));
}

int main(int argc, char** argv) {
    CLI::App app{"MPMCH2DQueue Throughput Benchmark"};

    ExperimentParams params;

    app.add_option("-c,--capacity", params.queue_capacity, "Queue capacity (power of 2)")
        ->default_val(1024)->check(CLI::PositiveNumber);
    app.add_option("-m,--messages", params.num_messages, "Total messages")
        ->default_val(1000000)->check(CLI::PositiveNumber);
    app.add_option("-t,--cpu-threads", params.num_cpu_threads, "CPU producer threads")
        ->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("-g,--gpu-blocks", params.num_gpu_blocks, "GPU consumer blocks (wavefronts)")
        ->default_val(1)->check(CLI::PositiveNumber);
    app.add_flag("-p,--pin-cpu", params.pin_cpu, "Pin producer threads to cores");
    app.add_flag("-v,--verbose", params.verbose, "Per-thread/block breakdown");

    CLI11_PARSE(app, argc, argv);

    check_numa_affinity();
    gpu_clock_warmup();

    std::cout << "\n=== MPMC H2D Throughput Benchmark ===\n";
    std::cout << "Queue capacity:    " << params.queue_capacity << "\n";
    std::cout << "CPU producers:     " << params.num_cpu_threads << " threads\n";
    std::cout << "GPU consumers:     " << params.num_gpu_blocks << " wavefronts\n";
    std::cout << "CPU pinning:       " << (params.pin_cpu ? "on" : "off") << "\n";

    benchmark_throughput(params);

    return 0;
}
