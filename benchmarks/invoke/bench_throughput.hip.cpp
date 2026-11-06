// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_throughput.hip.cpp
 * @brief Measure callback throughput (callbacks/second) with invoke
 *
 * Measures the sustainable rate of callback invocations from GPU to CPU.
 */

#include <kiwi/invoke/invoke.hpp>
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
#include <utility>
#include <string>

using namespace kiwi::invoke;

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_messages = 1000000;
    int num_endpoints = 1;
    int num_cpu_threads = 1;
    std::string poll_mode = "host-poll-state";
    bool verbose = false;
};

// Producer kernel: one wavefront owns one endpoint. Launch blocks are capped
// at 256 by packing additional endpoint wavefronts into each block.
__global__ void throughput_invoke_kernel(
    DeviceContext<>* ctx,
    uint8_t callback_id,
    int num_msgs_per_endpoint,
    int num_endpoints,
    int waves_per_block
) {
    int wave_in_block = threadIdx.x / 64;
    int endpoint = blockIdx.x * waves_per_block + wave_in_block;
    if (endpoint >= num_endpoints) return;

    auto handle = ctx->get_device_handle(endpoint);

    for (int i = 0; i < num_msgs_per_endpoint; ++i) {
        uint32_t value = static_cast<uint32_t>(endpoint * num_msgs_per_endpoint + i);

        while (!handle.invoke(callback_id, value)) {
            // Retry on full
        }
    }
}

// Consumer thread function
void consumer_thread_func(
    int tid,
    int num_endpoints,
    int num_cpu_threads,
    int total_messages,
    int worker_cpu,
    bool use_host_poll_state,
    Context<>* ctx,
    std::atomic<int>& per_thread_count,
    std::atomic<int>& total_processed,
    std::atomic<bool>& done
)
{
    pin_current_thread_to_cpu(worker_cpu);

    // Calculate which endpoints this thread should poll
    int endpoints_per_thread = num_endpoints / num_cpu_threads;
    int extra_endpoints = num_endpoints % num_cpu_threads;

    int start_endpoint = tid * endpoints_per_thread + std::min(tid, extra_endpoints);
    int end_endpoint = start_endpoint + endpoints_per_thread + (tid < extra_endpoints ? 1 : 0);

    if (start_endpoint >= num_endpoints) {
        return; // No endpoints for this thread
    }

    int local_count = 0;

    if (use_host_poll_state) {
        std::vector<std::pair<
            int, kiwi::queue::SPSCD2HQueue::HostPollState>> endpoint_poll_states;
        endpoint_poll_states.reserve(end_endpoint - start_endpoint);
        for (int ep = start_endpoint; ep < end_endpoint; ++ep) {
            endpoint_poll_states.emplace_back(
                ep, ctx->raw_queue(ep)->host_poll_state());
        }

        while (!done.load(std::memory_order_relaxed)) {
            for (auto& [ep, poll_state] : endpoint_poll_states) {
                poll_state.refresh();
                if (!poll_state.ready()) continue;
                local_count += ctx->progress(ep, 1);
                poll_state.refresh();
            }
        }
    } else {
        while (!done.load(std::memory_order_relaxed)) {
            for (int ep = start_endpoint; ep < end_endpoint; ++ep) {
                local_count += ctx->progress(ep, 1);
            }
        }
    }

    // Final drain
    bool made_progress = false;
    do {
        made_progress = false;
        for (int ep = start_endpoint; ep < end_endpoint; ++ep) {
            size_t processed = ctx->progress(ep, 1);
            local_count += processed;
            made_progress |= processed != 0;
        }
    } while (made_progress);

    per_thread_count = local_count;
    total_processed += local_count;
}

void benchmark_throughput(const ExperimentParams& params) {
    std::cout << "\n=== Throughput Benchmark ===\n";
    std::cout << "Queue capacity:   " << params.queue_capacity << "\n";
    std::cout << "Endpoints:        " << params.num_endpoints << "\n";
    std::cout << "CPU threads:      " << params.num_cpu_threads << "\n";
    std::cout << "Callback:         idle\n";
    std::cout << "Progress batch:   1 callback per endpoint visit\n";
    std::cout << "Polling:          " << params.poll_mode << "\n";
    std::cout << "Total messages:   " << params.num_messages << "\n\n";

    // Validate parameters
    if (params.num_endpoints < params.num_cpu_threads) {
        std::cerr << "Warning: More CPU threads than endpoints. "
                  << "Some CPU threads will have no endpoints to poll.\n";
    }

    // Create invoke context
    Context ctx(params.num_endpoints, params.queue_capacity);

    // Register an idle callback. Completion is validated by the number of
    // callbacks returned from progress(), avoiding a shared per-message
    // side effect that would distort worker scaling.
    std::function<void(uint32_t)> callback = [](uint32_t) {};
    uint8_t callback_id = ctx.register_callback<uint32_t>(callback);

    // Calculate messages per endpoint and cap the launch at 256 blocks.
    int num_msgs_per_endpoint = params.num_messages / params.num_endpoints;
    int total_messages = num_msgs_per_endpoint * params.num_endpoints;
    int waves_per_block = std::max(1, (params.num_endpoints + 255) / 256);
    int launch_blocks =
        (params.num_endpoints + waves_per_block - 1) / waves_per_block;
    std::vector<int> worker_cpus = worker_cpu_plan(params.num_cpu_threads);
    bool use_host_poll_state = params.poll_mode == "host-poll-state";
    std::cout << "Launch blocks:     " << launch_blocks << "\n";
    std::cout << "Waves per block:   " << waves_per_block << "\n";
    print_worker_cpu_plan(worker_cpus);

    // Shared counters
    std::atomic<int> total_processed{0};
    std::atomic<bool> done{false};
    std::vector<std::atomic<int>> per_thread_processed(params.num_cpu_threads);
    for (auto& counter : per_thread_processed) {
        counter = 0;
    }

    // Launch CPU consumer threads first
    std::vector<std::thread> consumers;

    for (int tid = 0; tid < params.num_cpu_threads; ++tid) {
        consumers.emplace_back(
            consumer_thread_func,
            tid,
            params.num_endpoints,
            params.num_cpu_threads,
            params.num_messages,
            worker_cpus[static_cast<size_t>(tid)],
            use_host_poll_state,
            &ctx,
            std::ref(per_thread_processed[tid]),
            std::ref(total_processed),
            std::ref(done)
        );
    }

    require_no_scratch(throughput_invoke_kernel, "throughput_invoke_kernel");
    auto start = std::chrono::high_resolution_clock::now();

    // Launch GPU producer (async) — DeviceContext already has handle array
    std::thread producer([&]() {
        throughput_invoke_kernel<<<launch_blocks, 64 * waves_per_block>>>(
            ctx.device_context(), callback_id, num_msgs_per_endpoint,
            params.num_endpoints, waves_per_block);
        KIWI_HIP_CHECK(hipDeviceSynchronize());
        done.store(true, std::memory_order_relaxed);
    });

    // Wait for all threads to complete
    for (auto& consumer : consumers) {
        consumer.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    producer.join();

    // Calculate throughput
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
    double duration_s = duration_ns / 1e9;

    double throughput_mps = total_messages / duration_s / 1e6;  // Million messages/sec
    double throughput_mbps = (total_messages * sizeof(uint32_t)) / duration_s / (1024 * 1024);  // MB/s

    // Report results
    std::cout << "Messages processed: " << total_processed.load() << " / " << total_messages << "\n";

    if (params.verbose) {
        std::cout << "\nPer-thread breakdown:\n";
        for (int i = 0; i < params.num_cpu_threads; ++i) {
            std::cout << "  Thread " << i << ": " << per_thread_processed[i].load() << " messages\n";
        }
        std::cout << "\n";
    }

    std::cout << "Duration:    " << std::fixed << std::setprecision(3)
              << duration_s << " seconds\n";
    std::cout << "Throughput:  " << std::setprecision(2)
              << throughput_mps << " M messages/sec\n";
    std::cout << "Bandwidth:   " << std::setprecision(2)
              << throughput_mbps << " MB/s\n";

    // Per-endpoint throughput
    double throughput_per_endpoint = throughput_mps / params.num_endpoints;
    std::cout << "Per-endpoint: " << std::setprecision(2)
              << throughput_per_endpoint << " M messages/sec\n";

    // Check target
    if (throughput_mps >= 1.0) {
        std::cout << "\n✓ Target met: throughput >= 1 M messages/sec\n";
    } else {
        std::cout << "\n✗ Target missed: throughput < 1 M messages/sec\n";
    }

}


int main(int argc, char** argv) {
    CLI::App app{"invoke Throughput Benchmark"};

    ExperimentParams params;

    // Configure CLI options
    app.add_option("-c,--capacity", params.queue_capacity,
                   "Queue capacity (64-byte lines)")
        ->default_val(1024)
        ->check(CLI::PositiveNumber);

    app.add_option("-m,--messages", params.num_messages,
                   "Total number of messages")
        ->default_val(1000000)
        ->check(CLI::PositiveNumber);

    app.add_option("-e,--endpoints", params.num_endpoints,
                   "Number of endpoints (parallel queues)")
        ->default_val(1)
        ->check(CLI::PositiveNumber);

    app.add_option("-t,--cpu-threads", params.num_cpu_threads,
                   "Number of CPU consumer threads")
        ->default_val(1)
        ->check(CLI::PositiveNumber);
    app.add_option("--poll-mode", params.poll_mode,
                   "Consumer polling: direct or host-poll-state")
        ->default_val("host-poll-state")
        ->check(CLI::IsMember({"direct", "host-poll-state"}));

    app.add_flag("-v,--verbose", params.verbose,
                 "Verbose output");

    // Parse command line
    CLI11_PARSE(app, argc, argv);

    if (params.num_endpoints > 4096) {
        std::cerr << "endpoint count must not exceed 4096\n";
        return 1;
    }

    // Print configuration
    std::cout << "invoke Throughput Benchmark\n";
    std::cout << "============================\n";

    benchmark_throughput(params);

    return 0;
}
