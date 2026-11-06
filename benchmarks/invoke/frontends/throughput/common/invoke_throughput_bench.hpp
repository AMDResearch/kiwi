// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the cross-frontend invoke callback throughput harness.
//
// One wavefront per endpoint pushes uint32 invokes to an idle callback while
// pinned CPU worker threads call progress(). A copy of the host code of the
// standalone bench_invoke_throughput (benchmarks/invoke/bench_throughput.hip.cpp)
// with the kernel launch taken from a launcher, so every frontend of the
// harness runs the same host code: the C++ kernel (cpp/) and the Python
// kernels (flydsl/, ...) through kiwi_invoke_throughput_bench_ext.
//
//   struct Launcher {
//     // Before the timed region (e.g. a scratch check).
//     void prepare();
//     // Launch the producer kernel asynchronously on the null stream:
//     // l.launch_blocks blocks of 64 * l.waves_per_block threads.
//     void launch(const ThroughputLaunch& l);
//   };

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/error.hpp>
#include <bench_common.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kiwi_bench {

struct ThroughputParams {
    int queue_capacity = 1024;
    int num_messages = 1000000;
    int num_endpoints = 1;
    int num_cpu_threads = 1;
    std::string poll_mode = "host-poll-state";
    bool verbose = false;
};

// Consumer polling modes.
inline std::vector<std::string> throughput_poll_modes() { return {"direct", "host-poll-state"}; }

// Largest supported endpoint count.
constexpr int kMaxThroughputEndpoints = 4096;

// One wavefront owns one endpoint; the launch is capped at 256 blocks by
// packing additional endpoint wavefronts into each block.
struct LaunchShape {
    int waves_per_block;
    int launch_blocks;
};

inline LaunchShape throughput_launch_shape(int num_endpoints) {
    const int waves_per_block = std::max(1, (num_endpoints + 255) / 256);
    return {waves_per_block, (num_endpoints + waves_per_block - 1) / waves_per_block};
}

// Arguments of the producer kernel launch.
struct ThroughputLaunch {
    kiwi::invoke::DeviceContext<>* ctx;  // device pointer
    uint8_t callback_id;
    int num_msgs_per_endpoint;
    int num_endpoints;
    int waves_per_block;
    int launch_blocks;
};

// Consumer thread function
inline void consumer_thread_func(
    int tid,
    int num_endpoints,
    int num_cpu_threads,
    int total_messages,
    int worker_cpu,
    bool use_host_poll_state,
    kiwi::invoke::Context<>* ctx,
    std::atomic<int>& per_thread_count,
    std::atomic<int>& total_processed,
    std::atomic<bool>& done
)
{
    (void)total_messages;
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

// Returns the number of messages processed; the caller checks it against
// the expected total (num_messages / num_endpoints * num_endpoints).
template<typename Launcher>
int benchmark_throughput(const ThroughputParams& params, Launcher& launcher) {
    using kiwi::invoke::Context;

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
    const LaunchShape shape = throughput_launch_shape(params.num_endpoints);
    const int waves_per_block = shape.waves_per_block;
    const int launch_blocks = shape.launch_blocks;
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

    launcher.prepare();
    auto start = std::chrono::high_resolution_clock::now();

    // Launch GPU producer (async) — DeviceContext already has handle array.
    // A launcher error ends the run (the consumers stop) and is rethrown.
    std::exception_ptr launch_error;
    std::thread producer([&]() {
        try {
            launcher.launch(ThroughputLaunch{
                ctx.device_context(), callback_id, num_msgs_per_endpoint,
                params.num_endpoints, waves_per_block, launch_blocks});
        } catch (...) {
            launch_error = std::current_exception();
        }
        KIWI_HIP_CHECK(hipDeviceSynchronize());
        done.store(true, std::memory_order_relaxed);
    });

    // Wait for all threads to complete
    for (auto& consumer : consumers) {
        consumer.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    producer.join();
    if (launch_error) std::rethrow_exception(launch_error);

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

    return total_processed.load();
}

// Prints the title and runs the benchmark; returns messages processed.
template<typename Launcher>
int run_throughput_benchmark(const ThroughputParams& params, Launcher& launcher,
                             const char* title = "invoke Throughput Benchmark") {
    std::cout << title << "\n";
    std::cout << "============================\n";
    return benchmark_throughput(params, launcher);
}

}  // namespace kiwi_bench
