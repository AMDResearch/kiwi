// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_throughput.hip.cpp
 * @brief Measure device-to-host message throughput (messages/second)
 *
 * SPSC (default): one queue per GPU producer wavefront. MPSC (--mpsc): wave w
 * pushes to queue w % Q (--queues, default one per consumer thread). Each
 * consumer thread polls its share of the queues. --bytes selects the message
 * size (8 to 512 bytes).
 */

#include <kiwi/queue/queue.hpp>
#include <kiwi/queue/error.hpp>
#include <bench_common.hpp>
#include "queue_bench.hip.hpp"
#include <hip/hip_runtime.h>
#include <CLI/CLI.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>

using namespace kiwi::queue;

// Keep queue handles in a device-resident context for the multi-endpoint
// producer benchmark.
template<typename Queue>
struct ProducerContext {
    typename Queue::DeviceHandle* handles;
    size_t num_queues;

    // Wave-uniform, like kiwi::invoke::DeviceContext::get_device_handle().
    __device__ typename Queue::DeviceHandle get_device_handle(size_t queue) {
        KIWI_QUEUE_ASSERT(queue < num_queues);
        return kiwi::queue::detail::wave_uniform(handles[queue]);
    }
};

struct ExperimentParams {
    int queue_capacity = 1024;
    int num_messages = 1000000;
    int num_gpu_blocks = 1;
    int num_cpu_threads = 1;
    int num_queues = 0;  // 0: one per wave (SPSC) or per consumer thread (MPSC)
    bool mpsc = false;
    std::string poll_mode = "host-poll-state";
    int message_bytes = 56;
    bool verbose = false;
};

// ---------------------------------------------------------------------------
// Producer kernel and consumer
// ---------------------------------------------------------------------------

// Wave w pushes num_msgs_per_wave messages to queue w % num_queues.
template<typename Queue, size_t Bytes>
__global__ void push_kernel(ProducerContext<Queue>* context,
                            int num_msgs_per_wave,
                            int num_waves,
                            int waves_per_block,
                            int num_queues) {
    int wave_in_block = threadIdx.x / 64;
    int wave = blockIdx.x * waves_per_block + wave_in_block;
    if (wave >= num_waves) return;

    auto handle = context->get_device_handle(wave % num_queues);
    queue_bench::Message<Bytes> payload{};
    for (int i = 0; i < num_msgs_per_wave; ++i) {
        payload.value = static_cast<uint32_t>(wave * num_msgs_per_wave + i);
        while (!handle.push_reg(payload)) {}
    }
}

// Thread tid pops a contiguous share of the queues; queue q carries
// expected_per_queue[q] messages.
template<typename Queue>
static void consumer_fn(int tid, int num_cpu_threads,
                        const std::vector<int>& expected_per_queue, int worker_cpu,
                        bool use_host_poll_state,
                        std::vector<std::unique_ptr<Queue>>& queues,
                        std::atomic<int>& thread_count,
                        std::atomic<int>& total_popped) {
    pin_current_thread_to_cpu(worker_cpu);

    int num_queues = static_cast<int>(queues.size());
    int queues_per_thread = num_queues / num_cpu_threads;
    int extra = num_queues % num_cpu_threads;
    int start = tid * queues_per_thread + std::min(tid, extra);
    int end = start + queues_per_thread + (tid < extra ? 1 : 0);
    if (start >= num_queues) return;

    int expected = 0;
    for (int q = start; q < end; ++q) expected += expected_per_queue[q];
    int local = 0;
    alignas(8) uint8_t buf[kMaxMessageBytes];

    if (use_host_poll_state) {
        std::vector<std::pair<int, typename Queue::HostPollState>> poll_states;
        poll_states.reserve(end - start);
        for (int q = start; q < end; ++q) {
            poll_states.emplace_back(q, queues[q]->host_poll_state());
        }
        while (local < expected) {
            for (auto& [q, poll_state] : poll_states) {
                poll_state.refresh();
                if (!poll_state.ready()) continue;
                if (queues[q]->pop(buf)) {
                    if (++local >= expected) break;
                }
                poll_state.refresh();
            }
        }
    } else {
        while (local < expected) {
            for (int q = start; q < end; ++q) {
                if (queues[q]->pop(buf)) {
                    if (++local >= expected) break;
                }
            }
        }
    }
    thread_count.store(local);
    total_popped += local;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

static void print_results(const ExperimentParams& params, int total_messages,
                           double duration_s,
                           const std::vector<std::atomic<int>>& per_thread) {
    double throughput_mps = total_messages / duration_s / 1e6;
    double bandwidth_mbps =
        (total_messages * static_cast<double>(params.message_bytes)) / duration_s / (1024 * 1024);

    std::cout << "\nTotal messages:    " << total_messages << "\n";
    std::cout << "Duration:          " << std::fixed << std::setprecision(3)
              << duration_s << " s\n";
    std::cout << "Throughput:        " << std::setprecision(2)
              << throughput_mps << " M msgs/s\n";
    std::cout << "Payload bandwidth: " << std::setprecision(2)
              << bandwidth_mbps << " MB/s\n";

    std::cout << "Per-queue:         " << std::setprecision(2)
              << throughput_mps / params.num_queues << " M msgs/s\n";

    if (params.verbose) {
        std::cout << "\nPer-consumer breakdown:\n";
        for (int t = 0; t < params.num_cpu_threads; ++t) {
            std::cout << "  Thread " << t << ": " << per_thread[t].load() << " msgs\n";
        }
    }

    if (throughput_mps >= 1.0)
        std::cout << "\n[PASS] throughput >= 1 M msgs/s\n";
    else
        std::cout << "\n[FAIL] throughput < 1 M msgs/s\n";
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

template<typename Queue, size_t Bytes>
static void benchmark(const ExperimentParams& params,
                      const std::vector<int>& worker_cpus,
                      int coordinator_cpu) {
    require_no_scratch(push_kernel<Queue, Bytes>, "push_kernel");
    using Handle = typename Queue::DeviceHandle;
    int num_waves = params.num_gpu_blocks;
    int num_queues = params.num_queues;
    int msgs_per_wave = params.num_messages / num_waves;
    int total_messages = msgs_per_wave * num_waves;
    int waves_per_block = std::max(1, (num_waves + 255) / 256);
    int launch_blocks = (num_waves + waves_per_block - 1) / waves_per_block;
    bool use_host_poll_state = params.poll_mode == "host-poll-state";

    std::vector<std::unique_ptr<Queue>> queues;
    std::vector<Handle> host_handles;
    std::vector<int> expected_per_queue(num_queues, 0);
    queues.reserve(num_queues);
    host_handles.reserve(num_queues);
    for (int q = 0; q < num_queues; ++q) {
        queues.push_back(std::make_unique<Queue>(params.queue_capacity));
        host_handles.push_back(queues.back()->device_handle());
    }
    for (int w = 0; w < num_waves; ++w) expected_per_queue[w % num_queues] += msgs_per_wave;

    Handle* d_handles;
    KIWI_HIP_CHECK(hipMalloc(&d_handles, num_queues * sizeof(Handle)));
    KIWI_HIP_CHECK(hipMemcpy(d_handles, host_handles.data(), num_queues * sizeof(Handle),
                             hipMemcpyHostToDevice));

    ProducerContext<Queue> host_context{d_handles, static_cast<size_t>(num_queues)};
    ProducerContext<Queue>* d_context;
    KIWI_HIP_CHECK(hipMalloc(&d_context, sizeof(host_context)));
    KIWI_HIP_CHECK(hipMemcpy(d_context, &host_context, sizeof(host_context), hipMemcpyHostToDevice));

    if (coordinator_cpu >= 0) {
        pin_current_thread_to_cpu(coordinator_cpu);
        fprintf(stderr, "# coordinator_cpu: %d\n", coordinator_cpu);
    }

    std::atomic<int> total_popped{0};
    std::vector<std::atomic<int>> per_thread(params.num_cpu_threads);
    for (auto& c : per_thread) c.store(0);

    std::vector<std::thread> consumers;
    for (int tid = 0; tid < params.num_cpu_threads; ++tid) {
        consumers.emplace_back(consumer_fn<Queue>, tid, params.num_cpu_threads,
                               std::cref(expected_per_queue),
                               worker_cpus[static_cast<size_t>(tid)],
                               use_host_poll_state,
                               std::ref(queues), std::ref(per_thread[tid]),
                               std::ref(total_popped));
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    push_kernel<Queue, Bytes><<<launch_blocks, 64 * waves_per_block>>>(
        d_context, msgs_per_wave, num_waves, waves_per_block, num_queues);
    KIWI_HIP_CHECK(hipDeviceSynchronize());
    for (auto& c : consumers) c.join();
    auto t_end = std::chrono::high_resolution_clock::now();

    KIWI_HIP_CHECK(hipFree(d_context));
    KIWI_HIP_CHECK(hipFree(d_handles));

    print_results(params, total_messages,
                  std::chrono::duration<double>(t_end - t_start).count(), per_thread);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    CLI::App app{"queue Throughput Benchmark"};

    ExperimentParams params;

    app.add_option("-c,--capacity", params.queue_capacity,
                   "Queue capacity in 64-byte lines (power of 2)")
        ->default_val(1024)->check(CLI::PositiveNumber);
    app.add_option("-m,--messages", params.num_messages, "Total messages")
        ->default_val(1000000)->check(CLI::PositiveNumber);
    app.add_option("-g,--gpu-blocks", params.num_gpu_blocks,
                   "GPU producer wavefronts")
        ->default_val(1)->check(CLI::PositiveNumber);
    app.add_option("-t,--cpu-threads", params.num_cpu_threads, "CPU consumer threads")
        ->default_val(1)->check(CLI::PositiveNumber);
    app.add_flag("--mpsc", params.mpsc, "MPSC queues shared by several producer waves");
    app.add_option("-q,--queues", params.num_queues,
                   "Queues (MPSC; default one per consumer thread)")
        ->check(CLI::PositiveNumber);
    app.add_option("--poll-mode", params.poll_mode,
                   "Consumer polling: direct or host-poll-state")
        ->default_val("host-poll-state")
        ->check(CLI::IsMember({"direct", "host-poll-state"}));
    app.add_option("--bytes", params.message_bytes, "Message bytes")
        ->default_val(56)->check(CLI::IsMember(queue_bench::sizes()));
    app.add_flag("-v,--verbose", params.verbose, "Per-thread breakdown");

    CLI11_PARSE(app, argc, argv);

    if (params.num_gpu_blocks > 4096) {
        std::cerr << "GPU producer wavefronts must not exceed 4096\n";
        return 1;
    }
    if (!params.mpsc) {
        if (params.num_queues != 0 && params.num_queues != params.num_gpu_blocks) {
            std::cerr << "SPSC uses one queue per producer wavefront (--queues needs --mpsc)\n";
            return 1;
        }
        params.num_queues = params.num_gpu_blocks;
    } else if (params.num_queues == 0) {
        params.num_queues = std::min(params.num_cpu_threads, params.num_gpu_blocks);
    }

    std::vector<int> worker_cpus = worker_cpu_plan(params.num_cpu_threads);
    int coordinator_cpu = coordinator_cpu_plan(worker_cpus);
    if (coordinator_cpu < 0) {
        fprintf(stderr, "# coordinator_cpu: none (no CPU outside worker plan)\n");
    }

    check_numa_affinity();
    gpu_clock_warmup();

    std::cout << "\n=== " << (params.mpsc ? "MPSC" : "SPSC") << " Throughput Benchmark ===\n";
    std::cout << "Queue capacity:    " << params.queue_capacity << " lines\n";
    std::cout << "Message bytes:     " << params.message_bytes << "\n";
    std::cout << "GPU producers:     " << params.num_gpu_blocks << " wavefronts\n";
    std::cout << "Queues:            " << params.num_queues << "\n";
    int waves_per_block = std::max(1, (params.num_gpu_blocks + 255) / 256);
    int launch_blocks =
        (params.num_gpu_blocks + waves_per_block - 1) / waves_per_block;
    std::cout << "Launch blocks:      " << launch_blocks << "\n";
    std::cout << "Waves per block:    " << waves_per_block << "\n";
    std::cout << "CPU consumers:     " << params.num_cpu_threads << " threads\n";
    std::cout << "Polling:           " << params.poll_mode << "\n";
    print_worker_cpu_plan(worker_cpus);

    queue_bench::dispatch(params.message_bytes, [&](auto bytes) {
        constexpr size_t B = decltype(bytes)::value;
        if (params.mpsc)
            benchmark<MPSCD2HQueue, B>(params, worker_cpus, coordinator_cpu);
        else
            benchmark<SPSCD2HQueue, B>(params, worker_cpus, coordinator_cpu);
    });

    return 0;
}
