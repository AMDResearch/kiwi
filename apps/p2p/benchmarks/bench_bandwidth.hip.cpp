// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <sched.h>
#include <CLI/CLI.hpp>
#include "context.hip.hpp"
#include "lci.hpp"
#include "readiness.hip.hpp"

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(err));                     \
            abort();                                                      \
        }                                                                 \
    } while (0)

static void pin_current_thread_to_core(int core, const char* role) {
    if (core < 0) return;

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        fprintf(stderr, "ERROR: failed to pin %s thread to CPU core %d: %s\n",
                role, core, std::strerror(errno));
        abort();
    }

    fprintf(stderr, "# affinity: %s requested_core=%d current_cpu=%d\n",
            role, core, sched_getcpu());
}

__global__ void gpu_clock_warmup_kernel(int niters) {
    uint64_t x = threadIdx.x;
    for (int i = 0; i < niters; i++) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    if (x == 0) asm volatile(""); // prevent dead-code elimination
}

// Unidirectional bandwidth kernel with multiple workgroups.
// Each WG uses its own queue (SPSC). Every payload PUT is followed immediately
// by a one-byte readiness PUT on the same proxy/device.
//
// Flow control is fully per-WG — no grid-level synchronization:
//
// PE0 (sender):
//   Each WG waits for its own per-WG ACK before starting the next iteration,
//   then posts `depth` payload-plus-readiness pairs at strided offsets.
//
// PE1 (receiver):
//   Each WG polls and resets its readiness bytes, then sends a per-WG ACK back
//   to the sender. All WGs work in parallel.
__global__ void bandwidth_kernel(kiwi::p2p::gcontext_t gctx, int niters, int depth,
                                  int my_rank, int peer,
                                  void* data, size_t size, int data_heap_id,
                                  uint8_t* ready, int ready_heap_id,
                                  size_t ack_base_offset) {
    auto handle = gctx.get_handle();
    int wg_id = blockIdx.x;
    int num_wgs = gridDim.x;

    // Each WG has its own ACK slot in the symmetric buffer
    size_t my_ack_offset = ack_base_offset + (size_t)wg_id * sizeof(uint64_t);
    volatile uint64_t* my_ack_ptr =
        (volatile uint64_t*)((char*)data + my_ack_offset);

    if (my_rank == 0) {
        // Sender: each WG independently waits for its own ACK, then sends
        for (int i = 0; i < niters; i++) {
            // Wait for per-WG ACK from receiver (skip first iteration)
            if (i > 0) {
                if (threadIdx.x == 0) {
                    while (*my_ack_ptr < (uint64_t)i) continue;
                }
                __syncthreads();
            }

            // All threads: post depth puts at strided offsets
            for (int d = 0; d < depth; d++) {
                size_t slot = (size_t)wg_id + (size_t)d * num_wgs;
                size_t offset = slot * size;
                handle.post_put(data_heap_id, ready_heap_id, peer,
                                (char*)data + offset, size, offset, slot);
            }
        }
        // Wait for final ACK
        if (threadIdx.x == 0) {
            while (*my_ack_ptr < (uint64_t)niters) continue;
        }
    } else {
        // Receiver: each WG polls its readiness bytes and sends per-WG ACK
        for (int i = 0; i < niters; i++) {
            for (int d = threadIdx.x; d < depth; d += blockDim.x) {
                size_t slot = (size_t)wg_id + (size_t)d * num_wgs;
                while (kiwi::p2p::load_readiness(ready + slot) == 0) continue;
                kiwi::p2p::reset_readiness(ready + slot);
            }
            __syncthreads();
            handle.post_p(data_heap_id, peer, (uint64_t)(i + 1),
                          my_ack_offset);
        }
    }
}

struct config_t {
    int num_wgs = 8;
    int num_devices = -1;        // -1 = num_proxy_threads
    int num_proxy_threads = -1;  // -1 = num_wgs
    int depth = 1024;
    int warmup_iters = 100;
    int measure_iters = 1000;
    size_t single_size = 0; // 0 = sweep all sizes
    int main_core = -1;
    int proxy_core = -1;
};

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{"kiwi-p2p unidirectional bandwidth benchmark (multi-WG, pipelined)"};
    app.add_option("--wgs", cfg.num_wgs, "Number of workgroups (= queues)")->default_val(8);
    app.add_option("--devices", cfg.num_devices, "Number of LCI network devices (-1 = num_proxy_threads)")->default_val(-1);
    app.add_option("--proxy-threads", cfg.num_proxy_threads, "Number of CPU proxy threads (-1 = num_wgs)")->default_val(-1);
    app.add_option("--depth", cfg.depth, "Total pipeline depth across all WGs per iteration")->default_val(1024);
    app.add_option("--warmup", cfg.warmup_iters, "Warmup iterations")->default_val(100);
    app.add_option("--iters", cfg.measure_iters, "Measurement iterations")->default_val(1000);
    app.add_option("--size", cfg.single_size, "Single message size (0 = sweep all)")->default_val(0);
    app.add_option("--main-core", cfg.main_core,
                   "Pin the main/control thread to this CPU core")
        ->default_val(-1);
    app.add_option("--proxy-core", cfg.proxy_core,
                   "Pin proxy thread 0 to this CPU core (additional proxies use consecutive cores)")
        ->default_val(-1);
    CLI11_PARSE(app, argc, argv);

    pin_current_thread_to_core(cfg.main_core, "main/control");

    if (cfg.depth % cfg.num_wgs != 0) {
        fprintf(stderr, "depth (%d) must be divisible by wgs (%d)\n",
                cfg.depth, cfg.num_wgs);
        return 1;
    }
    int depth_per_wg = cfg.depth / cfg.num_wgs;

    int num_proxy = cfg.num_proxy_threads == -1 ? cfg.num_wgs : cfg.num_proxy_threads;
    int num_devices = cfg.num_devices == -1 ? num_proxy : cfg.num_devices;

    // Message sizes to benchmark
    std::vector<size_t> sizes;
    if (cfg.single_size > 0) {
        sizes = {cfg.single_size};
    } else {
        sizes = {1024, 2048, 4096, 8192, 16384, 32768, 65536,
                 131072, 262144, 1048576, 4194304};
    }

    // One queue per WG; devices and proxy threads independently configurable
    kiwi::p2p::context_t ctx(cfg.num_wgs, num_devices);

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    if (nranks != 2) {
        fprintf(stderr, "bench_bw requires exactly 2 ranks (got %d)\n", nranks);
        return 1;
    }
    int peer = 1 - rank;

    // Buffer: depth * max_size for data + per-WG ack slots
    size_t max_size = *std::max_element(sizes.begin(), sizes.end());
    size_t data_size = (size_t)cfg.depth * max_size;
    size_t ack_base_offset = data_size;
    size_t buf_size = ack_base_offset + (size_t)cfg.num_wgs * sizeof(uint64_t);

    void* dev_buf;
    HIP_CHECK(hipMalloc(&dev_buf, buf_size));
    HIP_CHECK(hipMemset(dev_buf, 0, buf_size));
    // Default registration covers all context devices, which is required here
    // because proxy threads may be spread across multiple LCI devices.
    int heap_id = ctx.register_symmetric(dev_buf, buf_size);

    uint8_t* ready;
    HIP_CHECK(hipMalloc(&ready, cfg.depth * sizeof(uint8_t)));
    HIP_CHECK(hipMemset(ready, 0, cfg.depth * sizeof(uint8_t)));
    int ready_heap_id =
        ctx.register_symmetric(ready, cfg.depth * sizeof(uint8_t));

    lci::barrier();
    HIP_CHECK(hipDeviceSynchronize());

    ctx.spawn_proxy_threads(num_proxy, cfg.proxy_core);
    kiwi::p2p::gcontext_t gctx = ctx.get_device_context();

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warm GPU clock from deep sleep
    gpu_clock_warmup_kernel<<<1, 64>>>(2000000);
    HIP_CHECK(hipDeviceSynchronize());

    if (rank == 0) {
        printf("# kiwi-p2p bandwidth benchmark (multi-WG, pipelined)\n");
        printf("# protocol=payload-put+1B-readiness-put, bandwidth=payload-bytes-only\n");
        printf("# wgs=%d, devices=%d, proxy_threads=%d, depth=%d, warmup=%d, iters=%d, main_core=%d, proxy_core=%d\n",
               cfg.num_wgs, num_devices, num_proxy,
               cfg.depth, cfg.warmup_iters, cfg.measure_iters,
               cfg.main_core, cfg.proxy_core);
        printf("%-12s %12s %8s %6s %8s\n",
               "size(B)", "BW(GB/s)", "depth", "WGs", "iters");
        printf("%-12s %12s %8s %6s %8s\n",
               "--------", "--------", "-----", "---", "-----");
    }

    for (size_t size : sizes) {
        // Scale iterations for large messages
        int iters = cfg.measure_iters;
        if (size >= 1048576) iters = std::max(iters / 10, 10);
        if (size >= 4194304) iters = std::max(iters / 100, 5);
        int warmup = std::min(cfg.warmup_iters, iters);

        // Reset state
        HIP_CHECK(hipMemset((char*)dev_buf + ack_base_offset, 0,
                             cfg.num_wgs * sizeof(uint64_t)));
        HIP_CHECK(hipMemset(ready, 0, cfg.depth * sizeof(uint8_t)));
        lci::barrier();

        // Warmup
        bandwidth_kernel<<<cfg.num_wgs, 64>>>(
            gctx, warmup, depth_per_wg,
            rank, peer, dev_buf, size, heap_id,
            ready, ready_heap_id,
            ack_base_offset);
        HIP_CHECK(hipDeviceSynchronize());

        // Reset state
        HIP_CHECK(hipMemset((char*)dev_buf + ack_base_offset, 0,
                             cfg.num_wgs * sizeof(uint64_t)));
        HIP_CHECK(hipMemset(ready, 0, cfg.depth * sizeof(uint8_t)));
        lci::barrier();

        // Measurement
        HIP_CHECK(hipEventRecord(start));
        bandwidth_kernel<<<cfg.num_wgs, 64>>>(
            gctx, iters, depth_per_wg,
            rank, peer, dev_buf, size, heap_id,
            ready, ready_heap_id,
            ack_base_offset);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        lci::barrier();

        if (rank == 0) {
            double total_bytes = (double)iters * cfg.depth * size;
            double bw_gbs = total_bytes / (ms / 1e3) / 1e9;
            printf("%-12zu %12.3f %8d %6d %8d\n",
                   size, bw_gbs, cfg.depth, cfg.num_wgs, iters);
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    ctx.join_proxy_threads();
    lci::barrier();

    ctx.deregister_symmetric(ready_heap_id);
    ctx.deregister_symmetric(heap_id);
    HIP_CHECK(hipFree(ready));
    HIP_CHECK(hipFree(dev_buf));

    return 0;
}
