// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cerrno>
#include <vector>
#include <limits>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cctype>
#include <cstring>
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

// Dummy kernel to ramp GPU clock from deep sleep (131 MHz → 2100 MHz).
// MI300X takes ~3 seconds to reach full clock; 2M iterations is sufficient.
__global__ void gpu_clock_warmup_kernel(int niters) {
    uint64_t x = threadIdx.x;
    for (int i = 0; i < niters; i++) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    if (x == 0) asm volatile(""); // prevent dead-code elimination
}

// Ping-pong kernel: rank 0 sends, waits for reply; rank 1 waits, sends back.
// size > 0 uses one callback that posts the payload PUT followed immediately
// by a one-byte readiness PUT on the same proxy/device.
// size == 0 uses post_p to write a uint64_t signal into the peer's symmetric
// heap at offset 0; the peer polls that signal and replies the same way.
// When error_count != nullptr, rank 0 fills the buffer with check_pattern each
// iteration before sending, and both ranks verify received data after each put.
__global__ void ping_pong_kernel(kiwi::p2p::gcontext_t gctx, int niters,
                                  int rank_me, int peer_rank,
                                  void* data, size_t size,
                                  int data_heap_id,
                                  uint8_t* ready, int ready_heap_id,
                                  uint8_t check_pattern,
                                  uint64_t* error_count) {
    auto handle = gctx.get_handle();
    uint8_t* buf = (uint8_t*)data;
    volatile uint64_t* signal = (volatile uint64_t*)data;
    for (int i = 0; i < niters; i++) {
        uint64_t expected = (uint64_t)(i + 1);
        if (rank_me == 0) {
            if (size == 0) {
                handle.post_p(data_heap_id, peer_rank, expected, 0);
                while (*signal < expected) continue;
            } else {
                if (error_count) {
                    for (size_t b = threadIdx.x; b < size; b += blockDim.x)
                        buf[b] = check_pattern;
                    // Check-mode payloads are written by this GPU immediately
                    // before the CPU proxy/NIC reads them.  Publish every
                    // lane's stores to system scope before posting the put.
                    __syncthreads();
                    __threadfence_system();
                    __syncthreads();
                }
                handle.post_put(data_heap_id, ready_heap_id, peer_rank,
                                data, size, 0, 0);
                if (threadIdx.x == 0) {
                    while (kiwi::p2p::load_readiness(ready) == 0) continue;
                    kiwi::p2p::reset_readiness(ready);
                }
                __syncthreads();
                if (error_count) {
                    for (size_t b = threadIdx.x; b < size; b += blockDim.x) {
                        if (buf[b] != check_pattern)
                            atomicAdd(error_count, 1ULL);
                    }
                }
            }
        } else {
            if (size == 0) {
                while (*signal < expected) continue;
                handle.post_p(data_heap_id, peer_rank, expected, 0);
            } else {
                if (threadIdx.x == 0) {
                    while (kiwi::p2p::load_readiness(ready) == 0) continue;
                    kiwi::p2p::reset_readiness(ready);
                }
                __syncthreads();
                if (error_count) {
                    for (size_t b = threadIdx.x; b < size; b += blockDim.x) {
                        if (buf[b] != check_pattern)
                            atomicAdd(error_count, 1ULL);
                    }
                }
                handle.post_put(data_heap_id, ready_heap_id, peer_rank,
                                data, size, 0, 0);
            }
        }
    }
}

// Pin the calling thread when a benchmark requests a fixed control core.
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

// Check CPU-GPU NUMA affinity and warn if mismatched or unpinned.
static void check_numa_affinity() {
    int device;
    if (hipGetDevice(&device) != hipSuccess) return;

    char pciBusId[64];
    if (hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device) != hipSuccess) return;
    for (char* p = pciBusId; *p; p++) *p = tolower(*p);

    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", pciBusId);
    FILE* f = fopen(path, "r");
    if (!f) return;
    int gpu_numa = -1;
    if (fscanf(f, "%d", &gpu_numa) != 1) { fclose(f); return; }
    fclose(f);
    if (gpu_numa < 0) return;

    int cpu = sched_getcpu();
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpu%d", gpu_numa, cpu);
    struct stat st;
    bool on_correct_numa = (stat(path, &st) == 0);

    cpu_set_t mask;
    bool pinned = false;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        long total_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        pinned = (CPU_COUNT(&mask) < total_cpus);
    }

    if (!on_correct_numa) {
        fprintf(stderr,
            "WARNING: CPU %d is not on GPU %d's NUMA node %d. "
            "Cross-NUMA adds ~1-2 us latency.\n"
            "  Fix: taskset -c <cores-on-node-%d> ./benchmark\n\n",
            cpu, device, gpu_numa, gpu_numa);
    } else if (!pinned) {
        fprintf(stderr,
            "WARNING: Process not pinned — proxy thread may migrate to wrong "
            "NUMA node, adding ~1-2 us latency.\n"
            "  Fix: taskset -c <cores-on-node-%d> ./benchmark\n\n",
            gpu_numa);
    }
}

struct config_t {
    int num_queues = 1;
    int warmup_iters = 10000;
    int measure_iters = 100000;
    size_t only_size = std::numeric_limits<size_t>::max();
    bool use_cpu_mem = false;
    bool use_fine_mem = false;
    bool check = false;
    int main_core = -1;
    int proxy_core = -1;
};

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{"kiwi-p2p ping-pong latency benchmark"};
    app.add_option("--queues", cfg.num_queues, "Number of invoke queues")->default_val(1);
    app.add_option("--warmup", cfg.warmup_iters, "Warmup iterations")->default_val(10000);
    app.add_option("--iters", cfg.measure_iters, "Measurement iterations")->default_val(100000);
    app.add_option("--size", cfg.only_size, "Run one message size only (omit for built-in sweep)");
    app.add_flag("--cpu-mem", cfg.use_cpu_mem, "Use CPU memory for put source/destination (no --check)");
    app.add_flag("--fine-mem", cfg.use_fine_mem,
                 "Use fine-grained device memory for the benchmark buffer/signal");
    app.add_flag("--check", cfg.check, "Enable data correctness checking");
    app.add_option("--main-core", cfg.main_core,
                   "Pin the main/control thread to this CPU core")
        ->default_val(-1);
    app.add_option("--proxy-core", cfg.proxy_core,
                   "Pin proxy thread 0 to this CPU core (additional proxies use consecutive cores)")
        ->default_val(-1);
    CLI11_PARSE(app, argc, argv);
    if (cfg.use_cpu_mem && cfg.use_fine_mem) {
        fprintf(stderr, "--cpu-mem cannot be combined with --fine-mem\n");
        return 1;
    }
    if (cfg.use_cpu_mem && cfg.check) {
        fprintf(stderr, "--cpu-mem cannot be combined with --check: the GPU kernel does not dereference CPU buffers safely\n");
        return 1;
    }
    if (cfg.use_cpu_mem && cfg.only_size == 0) {
        fprintf(stderr, "--cpu-mem cannot be combined with --size 0: post_p requires GPU-visible signal polling, not CPU allocation\n");
        return 1;
    }

    pin_current_thread_to_core(cfg.main_core, "main/control");
    check_numa_affinity();

    // Message sizes to benchmark
    std::vector<size_t> sizes;
    if (cfg.only_size != std::numeric_limits<size_t>::max()) {
        sizes.push_back(cfg.only_size);
    } else {
        sizes = {8, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    }
    size_t max_size = sizes.back();
    size_t alloc_size = std::max<size_t>(max_size, sizeof(uint64_t));

    kiwi::p2p::context_t ctx(cfg.num_queues);

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    if (nranks != 1 && nranks != 2) {
        fprintf(stderr, "bench_pingpong requires 1 rank (loopback) or 2 ranks (got %d)\n", nranks);
        return 1;
    }
    bool loopback = (nranks == 1);
    int peer = loopback ? rank : 1 - rank;

    // Allocate and register memory. The default GPU-buffer path is the real
    // KIWI data path. --cpu-mem is an isolation mode: the GPU still enqueues
    // callbacks through invoke, but the CPU proxy asks LCI to put CPU memory,
    // separating proxy/queue latency from GPU-memory transport effects.
    void* buf;
    if (cfg.use_cpu_mem) {
        buf = calloc(1, alloc_size);
        if (!buf) {
            fprintf(stderr, "calloc failed\n");
            return 1;
        }
    } else if (cfg.use_fine_mem) {
        HIP_CHECK(hipExtMallocWithFlags(&buf, alloc_size,
                                        hipDeviceMallocFinegrained));
        HIP_CHECK(hipMemset(buf, 0, alloc_size));
    } else {
        HIP_CHECK(hipMalloc(&buf, alloc_size));
        HIP_CHECK(hipMemset(buf, 0, alloc_size));
    }

    int data_heap_id = ctx.register_symmetric(buf, alloc_size);

    uint8_t* ready;
    HIP_CHECK(hipMalloc(&ready, sizeof(uint8_t)));
    HIP_CHECK(hipMemset(ready, 0, sizeof(uint8_t)));
    int ready_heap_id = ctx.register_symmetric(ready, sizeof(uint8_t));

    lci::barrier();
    HIP_CHECK(hipDeviceSynchronize());

    ctx.spawn_proxy_threads(-1, cfg.proxy_core);
    kiwi::p2p::gcontext_t gctx = ctx.get_device_context();

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    uint64_t* d_error_count = nullptr;
    if (cfg.check) {
        HIP_CHECK(hipMallocManaged(&d_error_count, sizeof(uint64_t)));
    }

    // Warm GPU clock from deep sleep before any measurements
    gpu_clock_warmup_kernel<<<1, 64>>>(2000000);
    HIP_CHECK(hipDeviceSynchronize());

    if (rank == 0) {
        printf("# kiwi-p2p %s latency benchmark\n",
               loopback ? "loopback" : "ping-pong");
        printf("# positive-size protocol=payload-put+1B-readiness-put\n");
        printf("# queues=%d, warmup=%d, iters=%d, mem=%s, main_core=%d, proxy_core=%d\n",
               cfg.num_queues, cfg.warmup_iters, cfg.measure_iters,
               cfg.use_cpu_mem ? "cpu" :
               (cfg.use_fine_mem ? "fine-grained-gpu" : "gpu"),
               cfg.main_core, cfg.proxy_core);
        if (cfg.check) {
            printf("%-12s %12s %8s\n", "size(B)", "latency(us)", "check");
            printf("%-12s %12s %8s\n", "--------", "-----------", "-----");
        } else {
            printf("%-12s %12s\n", "size(B)", "latency(us)");
            printf("%-12s %12s\n", "--------", "-----------");
        }
    }

    for (size_t s = 0; s < sizes.size(); s++) {
        size_t size = sizes[s];

        uint8_t pattern = (uint8_t)((s + 1) * 37);
        if (pattern == 0) pattern = 1;

        // Warmup
        if (size == 0) {
            if (cfg.use_cpu_mem)
                memset(buf, 0, sizeof(uint64_t));
            else
                HIP_CHECK(hipMemset(buf, 0, sizeof(uint64_t)));
        } else {
            HIP_CHECK(hipMemset(ready, 0, sizeof(uint8_t)));
        }
        lci::barrier();
        ping_pong_kernel<<<1, 64>>>(gctx, cfg.warmup_iters, rank, peer,
                                    buf, size, data_heap_id,
                                    ready, ready_heap_id,
                                    pattern, nullptr);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Measurement
        if (d_error_count) *d_error_count = 0;
        if (size == 0) {
            if (cfg.use_cpu_mem)
                memset(buf, 0, sizeof(uint64_t));
            else
                HIP_CHECK(hipMemset(buf, 0, sizeof(uint64_t)));
        } else {
            HIP_CHECK(hipMemset(ready, 0, sizeof(uint8_t)));
        }
        lci::barrier();
        HIP_CHECK(hipEventRecord(start));
        ping_pong_kernel<<<1, 64>>>(gctx, cfg.measure_iters, rank, peer,
                                    buf, size, data_heap_id,
                                    ready, ready_heap_id,
                                    pattern, d_error_count);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        lci::barrier();

        // Loopback: one-way latency; Ping-pong: one-way = RTT / 2
        double latency_us = (double)ms / cfg.measure_iters * 1000.0;
        if (!loopback) latency_us /= 2.0;

        if (cfg.check) {
            bool pass = (*d_error_count == 0);
            if (!pass) {
                fprintf(stderr, "Rank %d: %lu byte errors at size %zu\n",
                        rank, (unsigned long)*d_error_count, size);
            }
            if (rank == 0) {
                printf("%-12zu %12.2f %8s\n", size, latency_us,
                       pass ? "PASS" : "FAIL");
            }
        } else if (rank == 0) {
            printf("%-12zu %12.2f\n", size, latency_us);
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    ctx.join_proxy_threads();

    lci::barrier();
    ctx.deregister_symmetric(ready_heap_id);
    ctx.deregister_symmetric(data_heap_id);
    HIP_CHECK(hipFree(ready));
    if (d_error_count) HIP_CHECK(hipFree(d_error_count));
    if (cfg.use_cpu_mem)
        free(buf);
    else
        HIP_CHECK(hipFree(buf));

    return 0;
}
