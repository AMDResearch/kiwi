// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Plain GPU load/store ping-pong latency benchmark.
// Two GPUs on the same node exchange data via direct peer memory access
// (XGMI/Infinity Fabric) using plain stores and system-scope atomic
// flag signaling.  No LCI or other communication library is used.
//
// Usage: ./bench_loadstore_latency --gpu0 0 --gpu1 1

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <thread>
#include <CLI/CLI.hpp>

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

// Ping-pong kernel using plain stores over XGMI.
//
// Memory layout per GPU:
//   my_data   - on this GPU, written by peer (for verification)
//   peer_data - on peer GPU, written by this GPU (remote store)
//   my_flag   - on this GPU, polled locally, written by peer remotely
//   peer_flag - on peer GPU, written by this GPU remotely
//
// rank 0: store data to peer -> fence -> signal -> poll own flag
// rank 1: poll own flag -> store data to peer -> fence -> signal
//
// Data is transferred using uint64_t (8-byte) stores for throughput.
// All message sizes must be multiples of 8 bytes.
__global__ void ping_pong_kernel(uint8_t* __restrict__ my_data,
                                  uint8_t* __restrict__ peer_data,
                                  uint64_t* my_flag,
                                  uint64_t* peer_flag,
                                  int niters, int rank, size_t size,
                                  uint8_t check_pattern,
                                  uint64_t* error_count) {
    uint64_t* my_words = (uint64_t*)my_data;
    uint64_t* peer_words = (uint64_t*)peer_data;
    size_t nwords = size / sizeof(uint64_t);
    uint64_t pat64 = (uint64_t)check_pattern * 0x0101010101010101ULL;

    for (int i = 0; i < niters; i++) {
        if (rank == 0) {
            // Store data to peer's buffer via XGMI
            for (size_t w = threadIdx.x; w < nwords; w += blockDim.x)
                peer_words[w] = pat64;
            if (size > 0) {
                __syncthreads();
                __threadfence_system();
            }
            // Signal peer
            if (threadIdx.x == 0)
                __hip_atomic_store(peer_flag, (uint64_t)(i + 1),
                                   __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            // Wait for peer's reply
            if (threadIdx.x == 0)
                while (__hip_atomic_load(my_flag, __ATOMIC_ACQUIRE,
                                          __HIP_MEMORY_SCOPE_SYSTEM) < (uint64_t)(i + 1));
            __syncthreads();
            // Verify received data
            if (error_count && nwords > 0) {
                for (size_t w = threadIdx.x; w < nwords; w += blockDim.x)
                    if (my_words[w] != pat64)
                        atomicAdd(error_count, 1ULL);
            }
        } else {
            // Wait for signal from rank 0
            if (threadIdx.x == 0)
                while (__hip_atomic_load(my_flag, __ATOMIC_ACQUIRE,
                                          __HIP_MEMORY_SCOPE_SYSTEM) < (uint64_t)(i + 1));
            __syncthreads();
            // Verify received data
            if (error_count && nwords > 0) {
                for (size_t w = threadIdx.x; w < nwords; w += blockDim.x)
                    if (my_words[w] != pat64)
                        atomicAdd(error_count, 1ULL);
            }
            // Store data back to peer's buffer via XGMI
            for (size_t w = threadIdx.x; w < nwords; w += blockDim.x)
                peer_words[w] = pat64;
            if (size > 0) {
                __syncthreads();
                __threadfence_system();
            }
            // Signal peer
            if (threadIdx.x == 0)
                __hip_atomic_store(peer_flag, (uint64_t)(i + 1),
                                   __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

struct config_t {
    int gpu0 = 0;
    int gpu1 = 1;
    int warmup_iters = 1000;
    int measure_iters = 200000;
    bool check = false;
};

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{"Plain load/store ping-pong latency benchmark (XGMI)"};
    app.add_option("--gpu0", cfg.gpu0, "First GPU device ID")->default_val(0);
    app.add_option("--gpu1", cfg.gpu1, "Second GPU device ID")->default_val(1);
    app.add_option("--warmup", cfg.warmup_iters, "Warmup iterations")->default_val(1000);
    app.add_option("--iters", cfg.measure_iters, "Measurement iterations")->default_val(200000);
    app.add_flag("--check", cfg.check, "Enable data correctness checking");
    CLI11_PARSE(app, argc, argv);

    // Validate GPU IDs
    int dev_count;
    HIP_CHECK(hipGetDeviceCount(&dev_count));
    if (cfg.gpu0 < 0 || cfg.gpu0 >= dev_count ||
        cfg.gpu1 < 0 || cfg.gpu1 >= dev_count || cfg.gpu0 == cfg.gpu1) {
        fprintf(stderr, "Invalid GPU IDs: gpu0=%d, gpu1=%d (available: 0-%d)\n",
                cfg.gpu0, cfg.gpu1, dev_count - 1);
        return 1;
    }

    // Check and enable peer access
    int can_0to1 = 0, can_1to0 = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_0to1, cfg.gpu0, cfg.gpu1));
    HIP_CHECK(hipDeviceCanAccessPeer(&can_1to0, cfg.gpu1, cfg.gpu0));
    if (!can_0to1 || !can_1to0) {
        fprintf(stderr, "Peer access not supported between GPU %d and GPU %d\n",
                cfg.gpu0, cfg.gpu1);
        return 1;
    }
    HIP_CHECK(hipSetDevice(cfg.gpu0));
    HIP_CHECK(hipDeviceEnablePeerAccess(cfg.gpu1, 0));
    HIP_CHECK(hipSetDevice(cfg.gpu1));
    HIP_CHECK(hipDeviceEnablePeerAccess(cfg.gpu0, 0));

    // Message sizes to benchmark
    size_t sizes[] = {0, 8, 64, 256, 1024, 2048, 4096, 8192,
                      16384, 32768, 65536, 131072, 262144};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    size_t max_size = sizes[num_sizes - 1];

    // Allocate data buffers on each GPU
    void *data0, *data1;
    HIP_CHECK(hipSetDevice(cfg.gpu0));
    HIP_CHECK(hipMalloc(&data0, max_size));
    HIP_CHECK(hipMemset(data0, 0, max_size));
    HIP_CHECK(hipSetDevice(cfg.gpu1));
    HIP_CHECK(hipMalloc(&data1, max_size));
    HIP_CHECK(hipMemset(data1, 0, max_size));

    // Allocate flags on each GPU
    uint64_t *flag0, *flag1;
    HIP_CHECK(hipSetDevice(cfg.gpu0));
    HIP_CHECK(hipMalloc(&flag0, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(flag0, 0, sizeof(uint64_t)));
    HIP_CHECK(hipSetDevice(cfg.gpu1));
    HIP_CHECK(hipMalloc(&flag1, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(flag1, 0, sizeof(uint64_t)));

    // Error counters (managed memory for host+device access)
    uint64_t *err0 = nullptr, *err1 = nullptr;
    if (cfg.check) {
        HIP_CHECK(hipMallocManaged(&err0, sizeof(uint64_t)));
        HIP_CHECK(hipMallocManaged(&err1, sizeof(uint64_t)));
    }

    // Create streams
    hipStream_t stream0, stream1;
    HIP_CHECK(hipSetDevice(cfg.gpu0));
    HIP_CHECK(hipStreamCreate(&stream0));
    HIP_CHECK(hipSetDevice(cfg.gpu1));
    HIP_CHECK(hipStreamCreate(&stream1));

    // Events on GPU 0 for timing
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipSetDevice(cfg.gpu0));
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));

    // Print header
    printf("# Plain load/store ping-pong latency benchmark\n");
    printf("# gpu0=%d, gpu1=%d, warmup=%d, iters=%d\n",
           cfg.gpu0, cfg.gpu1, cfg.warmup_iters, cfg.measure_iters);
    if (cfg.check) {
        printf("%-12s %12s %8s\n", "size(B)", "latency(us)", "check");
        printf("%-12s %12s %8s\n", "--------", "-----------", "-----");
    } else {
        printf("%-12s %12s\n", "size(B)", "latency(us)");
        printf("%-12s %12s\n", "--------", "-----------");
    }

    for (int s = 0; s < num_sizes; s++) {
        size_t sz = sizes[s];
        uint8_t pattern = (uint8_t)((s + 1) * 37);
        if (pattern == 0) pattern = 1;

        auto reset_flags = [&]() {
            HIP_CHECK(hipSetDevice(cfg.gpu0));
            HIP_CHECK(hipMemsetAsync(flag0, 0, sizeof(uint64_t), stream0));
            HIP_CHECK(hipStreamSynchronize(stream0));
            HIP_CHECK(hipSetDevice(cfg.gpu1));
            HIP_CHECK(hipMemsetAsync(flag1, 0, sizeof(uint64_t), stream1));
            HIP_CHECK(hipStreamSynchronize(stream1));
        };

        auto launch_pair = [&](int niters, uint64_t* e0, uint64_t* e1) {
            // Launch GPU 1 kernel in a separate thread (it starts by polling)
            std::thread t1([&, niters, e1]() {
                HIP_CHECK(hipSetDevice(cfg.gpu1));
                ping_pong_kernel<<<1, 64, 0, stream1>>>(
                    (uint8_t*)data1, (uint8_t*)data0,
                    flag1, flag0,
                    niters, 1, sz, pattern, e1);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipStreamSynchronize(stream1));
            });
            // GPU 0 kernel with event timing
            HIP_CHECK(hipSetDevice(cfg.gpu0));
            HIP_CHECK(hipEventRecord(ev_start, stream0));
            ping_pong_kernel<<<1, 64, 0, stream0>>>(
                (uint8_t*)data0, (uint8_t*)data1,
                flag0, flag1,
                niters, 0, sz, pattern, e0);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipEventRecord(ev_stop, stream0));
            HIP_CHECK(hipEventSynchronize(ev_stop));
            t1.join();
        };

        // Warmup
        reset_flags();
        launch_pair(cfg.warmup_iters, nullptr, nullptr);

        // Measurement
        reset_flags();
        if (err0) *err0 = 0;
        if (err1) *err1 = 0;
        launch_pair(cfg.measure_iters, err0, err1);

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        // One-way latency = total_time / num_iters / 2
        double latency_us = (double)ms / cfg.measure_iters * 1000.0 / 2.0;

        if (cfg.check) {
            bool pass = (!err0 || *err0 == 0) && (!err1 || *err1 == 0);
            if (!pass) {
                fprintf(stderr, "Check failed at size %zu: err0=%lu, err1=%lu\n",
                        sz, err0 ? (unsigned long)*err0 : 0,
                        err1 ? (unsigned long)*err1 : 0);
            }
            printf("%-12zu %12.2f %8s\n", sz, latency_us,
                   pass ? "PASS" : "FAIL");
        } else {
            printf("%-12zu %12.2f\n", sz, latency_us);
        }
    }

    // Cleanup
    HIP_CHECK(hipSetDevice(cfg.gpu0));
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipStreamDestroy(stream0));
    HIP_CHECK(hipFree(data0));
    HIP_CHECK(hipFree(flag0));

    HIP_CHECK(hipSetDevice(cfg.gpu1));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipFree(data1));
    HIP_CHECK(hipFree(flag1));

    if (err0) HIP_CHECK(hipFree(err0));
    if (err1) HIP_CHECK(hipFree(err1));

    return 0;
}
