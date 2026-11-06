// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <vector>
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

__device__ void device_memset(void* data, size_t size, uint8_t value) {
    for (size_t i = 0; i < size; ++i) {
        reinterpret_cast<uint8_t*>(data)[i] = value;
    }
}

__device__ void device_memcheck(void* data, size_t size, uint8_t value) {
    for (size_t i = 0; i < size; ++i) {
        assert(reinterpret_cast<uint8_t*>(data)[i] == value);
    }
}

// Ping-pong kernel (wave-level: all 64 threads in wavefront cooperate on post_put):
// Rank 0: fill buffer -> post_put -> wait for reply readiness
// Rank 1: wait for readiness -> verify -> fill -> post_put back
__global__ void ping_pong_kernel(kiwi::p2p::gcontext_t gctx, int niters,
                                  int rank_me, int peer_rank,
                                  void* data, size_t size,
                                  int data_heap_id,
                                  uint8_t* ready, int ready_heap_id) {
    auto handle = gctx.get_handle();
    for (int i = 0; i < niters; i++) {
        if (rank_me == 0) {
            // Fill buffer with rank_me pattern, send to peer
            device_memset(data, size, (uint8_t)rank_me);
            __threadfence_system();
            handle.post_put(data_heap_id, ready_heap_id, peer_rank,
                            data, size, 0, 0);
            if (threadIdx.x == 0) {
                while (kiwi::p2p::load_readiness(ready) == 0) continue;
                kiwi::p2p::reset_readiness(ready);
            }
            __syncthreads();
            // Verify received data from peer
            device_memcheck(data, size, (uint8_t)peer_rank);
        } else {
            if (threadIdx.x == 0) {
                while (kiwi::p2p::load_readiness(ready) == 0) continue;
                kiwi::p2p::reset_readiness(ready);
            }
            __syncthreads();
            // Verify received data from rank 0
            device_memcheck(data, size, (uint8_t)(peer_rank));
            // Fill buffer with rank_me pattern, send back
            device_memset(data, size, (uint8_t)rank_me);
            __threadfence_system();
            handle.post_put(data_heap_id, ready_heap_id, peer_rank,
                            data, size, 0, 0);
        }
    }
}

int main() {
    static constexpr int NUM_QUEUES = 4;
    static constexpr size_t BUF_SIZE = 4096;
    static constexpr int NITERS = 10;

    kiwi::p2p::context_t ctx(NUM_QUEUES);

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    assert(nranks == 2 && "Ping-pong test requires exactly 2 ranks");
    int peer = 1 - rank;

    printf("[Rank %d/%d] test_pingpong: started\n", rank, nranks);

    // Allocate device memory and register as symmetric heap
    void* dev_buf;
    HIP_CHECK(hipMalloc(&dev_buf, BUF_SIZE));
    HIP_CHECK(hipMemset(dev_buf, 0, BUF_SIZE));

    int heap_id = ctx.register_symmetric(dev_buf, BUF_SIZE);

    uint8_t* ready;
    HIP_CHECK(hipMalloc(&ready, sizeof(uint8_t)));
    HIP_CHECK(hipMemset(ready, 0, sizeof(uint8_t)));
    int ready_heap_id = ctx.register_symmetric(ready, sizeof(uint8_t));

    lci::barrier();
    HIP_CHECK(hipDeviceSynchronize());

    // Spawn proxy threads
    ctx.spawn_proxy_threads();

    kiwi::p2p::gcontext_t gctx = ctx.get_device_context();

    // Run ping-pong kernel with 1 block x 64 threads (1 wavefront)
    printf("[Rank %d] Running ping-pong: %d iterations, %zu bytes\n",
           rank, NITERS, BUF_SIZE);

    lci::barrier();

    ping_pong_kernel<<<1, 64>>>(gctx, NITERS, rank, peer,
                                dev_buf, BUF_SIZE, heap_id,
                                ready, ready_heap_id);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    printf("[Rank %d] Ping-pong completed successfully\n", rank);

    // Join proxy threads
    ctx.join_proxy_threads();

    // Cleanup
    lci::barrier();
    ctx.deregister_symmetric(ready_heap_id);
    ctx.deregister_symmetric(heap_id);
    HIP_CHECK(hipFree(ready));
    HIP_CHECK(hipFree(dev_buf));

    printf("[Rank %d] test_pingpong: PASSED\n", rank);
    return 0;
}
