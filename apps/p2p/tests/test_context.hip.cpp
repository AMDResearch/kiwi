// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include "context.hip.hpp"

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

// Simple kernel that verifies gcontext_t and get_handle() work on device
__global__ void check_gcontext_kernel(kiwi::p2p::gcontext_t gctx, int* result) {
    auto handle = gctx.get_handle();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // gcontext_t and ghandle_t are accessible on device
        *result = 1;
    }
}

int main() {
    // Create context with 4 queues (this initializes LCI)
    static constexpr int NUM_QUEUES = 4;
    kiwi::p2p::context_t ctx(NUM_QUEUES);

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    printf("[Rank %d/%d] test_context: context created with %d queues\n", rank, nranks, NUM_QUEUES);

    // Get device context
    kiwi::p2p::gcontext_t gctx = ctx.get_device_context();
    printf("[Rank %d] Device context obtained\n", rank);

    // Verify gcontext_t is accessible on GPU
    int* d_result;
    int h_result = 0;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));

    check_gcontext_kernel<<<1, 1>>>(gctx, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    assert(h_result == 1 && "gcontext_t not accessible on GPU");
    printf("[Rank %d] gcontext_t accessible on GPU: OK\n", rank);

    HIP_CHECK(hipFree(d_result));

    // Spawn proxy threads and then join
    printf("[Rank %d] Spawning proxy threads\n", rank);
    ctx.spawn_proxy_threads();
    printf("[Rank %d] Proxy threads spawned, joining...\n", rank);

    // Let them run briefly, then stop
    ctx.join_proxy_threads();
    printf("[Rank %d] Proxy threads joined successfully\n", rank);

    printf("[Rank %d] test_context: PASSED\n", rank);
    return 0;
}
