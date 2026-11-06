// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include <cstring>
#include "context.hip.hpp"
#include "lci.hpp"

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

// Kernel that verifies RMR data is accessible on GPU
__global__ void check_rmrs_kernel(lci::rmr_t* rmrs, int nranks, int* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int ok = 1;
        for (int i = 0; i < nranks; i++) {
            // Each rank registered memory, so base should be non-zero
            if (rmrs[i].base == 0 && rmrs[i].opaque_rkey == 0) {
                ok = 0;
            }
        }
        *result = ok;
    }
}

int main() {
    static constexpr int NUM_QUEUES = 1;
    static constexpr size_t BUF_SIZE = 4096;

    kiwi::p2p::context_t ctx(NUM_QUEUES);

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    printf("[Rank %d/%d] test_memory: started\n", rank, nranks);

    // 1. Allocate device memory
    void* dev_buf;
    HIP_CHECK(hipMalloc(&dev_buf, BUF_SIZE));
    HIP_CHECK(hipMemset(dev_buf, 0, BUF_SIZE));
    printf("[Rank %d] Allocated %zu bytes of device memory at %p\n",
           rank, BUF_SIZE, dev_buf);

    // 2. Register device memory with LCI
    lci::mr_t mr = lci::register_memory(dev_buf, BUF_SIZE);
    printf("[Rank %d] Memory registered with LCI\n", rank);

    // 3. Get local RMR
    lci::rmr_t my_rmr = lci::get_rmr(mr);
    printf("[Rank %d] Local RMR: base=0x%lx, rkey=0x%lx\n",
           rank, (unsigned long)my_rmr.base, (unsigned long)my_rmr.opaque_rkey);
    assert(!my_rmr.is_empty() && "Local RMR should not be empty");

    // 4. Exchange RMRs across all ranks
    lci::rmr_t my_rmr2 = lci::get_rmr(mr);
    std::vector<lci::rmr_t> rmrs(nranks);
    lci::allgather(&my_rmr2, rmrs.data(), sizeof(lci::rmr_t));
    printf("[Rank %d] RMR exchange complete (%d entries)\n", rank, nranks);

    // 5. Verify all RMRs are non-empty
    for (int i = 0; i < nranks; i++) {
        assert(!rmrs[i].is_empty() && "Exchanged RMR should not be empty");
        printf("[Rank %d] RMR[%d]: base=0x%lx, rkey=0x%lx\n",
               rank, i, (unsigned long)rmrs[i].base,
               (unsigned long)rmrs[i].opaque_rkey);
    }
    // Own RMR should match
    assert(rmrs[rank].base == my_rmr.base && "Own RMR base mismatch");
    assert(rmrs[rank].opaque_rkey == my_rmr.opaque_rkey && "Own RMR rkey mismatch");
    printf("[Rank %d] RMR verification: OK\n", rank);

    // 6. Pass RMR array to GPU kernel and verify accessibility
    lci::rmr_t* d_rmrs;
    HIP_CHECK(hipMalloc(&d_rmrs, nranks * sizeof(lci::rmr_t)));
    HIP_CHECK(hipMemcpy(d_rmrs, rmrs.data(), nranks * sizeof(lci::rmr_t),
                         hipMemcpyHostToDevice));

    int* d_result;
    int h_result = 0;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));

    check_rmrs_kernel<<<1, 1>>>(d_rmrs, nranks, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    assert(h_result == 1 && "RMRs not accessible or invalid on GPU");
    printf("[Rank %d] GPU RMR accessibility check: OK\n", rank);

    // 7. Cleanup
    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(d_rmrs));
    lci::barrier();
    lci::deregister_memory(&mr);
    HIP_CHECK(hipFree(dev_buf));

    printf("[Rank %d] test_memory: PASSED\n", rank);
    return 0;
}
