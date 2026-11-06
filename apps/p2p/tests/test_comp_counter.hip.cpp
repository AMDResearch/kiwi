// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include "comp_counter.hip.hpp"

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

// Kernel that reads the counter value and stores it
__global__ void read_counter_kernel(lci::comp_t comp, uint64_t* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = kiwi::p2p::get_counter(comp);
    }
}

// Kernel that sets the counter value
__global__ void set_counter_kernel(lci::comp_t comp, uint64_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        kiwi::p2p::set_counter(comp, value);
    }
}

int main() {
    printf("test_comp_counter: starting\n");

    // --- Test 1: Allocate counter and verify initial value ---
    lci::comp_t comp = kiwi::p2p::alloc_counter(0);
    assert(kiwi::p2p::get_counter(comp) == 0);
    printf("  Test 1 (alloc with initial=0): PASSED\n");

    // --- Test 2: Signal N times from CPU, verify from host ---
    static constexpr int N = 10;
    auto* impl = comp.p_impl;
    lci::status_t dummy_status;
    for (int i = 0; i < N; i++) {
        impl->signal(dummy_status);
    }
    uint64_t host_val = kiwi::p2p::get_counter(comp);
    assert(host_val == N);
    printf("  Test 2 (signal %d times, host read): PASSED (value=%lu)\n", N, host_val);

    // --- Test 3: Verify GPU kernel reads the correct value ---
    uint64_t* d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(uint64_t)));

    read_counter_kernel<<<1, 1>>>(comp, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t gpu_read;
    HIP_CHECK(hipMemcpy(&gpu_read, d_result, sizeof(uint64_t), hipMemcpyDeviceToHost));
    assert(gpu_read == N);
    printf("  Test 3 (GPU reads counter after CPU signal): PASSED (value=%lu)\n", gpu_read);

    // --- Test 4: set_counter from CPU, read from GPU ---
    kiwi::p2p::set_counter(comp, 42);
    assert(kiwi::p2p::get_counter(comp) == 42);

    read_counter_kernel<<<1, 1>>>(comp, d_result);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&gpu_read, d_result, sizeof(uint64_t), hipMemcpyDeviceToHost));
    assert(gpu_read == 42);
    printf("  Test 4 (CPU set, GPU read): PASSED (value=%lu)\n", gpu_read);

    // --- Test 5: set_counter from GPU, read from CPU ---
    set_counter_kernel<<<1, 1>>>(comp, 99);
    HIP_CHECK(hipDeviceSynchronize());

    host_val = kiwi::p2p::get_counter(comp);
    assert(host_val == 99);
    printf("  Test 5 (GPU set, CPU read): PASSED (value=%lu)\n", host_val);

    // --- Test 6: Allocate with non-zero initial value ---
    lci::comp_t comp2 = kiwi::p2p::alloc_counter(77);
    assert(kiwi::p2p::get_counter(comp2) == 77);
    printf("  Test 6 (alloc with initial=77): PASSED\n");
    kiwi::p2p::free_counter(comp2);

    // --- Test 7: Free counter ---
    HIP_CHECK(hipFree(d_result));
    kiwi::p2p::free_counter(comp);
    printf("  Test 7 (free_counter): PASSED\n");

    printf("test_comp_counter: ALL PASSED\n");
    return 0;
}
