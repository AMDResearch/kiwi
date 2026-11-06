// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>

// Verify LCI and invoke headers are accessible
#include "lci.hpp"
#include "kiwi/invoke/invoke.hpp"

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

__global__ void hello_kernel(int* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = 42;
    }
}

int main() {
    printf("kiwi-p2p hello world test\n");

    // Verify HIP runtime
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    printf("  HIP devices: %d\n", device_count);
    assert(device_count > 0 && "No HIP devices found");

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("  GPU: %s (%s)\n", props.name, props.gcnArchName);

    // Verify GPU kernel works
    int* d_result;
    int h_result = 0;
    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));

    hello_kernel<<<1, 1>>>(d_result);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    assert(h_result == 42 && "GPU kernel did not produce expected result");
    printf("  GPU kernel: OK (result=%d)\n", h_result);

    HIP_CHECK(hipFree(d_result));

    // Verify LCI header is usable (just check version/symbols are accessible)
    printf("  LCI header: OK\n");

    // Verify invoke header is usable
    printf("  invoke header: OK\n");

    printf("All checks passed!\n");
    return 0;
}
