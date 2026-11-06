// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include "lci.hpp"

namespace kiwi::p2p {

namespace internal {

class comp_counter_t : public lci::comp_impl_t {
public:
    explicit comp_counter_t(uint64_t initial_value = 0)
        : lci::comp_impl_t() {
        counter_value = initial_value;
    }

    ~comp_counter_t() = default;

    // Called from CPU by LCI completion handlers
    void signal(lci::status_t) override {
        __atomic_fetch_add(&counter_value, 1, __ATOMIC_RELAXED);
    }

    __host__ __device__ uint64_t get() const {
        return *reinterpret_cast<volatile const uint64_t*>(&counter_value);
    }

    __host__ __device__ void set(uint64_t value) {
        *reinterpret_cast<volatile uint64_t*>(&counter_value) = value;
    }

private:
    uint64_t counter_value;
};

} // namespace internal

// Allocate a completion counter in managed memory (fine-grained on MI300X).
// The counter implements lci::comp_impl_t so LCI can signal() it.
// Managed memory is coherent between CPU and GPU, so CPU atomic
// updates are immediately visible to GPU polling.
inline lci::comp_t alloc_counter(uint64_t initial_value = 0) {
    internal::comp_counter_t* counter_p;
    hipError_t err = hipMallocManaged(reinterpret_cast<void**>(&counter_p),
                                      sizeof(internal::comp_counter_t));
    if (err != hipSuccess) {
        fprintf(stderr, "kiwi::p2p::alloc_counter: hipMallocManaged failed: %s\n",
                hipGetErrorString(err));
        abort();
    }
    new (counter_p) internal::comp_counter_t(initial_value);
    return lci::comp_t(counter_p);
}

// Free a completion counter allocated with alloc_counter.
inline void free_counter(lci::comp_t& comp) {
    auto counter_p = static_cast<internal::comp_counter_t*>(comp.p_impl);
    counter_p->~comp_counter_t();
    (void)hipFree(counter_p);
    comp = lci::COMP_NULL;
}

// Read counter value. Callable from both GPU and CPU.
inline __host__ __device__ uint64_t get_counter(lci::comp_t comp) {
    auto counter_p = reinterpret_cast<internal::comp_counter_t*>(comp.p_impl);
    return counter_p->get();
}

// Set counter value. Callable from both GPU and CPU.
inline __host__ __device__ void set_counter(lci::comp_t comp, uint64_t value) {
    auto counter_p = reinterpret_cast<internal::comp_counter_t*>(comp.p_impl);
    counter_p->set(value);
}

// Device-side busy-wait until counter reaches target.
inline __device__ void poll_counter(lci::comp_t comp, uint64_t target) {
    auto counter_p = reinterpret_cast<internal::comp_counter_t*>(comp.p_impl);
    while (counter_p->get() < target) {
        // spin
    }
}

} // namespace kiwi::p2p
