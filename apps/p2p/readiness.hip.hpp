// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace kiwi::p2p {

__device__ inline uint8_t load_readiness(const uint8_t* ptr) {
    const unsigned long long addr =
        reinterpret_cast<unsigned long long>(ptr);
    unsigned int value;
    asm volatile("global_load_ubyte %0, %1, off sc0 sc1\n\t"
                 "s_waitcnt vmcnt(0)"
                 : "=v"(value)
                 : "v"(addr)
                 : "memory");
    return static_cast<uint8_t>(value);
}

__device__ inline void reset_readiness(uint8_t* ptr) {
    const unsigned long long addr =
        reinterpret_cast<unsigned long long>(ptr);
    const unsigned int value = 0;
    asm volatile("global_store_byte %0, %1, off sc0 sc1"
                 :
                 : "v"(addr), "v"(value)
                 : "memory");
    __builtin_amdgcn_s_waitcnt(0);
}

} // namespace kiwi::p2p
