// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file queue_bench.hip.hpp
 * @brief Message sizes shared by the device-to-host queue benchmarks
 *
 * Every benchmark message is --bytes bytes, pushed with push_reg(), and
 * carries a 32-bit sequence value in its first four bytes.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace queue_bench {

template <size_t... B>
struct SizeList {
    static std::vector<int> values() { return {static_cast<int>(B)...}; }

    /// Calls f(std::integral_constant<size_t, B>{}) for B == bytes.
    template <class F>
    static void dispatch(size_t bytes, F&& f) {
        ((bytes == B ? (f(std::integral_constant<size_t, B>{}), 0) : 0), ...);
    }
};

/// Supported --bytes values: one to sixteen 64-byte lines, and the KIWI-EP /
/// kiwi-p2p descriptor sizes.
using Sizes = SizeList<8, 24, 32, 48, 56, 64, 112, 168, 240, 448, 512>;

inline std::vector<int> sizes() { return Sizes::values(); }

template <class F>
inline void dispatch(size_t bytes, F&& f) {
    Sizes::dispatch(bytes, f);
}

/// A Bytes-byte message.
template <size_t Bytes>
struct Message {
    uint32_t value;
    uint8_t padding[Bytes - sizeof(uint32_t)];
};

}  // namespace queue_bench
