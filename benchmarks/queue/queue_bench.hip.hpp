// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file queue_bench.hip.hpp
 * @brief Message sizes and push patterns shared by the device-to-host queue
 *        benchmarks
 *
 * Every benchmark message is --bytes bytes and carries a 32-bit sequence
 * value in its first four bytes. --push selects how a producer wave pushes;
 * one step of the pattern is one push call (two for `mixed`):
 *
 *   collective   push_reg(): one message per step.
 *   lane-masked  push_lane_masked() with lanes [0, --lanes) active: one
 *                message per active lane (an SPSC call is retried while it
 *                returns false).
 *   mixed        collective, then lane-masked: 1 + --lanes messages per step.
 */

#pragma once

#include <kiwi/queue/queue.hpp>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
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

// ---------------------------------------------------------------------------
// Push patterns (--push, --lanes)
// ---------------------------------------------------------------------------

enum class Push { Collective, LaneMasked, Mixed };

inline const std::map<std::string, Push>& push_names() {
    static const std::map<std::string, Push> names{{"collective", Push::Collective},
                                                   {"lane-masked", Push::LaneMasked},
                                                   {"mixed", Push::Mixed}};
    return names;
}

inline const char* push_name(Push p) {
    for (const auto& [name, value] : push_names()) {
        if (value == p) return name.c_str();
    }
    return "?";
}

/// True if one lane-masked call of pattern `p` (`lanes` messages of `bytes`
/// bytes) fits a `capacity`-line ring; otherwise prints why: the call needs
/// more lines than the ring holds, and push_lane_masked() would trap.
inline bool lane_masked_call_fits(Push p, int lanes, size_t bytes, size_t capacity) {
    if (p == Push::Collective) return true;
    const uint64_t lines = static_cast<uint64_t>(lanes) * kiwi::queue::message_lines(bytes);
    if (lines <= capacity) return true;
    std::cerr << "ERROR: one lane-masked call takes " << lines << " lines, more than a "
              << capacity << "-line ring holds; raise --capacity\n";
    return false;
}

/// Messages one step pushes.
__host__ __device__ inline int step_messages(Push p, int lanes) {
    switch (p) {
        case Push::Collective: return 1;
        case Push::Mixed: return 1 + lanes;
        default: return lanes;
    }
}

/// Calls f(std::integral_constant<Push, P>{}) for P == p.
template <class F>
inline void dispatch_push(Push p, F&& f) {
    switch (p) {
        case Push::Collective: f(std::integral_constant<Push, Push::Collective>{}); break;
        case Push::LaneMasked: f(std::integral_constant<Push, Push::LaneMasked>{}); break;
        case Push::Mixed: f(std::integral_constant<Push, Push::Mixed>{}); break;
    }
}

/// One step of pattern P: messages seq, seq + 1, ... (step_messages() of
/// them; lane r of a lane-masked call pushes seq + r, after the collective
/// message in a mixed step). In a mixed step, *t_mid is set to the GPU wall
/// clock between the collective and the lane-masked push.
template <Push P, size_t Bytes, class Producer>
__device__ __forceinline__ void push_step(Producer& producer, uint32_t seq, int lanes,
                                          uint64_t* t_mid = nullptr) {
    Message<Bytes> msg{};
    if constexpr (P == Push::Collective || P == Push::Mixed) {
        msg.value = seq;
        while (!producer.push_reg(msg)) {}
    }
    if constexpr (P == Push::Mixed) {
        if (t_mid) *t_mid = wall_clock64();
        seq += 1;
    }
    if constexpr (P == Push::LaneMasked || P == Push::Mixed) {
        const uint32_t lane = __lane_id();
        msg.value = seq + lane;
        while (!producer.push_lane_masked(msg, Bytes, 0, lane < static_cast<uint32_t>(lanes))) {}
    }
}

}  // namespace queue_bench
