// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file qb_common.hpp
 * @brief Shared definitions for the GPU->CPU queue baseline microbenchmarks.
 *
 * Every queue adapter carries the same logical message (Msg) inside its own
 * native descriptor, so the harness can validate every handoff identically
 * while each adapter keeps its library's descriptor size and publish protocol.
 *
 * Adapter contract (checked at compile time by the harness templates):
 *
 *   struct Adapter {
 *     static constexpr const char* kName;
 *     static constexpr bool kMultiProducer;   // may several waves share a queue
 *     static constexpr bool kFifo;            // per-producer FIFO order guaranteed
 *     static constexpr int kDescriptorBytes;  // bytes the producer publishes
 *     using Handle = ...;                     // trivially copyable device handle
 *
 *     Adapter(int num_queues, int capacity);
 *     Handle handle(int queue);               // host: handle bound to `queue`
 *     __device__ static void globalize(Handle&);         // see as_global()
 *     __device__ static void push(Handle&, const Msg&);  // all 64 lanes call it
 *     bool pop(int queue, Msg& out);          // host: one consumer per queue
 *     static std::string notes();             // placement / protocol summary
 *   };
 *
 * push() is called by a full wavefront with wave-uniform arguments and returns
 * only after the message is published (it spins while the queue is full).
 */

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define QB_HIP_CHECK(expr)                                                     \
    do {                                                                       \
        hipError_t qb_err_ = (expr);                                           \
        if (qb_err_ != hipSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, \
                         #expr, hipGetErrorString(qb_err_));                   \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace qb {

/// Logical message carried by every queue (16 bytes).
struct Msg {
    uint32_t producer;
    uint32_t seq;
    uint64_t tag;
};

/// Per-message tag, derived from (producer, seq) so the consumer can detect
/// torn, stale, or misrouted descriptors. It is deliberately cheap (one NOT),
/// because message construction runs inside the timed push and must not be
/// mistaken for queue cost. Bit 63 is kept clear because some protocols
/// (e.g. MSCCL++ FIFO) reserve it as a commit bit.
__host__ __device__ inline uint64_t msg_tag(uint32_t producer, uint32_t seq) {
    return ~((static_cast<uint64_t>(seq) << 32) | producer) & 0x7fffffffffffffffULL;
}

__host__ __device__ inline Msg make_msg(uint32_t producer, uint32_t seq) {
    return Msg{producer, seq, msg_tag(producer, seq)};
}

/// Word 0 of the canonical encoding: (seq << 32) | producer.
__host__ __device__ inline uint64_t msg_word0(const Msg& m) {
    return (static_cast<uint64_t>(m.seq) << 32) | m.producer;
}

__host__ __device__ inline Msg msg_from_words(uint64_t w0, uint64_t w1) {
    return Msg{static_cast<uint32_t>(w0), static_cast<uint32_t>(w0 >> 32), w1};
}

/// Filler for descriptor words beyond the 16-byte message, so every adapter
/// publishes its full native descriptor with non-trivial content.
__host__ __device__ inline uint64_t filler_word(const Msg& m, int word) {
    return m.tag ^ (static_cast<uint64_t>(word) << 56);
}

__device__ inline int lane_id() { return __lane_id(); }

/// Tell the compiler that `p` points to global memory (device or mapped host).
///
/// Pointers loaded from memory are generic, so accesses through them compile
/// to flat_* instructions. On gfx9 a flat store also holds lgkmcnt, so the
/// next s_waitcnt lgkmcnt(0) (an LDS access, a scalar load) stalls until the
/// store to host memory is acknowledged. Handles passed by value as kernel
/// arguments do not have this problem; handles loaded from a per-wave array
/// (throughput kernel) do. The empty asm makes the address-space-1 value
/// opaque, so the cast pair is not folded away and InferAddressSpaces
/// rewrites every access derived from the result to global_*.
template <typename T>
__device__ inline T* as_global(T* p) {
    // Handles are wave-uniform; keep the pointer in SGPRs so the address
    // arithmetic stays scalar, as for a by-value kernel argument.
    const auto bits = reinterpret_cast<uintptr_t>(p);
    const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(bits));
    const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(bits >> 32));
    using G = __attribute__((address_space(1))) T*;
    G g = (G)((static_cast<uint64_t>(hi) << 32) | lo);
    __asm__ volatile("" : "+s"(g));
    return (T*)g;
}

/// `v` with every 32-bit word taken from the first lane, so it lives in
/// SGPRs like a by-value kernel argument. For a handle whose indices live in
/// it (not only pointers, see as_global()) loaded from a per-wave array: left
/// in VGPRs, those indices are carried around the push loop in VGPRs, and the
/// s_waitcnt vmcnt(0) the compiler then puts at the top of the loop also
/// waits for the previous store to host memory to be acknowledged (vmcnt
/// counts stores too). Requires the whole wavefront to be active.
template <typename T>
__device__ inline T wave_uniform(const T& v) {
    static_assert(sizeof(T) % sizeof(uint32_t) == 0, "whole 32-bit words only");
    uint32_t words[sizeof(T) / sizeof(uint32_t)];
    __builtin_memcpy(words, &v, sizeof(T));
#pragma unroll
    for (uint32_t& w : words) w = __builtin_amdgcn_readfirstlane(w);
    T out;
    __builtin_memcpy(&out, words, sizeof(T));
    return out;
}

}  // namespace qb
