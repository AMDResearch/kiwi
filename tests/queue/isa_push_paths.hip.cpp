// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file isa_push_paths.hip.cpp
 * @brief Push kernels whose gfx942 ISA check_push_isa.cmake inspects
 *
 * Compiled with --cuda-device-only -S only; never run. For each
 * device-to-host queue, one extern "C" kernel per push path, plus an invoke
 * kernel that loads its handle from a device-resident DeviceContext. The SPSC
 * invoke loop is the benchmarks' producer loop, where the credit check must
 * stay on the scalar unit.
 */

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/queue.hpp>

using kiwi::queue::MPSCD2HQueue;
using kiwi::queue::SPSCD2HQueue;
using Spsc = SPSCD2HQueue::DeviceHandle;
using Mpsc = MPSCD2HQueue::DeviceHandle;

namespace {

template <size_t Bytes>
struct Msg {
    uint8_t b[Bytes];
};

template <size_t Bytes, typename Handle>
__device__ void push_reg_one(Handle& h, uint64_t x) {
    Msg<Bytes> m;
#pragma unroll
    for (size_t k = 0; k < Bytes; ++k) m.b[k] = static_cast<uint8_t>(x + k);
    while (!h.push_reg(m, 7)) {}
}

template <typename Handle>
__device__ void push_reg(Handle& h, uint64_t x) {
    push_reg_one<13>(h, x);   // partial last word
    push_reg_one<56>(h, x);   // a line plus padding
    push_reg_one<512>(h, x);  // the largest message
}

template <typename Handle>
__device__ void push_shared(Handle& h, uint64_t x, size_t bytes) {
    __shared__ uint64_t row[64];
    row[__lane_id()] = x + __lane_id();
    while (!h.push_shared(row, bytes, 7)) {}
}

template <typename Queue>
__device__ void invoke_from_context(kiwi::invoke::DeviceContext<Queue>* ctx, uint64_t x,
                                    size_t bytes) {
    __shared__ uint64_t row[64];
    auto h = ctx->get_device_handle(0);
    while (!h.invoke(0)) {}
    while (!h.invoke(1, static_cast<uint8_t>(x), static_cast<uint32_t>(x), x)) {}  // 13 B
    while (!h.invoke(1, Msg<168>{})) {}  // the KIWI-EP multicast descriptor size
    row[__lane_id()] = x;
    while (!h.invoke_shared(2, row, bytes)) {}
}

}  // namespace

extern "C" {

__global__ void spsc_push_reg(Spsc h, uint64_t x) { push_reg(h, x); }
__global__ void spsc_push_shared(Spsc h, uint64_t x, size_t bytes) { push_shared(h, x, bytes); }
__global__ void spsc_push_global(Spsc h, const uint64_t* p, size_t n) {
    while (!h.push_global(p, n, 7)) {}
}
__global__ void spsc_push_lanes(Spsc h, uint64_t x, uint32_t bytes) {
    while (!h.push_lanes(x + __lane_id(), bytes, 7)) {}
}
__global__ void spsc_invoke_context(kiwi::invoke::DeviceContext<SPSCD2HQueue>* ctx, uint64_t x,
                                    size_t bytes) {
    invoke_from_context(ctx, x, bytes);
}

__global__ void spsc_invoke_loop(kiwi::invoke::DeviceContext<SPSCD2HQueue>* ctx, uint8_t cb,
                                 int n) {
    auto h = ctx->get_device_handle(0);
    for (int i = 0; i < n; ++i) {
        while (!h.invoke(cb, static_cast<uint32_t>(i))) {}
    }
}

__global__ void mpsc_push_reg(Mpsc h, uint64_t x) { push_reg(h, x); }
__global__ void mpsc_push_shared(Mpsc h, uint64_t x, size_t bytes) { push_shared(h, x, bytes); }
__global__ void mpsc_push_global(Mpsc h, const uint64_t* p, size_t n) {
    while (!h.push_global(p, n, 7)) {}
}
__global__ void mpsc_push_lanes(Mpsc h, uint64_t x, uint32_t bytes) {
    while (!h.push_lanes(x + __lane_id(), bytes, 7)) {}
}
__global__ void mpsc_invoke_context(kiwi::invoke::DeviceContext<MPSCD2HQueue>* ctx, uint64_t x,
                                    size_t bytes) {
    invoke_from_context(ctx, x, bytes);
}

}  // extern "C"
