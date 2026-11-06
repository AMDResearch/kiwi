// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// invoke over both device-to-host queues (kiwi::invoke::Context<Q>).
//
// Queue mechanics (wraparound, push paths, partial messages) are covered by
// test_queue_chunk; this file checks what invoke adds: argument
// serialization of 0 to 512 bytes through invoke() and invoke_shared(),
// handles saved across launches, the message length checked against the
// registered argument size, the ring-size check at registration, and many
// waves sharing an MPSC endpoint.

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/queue.hpp>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {

/// N-word argument; w[k] = i * 1000 + k.
template <int N>
struct Blob {
    uint64_t w[N];
};

template <int N>
__host__ __device__ inline Blob<N> make_blob(uint64_t i) {
    Blob<N> b;
    for (int k = 0; k < N; ++k) b.w[k] = i * 1000 + k;
    return b;
}

/// Three arguments, 13 bytes in all (a partial last word).
struct Odd {
    uint8_t a;
    uint32_t b;
    uint64_t c;
} __attribute__((packed));

// Kinds: 0 no arguments, 1 Odd (13 B), 2 Blob<21> (168 B, the KIWI-EP
// multicast descriptor), 3 the same through invoke_shared(), 4 Blob<64>
// (512 B) through invoke_shared().
constexpr int kKinds = 5;

__host__ __device__ inline int kind_of(uint64_t i) { return (i * 3 + i / kKinds) % kKinds; }

template <int N, typename H>
__device__ void send_shared(H& h, uint8_t cb, uint64_t* stage, uint64_t i) {
    for (int k = __lane_id(); k < N; k += 64) stage[k] = i * 1000 + k;
    __syncthreads();
    while (!h.invoke_shared(cb, stage, sizeof(Blob<N>))) {}
    __syncthreads();  // the next message reuses the stage
}

/// One wave per block; block b sends callbacks first .. first + msgs - 1 to
/// endpoint b, then saves its handle for the next launch.
template <typename DevCtx>
__global__ void sender(DevCtx* ctx, const uint8_t* ids, uint64_t first, int msgs) {
    __shared__ uint64_t stage[64];
    auto h = ctx->get_device_handle();
    for (uint64_t i = first; i < first + msgs; ++i) {
        const uint8_t cb = ids[kind_of(i)];
        switch (kind_of(i)) {
            case 0: while (!h.invoke(cb)) {} break;
            case 1: while (!h.invoke(cb, Odd{static_cast<uint8_t>(i), static_cast<uint32_t>(i), i})) {} break;
            case 2: while (!h.invoke(cb, make_blob<21>(i))) {} break;
            case 3: send_shared<21>(h, cb, stage, i); break;
            default: send_shared<64>(h, cb, stage, i); break;
        }
    }
    ctx->save_device_handle(blockIdx.x, h);
}

/// Next expected message of the endpoint being progressed, and errors seen.
struct Receiver {
    uint64_t next = 0;
    uint64_t errors = 0;
    void check(int kind, bool ok) {
        errors += kind_of(next) != kind || !ok;
        ++next;
    }
};

template <int N>
std::function<void(Blob<N>)> blob_checker(Receiver** r, int kind) {
    return [r, kind](Blob<N> b) {
        bool ok = true;
        for (int k = 0; k < N; ++k) ok = ok && b.w[k] == (*r)->next * 1000 + k;
        (*r)->check(kind, ok);
    };
}

/// Eight endpoints in 32-line rings, relaunched 10 times; each launch
/// continues from the handles the previous one saved. Every callback must
/// run once, in order, with its exact arguments.
template <typename Q>
void run_stream() {
    constexpr size_t kEndpoints = 8;
    constexpr int kMsgs = kKinds * 60;
    kiwi::invoke::Context<Q> ctx(kEndpoints, 32);
    std::vector<Receiver> receivers(kEndpoints);
    Receiver* cur = nullptr;
    Receiver** r = &cur;
    uint8_t ids[kKinds];
    ids[0] = ctx.template register_callback<>(std::function<void()>([r] { (*r)->check(0, true); }));
    ids[1] = ctx.template register_callback<Odd>(std::function<void(Odd)>([r](Odd o) {
        const uint64_t i = (*r)->next;
        (*r)->check(1, o.a == static_cast<uint8_t>(i) && o.b == static_cast<uint32_t>(i) && o.c == i);
    }));
    ids[2] = ctx.template register_callback<Blob<21>>(blob_checker<21>(r, 2));
    ids[3] = ctx.template register_callback<Blob<21>>(blob_checker<21>(r, 3));
    ids[4] = ctx.template register_callback<Blob<64>>(blob_checker<64>(r, 4));
    uint8_t* d_ids;
    ASSERT_EQ(hipMalloc(&d_ids, sizeof(ids)), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_ids, ids, sizeof(ids), hipMemcpyHostToDevice), hipSuccess);
    std::vector<typename Q::HostPollState> states;
    for (size_t e = 0; e < kEndpoints; ++e) states.push_back(ctx.raw_queue(e)->host_poll_state());

    for (int launch = 0; launch < 10; ++launch) {
        const uint64_t target = static_cast<uint64_t>(launch + 1) * kMsgs;
        sender<<<kEndpoints, 64>>>(ctx.device_context(), d_ids, target - kMsgs, kMsgs);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
        auto last = std::chrono::steady_clock::now();
        for (size_t done = 0; done < kEndpoints;) {
            done = 0;
            for (size_t e = 0; e < kEndpoints; ++e) {
                if (receivers[e].next == target) {
                    ++done;
                } else if (states[e].ready()) {  // KIWI-EP-style sweep
                    cur = &receivers[e];
                    if (ctx.progress(e) > 0) last = std::chrono::steady_clock::now();
                    states[e].refresh();
                }
            }
            ASSERT_LT(std::chrono::steady_clock::now() - last, std::chrono::seconds(20))
                << "timed out in launch " << launch;
        }
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }
    for (size_t e = 0; e < kEndpoints; ++e) {
        cur = &receivers[e];
        EXPECT_EQ(ctx.progress(e), 0u);
        EXPECT_EQ(receivers[e].errors, 0u) << "endpoint " << e;
    }
    ASSERT_EQ(hipFree(d_ids), hipSuccess);
}

template <typename DevCtx>
__global__ void mislabeled_sender(DevCtx* ctx, uint8_t cb) {
    auto h = ctx->get_device_handle(0);
    while (!h.invoke(cb, make_blob<2>(7))) {}  // 16 bytes: wrong size
    while (!h.invoke(cb, make_blob<3>(8))) {}  // 24 bytes: right size
}

}  // namespace

template <typename Q>
class InvokeChunkQueue : public ::testing::Test {};
using Queues = ::testing::Types<kiwi::queue::SPSCD2HQueue, kiwi::queue::MPSCD2HQueue>;
TYPED_TEST_SUITE(InvokeChunkQueue, Queues);

TYPED_TEST(InvokeChunkQueue, MixedArgumentsRelaunchWithSavedHandles) { run_stream<TypeParam>(); }

// Messages carry their length: a 16-byte message sent to a callback
// registered for 24 bytes is rejected instead of being read past its end. The
// bad message is consumed, so the endpoint keeps working.
TYPED_TEST(InvokeChunkQueue, LengthMismatchIsRejected) {
    kiwi::invoke::Context<TypeParam> ctx(1, 16);
    int calls = 0;
    const uint8_t cb = ctx.template register_callback<Blob<3>>(
        std::function<void(Blob<3>)>([&calls](Blob<3>) { ++calls; }));
    mislabeled_sender<<<1, 64>>>(ctx.device_context(), cb);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_THROW(ctx.progress(0), std::runtime_error);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(ctx.progress(0), 1u);
    EXPECT_EQ(calls, 1);
}

// Arguments that need more lines than the ring holds are rejected on the
// host, not left to trap in the device push.
TYPED_TEST(InvokeChunkQueue, RegisterRejectsArgumentsLargerThanRing) {
    kiwi::invoke::Context<TypeParam> ctx(2, 4);
    const auto blob21 = std::function<void(Blob<21>)>([](Blob<21>) {});  // 168 B: 6 lines
    const auto blob16 = std::function<void(Blob<16>)>([](Blob<16>) {});  // 128 B: 4 lines
    EXPECT_THROW(ctx.template register_callback<Blob<21>>(blob21), std::invalid_argument);
    EXPECT_EQ(ctx.template register_callback<Blob<16>>(blob16), 0u);
    kiwi::invoke::Context<TypeParam> big(1, 8);
    EXPECT_EQ(big.template register_callback<Blob<21>>(blob21), 0u);
}

namespace {

using MpscCtx = kiwi::invoke::DeviceContext<kiwi::queue::MPSCD2HQueue>;

/// Every wave sends `msgs` callbacks (wave, i, Blob<5>) to endpoint
/// wave % endpoints, alternating invoke() and invoke_shared().
__global__ void shared_endpoint_sender(MpscCtx* ctx, uint8_t cb, int endpoints, int msgs) {
    __shared__ uint64_t stage[4][8];
    const uint32_t local = threadIdx.x / 64;
    const uint32_t wave = blockIdx.x * (blockDim.x / 64) + local;
    auto h = ctx->get_device_handle(wave % endpoints);
    for (int i = 0; i < msgs; ++i) {
        const uint64_t key = (static_cast<uint64_t>(wave) << 32) | i;
        if (i % 2 == 0) {
            while (!h.invoke(cb, key, make_blob<5>(key))) {}
        } else {
            uint64_t* row = stage[local];
            if (__lane_id() == 0) row[0] = key;
            if (__lane_id() < 5) row[1 + __lane_id()] = key * 1000 + __lane_id();
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            while (!h.invoke_shared(cb, row, 6 * sizeof(uint64_t))) {}
            __builtin_amdgcn_wave_barrier();  // the next message reuses the row
        }
    }
}

}  // namespace

// 64 waves share 4 MPSC endpoints of 8 lines each (many laps). One progress
// loop per endpoint must run every callback once, with its exact arguments,
// and each wave's callbacks in the order it sent them.
TEST(InvokeMpsc, ManyWavesShareEachEndpoint) {
    constexpr int kEndpoints = 4, kBlocks = 16, kWavesPerBlock = 4, kMsgs = 200;
    constexpr int kWaves = kBlocks * kWavesPerBlock;
    kiwi::invoke::Context<kiwi::queue::MPSCD2HQueue> ctx(kEndpoints, 8);
    std::vector<int> next(kWaves, 0);
    uint64_t errors = 0, calls = 0;
    const uint8_t cb = ctx.register_callback<uint64_t, Blob<5>>(
        std::function<void(uint64_t, Blob<5>)>([&](uint64_t key, Blob<5> b) {
            const uint64_t wave = key >> 32, i = key & 0xffffffffu;
            bool ok = wave < kWaves && static_cast<int>(i) == next[wave]++;
            for (int k = 0; k < 5; ++k) ok = ok && b.w[k] == key * 1000 + k;
            errors += !ok;
            ++calls;
        }));
    shared_endpoint_sender<<<kBlocks, 64 * kWavesPerBlock>>>(ctx.device_context(), cb, kEndpoints,
                                                             kMsgs);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    auto last = std::chrono::steady_clock::now();
    while (calls < static_cast<uint64_t>(kWaves) * kMsgs) {
        for (int e = 0; e < kEndpoints; ++e) {
            if (ctx.progress(e) > 0) last = std::chrono::steady_clock::now();
        }
        ASSERT_LT(std::chrono::steady_clock::now() - last, std::chrono::seconds(20))
            << "timed out after " << calls << " callbacks";
    }
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    for (int e = 0; e < kEndpoints; ++e) EXPECT_EQ(ctx.progress(e), 0u);
    EXPECT_EQ(errors, 0u);
}
