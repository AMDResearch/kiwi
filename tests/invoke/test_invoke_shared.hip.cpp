// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// End-to-end test for the shared-memory-sourced push path:
//   device push_shared / invoke invoke_shared -> CPU dispatcher -> callback.
//
// A wavefront fills a descriptor in __shared__ memory and submits it via
// invoke_shared(); the CPU dispatcher must invoke the registered callback
// exactly once with byte-identical arguments. Exercises a partial last word
// and one-, two- and eight-line messages.

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/error.hpp>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <functional>

using DeviceContext = kiwi::invoke::DeviceContext<>;

template<int N>
struct Payload {
    uint8_t b[N];
};

// Full-wavefront kernel: cooperatively stage the descriptor in shared memory,
// then submit it as one op via invoke_shared().
template<int N>
__global__ void shared_invoke_kernel(DeviceContext* ctx, uint8_t cb_id,
                                     Payload<N> value) {
    // 8-byte aligned and padded to whole words: invoke_shared() reads
    // round_up(N, 8) bytes.
    __shared__ uint64_t stage_words[(sizeof(Payload<N>) + 7) / 8];
    unsigned char* stage = reinterpret_cast<unsigned char*>(stage_words);

    auto handle = ctx->get_device_handle(0);

    for (int i = threadIdx.x; i < static_cast<int>(sizeof(Payload<N>));
         i += blockDim.x) {
        stage[i] = value.b[i];
    }
    __syncthreads();  // publish the shared descriptor before the collective push

    while (!handle.invoke_shared(cb_id, stage, sizeof(Payload<N>))) {
        // retry on full queue
    }
}

template<int N>
static void run_shared_invoke_case() {
    kiwi::invoke::Context ctx(1, 1024);

    Payload<N> got{};
    int calls = 0;
    std::function<void(Payload<N>)> func = [&](Payload<N> p) {
        got = p;
        ++calls;
    };
    uint8_t cb = ctx.register_callback<Payload<N>>(func);

    Payload<N> in{};
    for (int i = 0; i < N; ++i) {
        in.b[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);
    }

    shared_invoke_kernel<N><<<1, 64>>>(ctx.device_context(), cb, in);
    KIWI_HIP_CHECK(hipDeviceSynchronize());

    size_t executed = ctx.progress(0);
    EXPECT_EQ(executed, 1u);
    EXPECT_EQ(calls, 1);
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(got.b[i], in.b[i]) << "mismatch at byte " << i << " (N=" << N << ")";
    }
}

TEST(KiwiInvokeSharedTest, OneLine)       { run_shared_invoke_case<20>(); }   // partial last word
TEST(KiwiInvokeSharedTest, TwoLines)      { run_shared_invoke_case<56>(); }
TEST(KiwiInvokeSharedTest, FourLines)     { run_shared_invoke_case<128>(); }
TEST(KiwiInvokeSharedTest, MaxMessage)    { run_shared_invoke_case<512>(); }  // 16 lines

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
