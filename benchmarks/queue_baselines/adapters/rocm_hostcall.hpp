// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file rocm_hostcall.hpp
 * @brief ROCm hostcall adapter (device-libs ockl + CLR listener, unmodified).
 *
 * A push is __ockl_call_host_function() (ockl services.cl), which the stock
 * HIP runtime serves with SERVICE_FUNCTION_CALL (CLR devhostcall.cpp): the
 * listener thread calls a host function pointer with the seven argument
 * words and writes two result words back. The call is synchronous, so this
 * adapter has no host pop(): the "consumer" is the CLR listener thread, and
 * the harness validates messages from inside the host function.
 *
 * Only lane 0 calls, so one 64B lane packet (fptr + 7 args) is used. The
 * listener is interrupt-driven (HSA signal wait) and unpinned; that is the
 * runtime's behavior and is not changed here.
 */

#pragma once

#include "../qb_common.hpp"

#include <functional>
#include <string>

typedef long qb_long2 __attribute__((ext_vector_type(2)));
extern "C" __device__ qb_long2 __ockl_call_host_function(
    unsigned long fptr, unsigned long arg0, unsigned long arg1, unsigned long arg2,
    unsigned long arg3, unsigned long arg4, unsigned long arg5, unsigned long arg6);

namespace qb {

struct RocmHostcallAdapter {
    static constexpr const char* kName = "rocm-hostcall";
    static constexpr bool kMultiProducer = true;
    static constexpr bool kFifo = true;  // calls are synchronous per producer
    static constexpr bool kSynchronous = true;
    static constexpr int kDescriptorBytes = 64;
    using Handle = uint64_t;  // host function pointer

    /// Called by the CLR listener thread for every message.
    static inline std::function<void(const Msg&)> sink;

    RocmHostcallAdapter(int /*num_queues*/, int /*capacity*/) {}

    Handle handle(int) { return reinterpret_cast<uint64_t>(&host_function); }

    __device__ static void globalize(Handle&) {}

    __device__ static void push(Handle& fptr, const Msg& m) {
        if (lane_id() == 0) {
            (void)__ockl_call_host_function(fptr, msg_word0(m), m.tag, filler_word(m, 2),
                                            filler_word(m, 3), filler_word(m, 4),
                                            filler_word(m, 5), filler_word(m, 6));
        }
    }

    bool pop(int, Msg&) { return false; }

    // CLR's HostcallFunctionCall signature: void(uint64_t* output, uint64_t* input).
    static void host_function(uint64_t* output, uint64_t* input) {
        sink(msg_from_words(input[0], input[1]));
        output[0] = 0;
        output[1] = 0;
    }

    static std::string notes() {
        return "synchronous: lane-0 system-scope CAS pop of a free packet, 64B lane "
               "slot, CAS push to ready stack, doorbell signal + s_sendmsg interrupt; "
               "CLR listener runs the host function and releases the wave";
    }
};

}  // namespace qb
