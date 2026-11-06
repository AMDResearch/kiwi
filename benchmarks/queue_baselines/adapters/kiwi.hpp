// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file kiwi.hpp
 * @brief KIWI adapters: the SPSC and MPSC device-to-host queues, and
 *        KIWI-Invoke over the SPSC queue.
 */

#pragma once

#include "../qb_common.hpp"

#include <kiwi/invoke/invoke.hpp>
#include <kiwi/queue/queue.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qb {

namespace detail {

/// Both KIWI device-to-host queues: same ring format, push API and consumer.
template <typename Q, bool MultiProducer>
struct KiwiAdapter {
    static constexpr bool kMultiProducer = MultiProducer;
    static constexpr bool kFifo = true;
    static constexpr int kDescriptorBytes = 64;
    using Queue = Q;
    using Handle = typename Queue::DeviceHandle;

    KiwiAdapter(int num_queues, int capacity) {
        for (int q = 0; q < num_queues; ++q) {
            queues_.push_back(std::make_unique<Queue>(capacity));
            poll_.push_back(queues_.back()->host_poll_state());
        }
    }

    Handle handle(int queue) { return queues_[queue]->device_handle(); }

    // The pointers are already global; keep the whole handle wave-uniform.
    __device__ static void globalize(Handle& h) { h = qb::wave_uniform(h); }

    // One 16-byte message (lane 0: word0, lane 1: tag) straight from
    // registers; chunks 2-3 pad the 64-byte line.
    __device__ static void push(Handle& h, const Msg& m) {
        const uint64_t word = lane_id() == 0 ? msg_word0(m) : m.tag;
        while (!h.push_lanes(word, 2 * sizeof(uint64_t), 0)) {}
    }

    // Native proxy pattern: cached HostPollState check, then pop + refresh.
    bool pop(int queue, Msg& out) {
        auto& ps = poll_[queue];
        if (!ps.ready()) return false;
        uint64_t buf[2];
        bool ok = queues_[queue]->pop(buf);
        ps.refresh();
        if (!ok) return false;
        out = msg_from_words(buf[0], buf[1]);
        return true;
    }

    std::vector<std::unique_ptr<Queue>> queues_;
    std::vector<typename Queue::HostPollState> poll_;
};

}  // namespace detail

/// KIWI SPSC device-to-host queue: one producer wave per queue.
struct KiwiSpscAdapter : detail::KiwiAdapter<kiwi::queue::SPSCD2HQueue, false> {
    static constexpr const char* kName = "kiwi-spsc";
    using KiwiAdapter::KiwiAdapter;

    static std::string notes() {
        return "16B message in one 64B line of 16B chunks (8B payload + 8B epoch "
               "signal each), one 16B store per lane; producer/cached-consumer "
               "index in registers; batched consumer index; no atomics";
    }
};

/// KIWI MPSC device-to-host queue: any number of producer waves per queue,
/// one consumer thread.
struct KiwiMpscAdapter : detail::KiwiAdapter<kiwi::queue::MPSCD2HQueue, true> {
    static constexpr const char* kName = "kiwi-mpsc";
    using KiwiAdapter::KiwiAdapter;

    static std::string notes() {
        return "16B message in one 64B line of 16B chunks, one 16B store per lane; "
               "lane-0 agent-scope fetch-add line claim on a VRAM index; cached "
               "credit; single host consumer, batched consumer index";
    }
};

/// KIWI-Invoke over its default SPSC device-to-host queue: one endpoint per
/// wave. A push is a callback invocation with the 16-byte message as its two
/// arguments; a pop runs the callback on the proxy thread.
struct KiwiInvokeAdapter {
    static constexpr const char* kName = "kiwi-invoke";
    static constexpr bool kMultiProducer = false;
    static constexpr bool kFifo = true;
    static constexpr int kDescriptorBytes = 64;
    using Context = kiwi::invoke::Context<>;
    using Handle = Context::DeviceHandle;
    /// The adapter registers exactly one callback, so its id is 0.
    static constexpr uint8_t kCallback = 0;

    KiwiInvokeAdapter(int num_queues, int capacity) : ctx_(num_queues, capacity) {
        const uint8_t id = ctx_.register_callback<uint64_t, uint64_t>(
            std::function<void(uint64_t, uint64_t)>(
                [](uint64_t w0, uint64_t tag) { *sink_ = msg_from_words(w0, tag); }));
        if (id != kCallback) std::abort();
        for (int q = 0; q < num_queues; ++q) poll_.push_back(ctx_.raw_queue(q)->host_poll_state());
    }

    // The endpoint handle Context places in its DeviceContext. The harness
    // loads it per wave from VRAM and makes it wave-uniform (globalize), as
    // DeviceContext::get_device_handle() does.
    Handle handle(int queue) { return Handle{ctx_.raw_queue(queue)->device_handle()}; }

    __device__ static void globalize(Handle& h) { h = wave_uniform(h); }

    // invoke(id, word0, tag): the arguments are serialized in registers into
    // one 16-byte message (as kiwi-spsc's), the id rides in the metadata.
    __device__ static void push(Handle& h, const Msg& m) {
        while (!h.invoke(kCallback, msg_word0(m), m.tag)) {}
    }

    // KIWI-EP proxy pattern: cached HostPollState check inline, then
    // progress() of one callback (pop, registry lookup, argument
    // deserialization, call) only when a message is ready. progress() is
    // kept out of line so the empty-queue check stays inlined in the poll
    // loop, as in a proxy.
    bool pop(int queue, Msg& out) {
        if (!poll_[queue].ready()) return false;
        return progress_one(queue, out);
    }

    __attribute__((noinline)) bool progress_one(int queue, Msg& out) {
        sink_ = &out;
        const size_t ran = ctx_.progress(static_cast<size_t>(queue), 1);
        poll_[queue].refresh();
        return ran == 1;
    }

    static std::string notes() {
        return "KIWI-Invoke over the SPSC queue: invoke(cb, word0, tag), a 16B "
               "argument message in one 64B line of 16B chunks; host progress() "
               "pops it and runs the registered std::function callback";
    }

    static inline thread_local Msg* sink_ = nullptr;  ///< the consuming pop()'s output
    Context ctx_;
    std::vector<kiwi::queue::SPSCD2HQueue::HostPollState> poll_;
};

}  // namespace qb
