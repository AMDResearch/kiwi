// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file raw_store.hpp
 * @brief Lower bound, not a queue: one wave-wide 64B store per message.
 *
 * Lanes 0-7 write one 64-byte slot of an uncached host ring (the same
 * host_malloc_uc memory KIWI uses); lane 7 carries seq+1 as the ready flag.
 * There is no capacity check and no consumer index, so it is only valid
 * while the ring never wraps with a message unconsumed: latency mode (one
 * message outstanding) or push-cost with a ring that holds the whole run.
 * The harness rejects other modes (kUnbounded).
 */

#pragma once

#include "../qb_common.hpp"

#include <kiwi/queue/detail/host_memory.hpp>

#include <string>
#include <vector>

namespace qb {

struct RawStoreAdapter {
    static constexpr const char* kName = "raw-store";
    static constexpr bool kMultiProducer = false;
    static constexpr bool kFifo = true;
    static constexpr bool kUnbounded = true;
    static constexpr int kDescriptorBytes = 64;

    struct Handle {
        uint64_t* slots;  // capacity x 8 words, uncached host memory
        uint64_t seq;     // register
        uint64_t mask;    // capacity - 1
    };

    struct Ring {
        uint64_t* slots;
        uint64_t next = 0;  // host consumer position
    };

    RawStoreAdapter(int num_queues, int capacity) : mask_(static_cast<uint64_t>(capacity) - 1) {
        for (int q = 0; q < num_queues; ++q) {
            Ring r;
            r.slots = kiwi::queue::detail::host_malloc_uc<uint64_t>(
                static_cast<size_t>(capacity) * 8);
            for (size_t i = 0; i < static_cast<size_t>(capacity) * 8; ++i) r.slots[i] = 0;
            rings_.push_back(r);
        }
    }

    ~RawStoreAdapter() {
        for (auto& r : rings_) kiwi::queue::detail::host_free_uc(r.slots);
    }

    RawStoreAdapter(const RawStoreAdapter&) = delete;
    RawStoreAdapter& operator=(const RawStoreAdapter&) = delete;

    Handle handle(int queue) { return Handle{rings_[queue].slots, 0, mask_}; }

    __device__ static void globalize(Handle& h) { h.slots = as_global(h.slots); }

    __device__ static void push(Handle& h, const Msg& m) {
        const int lane = lane_id();
        const uint64_t base = lane == 0 ? msg_word0(m) : m.tag;
        uint64_t word = base ^ (lane >= 2 ? static_cast<uint64_t>(lane) << 56 : 0);
        word = lane == 7 ? h.seq + 1 : word;
        if (lane < 8) h.slots[(h.seq & h.mask) * 8 + lane] = word;
        ++h.seq;
    }

    bool pop(int queue, Msg& out) {
        Ring& r = rings_[queue];
        volatile uint64_t* slot = r.slots + (r.next & mask_) * 8;
        if (slot[7] != r.next + 1) return false;
        out = msg_from_words(slot[0], slot[1]);
        ++r.next;
        return true;
    }

    static std::string notes() {
        return "lower bound: one wave-wide 64B store to uncached host memory, seq in "
               "lane 7; no capacity check, no consumer index";
    }

    uint64_t mask_;
    std::vector<Ring> rings_;
};

}  // namespace qb
