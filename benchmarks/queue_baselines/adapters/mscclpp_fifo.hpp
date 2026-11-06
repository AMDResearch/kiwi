// Copyright (c) Microsoft Corporation.
// Modifications Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Portions adapted from MSCCL++ (https://github.com/microsoft/mscclpp, commit
// 07ce0f3e), src/core/fifo.cc, licensed under the MIT license. Modified by AMD
// to run as a HIP queue-baseline adapter. See NOTICES.txt.

/**
 * @file mscclpp_fifo.hpp
 * @brief MSCCL++ PortChannel proxy FIFO adapter.
 *
 * Device side: the unmodified upstream mscclpp::FifoDeviceHandle::push()
 * (include/mscclpp/fifo_device.hpp at the pinned commit), called by lane 0
 * as PortChannel device methods do.
 *
 * Host side: mscclpp::Fifo lives in libmscclpp, so its allocation, poll() and
 * pop() are reproduced here line for line from src/core/fifo.cc at the same
 * commit (allocation flags, acquire/relaxed loads, release tail store).
 * ProxyService calls poll(), handles the trigger, then pop(); the handoff
 * benchmark handles the trigger by copying it out.
 */

#pragma once

#include "../qb_common.hpp"

#include <mscclpp/fifo_device.hpp>

#include <cstring>
#include <string>
#include <vector>

namespace qb {

struct MscclppFifoAdapter {
    static constexpr const char* kName = "mscclpp-fifo";
    static constexpr bool kMultiProducer = true;
    static constexpr bool kFifo = true;
    static constexpr int kDescriptorBytes = sizeof(mscclpp::ProxyTrigger);
    using Handle = mscclpp::FifoDeviceHandle;

    struct Fifo {
        mscclpp::ProxyTrigger* triggers;  // host, hipHostMallocMapped (fifo.cc)
        uint64_t* head;                   // device
        uint64_t* tail;                   // host, hipHostMallocMapped
        uint64_t* tail_cache;             // device
        int size;
        uint64_t size_mask;
        uint64_t size_shift;
    };

    MscclppFifoAdapter(int num_queues, int capacity) {
        for (int q = 0; q < num_queues; ++q) {
            Fifo f{};
            // detail::gpuCallocHostUnique -> cudaHostAlloc(cudaHostAllocMapped)
            // -> hipHostMalloc(hipHostMallocMapped), then memset 0.
            QB_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&f.triggers),
                                       capacity * sizeof(mscclpp::ProxyTrigger),
                                       hipHostMallocMapped));
            std::memset(f.triggers, 0, capacity * sizeof(mscclpp::ProxyTrigger));
            QB_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&f.tail),
                                       sizeof(uint64_t), hipHostMallocMapped));
            *f.tail = 0;
            // detail::gpuCallocUnique -> cudaMalloc + memset 0.
            QB_HIP_CHECK(hipMalloc(&f.head, sizeof(uint64_t)));
            QB_HIP_CHECK(hipMemset(f.head, 0, sizeof(uint64_t)));
            QB_HIP_CHECK(hipMalloc(&f.tail_cache, sizeof(uint64_t)));
            QB_HIP_CHECK(hipMemset(f.tail_cache, 0, sizeof(uint64_t)));
            f.size = capacity;
            f.size_mask = static_cast<uint64_t>(capacity) - 1;
            f.size_shift = static_cast<uint64_t>(__builtin_ctz(capacity));
            fifos_.push_back(f);
        }
    }

    ~MscclppFifoAdapter() {
        for (auto& f : fifos_) {
            (void)hipHostFree(f.triggers);
            (void)hipHostFree(f.tail);
            (void)hipFree(f.head);
            (void)hipFree(f.tail_cache);
        }
    }

    MscclppFifoAdapter(const MscclppFifoAdapter&) = delete;
    MscclppFifoAdapter& operator=(const MscclppFifoAdapter&) = delete;

    // Fifo::deviceHandle()
    Handle handle(int queue) {
        const Fifo& f = fifos_[queue];
        Handle h;
        h.triggers = f.triggers;
        h.head = f.head;
        h.tail = f.tail;
        h.tailCache = f.tail_cache;
        h.size = f.size;
        h.sizeMask = f.size_mask;
        h.sizeShift = f.size_shift;
        return h;
    }

    __device__ static void globalize(Handle& h) {
        h.triggers = as_global(h.triggers);
        h.head = as_global(h.head);
        h.tail = as_global(h.tail);
        h.tailCache = as_global(h.tailCache);
    }

    __device__ static void push(Handle& h, const Msg& m) {
        if (lane_id() == 0) {
            mscclpp::ProxyTrigger t;
            t.fst = msg_word0(m);
            t.snd = m.tag;  // bit 63 clear: reserved for the commit bit
            h.push(t, /*maxSpinCount=*/-1);
        }
    }

    bool pop(int queue, Msg& out) {
        Fifo& f = fifos_[queue];
        // Fifo::poll()
        const uint64_t cur_tail = *f.tail;
        mscclpp::ProxyTrigger* ptr = &f.triggers[cur_tail & f.size_mask];
        const uint64_t expected_parity = ((cur_tail >> f.size_shift) & 1ULL) ^ 1ULL;
        mscclpp::ProxyTrigger candidate;
        candidate.snd = __atomic_load_n(&ptr->snd, __ATOMIC_ACQUIRE);
        if (candidate.fields.reserved != expected_parity) return false;
        candidate.fields.reserved = 0;
        candidate.fst = __atomic_load_n(&ptr->fst, __ATOMIC_RELAXED);
        out = msg_from_words(candidate.fst, candidate.snd);
        // Fifo::pop()
        __atomic_store_n(f.tail, *f.tail + 1, __ATOMIC_RELEASE);
        return true;
    }

    static std::string notes() {
        return "16B trigger in mapped host memory; lane-0 agent-scope fetch-add on "
               "device head; fst store + system release store of snd (lap-parity "
               "commit bit); host releases tail per pop";
    }

    std::vector<Fifo> fifos_;
};

}  // namespace qb
