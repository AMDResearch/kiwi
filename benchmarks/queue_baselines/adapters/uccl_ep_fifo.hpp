// Copyright (c) Microsoft Corporation.
// Modifications Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// Portions adapted from UCCL (https://github.com/uccl-project/uccl, commit
// 3ec055f3): ep/src/fifo.cpp (MSCCL++ FIFO, Copyright (c) Microsoft
// Corporation, MIT license) and ep/src/proxy.cpp (Apache License 2.0).
// Modified by AMD to run as a HIP queue-baseline adapter. See NOTICES.txt.

/**
 * @file uccl_ep_fifo.hpp
 * @brief UCCL-EP GPU->CPU command FIFO adapter (default USE_MSCCLPP_FIFO_BACKEND).
 *
 * UCCL-EP (uccl-project/uccl @ 3ec055f3, main 2026-10-15) vendors an intermediate MSCCL++ FIFO
 * as ep/include/fifo_device.hpp + fifo_util.hpp and selects it by default
 * (ep/include/common.hpp:34). Line references below are to that commit.
 *
 * Device side: the unmodified vendored FifoDeviceHandle::push(), called by
 * lane 0 with a TransferCmd-packed trigger, exactly as
 * nvshmemi_ibgda_put_nbi_warp() -> D2HHandle::atomic_set_and_commit() does.
 * The headers are fetched at the pinned commit (SHA-256 checked) and compiled
 * through a hipify-equivalent include shim (adapters/uccl_ep_fifo_shim).
 *
 * Host side: mscclpp::Fifo lives in ep/src/fifo.cpp, so its allocation,
 * poll() and pop() are reproduced line for line, together with the proxy's
 * consume step (Proxy::post_gpu_command, ep/src/proxy.cpp:849-898).
 */

#pragma once

#include "../qb_common.hpp"

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// The vendored headers declare namespace mscclpp and reuse the include guard
// and macro names of upstream MSCCL++ (used by adapters/mscclpp_fifo.hpp in
// the same TU). Compile them under a private namespace name and give them
// their own macro definitions; restore everything afterwards. Preprocessor
// only: the upstream bytes are unchanged.
#pragma push_macro("mscclpp")
#pragma push_macro("MSCCLPP_FIFO_DEVICE_HPP_")
#pragma push_macro("MSCCLPP_DEVICE_COMPILE")
#pragma push_macro("MSCCLPP_DEVICE_HIP")
#pragma push_macro("MSCCLPP_INLINE")
#pragma push_macro("MSCCLPP_DEVICE_INLINE")
#pragma push_macro("MSCCLPP_HOST_DEVICE_INLINE")
#pragma push_macro("MSCCLPP_ASSERT_DEVICE")
#pragma push_macro("POLL_MAYBE_JAILBREAK")
#pragma push_macro("MSCCLPP_CUDATHROW")
#pragma push_macro("MSCCLPP_CUTHROW")
#undef MSCCLPP_FIFO_DEVICE_HPP_
#undef MSCCLPP_DEVICE_COMPILE
#undef MSCCLPP_DEVICE_HIP
#undef MSCCLPP_INLINE
#undef MSCCLPP_DEVICE_INLINE
#undef MSCCLPP_HOST_DEVICE_INLINE
#undef MSCCLPP_ASSERT_DEVICE
#undef POLL_MAYBE_JAILBREAK
#undef MSCCLPP_CUDATHROW
#undef MSCCLPP_CUTHROW
#define mscclpp uccl_ep_mscclpp
#include <uccl_ep/fifo_device.hpp>
#pragma pop_macro("MSCCLPP_CUTHROW")
#pragma pop_macro("MSCCLPP_CUDATHROW")
#pragma pop_macro("POLL_MAYBE_JAILBREAK")
#pragma pop_macro("MSCCLPP_ASSERT_DEVICE")
#pragma pop_macro("MSCCLPP_HOST_DEVICE_INLINE")
#pragma pop_macro("MSCCLPP_DEVICE_INLINE")
#pragma pop_macro("MSCCLPP_INLINE")
#pragma pop_macro("MSCCLPP_DEVICE_HIP")
#pragma pop_macro("MSCCLPP_DEVICE_COMPILE")
#pragma pop_macro("MSCCLPP_FIFO_DEVICE_HPP_")
#pragma pop_macro("mscclpp")

namespace qb {

struct UcclEpFifoAdapter {
    static constexpr const char* kName = "uccl-ep-fifo";
    static constexpr bool kMultiProducer = true;  // many warps share a FIFO
    static constexpr bool kFifo = true;
    using Trigger = uccl_ep_mscclpp::ProxyTrigger;
    using Handle = uccl_ep_mscclpp::FifoDeviceHandle;
    // push() stores fst and snd (16B). The slot is 32B: in this version the
    // zero-width bit-field before `reserved` (fifo_device.hpp:45-48) pushes
    // `reserved` into a third word, so sizeof(ProxyTrigger) == 32.
    static constexpr int kDescriptorBytes = 2 * sizeof(uint64_t);
    static_assert(sizeof(Trigger) == 32, "UCCL-EP ProxyTrigger is 32B at 3ec055f3");

    // CmdType::WRITE (ring_buffer.cuh:27) via make_cmd_type(WRITE, false, 0)
    // (ring_buffer.cuh:37-43). Non-zero: the proxy treats fst == 0 as empty.
    static constexpr uint64_t kCmdWrite = 1;

    // Fifo::Impl (fifo.cpp:13-26).
    struct Fifo {
        uccl_ep_mscclpp::detail::UniqueGpuHostPtr<Trigger> triggers;
        uccl_ep_mscclpp::detail::UniqueGpuPtr<uint64_t> head;
        uccl_ep_mscclpp::detail::UniqueGpuHostPtr<uint64_t> tail;
        uccl_ep_mscclpp::detail::UniqueGpuPtr<uint64_t> tailCache;
        int const size;

        explicit Fifo(int size)
            // gpuCallocHostUnique: cudaHostAlloc(cudaHostAllocMapped) + memset
            // (fifo_util.hpp:326-331,407-412), hipified to
            // hipHostMalloc(hipHostMallocMapped). gpuCallocUnique: cudaMalloc +
            // cudaMemset (fifo_util.hpp:319-324,401-405).
            : triggers(uccl_ep_mscclpp::detail::gpuCallocHostUnique<Trigger>(size)),
              head(uccl_ep_mscclpp::detail::gpuCallocUnique<uint64_t>()),
              tail(uccl_ep_mscclpp::detail::gpuCallocHostUnique<uint64_t>()),
              tailCache(uccl_ep_mscclpp::detail::gpuCallocUnique<uint64_t>()),
              size(size) {}
    };

    UcclEpFifoAdapter(int num_queues, int capacity) {
        // Fifo::Fifo (fifo.cpp:28-44): prefer the GPU's NUMA node for the
        // calling thread's allocations. libnuma's set_mempolicy() is this
        // syscall; the policy stays in effect, as upstream.
        int device = 0;
        QB_HIP_CHECK(hipGetDevice(&device));
        const int numa_node = uccl_ep_mscclpp::getDeviceNumaNode(device);
        if (numa_node >= 0) {
            unsigned long nodemask = 1UL << numa_node;
            if (syscall(SYS_set_mempolicy, MPOL_PREFERRED, &nodemask,
                        8 * sizeof(nodemask)) != 0)
                std::fprintf(stderr, "Warning: set_mempolicy failed: %s\n",
                             std::strerror(errno));
        }
        // UCCL-EP: kChannelPerProxy = 8 FIFOs per proxy thread, 4 threads,
        // kQueueSize = 2048 slots each (common.hpp:59,68-69;
        // uccl_proxy.cpp:32-44). Here the harness chooses count and size.
        for (int q = 0; q < num_queues; ++q)
            fifos_.push_back(std::make_unique<Fifo>(capacity));
    }

    UcclEpFifoAdapter(const UcclEpFifoAdapter&) = delete;
    UcclEpFifoAdapter& operator=(const UcclEpFifoAdapter&) = delete;

    // Fifo::deviceHandle() (fifo.cpp:65-73).
    Handle handle(int queue) {
        const Fifo& f = *fifos_[queue];
        Handle h;
        h.triggers = f.triggers.get();
        h.head = f.head.get();
        h.tail = f.tail.get();
        h.tailCache = f.tailCache.get();
        h.size = f.size;
        return h;
    }

    __device__ static void globalize(Handle& h) {
        h.triggers = as_global(h.triggers);
        h.head = as_global(h.head);
        h.tail = as_global(h.tail);
        h.tailCache = as_global(h.tailCache);
    }

    // Msg in TransferCmd fields (d2h_queue_device.cuh:14-19):
    //   cmd_type = WRITE, dst_rank = filler byte, bytes_and_val = seq,
    //   expert_idx = producer (16 bits), req_rptr | req_lptr << 32 = tag.
    __host__ __device__ static uint8_t dst_rank_filler(const Msg& m) {
        return static_cast<uint8_t>(filler_word(m, 2));
    }

    __device__ static void push(Handle& h, const Msg& m) {
        // nvshmemi_ibgda_put_nbi_warp: "if (lane_id != 0) return;"
        // (uccl_ibgda.cuh:34); UCCL does not warp-cooperate.
        if (lane_id() == 0) {
            // pack_transfer_cmd (d2h_queue_device.cuh:21-44).
            uint64_t fst = 0;
            uint64_t snd = 0;
            fst |= (kCmdWrite & 0xFFull);
            fst |= ((static_cast<uint64_t>(dst_rank_filler(m)) & 0xFFull) << 8);
            fst |= ((static_cast<uint64_t>(m.seq) & 0xFFFFFFFFull) << 16);
            fst |= ((static_cast<uint64_t>(m.producer) & 0xFFFFull) << 48);
            snd |= ((m.tag & 0xFFFFFFFFull) << 0);
            snd |= (((m.tag >> 32) & 0xFFFFFFFFull) << 32);
            // D2HHandle::atomic_set_and_commit (d2h_queue_device.cuh:77-82).
            Trigger trig;
            trig.fst = fst;
            trig.snd = snd;
            h.push(trig, /*maxSpinCount=*/-1);
        }
    }

    bool pop(int queue, Msg& out) {
        Fifo& f = *fifos_[queue];
        // Fifo::poll() (fifo.cpp:48-55).
        Trigger trigger;
        Trigger* ptr = &f.triggers.get()[*(f.tail) % f.size];
        trigger.fst = uccl_ep_mscclpp::atomicLoad(&(ptr->fst),
                                                  uccl_ep_mscclpp::memoryOrderAcquire);
        trigger.snd = ptr->snd;
        // Proxy::post_gpu_command (proxy.cpp:855): empty slot.
        if (trigger.fst == 0) return false;
        // decode_packed_transfer_cmd (d2h_queue_host.hpp:17-44).
        const uint64_t fst = trigger.fst;
        const uint64_t snd = trigger.snd ^ (1ULL << 63);  // revert push's flip
        out.producer = static_cast<uint32_t>((fst >> 48) & 0xFFFFull);
        out.seq = static_cast<uint32_t>((fst >> 16) & 0xFFFFFFFFull);
        out.tag = snd;
        // Fold the other fields into validation: a bad cmd_type or dst_rank
        // corrupts the tag, which the harness rejects.
        if ((fst & 0xFFull) != kCmdWrite ||
            static_cast<uint8_t>(fst >> 8) != dst_rank_filler(out))
            out.tag = ~out.tag;
        // Fifo::pop() (fifo.cpp:57-61), called right after the trigger is
        // consumed for WRITE/ATOMIC commands (proxy.cpp:895-898).
        uint64_t curTail = *(f.tail);
        f.triggers.get()[curTail % f.size].fst = 0;
        uccl_ep_mscclpp::atomicStore(f.tail.get(), curTail + 1,
                                     uccl_ep_mscclpp::memoryOrderRelease);
        return true;
    }

    static std::string notes() {
        return "16B fst/snd in a 32B slot, mapped host memory; lane-0 system-scope "
               "fetch-add on device head; 64-bit % size; snd relaxed + fst system "
               "release store (fst != 0 = ready); host clears fst, releases tail per pop";
    }

    std::vector<std::unique_ptr<Fifo>> fifos_;
};

}  // namespace qb
