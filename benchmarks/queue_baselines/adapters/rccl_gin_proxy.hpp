// Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Modifications Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions adapted from RCCL (https://github.com/ROCm/rocm-systems, commit
// 5909ad99, projects/rccl/src/gin/gin_host_proxy.cc and gin/gin_host.cc),
// licensed under the Apache License 2.0. Modified by AMD to run as a single-
// context queue-baseline adapter. See NOTICES.txt.

/**
 * @file rccl_gin_proxy.hpp
 * @brief RCCL GIN proxy-backend GFD queue adapter.
 *
 * Device side: the unmodified upstream RCCL GIN proxy backend,
 * ncclGinApi_Put<NCCL_NET_DEVICE_GIN_PROXY>::call (gin_proxy.h:528-542), which
 * builds the GFD in a thread-local descriptor (buildGfd) and posts it
 * (postGfd). It is what a device-API put on a proxy GIN context dispatches to.
 * The headers are fetched from ROCm/rocm-systems projects/rccl at commit
 * 5909ad99 (develop, 2026-10-16; SHA-256 checked) and prepared with RCCL's
 * own build steps (see CMakeLists.txt): hipify-perl, hip_compat.h as is,
 * nccl.h from nccl.h.in.
 *
 * Host side: the GIN proxy lives in librccl (gin/gin_host_proxy.cc), so its
 * allocation, progress pass, GFD poll and completion release are reproduced
 * here line for line at the same commit, using the upstream GFD types.
 * Citations are relative to projects/rccl/src/:
 *
 *   dev   include/nccl_device/gin/proxy/gin_proxy.h                    (gin_proxy.h)
 *   fmt   include/nccl_device/gin/proxy/gin_proxy_device_host_common.h (gfd.h)
 *   host  gin/gin_host_proxy.cc (host_proxy.cc), gin/gin_host.cc (gin_host.cc)
 *   alloc include/gdrwrap.h, init.cc
 *
 * One harness queue is one GIN context with nRanks = 1: upstream allocates
 * the ring, pis and cis separately per context (host_proxy.cc:578-610) and
 * has nContexts x nRanks rings, each multi-producer.
 *
 * Placement (host_proxy.cc:592, 600-601, 604-614; gdrwrap.h:398-413):
 *   ring  allocMemCPUAccessible(forceHost) -> hipHostMalloc with no flags
 *         (gdrwrap.h:406), which CLR allocates coherent
 *         => hipHostMalloc(hipHostMallocCoherent).
 *   pis   ncclCudaCalloc => hipMalloc + hipMemset 0.
 *   cis   allocMemCPUAccessible. RCCL_GDRCOPY_ENABLE defaults to 0
 *         (init.cc:246), so cis is pinned host memory (gdrwrap.h:406; the
 *         CU_MEMHOSTALLOC_WRITECOMBINED flag is ignored on HIP).
 *         => default: hipHostMalloc coherent. QB_RCCL_GIN_CIS=device selects
 *         the GDRCopy-equivalent path RCCL takes with RCCL_GDRCOPY_ENABLE=1 on
 *         gfx942/gfx950 (gdrwrap.h:170-200): hipDeviceMallocUncached VRAM, the
 *         flag RCCL builds with whenever HIP provides it (gdrwrap.h:186-187).
 *   ctx   ncclGinProxyGpuCtx_t array in device memory (ncclCudaCalloc +
 *         cudaMemcpy, host_proxy.cc:612-614); the device handle points at it
 *         and postGfd() loads queues/pis/cis/queueSize from it on every push.
 *         The fields this commit added (signalOffsets, lastIssuedGet,
 *         lastVisibleGet, flushesAllPutsOnAnySignal) serve signals, gets and
 *         flushes, which a plain put does not touch; they stay zero.
 *
 * Call arguments: the Msg rides in the put's srcOff (seq), srcWin (producer
 * + 1; put() posts a GFD only for a non-null source window, gin_proxy.h:320)
 * and dstOff (tag), all 63-bit GFD fields; size is 8 bytes. No signal, no
 * counter, no caller descriptor. givenRelease = thread_scope_system: put()'s
 * source-release fence (gin_proxy.h:277-286) orders a source buffer before
 * the NIC reads it, and the harness message has no source buffer.
 */

#pragma once

#include "../qb_common.hpp"

#define NCCL_GIN_PROXY_ENABLE 1
#include <nccl_device/gin/gin_device_api.h>

#include <immintrin.h>

#include <cstdlib>
#include <cstring>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

namespace qb {

struct RcclGinProxyAdapter {
    static constexpr const char* kName = "rccl-gin-proxy";
    static constexpr bool kMultiProducer = true;  // pi.fetch_add claim
    // Slots are claimed with fetch_add and the proxy consumes them in index
    // order, and a single wave's pushes claim increasing indices.
    static constexpr bool kFifo = true;
    static constexpr int kDescriptorBytes = sizeof(ncclGinProxyGfd_t);
    static_assert(sizeof(ncclGinProxyGfd_t) == 128, "GFD must be 128 bytes");
    static constexpr size_t kPutBytes = 8;

    // Device view: the context RCCL's device API hands the backend
    // (ncclGinCtx::handle points at the ncclGinProxyGpuCtx_t array).
    struct Handle {
        ncclGinProxyGpuCtx_t* ctx;
    };

    // ginProxyGfdState, host_proxy.cc:25-30.
    struct GfdState {
        ncclGinProxyOp_t op;
        uint16_t counterId;
        int done;
        void* request;
    };

    // ginProxyHostGpuCtx (host_proxy.cc:33-59) for one context, nRanks = 1,
    // plus the progress threads' reader-writer lock and write-pending flag
    // (ncclGinState::devCommRwMutex / writePending, gin_host.cc:47-56, 67-72).
    // Deviation (placement only): upstream ncclCallocs sis/cisShadow as tiny
    // separate arrays and shares one lock across every context of a
    // communicator; here each queue has its own consumer thread, so the
    // mutable host state and the (uncontended) lock are cache-line aligned per
    // queue. `ready` holds the GFDs of the current progress pass that the
    // harness has not popped yet.
    struct alignas(64) Ctx {
        ncclGinProxyGfd_t* queues = nullptr;  // host (forceHost)
        uint32_t* pis = nullptr;              // device
        uint32_t* cis = nullptr;              // host (default) or uncached VRAM
        uint32_t cisShadow = 0;
        uint32_t sis = 0;
        uint32_t queueSize = 0;
        GfdState* states = nullptr;  // queueSize entries
        ncclGinProxyGpuCtx_t* devGpuCtx = nullptr;
        std::atomic<bool> writePending{false};
        std::shared_timed_mutex devCommRwMutex;
        std::vector<Msg> ready;  // pollBatch entries
        int readyHead = 0;
        int readyCount = 0;
    };

    static bool cis_on_host() {
        const char* e = std::getenv("QB_RCCL_GIN_CIS");
        return !(e && std::strcmp(e, "device") == 0);
    }

    // NCCL_GIN_PROXY_POLL_BATCH (host_proxy.cc:21, default 32), sanitized as
    // in ncclGinProxyCreateContext (host_proxy.cc:523-524).
    static int poll_batch() {
        const char* e = std::getenv("NCCL_GIN_PROXY_POLL_BATCH");
        const long v = e ? std::strtol(e, nullptr, 10) : 32;
        return v < 1 ? 1 : static_cast<int>(v);
    }

    RcclGinProxyAdapter(int num_queues, int capacity)
        : cis_host_(cis_on_host()), poll_batch_(poll_batch()), num_queues_(num_queues),
          ctxs_(new Ctx[num_queues]) {
        for (int q = 0; q < num_queues_; ++q) {
            Ctx& c = ctxs_[q];
            c.queueSize = static_cast<uint32_t>(capacity);  // power of two
            c.states = new GfdState[capacity]();              // ncclCalloc, host_proxy.cc:586
            c.ready.resize(poll_batch_);
            // host_proxy.cc:592: pis = ncclCudaCalloc(nRanks).
            QB_HIP_CHECK(hipMalloc(&c.pis, sizeof(uint32_t)));
            QB_HIP_CHECK(hipMemset(c.pis, 0, sizeof(uint32_t)));
            // host_proxy.cc:600 + gdrwrap.h:406-408: pinned host ring, zeroed.
            QB_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&c.queues),
                                       static_cast<size_t>(capacity) * sizeof(ncclGinProxyGfd_t),
                                       hipHostMallocCoherent));
            std::memset(c.queues, 0, static_cast<size_t>(capacity) * sizeof(ncclGinProxyGfd_t));
            // host_proxy.cc:601: cis in pinned host memory (GDRCopy off), or
            // the GDRCopy-equivalent uncached VRAM (gdrwrap.h:186-187).
            if (cis_host_) {
                QB_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&c.cis),
                                           sizeof(uint32_t), hipHostMallocCoherent));
                *c.cis = 0;
            } else {
                QB_HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&c.cis),
                                                   sizeof(uint32_t), hipDeviceMallocUncached));
                QB_HIP_CHECK(hipMemset(c.cis, 0, sizeof(uint32_t)));
            }
            // host_proxy.cc:604-614: the device context.
            ncclGinProxyGpuCtx_t h{};
            h.nranks = 1;
            h.queueSize = c.queueSize;
            h.queues = c.queues;
            h.pis = c.pis;
            h.cis = c.cis;
            QB_HIP_CHECK(hipMalloc(&c.devGpuCtx, sizeof(h)));
            QB_HIP_CHECK(hipMemcpy(c.devGpuCtx, &h, sizeof(h), hipMemcpyHostToDevice));
        }
        QB_HIP_CHECK(hipDeviceSynchronize());
    }

    ~RcclGinProxyAdapter() {
        for (int q = 0; q < num_queues_; ++q) {
            Ctx& c = ctxs_[q];
            (void)hipFree(c.devGpuCtx);
            (void)hipFree(c.pis);
            (void)hipHostFree(c.queues);
            if (cis_host_) (void)hipHostFree(c.cis);
            else (void)hipFree(c.cis);
            delete[] c.states;
        }
    }

    RcclGinProxyAdapter(const RcclGinProxyAdapter&) = delete;
    RcclGinProxyAdapter& operator=(const RcclGinProxyAdapter&) = delete;

    Handle handle(int queue) { return Handle{ctxs_[queue].devGpuCtx}; }

    __device__ static void globalize(Handle& h) { h.ctx = as_global(h.ctx); }

    /// A wave-cooperative put on context 0 to peer 0, through RCCL's proxy
    /// backend; lane 0 posts the GFD (gin_proxy.h:119).
    __device__ static void push(Handle& h, const Msg& m) {
        ncclGinCtx ctx{};
        ctx.backend = NCCL_NET_DEVICE_GIN_PROXY;
        ctx.nRanks = 1;
        ctx.handle = h.ctx;
        ctx.contextId = 0;
        ncclGinSignalDescriptor signal{};
        signal.type = NCCL_GIN_SIGNAL_TYPE_NONE;
        ncclGinApi_Put<NCCL_NET_DEVICE_GIN_PROXY>::call(
            ctx, ncclCoopWarp(), /*peer=*/0, /*hasWins=*/true,
            /*dstWin=*/nullptr, /*dstOff=*/m.tag,
            /*srcWin=*/reinterpret_cast<ncclGinWindow_t>(static_cast<uintptr_t>(m.producer) + 1),
            /*srcOff=*/m.seq, kPutBytes, signal, ncclGinSignalInc, /*signalOpArg=*/0,
            /*hasCounter=*/false, /*counterId=*/0, /*hasDescriptor=*/false, /*descriptor=*/nullptr,
            /*required=*/cuda::thread_scope_device, /*given=*/cuda::thread_scope_system);
    }

    static Msg decode(const ncclGinProxyGfd_t& g) {
        return Msg{static_cast<uint32_t>(g.qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle - 1),
                   static_cast<uint32_t>(g.qword[ncclGinProxyGfdSrcOff].srcOff.srcOff),
                   g.qword[ncclGinProxyGfdDstOff].dstOff.dstOff};
    }

    /// Returns the next GFD of the current progress pass; when the pass is
    /// used up, runs the next one.
    bool pop(int queue, Msg& out) {
        Ctx& c = ctxs_[queue];
        if (c.readyHead == c.readyCount && !progress(c)) return false;
        out = c.ready[c.readyHead++];
        return true;
    }

    static std::string notes() {
        return std::string(
                   "unmodified RCCL ncclGinApi_Put proxy backend (buildGfd + postGfd): 128B GFD "
                   "in pinned host ring; lane-0 system-scope fetch_add on pi; ci load every push "
                   "(cis in ") +
               (cis_on_host() ? "pinned host memory, RCCL default"
                              : "uncached VRAM, RCCL GDRCopy-equivalent") +
               "); 8 x 16B system-scope (sc0 sc1) stores, no fence; host spins per-qword "
               "flags, zeroes the slot with nt stores + sfence, relaxed ci store per GFD";
    }

private:
    /// One progress-thread pass over this context's single peer
    /// (ncclGinProgress, gin_host.cc:60-91): skip the pass while a writer
    /// waits (gin_host.cc:67-70), take the reader lock (gin_host.cc:72), and
    /// run ncclGinProxyProgress (host_proxy.cc:684-705):
    /// proxyGinPollCompletions, then up to pollBatch GFDs, each polled
    /// (proxyGinPollGfd) and processed (proxyGinProcessGfd). The network
    /// plugin's rmaProgress() at the end of the pass is skipped, as there is
    /// no network. False if the pass found no GFD.
    ///
    /// The per-pass lock/unlock is kept as upstream does it. With
    /// QB_RCCL_GIN_CIS=device, its LOCK-prefixed RMWs also drain the x86
    /// write-combining buffer that holds the relaxed ci store to CPU-mapped
    /// device memory. The std::this_thread::yield() after each pass
    /// (gin_host.cc:89) is omitted: it is a scheduler courtesy of the outer
    /// loop, which the harness replaces for every queue.
    bool progress(Ctx& c) {
        c.readyHead = c.readyCount = 0;
        if (c.writePending.load()) return false;
        std::shared_lock<std::shared_timed_mutex> rlock(c.devCommRwMutex);
        poll_completions(c);
        for (int p = 0; p < poll_batch_; p++) {
            ncclGinProxyGfd_t gfd;
            GfdState* state = nullptr;
            if (!poll_gfd(c, &gfd, &state)) break;
            // proxyGinProcessGfd (host_proxy.cc:282-288) first peeks at the
            // peer's next slot to decide whether the plugin may aggregate
            // this request with the next one, then posts the network op;
            // ginBackend->test() later sets state->done. Here the GFD is
            // handled by copying it out, so it is done at consume time; its
            // ci writeback happens in the next pass's poll_completions,
            // because upstream orders completions before polling
            // (host_proxy.cc:689-694).
            (void)gfd_available(c);
            c.ready[c.readyCount++] = decode(gfd);
            state->done = 1;
        }
        return c.readyCount > 0;
    }

    /// proxyGinPollCompletions, host_proxy.cc:124-167 (nRanks = 1).
    static void poll_completions(Ctx& c) {
        for (uint32_t i = c.cisShadow; i < c.sis; i++) {
            const uint32_t idx = i & (c.queueSize - 1);
            GfdState* state = &c.states[idx];
            // host_proxy.cc:132-153: rmaBackend->test() and counter update
            // are skipped; done was set at consume time.
            // host_proxy.cc:155-160: in-order release, one relaxed store per GFD.
            if (state->done && i == c.cisShadow)
                __atomic_store_n(c.cis, ++c.cisShadow, __ATOMIC_RELAXED);
        }
    }

    /// isGfdAvailable, host_proxy.cc:187-200 (HIP branch): relaxed load of
    /// the header of slot sis, test the flag.
    static bool gfd_available(Ctx& c) {
        ncclGinProxyGfd_t* q = c.queues;  // + targetRank * queueSize, targetRank = 0
        const uint32_t idx = c.sis & (c.queueSize - 1);
        ncclGinProxyQword_t header;
        uint64_t* headerPtr = (uint64_t*)__builtin_assume_aligned(
            &q[idx].qword[ncclGinProxyGfdHeader].raw, alignof(uint64_t));
        header.raw = __atomic_load_n(headerPtr, __ATOMIC_RELAXED);
        return header.flag.v != 0;
    }

    /// proxyGinPollGfd, host_proxy.cc:202-267 (HIP branches).
    static bool poll_gfd(Ctx& c, ncclGinProxyGfd_t* gfd, GfdState** state) {
        // host_proxy.cc:204-206: isGfdAvailable.
        if (!gfd_available(c)) return false;
        ncclGinProxyGfd_t* q = c.queues;
        const uint32_t idx = c.sis & (c.queueSize - 1);
        ncclGinProxyQword_t qword;
        // host_proxy.cc:213: copy the header (a second, plain read).
        gfd->qword[ncclGinProxyGfdHeader] = q[idx].qword[ncclGinProxyGfdHeader];
        // host_proxy.cc:215-227: spin on and copy each remaining qword.
        for (int k = 1; k < ncclGinProxyGfdQwords; k++) {
            uint64_t* qwordPtr =
                (uint64_t*)__builtin_assume_aligned(&q[idx].qword[k].raw, alignof(uint64_t));
            do {
                qword.raw = __atomic_load_n(qwordPtr, __ATOMIC_RELAXED);
            } while (qword.flag.v == 0);
            gfd->qword[k] = qword;
        }
        // host_proxy.cc:230-239: reset the slot with 16 nontemporal stores of
        // 0, then wc_store_fence() (_mm_sfence on x86, gdrwrap.h:40-42) so the
        // zeros land before the ci advance.
        for (int k = 0; k < ncclGinProxyGfdQwords; k++)
            __builtin_nontemporal_store((uint64_t)0, &q[idx].qword[k].raw);
        _mm_sfence();
        // host_proxy.cc:246-252: record the state for completion tracking.
        *state = &c.states[idx];
        (*state)->op = (ncclGinProxyOp_t)gfd->qword[ncclGinProxyGfdHeaderExt].headerExt.op;
        (*state)->counterId = gfd->qword[ncclGinProxyGfdCompletion].completion.counterId;
        (*state)->done = 0;
        (*state)->request = nullptr;
        c.sis++;  // host_proxy.cc:264
        return true;
    }

    bool cis_host_;
    int poll_batch_;
    int num_queues_;
    std::unique_ptr<Ctx[]> ctxs_;
};

}  // namespace qb
