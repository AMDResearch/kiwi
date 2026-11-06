// Copyright (c) 2014, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Modifications Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The U.S. Department of Energy funded the development of this software
// under subcontract 7078610 with Lawrence Berkeley National Laboratory.
//
// Portions adapted from NVSHMEM (https://github.com/NVIDIA/nvshmem, commit
// cffd2a5), proxy channel sources cited below, licensed under the Apache
// License 2.0. Modified by AMD: ported to HIP as a queue-baseline adapter.
// See NOTICES.txt.

/**
 * @file nvshmem_proxy.hpp
 * @brief NVSHMEM reverse-offload proxy channel adapter (HIP re-implementation).
 *
 * NVSHMEM is CUDA-only, so its proxy channel protocol is re-implemented here
 * from the upstream sources at github.com/NVIDIA/nvshmem commit cffd2a5 (devel,
 * 2026-10-03). Every step cites the upstream file:line it reproduces. Paths
 * are relative to the NVSHMEM repository:
 *   DEV  = src/include/non_abi/device/pt-to-pt/proxy_device.cuh
 *   WAIT = src/include/non_abi/device/wait/nvshmemi_wait_until_apis.cuh
 *   XFER = src/include/non_abi/device/pt-to-pt/transfer_device.cuh.in
 *   CHAN = src/include/non_abi/device/transport/nvshmem_proxy_channel.h
 *   TC   = src/include/non_abi/device/transport/transport_constants.h
 *   PROX = src/host/proxy/proxy.cpp
 *   PH   = src/host/proxy/proxy_host.h
 *
 * Producer: nvshmemx_putmem_nbi_warp() over a proxy transport (IBRC, UCX,
 * libfabric) -> nvshmemi_transfer_rma_nbi (XFER:266-281; lane 0 of the group
 * only) -> nvshmemi_proxy_rma_nbi -> transfer_dma<false, NONE> (DEV:287-372):
 * one 32B put DMA request written as four 8B entries.
 *
 * Consumer: the proxy thread's progress_channels() (PROX:1428-1547) and
 * process_channel_dma() (PROX:515-613), and proxy_update_processed()
 * (PROX:159-167), which writes `complete` back only every >= 1024 bytes.
 *
 * Mapping to the harness:
 *  - One harness queue = one NVSHMEM channel (CHANNEL_COUNT is 1 per PE,
 *    PH:15), shared by every producer on the PE.
 *  - Ring bytes = capacity * PROXY_DMA_REQ_BYTES (one 32B put request per
 *    harness slot). Upstream uses a fixed 4 MiB ring (TC:17-18,
 *    CHANNEL_BUF_SIZE_LOG 22 = 131072 put requests).
 *  - The Msg rides in the native put fields: laddr (64 bit) = msg_word0,
 *    size (48 bit) = tag[47:0], pe (16 bit) = tag[63:48]. roffset (40 bit)
 *    and qp_index (32 bit) carry filler_word(). op = NVSHMEMI_OP_PUT and
 *    group size = PROXY_GROUP_SIZE_SINGLE, so the host dispatches it as a PUT.
 */

#pragma once

#include "../qb_common.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace qb {

namespace nvshmem_detail {

// TC:17-19, 40-43; non_abi/device/transport/nvshmem_common_transport.h:21.
constexpr uint64_t kChannelEntryBytes = 8;     // CHANNEL_ENTRY_BYTES
constexpr uint64_t kDmaReqBytes = 32;          // PROXY_DMA_REQ_BYTES
constexpr uint64_t kGroupSizeSingle = 0x01;    // PROXY_GROUP_SIZE_SINGLE
constexpr uint8_t kGroupRegion = 0x80;         // PROXY_GROUP_REGION (TC:46)
constexpr uint64_t kOpPut = 1;                 // NVSHMEMI_OP_PUT
constexpr uint64_t kSyncBytes = 1024;          // proxy_update_processed (PROX:162)

// Request entry layouts (CHAN:26-32, 63-91). Byte 0 of every entry is the
// flag; bit 0 is the lap parity, bit 4 a continuation mark (inline copies).
struct __attribute__((packed)) BaseRequest {
    volatile uint8_t flag;
    uint8_t groupsize;
    uint8_t op;
    uint8_t roffset_low;
    uint32_t roffset_high;
};
struct __attribute__((packed)) PutDmaRequest0 {
    volatile uint8_t flag;
    uint8_t laddr_2;
    uint16_t laddr_3;
    uint32_t laddr_high;
};
struct __attribute__((packed)) PutDmaRequest1 {
    volatile uint8_t flag;
    uint8_t laddr_low;
    uint16_t size_low;
    uint32_t size_high;
};
struct __attribute__((packed)) PutDmaRequest2 {
    volatile uint8_t flag;
    uint8_t resv;
    uint16_t pe;
    uint32_t qp_index;
};
static_assert(sizeof(BaseRequest) == 8 && sizeof(PutDmaRequest0) == 8 &&
                  sizeof(PutDmaRequest1) == 8 && sizeof(PutDmaRequest2) == 8,
              "request entries are 8 bytes");

// CUDA `volatile` accesses in the device path. NVCC emits ld/st.volatile,
// which the PTX memory model treats as relaxed system-scope accesses: no
// ordering between different addresses. HIP `volatile` is stronger on gfx942:
// the AMDGPU memory model adds `s_waitcnt vmcnt(0)` after every volatile
// global access so that volatile accesses are never reordered in hardware,
// which would turn each of the four entry stores into a round trip to host
// memory. The faithful mapping is therefore a relaxed system-scope atomic
// (global_* sc0 sc1, no wait). Define QB_NVSHMEM_HIP_VOLATILE to use literal
// HIP volatile instead (sensitivity check only).
__device__ inline uint64_t ld_volatile(const uint64_t* p) {
#ifdef QB_NVSHMEM_HIP_VOLATILE
    return *reinterpret_cast<const volatile uint64_t*>(p);
#else
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}
__device__ inline void st_volatile(uint64_t* p, uint64_t v) {
#ifdef QB_NVSHMEM_HIP_VOLATILE
    *reinterpret_cast<volatile uint64_t*>(p) = v;
#else
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}

}  // namespace nvshmem_detail

struct NvshmemProxyAdapter {
    static constexpr const char* kName = "nvshmem-proxy";
    // Every thread of every kernel on the PE reserves from the one channel.
    static constexpr bool kMultiProducer = true;
    // One producer's reservations increase monotonically and the proxy
    // consumes strictly in counter order, so per-producer order holds.
    static constexpr bool kFifo = true;
    static constexpr int kDescriptorBytes = static_cast<int>(nvshmem_detail::kDmaReqBytes);

    // Subset of nvshmemi_device_host_state_t read by transfer_dma (upstream
    // keeps it in __constant__ nvshmemi_device_state_d; PROX:89-90, 120-122, 148).
    struct Handle {
        uint8_t* buf;              // proxy_channels_buf: pinned host ring (device ptr)
        uint64_t* issue;           // proxy_channels_issue: device memory
        uint64_t* complete;        // proxy_channels_complete: pinned host (device ptr)
        uint64_t* complete_local;  // proxy_channels_complete_local_ptr: device memory
        uint64_t buf_size;         // proxy_channel_buf_size (bytes, power of two)
        uint64_t buf_logsize;      // proxy_channel_buf_logsize
    };

    // proxy_channel_t (PH:27-39) fields used by the channel path, plus
    // proxy_state_t's bufsize/log (PH:49-50). One consumer thread per channel;
    // cache-line aligned so channels polled by different threads don't share.
    struct alignas(64) Channel {
        char* buf = nullptr;         // host view of the ring
        uint64_t* issue = nullptr;   // device
        uint64_t* complete = nullptr;  // host view
        uint64_t* complete_local = nullptr;  // device
        uint64_t processed = 0;
        uint64_t last_sync = 0;
        uint64_t bufsize = 0;
        uint64_t bufsize_log = 0;
        Handle handle{};
    };

    NvshmemProxyAdapter(int num_queues, int capacity) : chans_(num_queues) {
        const uint64_t bufsize = static_cast<uint64_t>(capacity) * nvshmem_detail::kDmaReqBytes;
        // `complete` is written back only every >= 1024 bytes (PROX:162), so a
        // ring smaller than that could never be released. Upstream is 4 MiB.
        if (bufsize < nvshmem_detail::kSyncBytes)
            throw std::runtime_error("nvshmem-proxy: capacity must be >= 32 (1 KiB ring)");
        for (Channel& ch : chans_) {
            ch.bufsize = bufsize;
            ch.bufsize_log = static_cast<uint64_t>(__builtin_ctzll(bufsize));
            // nvshmemi_proxy_create_channels (PROX:281-311).
            // cudaMallocHost(&buf, size, 0) -> hipHostMalloc(..., hipHostMallocDefault);
            // CLR maps flags == 0 to fine-grained coherent host memory
            // (CL_MEM_SVM_ATOMICS), the same as hipHostMallocMapped.
            // Upstream appends per-slot region metadata after the ring
            // (PROX:283-286); it is only touched by region requests
            // (CHECK_REGION), which this path never issues, so it is omitted.
            QB_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ch.buf), bufsize,
                                       hipHostMallocDefault));
            std::memset(ch.buf, 0, bufsize);  // PROX:288
            // issue: cudaMalloc + memset 0, never touched by the CPU (PROX:290-293).
            QB_HIP_CHECK(hipMalloc(&ch.issue, sizeof(uint64_t)));
            QB_HIP_CHECK(hipMemset(ch.issue, 0, sizeof(uint64_t)));
            // complete: cudaMallocHost(.., 0), CPU writes, GPU reads (PROX:295-306).
            QB_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ch.complete),
                                       sizeof(uint64_t), hipHostMallocDefault));
            *ch.complete = 0;
            // complete_local: cudaMalloc + memset 0 (PROX:134-136).
            QB_HIP_CHECK(hipMalloc(&ch.complete_local, sizeof(uint64_t)));
            QB_HIP_CHECK(hipMemset(ch.complete_local, 0, sizeof(uint64_t)));

            // nvshmemi_proxy_setup_device_channels (PROX:86-148).
            Handle& h = ch.handle;
            QB_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&h.buf), ch.buf, 0));
            QB_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&h.complete),
                                                 ch.complete, 0));
            h.issue = ch.issue;
            h.complete_local = ch.complete_local;
            h.buf_size = ch.bufsize;
            h.buf_logsize = ch.bufsize_log;
        }
    }

    ~NvshmemProxyAdapter() {
        for (Channel& ch : chans_) {
            (void)hipHostFree(ch.buf);
            (void)hipHostFree(ch.complete);
            (void)hipFree(ch.issue);
            (void)hipFree(ch.complete_local);
        }
    }

    NvshmemProxyAdapter(const NvshmemProxyAdapter&) = delete;
    NvshmemProxyAdapter& operator=(const NvshmemProxyAdapter&) = delete;

    Handle handle(int queue) { return chans_[queue].handle; }

    __device__ static void globalize(Handle& h) {
        h.buf = as_global(h.buf);
        h.issue = as_global(h.issue);
        h.complete = as_global(h.complete);
        h.complete_local = as_global(h.complete_local);
    }

    // ---------------------------------------------------------------------
    // Device: transfer_dma (DEV:287-372)
    // ---------------------------------------------------------------------

    // check_channel_availability (DEV:62-77) with the inlined
    // nvshmemi_wait_until_greater_than_equals_add (WAIT:121-133; the timeout
    // check is compiled only with NVSHMEM_TIMEOUT_DEVICE_POLLING).
    __device__ static void check_channel_availability(const Handle& h, uint64_t tail_idx) {
        using nvshmem_detail::ld_volatile;
        const uint64_t complete = ld_volatile(h.complete_local);  // DEV:65
        if ((complete + h.buf_size - 1) < tail_idx) {             // DEV:66
            uint64_t valataddr;
            do {
                valataddr = ld_volatile(h.complete);
            } while (valataddr + (h.buf_size - 1) < tail_idx);
            // CUDA atomicMax(ull*) -> HIP atomicMax: relaxed, agent scope.
            atomicMax(reinterpret_cast<unsigned long long*>(h.complete_local),
                      static_cast<unsigned long long>(*h.complete));
            __threadfence_system();  // DEV:74
        }
    }

    __device__ static void transfer_dma(const Handle& h, uint64_t roffset, uint64_t laddr,
                                        uint64_t size_u64, uint16_t pe_u16, uint32_t qp_index) {
        using nvshmem_detail::st_volatile;
        constexpr uint64_t size = nvshmem_detail::kDmaReqBytes;
        uint8_t* buf_ptr = h.buf;
        const uint64_t mask = h.buf_size - 1;  // CHANNEL_BUF_SIZE - 1 upstream
        const uint64_t mask_lowest_byte = 0xFFFFFFFFFFFFFF00u;
        const uint64_t mask_upper_7_bytes = 0x00000000000000FFu;
        uint64_t* req;

        __threadfence();  // DEV:302 (CUDA device-scope fence -> HIP agent seq_cst fence)

        // nvshmemi_proxy_channel_reserve (DEV:55-60): CUDA atomicAdd(ull*) is
        // relaxed, device scope -> HIP atomicAdd, relaxed, agent scope. The
        // counter counts bytes and never wraps.
        uint64_t idx = atomicAdd(reinterpret_cast<unsigned long long*>(h.issue), size);
        const uint64_t tail_idx = idx + (size - 1);  // DEV:307

        check_channel_availability(h, tail_idx);  // DEV:315

        // Entry 0 (base request) is built now but stored last (DEV:317-334, 371).
        uint64_t* base_request = reinterpret_cast<uint64_t*>(buf_ptr + (idx & mask));
        uint64_t curr_flag = !((idx >> h.buf_logsize) & 1);
        const uint64_t op = nvshmem_detail::kOpPut;
        const uint64_t base_request_value =
            (roffset << 24) | (op << 16) | (nvshmem_detail::kGroupSizeSingle << 8) | curr_flag;

        // put_dma_request_0: laddr_high | flag (DEV:339-343).
        idx += nvshmem_detail::kChannelEntryBytes;
        const uint64_t laddr_high = laddr & mask_lowest_byte;
        req = reinterpret_cast<uint64_t*>(buf_ptr + (idx & mask));
        curr_flag = !((idx >> h.buf_logsize) & 1);
        st_volatile(req, laddr_high | curr_flag);

        // put_dma_request_1: size | laddr_low | flag (DEV:348-353).
        idx += nvshmem_detail::kChannelEntryBytes;
        const uint64_t laddr_low = laddr & mask_upper_7_bytes;
        req = reinterpret_cast<uint64_t*>(buf_ptr + (idx & mask));
        curr_flag = !((idx >> h.buf_logsize) & 1);
        st_volatile(req, size_u64 << 16 | laddr_low << 8 | curr_flag);

        // put_dma_request_2: qp_index | pe | resv | flag (DEV:358-362).
        idx += nvshmem_detail::kChannelEntryBytes;
        req = reinterpret_cast<uint64_t*>(buf_ptr + (idx & mask));
        curr_flag = !((idx >> h.buf_logsize) & 1);
        st_volatile(req, (static_cast<uint64_t>(qp_index) << 32) |
                             (static_cast<uint64_t>(pe_u16) << 16) | curr_flag);

        // Entry 0 last (DEV:371). No fence: each entry has its own parity flag
        // and the proxy spins on every entry's flag (PROX:533-545).
        st_volatile(base_request, base_request_value);
    }

    __device__ static void push(Handle& h, const Msg& m) {
        // nvshmemx_putmem_nbi_warp over a proxy transport: only thread 0 of
        // the group issues the request (XFER:270-281).
        if (lane_id() != 0) return;
        transfer_dma(h,
                     /*roffset=*/filler_word(m, 2),  // 40 bits survive the << 24
                     /*laddr=*/msg_word0(m),
                     /*size=*/m.tag & 0x0000FFFFFFFFFFFFULL,
                     /*pe=*/static_cast<uint16_t>(m.tag >> 48),
                     /*qp_index=*/static_cast<uint32_t>(filler_word(m, 3)));
    }

    // ---------------------------------------------------------------------
    // Host: progress_channels + process_channel_dma
    // ---------------------------------------------------------------------

    // COUNTER_TO_FLAG / WRAPPED_CHANNEL_BUF (PH:16-17).
    static uint8_t counter_to_flag(const Channel& ch, uint64_t counter) {
        return static_cast<uint8_t>(!((counter >> ch.bufsize_log) & 1));
    }
    static char* wrapped(const Channel& ch, uint64_t counter) {
        return ch.buf + (counter & (ch.bufsize - 1));
    }

    // proxy_update_processed (PROX:159-167). Upstream writes `complete` with a
    // plain store; volatile keeps the same single x86 mov.
    static void update_processed(Channel& ch, uint64_t bytes) {
        ch.processed += bytes;
        if ((ch.processed - ch.last_sync) >= nvshmem_detail::kSyncBytes) {
            *reinterpret_cast<volatile uint64_t*>(ch.complete) = ch.processed;
            ch.last_sync = ch.processed;
        }
    }

    // One iteration of the progress_channels batch loop (PROX:1438-1547).
    // The harness's --batch (default 32) plays PROXY_REQUEST_BATCH_MAX (32).
    bool pop(int queue, Msg& out) {
        using namespace nvshmem_detail;
        Channel& ch = chans_[queue];
        const uint64_t counter = ch.processed;                              // PROX:1441
        const uint8_t flag = counter_to_flag(ch, counter);                  // PROX:1444
        auto* base_req = reinterpret_cast<BaseRequest*>(wrapped(ch, counter));  // PROX:1445
        // Mask out the continuation bit (PROX:1446-1447).
        const uint8_t flag_value = static_cast<uint8_t>(base_req->flag & 1);
        if (flag_value != flag) return false;  // PROX:1448-1451, 1542-1544

        // switch (channel_req->op) (PROX:1471-1478): only PUT is ever issued.
        const uint8_t op = *reinterpret_cast<volatile uint8_t*>(&base_req->op);
        if (op != kOpPut) {
            std::fprintf(stderr, "nvshmem-proxy: invalid op %u in channel %d\n", op, queue);
            std::abort();
        }

        // process_channel_dma (PROX:515-613).
        const bool explicit_region =
            *reinterpret_cast<volatile uint8_t*>(&base_req->groupsize) & kGroupRegion;
        (void)explicit_region;  // PROX:530; never set on this path
        auto* dma_req_0 = reinterpret_cast<PutDmaRequest0*>(wrapped(ch, counter + 8));
        while (dma_req_0->flag != counter_to_flag(ch, counter + 8)) {}  // PROX:533-535
        auto* dma_req_1 = reinterpret_cast<PutDmaRequest1*>(wrapped(ch, counter + 16));
        while (dma_req_1->flag != counter_to_flag(ch, counter + 16)) {}  // PROX:537-540
        auto* dma_req_2 = reinterpret_cast<PutDmaRequest2*>(wrapped(ch, counter + 24));
        while (dma_req_2->flag != counter_to_flag(ch, counter + 24)) {}  // PROX:542-545
        // qp_index is read only for *_QP ops (PROX:547-549).

        // x86: compiler barrier only (PROX:551-555).
        asm volatile("" : : : "memory");
        const uint64_t laddr = (static_cast<uint64_t>(dma_req_0->laddr_high) << 32) |
                               (static_cast<uint64_t>(dma_req_0->laddr_3) << 16) |
                               (static_cast<uint64_t>(dma_req_0->laddr_2) << 8) |
                               dma_req_1->laddr_low;  // PROX:556-558
        const uint64_t size = (static_cast<uint64_t>(dma_req_1->size_high) << 16) |
                              dma_req_1->size_low;  // PROX:559
        const uint64_t pe = dma_req_2->pe;          // PROX:560

        // "Issue transport DMA" (PROX:563-599) = copy the request out.
        out = msg_from_words(laddr, size | (pe << 48));

        // x86: no fence before the complete update (PROX:600-603 is PPC/ARM only).
        update_processed(ch, kDmaReqBytes);  // PROX:607
        return true;
    }

    static std::string notes() {
        return "NVSHMEM cffd2a5 proxy channel (HIP re-impl); ring = capacity*32B pinned host, "
               "byte counter; lane-0 agent fence + device atomicAdd(32) on device `issue`, "
               "relaxed-sys load of device complete_local, 4 relaxed-sys 8B entry stores (entry "
               "0 last) with per-entry lap-parity flag; proxy writes host `complete` every "
               ">=1024B";
    }

    std::vector<Channel> chans_;
};

}  // namespace qb
