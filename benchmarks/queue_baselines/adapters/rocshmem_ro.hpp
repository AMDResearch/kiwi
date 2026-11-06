// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file rocshmem_ro.hpp
 * @brief rocSHMEM reverse-offload (RO) backend GPU->CPU queue adapter.
 *
 * Re-implementation (not vendored) of the RO command queue from
 * ROCm/rocm-systems projects/rocshmem (rocSHMEM 3.8.0), monorepo commit
 * 5909ad99a98c44a9f4e65ba7c416b3be8330e355 (develop, 2026-10-16). Citations
 * are src/<file>:<line> at that commit.
 *
 * One queue == one rocSHMEM context (BlockHandle + queue descriptor + ring).
 * A push is the queue part of ROContext::putmem_nbi_wave()
 * (reverse_offload/context_ro_device.cpp:393-407): only thread 0 of the wave
 * calls build_queue_element(RO_NET_PUT_NBI, ..., blocking=false), which
 * claims one slot under the per-context spinlock, writes the PUT_NBI fields,
 * fences, sets valid=1, fences, and returns.
 *
 * The consumer is Queue::process() (reverse_offload/queue.cpp:47-57) with
 * the library defaults: ROCSHMEM_RO_NET_CPU_QUEUE=0 (memcpy-to-cache path)
 * and USE_HDP_FLUSH=OFF (NoHdpPolicy, so hdp_flush() is empty).
 *
 * Deviations, all in the baseline's favor:
 *  - The element is handled in the polling thread itself; upstream copies it
 *    into a mutex-protected std::queue (mpi_transport.cpp:74-78) that a
 *    second thread drains one element per loop (mpi_transport.cpp:63-90),
 *    sleeping ROCSHMEM_RO_PROGRESS_DELAY=3 us when it has no outstanding MPI
 *    requests (mpi_transport.cpp:611-619, envvar.cpp:175-177).
 *  - The IPC-availability check and the context-object indirection of
 *    putmem_nbi_wave are not modeled; only the queue protocol is.
 *  - The BlockHandle pointer is address-space-cast to global (harness rule).
 *    Upstream reaches it through ROContext, so those accesses are flat_*.
 */

#pragma once

#include "../qb_common.hpp"

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace qb {

namespace rocshmem_ro {

// commands_types.hpp:31-48 (enum ro_net_cmds): RO_NET_PUT_NBI is value 3.
enum ro_net_cmds : int { RO_NET_PUT = 0, RO_NET_P, RO_NET_GET, RO_NET_PUT_NBI };

// queue_proxy.hpp:38-41
struct cacheline_t {
    volatile char valid;
    volatile char padding[63];
} __attribute__((__aligned__(64)));

// queue_proxy.hpp:43-76. Field order and types are upstream's; the default
// member initializers are dropped because the ring is memset to 0 after
// allocation anyway (queue_proxy.hpp:131-133).
typedef struct queue_element {
    cacheline_t notify_cpu;  // polled by the CPU; own 64B line
    ro_net_cmds type;
    int PE;
    void* src;
    void* dst;
    int ro_net_win_id;
    int logPE_stride;
    int PE_size;
    long* pSync;
    int op;
    int datatype;
    int PE_root;
    volatile char* status;
    intptr_t team_comm;
    union {
        size_t size;
        unsigned long long atomic_value;
    } ol1;
    union {
        void* pWrk;
        unsigned long long atomic_cond;
    } ol2;
} __attribute__((__aligned__(64))) queue_element_t;

static_assert(sizeof(queue_element_t) == 192, "upstream element is 192B");
static_assert(offsetof(queue_element_t, type) == 64, "header line is 64B");
static_assert(offsetof(queue_element_t, ol1) == 144, "layout mismatch");

// queue_desc_proxy.hpp:33-51: read_index and write_index on separate lines.
typedef struct queue_desc {
    uint64_t read_index;
    char padding1[56];
    uint64_t write_index;
    char padding2[56];
} __attribute__((__aligned__(64))) queue_desc_t;

// block_handle.hpp:40-54. `profiler` is NullStats (profiler.hpp, PROFILE=OFF
// by default), an empty class; a 1-byte stand-in keeps the 104B layout, so
// contexts' handles share cache lines exactly as in the upstream array.
struct BlockHandle {
    char profiler;
    queue_element_t* queue;
    uint64_t queue_size;
    volatile uint64_t read_index;   // device-side cached copy of the read index
    volatile uint64_t write_index;
    volatile uint64_t* host_read_index;  // &queue_desc_t::read_index
    volatile char* status;
    void* g_ret;
    void* atomic_ret;
    volatile uint64_t lock;
    void* default_ctx_status;
    void* default_ctx_g_ret;
    void* default_ctx_atomic_ret;
};
static_assert(sizeof(BlockHandle) == 104, "upstream BlockHandle is 104B");

constexpr int kWfSize = 64;  // constants.hpp:64

// util.hpp:204-207
__device__ __forceinline__ int get_flat_block_id() {
    return threadIdx.x + threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
}

// util.hpp:235-237
__device__ __forceinline__ bool is_thread_zero_in_wave() {
    return (get_flat_block_id() % kWfSize) == 0;
}

// assembly.hpp:124-139 (gfx942/gfx950 branch): uncached system-scope load.
__device__ __forceinline__ void refresh_volatile_dwordx2(volatile uint64_t* assigned_value,
                                                         volatile uint64_t* read_value) {
#if defined(__gfx942__) || defined(__gfx950__)
    asm volatile(
        "global_load_dwordx2 %0 %1 off sc0 sc1\n "
        "s_waitcnt vmcnt(0)"
        : "=v"(*assigned_value)
        : "v"(read_value)
        : "memory");
#else
    *assigned_value = __hip_atomic_load(const_cast<uint64_t*>(read_value), __ATOMIC_RELAXED,
                                        __HIP_MEMORY_SCOPE_SYSTEM);
#endif
}

// context_ro_device.cpp:532-534
__device__ __forceinline__ uint64_t number_active_lanes() { return __popcll(__ballot(1)); }

// context_ro_device.cpp:536-545
__device__ __forceinline__ uint64_t active_logical_lane_id() {
    uint64_t ballot{__ballot(1)};
    uint64_t my_physical_lane_id{__lane_id()};
    uint64_t all_ones_mask = -1;
    uint64_t lane_mask{all_ones_mask << my_physical_lane_id};
    uint64_t inverted_mask{~lane_mask};
    uint64_t lower_active_lanes{ballot & inverted_mask};
    return __popcll(lower_active_lanes);
}

// context_ro_device.cpp:547-556 (broadcast() -> broadcast_lds(), :568-570)
__device__ __forceinline__ uint64_t broadcast_lds(bool lowest_active, uint64_t value) {
    constexpr size_t SIZE = 1024 / kWfSize;
    __shared__ uint64_t value_per_warp[SIZE];
    auto wavefront_number{get_flat_block_id() / kWfSize};
    if (lowest_active) {
        value_per_warp[wavefront_number] = value;
        __threadfence_block();
    }
    return value_per_warp[wavefront_number];
}

// context_ro_device.cpp:581-583
__device__ __forceinline__ bool enough_space(BlockHandle* h, uint64_t required) {
    return (h->queue_size - (h->write_index - h->read_index)) >= required;
}

// context_ro_device.cpp:585-587: relaxed, agent-scope CAS (HIP atomicCAS).
__device__ __forceinline__ void acquire_lock(BlockHandle* handle) {
    while (atomicCAS((unsigned long*)&handle->lock, 0UL, 1UL) == 1) {}
}

// context_ro_device.cpp:589-592
__device__ __forceinline__ void release_lock(BlockHandle* handle) {
    handle->lock = 0;
    __threadfence();
}

// context_ro_device.cpp:594-598
__device__ __forceinline__ void wait_until_space_available(BlockHandle* handle,
                                                           uint64_t required) {
    while (!enough_space(handle, required))
        refresh_volatile_dwordx2(&handle->read_index, handle->host_read_index);
}

// context_ro_device.cpp:636-652 (next_write_slot() selects _o_m_m, :654-659)
__device__ __forceinline__ uint64_t next_write_slot_o_m_m(BlockHandle* handle) {
    auto num_active_lanes{number_active_lanes()};
    uint64_t write_slot{0};
    auto my_active_lane_id{active_logical_lane_id()};
    bool is_lowest_active_lane{my_active_lane_id == 0};
    if (is_lowest_active_lane) {
        acquire_lock(handle);
        wait_until_space_available(handle, num_active_lanes);
        write_slot = handle->write_index;
        handle->write_index += num_active_lanes;
        __threadfence();
        release_lock(handle);
    }
    write_slot = broadcast_lds(is_lowest_active_lane, write_slot);
    write_slot += my_active_lane_id;
    return write_slot % handle->queue_size;
}

// build_queue_element() (context_ro_device.cpp:661-742) specialised to
// type=RO_NET_PUT_NBI, blocking=false, default_ctx=false, the arguments
// putmem_nbi_wave passes (:403-405). The Msg rides in the native fields:
// dst = word0 (seq<<32 | producer), src = tag, the rest is filler.
__device__ __forceinline__ void build_put_nbi_element(BlockHandle* handle, const Msg& m) {
    auto write_slot{next_write_slot_o_m_m(handle)};
    auto queue_element = &handle->queue[write_slot];

    // :670-682 (status is written only when blocking; RO_NET_P only memcpys)
    queue_element->type = RO_NET_PUT_NBI;
    queue_element->PE = static_cast<int>(filler_word(m, 2));
    queue_element->ol1.size = filler_word(m, 3);
    queue_element->dst = reinterpret_cast<void*>(msg_word0(m));
    queue_element->ro_net_win_id = static_cast<int>(filler_word(m, 4));
    queue_element->src = reinterpret_cast<void*>(m.tag);

    // :722-727
    __threadfence();
    queue_element->notify_cpu.valid = 1;
    __threadfence();
    // :730-741: non-blocking, not the default context -> return.
}

}  // namespace rocshmem_ro

struct RocshmemRoAdapter {
    using BlockHandle = rocshmem_ro::BlockHandle;
    using queue_element_t = rocshmem_ro::queue_element_t;
    using queue_desc_t = rocshmem_ro::queue_desc_t;

    static constexpr const char* kName = "rocshmem-ro";
    static constexpr bool kMultiProducer = true;  // waves share a context
    static constexpr bool kFifo = true;           // slots claimed in order under the lock
    // Native slot size (what the host copies per poll). A PUT_NBI writes only
    // 36B of fields in lines 1-2 plus the valid byte in line 0.
    static constexpr int kDescriptorBytes = sizeof(queue_element_t);
    using Handle = BlockHandle*;

    /// Host-side per-queue consumer state (one consumer thread per queue).
    struct alignas(64) Consumer {
        queue_element_t cache;       // queue_element_cache_proxy_ (queue.hpp:69)
        queue_element_t* ring;       // queues[queue_index]
        volatile uint64_t* read_index;  // &queue_desc_t::read_index (device memory)
    };

    RocshmemRoAdapter(int num_queues, int capacity)
        : num_queues_(num_queues), queue_size_(static_cast<uint64_t>(capacity)),
          consumers_(num_queues) {
        // QueueProxy (queue_proxy.hpp:118-134): one HIPHostAllocator
        // allocation (hipHostMalloc(hipHostMallocCoherent),
        // hip_allocator.hpp:526-530) holding every context's ring, memset 0.
        const size_t ring_bytes = sizeof(queue_element_t) * queue_size_ * num_queues;
        QB_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&rings_), ring_bytes,
                                   hipHostMallocCoherent));
        std::memset(rings_, 0, ring_bytes);

        // QueueDescProxy (queue_desc_proxy.hpp:59-68): HIPAllocatorFinegrained
        // = hipExtMallocWithFlags(hipDeviceMallocFinegrained)
        // (hip_allocator.hpp:400-407); indices zeroed from the host.
        QB_HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&descs_),
                                           sizeof(queue_desc_t) * num_queues,
                                           hipDeviceMallocFinegrained));
        std::memset(descs_, 0, sizeof(queue_desc_t) * num_queues);

        // BlockHandleProxy (block_handle.hpp:110-131): one contiguous
        // fine-grained device array of handles, filled from the host.
        QB_HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&handles_),
                                           sizeof(BlockHandle) * num_queues,
                                           hipDeviceMallocFinegrained));
        std::memset(handles_, 0, sizeof(BlockHandle) * num_queues);
        for (int q = 0; q < num_queues; ++q) {
            BlockHandle* h = &handles_[q];
            h->queue = rings_ + static_cast<size_t>(q) * queue_size_;
            h->queue_size = queue_size_;
            h->read_index = descs_[q].read_index;
            h->write_index = descs_[q].write_index;
            h->host_read_index = &descs_[q].read_index;
            h->lock = 0;

            Consumer& c = consumers_[q];
            std::memset(&c.cache, 0, sizeof(c.cache));
            c.ring = h->queue;
            c.read_index = &descs_[q].read_index;
        }
    }

    ~RocshmemRoAdapter() {
        (void)hipFree(handles_);
        (void)hipFree(descs_);
        (void)hipHostFree(rings_);
    }

    RocshmemRoAdapter(const RocshmemRoAdapter&) = delete;
    RocshmemRoAdapter& operator=(const RocshmemRoAdapter&) = delete;

    Handle handle(int queue) { return &handles_[queue]; }

    // Only the handle pointer can be marked: the ring and host_read_index
    // pointers are loaded from the BlockHandle on every push, as upstream
    // does, so those accesses stay flat_*.
    __device__ static void globalize(Handle& h) { h = as_global(h); }

    // putmem_nbi_wave (context_ro_device.cpp:393-407): thread 0 of the wave
    // issues the command, so number_active_lanes() == 1 and one slot is used.
    __device__ static void push(Handle& h, const Msg& m) {
        if (rocshmem_ro::is_thread_zero_in_wave()) rocshmem_ro::build_put_nbi_element(h, m);
    }

    // Queue::process() (queue.cpp:47-57), with the upstream default path of
    // next_element() (queue.cpp:59-71): ROCSHMEM_RO_NET_CPU_QUEUE=0
    // (envvar.cpp:178-180), so hdp_flush() then memcpy into the cache
    // element. hdp_flush() is NoHdpPolicy::hdp_flush(), empty, because
    // USE_HDP_FLUSH is OFF by default (CMakeLists.txt:56, hdp_policy.hpp:187-197).
    // read_index lives in fine-grained device memory; every source-level
    // access is a separate load (volatile here; upstream re-loads because the
    // memcpy/char stores and insertRequest() may alias it).
    bool pop(int queue, Msg& out) {
        Consumer& c = consumers_[queue];
        // copy_element_to_cache (queue.cpp:73-78)
        const uint64_t read_slot = *c.read_index % queue_size_;
        std::memcpy(&c.cache, &c.ring[read_slot], sizeof(queue_element_t));
        if (!c.cache.notify_cpu.valid) return false;
        // transport->insertRequest(next_elem, ...) (queue.cpp:50) copies the
        // element into the transport's std::queue; here the copy-out is the Msg.
        out = msg_from_words(reinterpret_cast<uint64_t>(c.cache.dst),
                             reinterpret_cast<uint64_t>(c.cache.src));
        // queue.cpp:52: clear valid at get_read_index() (queue.cpp:39-41)
        c.ring[*c.read_index % queue_size_].notify_cpu.valid = 0;
        // queue.cpp:53 -> increment_read_index (queue.cpp:43-45): RMW over PCIe
        *c.read_index = *c.read_index + 1;
        return true;
    }

    static std::string notes() {
        return "192B element (64B valid line + fields) in hipHostMallocCoherent ring; "
               "lane-0 agent-scope CAS spinlock + write_index bump in fine-grained "
               "device memory; PUT_NBI fields, fence, valid=1, fence; host memcpy+"
               "valid check, clear valid, read_index++ in device memory per pop";
    }

    int num_queues_;
    uint64_t queue_size_;
    queue_element_t* rings_ = nullptr;
    queue_desc_t* descs_ = nullptr;
    BlockHandle* handles_ = nullptr;
    std::vector<Consumer> consumers_;
};

}  // namespace qb
