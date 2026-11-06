// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include "cmd_desc.hip.hpp"
#include "kiwi/invoke/invoke.hpp"
#include "lci.hpp"

namespace kiwi::p2p {

namespace internal {
// Thread-local LCI device used by proxy threads / callbacks.
// tl_device_idx is this device's index in context_t::devices and lets
// callbacks resolve heap registrations without scanning per operation.
inline thread_local lci::device_t tl_device;
inline thread_local int tl_device_idx = -1;
inline thread_local int tl_queue_idx = -1;

// Symmetric heap registration for one LCI device.
struct heap_device_entry_t {
    lci::mr_t mr;
    std::vector<lci::rmr_t> rmrs;  // indexed by rank
};

// Symmetric heap entry: one registration per context-owned LCI device.
// device_entries[i] is registered on context_t::devices[i].
struct heap_entry_t {
    std::vector<heap_device_entry_t> device_entries;
};

// queue moved kiwi::invoke::DeviceContext under kiwi::invoke::internal in newer
// revisions.  Refer to the public Context::device_context() return type so KIWI
// builds against both layouts.
using gc_device_context_t = typename std::remove_pointer<
    decltype(std::declval<kiwi::invoke::Context<>&>().device_context())>::type;

} // namespace internal

// Queue currently being progressed by this proxy thread, or -1 outside a
// invoke callback/proxy-progress hook.  Higher-level transports can use this
// to keep per-queue callback state without baking their semantics into P2P.
inline int current_queue_index() {
    return internal::tl_queue_idx;
}

class gcontext_t;

// GPU-side RAII handle for a single endpoint. Wraps a invoke DeviceHandle
// and borrows callback IDs from the owning KIWI device context. All methods are
// wave-level operations: the full 64-thread wavefront must cooperate on each call.
//
// Usage:
//   auto handle = gctx.get_handle();   // load from device memory
//   handle.post_put(...);              // use for all messaging
//   // destructor persists producer_index back to device memory automatically
//
// The handle is move-only. Copying is deleted to prevent double write-back.
class ghandle_t {
public:
    __device__ ghandle_t(const gcontext_t& gctx_,
                         internal::gc_device_context_t::DeviceHandle h,
                         size_t ep_id)
        : gctx(gctx_), handle(h), endpoint_id(ep_id), save_on_destroy(true) {}

    __device__ __forceinline__ ~ghandle_t();

    // Move OK (source disarmed, won't double-save)
    __device__ ghandle_t(ghandle_t&& other) noexcept
        : gctx(other.gctx), handle(other.handle), endpoint_id(other.endpoint_id),
          save_on_destroy(other.save_on_destroy) {
        other.save_on_destroy = false;
    }

    // No copy (prevents double write-back)
    ghandle_t(const ghandle_t&) = delete;
    ghandle_t& operator=(const ghandle_t&) = delete;

    __device__ void post_p(int heap_id, int rank, uint64_t value,
                           uintptr_t remote_disp);

    template <typename Desc>
    __device__ bool try_invoke(uint8_t cb_id, const Desc& desc) {
        return handle.invoke(cb_id, desc);
    }

    template <typename Desc>
    __device__ void post_desc(uint8_t cb_id, const Desc& desc) {
        while (!handle.invoke(cb_id, desc)) {
            // Queue full, spin-wait
        }
    }

    template <typename Desc>
    __device__ void post_desc_shared(uint8_t cb_id, const Desc* lds_desc,
                                     size_t bytes) {
        while (!handle.invoke_shared(cb_id, lds_desc, bytes)) {
            // Queue full, spin-wait
        }
    }

    // Post a payload PUT followed by a one-byte readiness PUT on the same
    // network device/QP. The receiver owns and resets the readiness byte.
    __device__ void post_put(int data_heap_id, int ready_heap_id, int rank,
                             void* local_buffer, size_t size,
                             uintptr_t remote_disp,
                             uintptr_t ready_remote_disp);

    template <typename Comp>
    __device__ void post_barrier(Comp* comp);

    template <typename SendBuf, typename RecvBuf, typename Comp>
    __device__ void post_all2all(SendBuf* send_buf, RecvBuf* recv_buf,
                                 Comp* comp);

private:
    const gcontext_t& gctx;
    internal::gc_device_context_t::DeviceHandle handle;
    size_t endpoint_id;
    bool save_on_destroy;
};

// GPU-side context. Passed by value to kernels. Call get_handle() to obtain
// a RAII ghandle_t for the current block's endpoint. The handle's destructor
// automatically persists state back to device memory.
class gcontext_t {
public:
    __host__ __device__ gcontext_t()
        : dev_ctx(nullptr), num_endpoints(0), p_cb_id(0), put_cb_id(0),
          barrier_cb_id(0), all2all_cb_id(0) {}

    __host__ __device__ gcontext_t(internal::gc_device_context_t* ctx,
                                    size_t num_ep,
                                    uint8_t p_id,
                                    uint8_t put_id)
        : dev_ctx(ctx), num_endpoints(num_ep), p_cb_id(p_id), put_cb_id(put_id),
          barrier_cb_id(0), all2all_cb_id(0) {}

    // Get RAII handle for this block's endpoint (blockIdx → endpoint mapping).
    // The returned handle persists state automatically on destruction.
    __device__ ghandle_t get_handle() const {
        int blockId = blockIdx.x
                    + blockIdx.y * gridDim.x
                    + blockIdx.z * gridDim.x * gridDim.y;
        size_t ep = blockId % num_endpoints;
        return ghandle_t(*this, dev_ctx->get_device_handle(ep), ep);
    }

    // Get RAII handle for a specific endpoint.
    __device__ ghandle_t get_handle(size_t endpoint_id) const {
        return ghandle_t(*this, dev_ctx->get_device_handle(endpoint_id),
                         endpoint_id);
    }

    __host__ __device__ size_t get_num_endpoints() const { return num_endpoints; }

    // Device-side kiwi::invoke::DeviceContext, for kernels that are not
    // written in HIP C++ (for example FlyDSL through kiwi_flydsl.Endpoints).
    __host__ __device__ internal::gc_device_context_t* invoke_device_context() const {
        return dev_ctx;
    }

    __host__ __device__ void set_barrier_cb_id(uint8_t barrier_id) {
        barrier_cb_id = barrier_id;
    }

    __host__ __device__ void set_all2all_cb_id(uint8_t all2all_id) {
        all2all_cb_id = all2all_id;
    }

private:
    internal::gc_device_context_t* dev_ctx;
    size_t num_endpoints;
    uint8_t p_cb_id;
    uint8_t put_cb_id;
    uint8_t barrier_cb_id;
    uint8_t all2all_cb_id;

    friend class ghandle_t;
};

// Force-inlined so the handle never needs an address (scratch); see
// kiwi::ep::ep_comm_handle_t.
__device__ __forceinline__ ghandle_t::~ghandle_t() {
    if (save_on_destroy) {
        gctx.dev_ctx->save_device_handle(endpoint_id, handle);
    }
}

__device__ inline void ghandle_t::post_p(int heap_id, int rank, uint64_t value,
                                         uintptr_t remote_disp) {
    internal::cmd_desc_p_t desc;
    desc.value = value;
    desc.remote_disp = remote_disp;
    desc.rank = rank;
    desc.heap_id = heap_id;

    while (!handle.invoke(gctx.p_cb_id, desc)) {
        // Queue full, spin-wait
    }
}

template <typename Comp>
__device__ inline void ghandle_t::post_barrier(Comp* comp) {
    internal::cmd_desc_barrier_t desc;
    desc.comp = reinterpret_cast<uintptr_t>(comp);

    while (!handle.invoke(gctx.barrier_cb_id, desc)) {
        // Queue full, spin-wait
    }
}

template <typename SendBuf, typename RecvBuf, typename Comp>
__device__ inline void ghandle_t::post_all2all(SendBuf* send_buf,
                                               RecvBuf* recv_buf,
                                               Comp* comp) {
    internal::cmd_desc_all2all_t desc;
    desc.send_buf = reinterpret_cast<uintptr_t>(send_buf);
    desc.recv_buf = reinterpret_cast<uintptr_t>(recv_buf);
    desc.comp = reinterpret_cast<uintptr_t>(comp);

    while (!handle.invoke(gctx.all2all_cb_id, desc)) {
        // Queue full, spin-wait
    }
}

__device__ inline void ghandle_t::post_put(
        int data_heap_id, int ready_heap_id, int rank,
        void* local_buffer, size_t size, uintptr_t remote_disp,
        uintptr_t ready_remote_disp) {
    internal::cmd_desc_put_t desc{
        reinterpret_cast<uintptr_t>(local_buffer),
        size,
        remote_disp,
        ready_remote_disp,
        rank,
        data_heap_id,
        ready_heap_id,
    };

    while (!handle.invoke(gctx.put_cb_id, desc)) {
        // Queue full, spin-wait
    }
}

class context_t {
public:
    // `requested_device_name` is only consulted when the first context in the
    // process initializes the shared LCI runtime. Later contexts reuse that
    // runtime and cannot change the selected network device.
    // queue_capacity is the per-queue ring size in 64-byte lines; a line
    // carries up to 32 B of payload (a 48-byte put descriptor uses 2). Each queue
    // allocates queue_capacity * 64 bytes of pinned coherent host memory, so total
    // pinned cost scales with num_queues * queue_capacity. EP only needs a
    // shallow queue because puts drain continuously; keep this small so the
    // kiwi/queue/wavefront count can scale cheaply. Must be a power of two.
    explicit context_t(int num_queues = 1, int num_devices = 1,
                       std::string requested_device_name = {},
                       size_t queue_capacity = 2048);
    ~context_t();

    // Not copyable or movable (owns GPU and thread resources)
    context_t(const context_t&) = delete;
    context_t& operator=(const context_t&) = delete;

    gcontext_t get_device_context();

    // Symmetric heap: collective (all ranks must call).
    // Registers buffer with LCI on every context-owned LCI device, then
    // exchanges an RMR table for each registration.
    // Returns heap_id for use in post_p/post_put.
    // Must be called before spawn_proxy_threads() (not thread-safe with callbacks).
    int register_symmetric(void* buf, size_t size);
    // Must be called after join_proxy_threads() (not thread-safe with callbacks).
    void deregister_symmetric(int heap_id);

    // Spawn proxy/progress threads. If first_core >= 0, proxy thread i
    // pins itself to CPU core first_core + i. Otherwise threads use the
    // automatic GPU-local affinity policy when available.
    void spawn_proxy_threads(int num_threads = -1, int first_core = -1);
    void join_proxy_threads();

    // Register a custom callback on the underlying invoke context.
    // Returns the callback ID for use with ghandle_t::try_invoke().
    template <typename Desc>
    uint8_t register_callback(std::function<void(Desc)> cb) {
        return gc_context->register_callback(std::move(cb));
    }

    // Optional generic hook that each proxy thread invokes once per sweep for
    // every queue it owns, whether or not the queue had messages, so a hook
    // waiting on network completions keeps advancing while its queue is idle.
    // For a ready queue it runs after that queue's progress step and the LCI
    // progress that follows it; an idle queue's hook sees the LCI progress of
    // an earlier ready queue or of the previous sweep.  It is intentionally
    // transport-neutral: P2P only owns queue/proxy mechanics, while EP owns
    // payload semantics in apps/ep.
    void set_proxy_progress_hook(std::function<void(int)> hook) {
        proxy_progress_hook = std::move(hook);
    }

    // Access LCI device for direct transport operations outside invoke.
    lci::device_t get_device(int index = 0) const { return devices[index]; }
    int num_devices_count() const { return num_devices; }

    // Access heap table (for custom callbacks that need mr/rmr resolution)
    const std::vector<internal::heap_entry_t>& get_heap_table() const { return heap_table; }

private:
    // Drain outstanding LCI operations on all context-owned devices before
    // deregistering or freeing buffers used by asynchronous proxy-thread puts.
    void wait_drained();

    void proxy_thread_func(int thread_id, int num_proxy_threads,
                           int selected_cpu);

    int num_queues;
    int num_devices;
    std::vector<lci::device_t> devices;
    std::unique_ptr<kiwi::invoke::Context<>> gc_context;

    // Callback IDs
    uint8_t p_cb_id;
    uint8_t put_cb_id;
    std::function<void(int)> proxy_progress_hook;

    // Symmetric heap table (read-only during callbacks)
    std::vector<internal::heap_entry_t> heap_table;

    // Proxy threads
    std::vector<std::thread> proxy_threads;
    std::atomic<bool> stop_proxy_threads{false};

    // Reference counting for LCI global init/fini
    static int init_counter;
};

} // namespace kiwi::p2p
