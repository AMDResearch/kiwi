// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "context.hip.hpp"
#include "topology.hpp"
#include <hip/hip_runtime.h>
#include <cassert>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(err));                     \
            abort();                                                      \
        }                                                                 \
    } while (0)

namespace kiwi::p2p {

namespace internal {
// No-op completion: avoids COMP_NULL's allow_retry=false (backlog queue)
// without paying for real counter signaling.
class comp_noop_t : public lci::comp_impl_t {
public:
    void signal(lci::status_t) override {}
};
inline thread_local lci::comp_t tl_noop_comp;

} // namespace internal

int context_t::init_counter = 0;

context_t::context_t(int num_queues_, int num_devices_,
                     std::string requested_device_name,
                     size_t queue_capacity)
    : num_queues(num_queues_), num_devices(num_devices_) {
    // queue_capacity is the per-queue depth and must be a non-zero power of two
    // for queue ring indexing.
    if (queue_capacity == 0 || (queue_capacity & (queue_capacity - 1)) != 0) {
        throw std::invalid_argument(
            "kiwi::p2p::context_t: queue_capacity=" +
            std::to_string(queue_capacity) + " is not a non-zero power of two");
    }

    // Global LCI init (reference counted)
    if (init_counter++ == 0) {
        if (requested_device_name.empty()) {
            lci::g_runtime_init();
        } else {
            lci::g_runtime_init_x().device_name(std::move(requested_device_name))();
        }
    }

    // Create invoke context with num_queues endpoints. queue_capacity is the
    // per-queue depth; each queue costs queue_capacity * 64 bytes of pinned
    // coherent host memory, so total pinned cost scales with
    // num_queues * queue_capacity.
    gc_context = std::make_unique<kiwi::invoke::Context<>>(num_queues, queue_capacity);

    // Register callbacks with heap table capture for mr/rmr lookup.
    auto* table = &heap_table;

    p_cb_id = gc_context->register_callback(
        std::function<void(internal::cmd_desc_p_t)>(
            [table](internal::cmd_desc_p_t desc) {
                auto& heap = (*table)[desc.heap_id];
                auto& entry = heap.device_entries[internal::tl_device_idx];
                lci::rmr_t rmr = entry.rmrs[desc.rank];

                uint64_t value = desc.value;
                lci::status_t status;
                do {
                    status = lci::post_put_x(desc.rank, &value, sizeof(uint64_t),
                                             internal::tl_noop_comp,
                                             desc.remote_disp, rmr)
                        .device(internal::tl_device)
                        .allow_done(false)();
                    if (status.is_retry()) {
                        lci::progress_x().device(internal::tl_device)();
                    }
                } while (status.is_retry());
            }));

    put_cb_id = gc_context->register_callback(
        std::function<void(internal::cmd_desc_put_t)>(
            [table](internal::cmd_desc_put_t desc) {
                auto& data_heap = (*table)[desc.data_heap_id];
                auto& ready_heap = (*table)[desc.ready_heap_id];
                auto& data_entry =
                    data_heap.device_entries[internal::tl_device_idx];
                auto& ready_entry =
                    ready_heap.device_entries[internal::tl_device_idx];

                lci::error_t error;
                do {
                    error = lci::net_post_put_x(
                                desc.rank,
                                reinterpret_cast<void*>(desc.local_buffer),
                                desc.size, data_entry.mr, desc.remote_disp,
                                data_entry.rmrs[desc.rank])
                                .device(internal::tl_device)();
                    if (error.is_retry()) {
                        lci::progress_x().device(internal::tl_device)();
                    }
                } while (error.is_retry());
                if (!error.is_posted()) {
                    std::fprintf(stderr, "payload network PUT did not post\n");
                    std::abort();
                }

                uint8_t ready = 1;
                do {
                    error = lci::net_post_puts_x(
                                desc.rank, &ready, sizeof(ready),
                                desc.ready_remote_disp,
                                ready_entry.rmrs[desc.rank])
                                .device(internal::tl_device)();
                    if (error.is_retry()) {
                        lci::progress_x().device(internal::tl_device)();
                    }
                } while (error.is_retry());
                if (!error.is_done()) {
                    std::fprintf(stderr,
                                 "readiness network PUT did not inject\n");
                    std::abort();
                }
            }));

    // Allocate LCI devices
    size_t npackets = lci::get_default_packet_pool().get_attr_npackets();
    size_t max_nrecvs = std::min(npackets / 4 / num_devices, 4096UL);
    size_t max_nsends = std::min(npackets / 4 / lci::get_rank_n() / num_devices, 64UL);
    max_nsends = std::max(max_nsends, 8UL);

    devices.resize(num_devices);
    for (int i = 0; i < num_devices; i++) {
        devices[i] = lci::alloc_device_x()
            .net_max_sends(max_nsends)
            .net_max_recvs(max_nrecvs)();
    }
}

context_t::~context_t() {
    // Free LCI devices
    for (auto& device : devices) {
        lci::free_device(&device);
    }

    // Destroy invoke context (frees GPU resources)
    gc_context.reset();

    // Global LCI fini (reference counted)
    if (--init_counter == 0) {
        lci::g_runtime_fina();
    }
}

gcontext_t context_t::get_device_context() {
    return gcontext_t(gc_context->device_context(),
                      gc_context->num_endpoints(),
                      p_cb_id, put_cb_id);
}

int context_t::register_symmetric(void* buf, size_t size) {
    assert(!devices.empty());

    int nranks = lci::get_rank_n();
    internal::heap_entry_t heap_entry;
    heap_entry.device_entries.reserve(devices.size());

    for (auto device : devices) {
        assert(!device.is_empty());
        lci::mr_t mr = lci::register_memory_x(buf, size).device(device)();
        lci::rmr_t my_rmr = lci::get_rmr(mr);

        std::vector<lci::rmr_t> rmrs(nranks);
        lci::allgather_x(&my_rmr, rmrs.data(), sizeof(lci::rmr_t))
            .device(device)();

        heap_entry.device_entries.push_back({mr, std::move(rmrs)});
    }

    int heap_id = static_cast<int>(heap_table.size());
    heap_table.push_back(std::move(heap_entry));
    return heap_id;
}

void context_t::wait_drained() {
    for (auto& device : devices) {
        lci::wait_drained_x().device(device)();
    }
}

void context_t::deregister_symmetric(int heap_id) {
    assert(heap_id >= 0 && heap_id < static_cast<int>(heap_table.size()));
    wait_drained();
    for (auto& entry : heap_table[heap_id].device_entries) {
        lci::deregister_memory(&entry.mr);
    }
}

void context_t::spawn_proxy_threads(int num_threads, int first_core) {
    if (num_threads == -1) {
        num_threads = num_queues;
    }
    assert(proxy_threads.empty());
    stop_proxy_threads = false;

    // Resolve the current HIP device's nearby CPUs in the spawning thread so
    // callers that selected a device with hipSetDevice() get the expected GPU
    // locality even if newly-created host threads would otherwise default to
    // HIP device 0. If the caller supplies first_core, proxy thread i is bound
    // to CPU first_core + i instead.
    const std::vector<int> proxy_cpus =
        first_core >= 0 ? std::vector<int>{} : topology::proxy_thread_cpu_set();

    for (int i = 0; i < num_threads; i++) {
        proxy_threads.emplace_back(
            &context_t::proxy_thread_func, this, i, num_threads,
            first_core >= 0 ? std::vector<int>{first_core + i} : proxy_cpus);
    }
}

void context_t::join_proxy_threads() {
    stop_proxy_threads = true;
    for (auto& t : proxy_threads) {
        t.join();
    }
    proxy_threads.clear();
}

void context_t::proxy_thread_func(int thread_id, int num_proxy_threads,
                                  std::vector<int> cpus) {
    int device_idx = thread_id % num_devices;
    topology::bind_current_thread_to_cpus(cpus);
    internal::tl_device_idx = device_idx;
    internal::tl_device = devices[device_idx];
    auto* noop = new internal::comp_noop_t();
    internal::tl_noop_comp = lci::comp_t(noop);

    // Each proxy thread handles a strided set of queues.
    // With 16 queues and 8 proxies: thread 0 handles {0,8}, thread 1 handles {1,9}, etc.
    // A sweep checks each queue's cached readiness (one signal load) and runs
    // progress(q) only on ready queues. LCI progress runs after each ready
    // queue and once per sweep that found none. Measured, this neither starves
    // the CQ nor costs bandwidth, and idle queues no longer pay for network
    // progress.
    // Only this thread consumes its queues, so the cached state can go stale
    // only through its own progress(q); refresh it after each one, including
    // one that consumed only part of a message. The progress hook still runs
    // for every owned queue in every sweep (see set_proxy_progress_hook()).
    using poll_state_t = decltype(gc_context->raw_queue(0)->host_poll_state());
    std::vector<std::pair<int, poll_state_t>> polls;
    for (int q = thread_id; q < num_queues; q += num_proxy_threads) {
        polls.emplace_back(q, gc_context->raw_queue(q)->host_poll_state());
    }
    while (!stop_proxy_threads) {
        bool any_ready = false;
        for (auto& [q, poll] : polls) {
            internal::tl_queue_idx = q;
            if (poll.ready()) {
                any_ready = true;
                gc_context->progress(q);
                poll.refresh();
                lci::progress_x().device(internal::tl_device)();
            }
            if (proxy_progress_hook) {
                proxy_progress_hook(q);
            }
            internal::tl_queue_idx = -1;
        }
        if (!any_ready) {
            lci::progress_x().device(internal::tl_device)();
        }
    }

    // The default "no local completion requested" path still posts LCI
    // operations with this proxy thread's no-op completion object.  Drain the
    // thread's queues and LCI device before destroying that thread-local
    // completion object; otherwise teardown can race a late local-completion
    // signal into freed memory.
    bool made_progress = false;
    do {
        made_progress = false;
        for (int q = thread_id; q < num_queues; q += num_proxy_threads) {
            internal::tl_queue_idx = q;
            made_progress |= (gc_context->progress(q) != 0);
            lci::progress_x().device(internal::tl_device)();
            if (proxy_progress_hook) {
                proxy_progress_hook(q);
            }
            internal::tl_queue_idx = -1;
        }
    } while (made_progress);
    lci::wait_drained_x().device(internal::tl_device)();

    internal::tl_queue_idx = -1;
    internal::tl_noop_comp = lci::comp_t();
    internal::tl_device_idx = -1;
    delete noop;
}

} // namespace kiwi::p2p
