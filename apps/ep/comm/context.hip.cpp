// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "comm/context.hip.hpp"
#include "comm/proxy_network.hip.hpp"
#include "assert.hpp"
#include "ep_context.hip.hpp"
#include "topology.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace kiwi::ep {

namespace internal {
namespace {

// The largest EP setup message is the 256-byte hostname exchange. A 512-byte
// packet leaves room for LCI metadata while keeping the control pool small.
constexpr size_t kControlPacketSize = 512;
constexpr size_t kControlPacketCount = 8192;
constexpr size_t kControlMaxRecvs = kControlPacketCount / 2;
constexpr size_t kRawNetworkMaxSends = 64;
int runtime_ref_count = 0;

void initialize_runtime(std::string requested_device_name) {
    // LCI initializes its global attributes from the environment before it
    // constructs the default runtime. Tune only unset process-wide defaults;
    // explicit environment settings remain authoritative.
    lci::global_initialize();
    const lci::global_attr_t original_attr = lci::get_g_default_attr();
    lci::global_attr_t ep_attr = original_attr;
    if (std::getenv("LCI_ATTR_PACKET_SIZE") == nullptr) {
        ep_attr.packet_size = kControlPacketSize;
    }
    const bool packet_count_overridden =
        std::getenv("LCI_ATTR_NPACKETS") != nullptr;
    const bool max_recvs_overridden =
        std::getenv("LCI_ATTR_NET_MAX_RECVS") != nullptr;
    if (!packet_count_overridden) {
        ep_attr.npackets = max_recvs_overridden
            ? std::max(kControlPacketCount, ep_attr.net_max_recvs * 2)
            : kControlPacketCount;
    }
    if (!max_recvs_overridden) {
        ep_attr.net_max_recvs = std::min(kControlMaxRecvs, ep_attr.npackets / 2);
    }
    lci::set_g_default_attr(ep_attr);

    try {
        if (requested_device_name.empty()) {
            lci::g_runtime_init();
        } else {
            lci::g_runtime_init_x()
                .device_name(std::move(requested_device_name))();
        }
    } catch (...) {
        lci::set_g_default_attr(original_attr);
        lci::global_finalize();
        throw;
    }

    // Balance the explicit initialization above. g_runtime_init() retains its
    // own global reference until g_runtime_fina().
    lci::global_finalize();
}

} // namespace

ep_runtime_ref_t::ep_runtime_ref_t(std::string requested_device_name) {
    if (runtime_ref_count == 0) {
        initialize_runtime(std::move(requested_device_name));
    }
    ++runtime_ref_count;
}

ep_runtime_ref_t::~ep_runtime_ref_t() {
    if (--runtime_ref_count == 0) {
        lci::g_runtime_fina();
    }
}

static thread_local size_t tl_raw_network_outstanding = 0;

enum class raw_network_cq_completion_t {
    WRITE,
    ASYNCHRONOUS_ERROR,
    UNEXPECTED_OPCODE,
    NO_OUTSTANDING_WORK,
};

static raw_network_cq_completion_t raw_network_account_completion(
        lci::net_opcode_t opcode, size_t& outstanding) {
    if (outstanding == 0) {
        return raw_network_cq_completion_t::NO_OUTSTANDING_WORK;
    }
    if (opcode == lci::net_opcode_t::WRITE) {
        --outstanding;
        return raw_network_cq_completion_t::WRITE;
    }
    if (opcode == lci::net_opcode_t::ERROR) {
        --outstanding;
        return raw_network_cq_completion_t::ASYNCHRONOUS_ERROR;
    }
    return raw_network_cq_completion_t::UNEXPECTED_OPCODE;
}

static void raw_network_report_bad_completion(
        const char* reason, const lci::net_status_t& status,
        size_t outstanding) {
    std::fprintf(
        stderr,
        "kiwi EP raw network CQ %s: opcode=%s(%d) rank=%d "
        "user_context=%p length=%zu imm_data=0x%x outstanding=%zu\n",
        reason, lci::get_net_opcode_str(status.opcode),
        static_cast<int>(status.opcode), status.rank, status.user_context,
        status.length, static_cast<unsigned int>(status.imm_data),
        outstanding);
}

static void raw_network_progress() {
    if (tl_raw_network_outstanding == 0) {
        return;
    }
    // net_poll_cq_x initializes only the returned entries. Avoid clearing the
    // full batch on this hot path.
    std::array<lci::net_status_t, LCI_BACKEND_MAX_POLLS> statuses;
    const size_t count =
        lci::net_poll_cq_x(statuses.size(), statuses.data())
            .device(tl_device)();
    for (size_t i = 0; i < count; ++i) {
        const lci::net_status_t& status = statuses[i];
        const raw_network_cq_completion_t completion =
            raw_network_account_completion(
                status.opcode, tl_raw_network_outstanding);
        if (completion == raw_network_cq_completion_t::WRITE) {
            continue;
        }
        if (completion
            == raw_network_cq_completion_t::ASYNCHRONOUS_ERROR) {
            raw_network_report_bad_completion(
                "reported an asynchronous error", status,
                tl_raw_network_outstanding);
            KIWI_EP_ASSERT(
                false, "raw network CQ reported an asynchronous error");
        }
        if (completion
            == raw_network_cq_completion_t::NO_OUTSTANDING_WORK) {
            raw_network_report_bad_completion(
                "has no tracked raw work request", status,
                tl_raw_network_outstanding);
            KIWI_EP_ASSERT(
                false, "raw network CQ completion has no outstanding work request");
        }
        raw_network_report_bad_completion(
            "received a CQE not owned by the raw proxy", status,
            tl_raw_network_outstanding);
        KIWI_EP_ASSERT(
            false, "raw network CQ completion violates raw-device ownership");
    }
}

void proxy_network_record_injected_write(lci::error_t error) {
    KIWI_EP_ASSERT(
        error.is_done(),
        "raw short/inject WRITE did not complete its source-buffer lifetime");
    // `done` here does not mean that the signaled network WRITE is absent or
    // remotely complete. Its CQE is still pending and must be drained before
    // this proxy can be torn down.
    ++tl_raw_network_outstanding;
}

static void raw_network_drain() {
    while (tl_raw_network_outstanding != 0) {
        raw_network_progress();
    }
}

void proxy_network_progress() {
    raw_network_progress();
}

void proxy_network_record_posted_write(lci::error_t error) {
    KIWI_EP_ASSERT(
        error.is_posted(),
        "raw network WRITE did not post");
    ++tl_raw_network_outstanding;
}

} // namespace internal

ep_comm_context_t::ep_comm_context_t(
        int num_queues, int num_devices, std::string requested_device_name,
        size_t queue_capacity)
    : runtime_(std::move(requested_device_name)), num_queues_(num_queues),
      num_devices_(num_devices),
      active_queue_request_(num_queues) {
    if (num_queues_ <= 0) {
        throw std::invalid_argument(
            "kiwi::ep::ep_comm_context_t: num_endpoints=" +
            std::to_string(num_queues_) + " must be positive");
    }
    if (num_devices_ <= 0) {
        throw std::invalid_argument(
            "kiwi::ep::ep_comm_context_t: num_devices=" +
            std::to_string(num_devices_) + " must be positive");
    }
    // Create EP-owned invoke context. queue_capacity is the per-queue
    // depth and must be a non-zero power of two for queue ring indexing.
    if (queue_capacity == 0 || (queue_capacity & (queue_capacity - 1)) != 0) {
        throw std::invalid_argument(
            "kiwi::ep::ep_comm_context_t: queue_capacity=" +
            std::to_string(queue_capacity) + " is not a non-zero power of two");
    }

    gc_context_ = std::make_unique<kiwi::invoke::Context<ep_queue_t>>(num_queues_,
                                                         queue_capacity);

    // Allocate EP-owned LCI devices. Their raw WRITE path does not consume
    // packets, so its send depth is independent of the control packet pool.
    devices_.resize(num_devices_);
    for (int i = 0; i < num_devices_; i++) {
        // The runtime default device owns LCI setup traffic and SHM. These
        // proxy devices expose only the raw-network path.
        devices_[i] = lci::alloc_device_x()
            .net_max_sends(internal::kRawNetworkMaxSends)
            .net_max_recvs(0)
            .shm_enable(false)
            .packet_pool(lci::packet_pool_t())();
    }
}

ep_comm_context_t::~ep_comm_context_t() {
    for (auto& device : devices_) {
        lci::free_device(&device);
    }

    gc_context_.reset();
}

ep_comm_device_context_t ep_comm_context_t::get_device_context(
        uint8_t all2all_cb_id) {
    return ep_comm_device_context_t(
        gc_context_->device_context(), gc_context_->num_endpoints(),
        all2all_cb_id);
}

int ep_comm_context_t::alloc_symmetric(void* buf, size_t size) {
    assert(!devices_.empty());

    int nranks = lci::get_rank_n();
    internal::ep_heap_entry_t heap_entry;
    heap_entry.device_entries.reserve(devices_.size());

    // Exchange per-device RMRs through the runtime default setup device. Raw
    // proxy devices have no high-level receive path.
    const lci::device_t setup_device = get_setup_device();
    for (auto device : devices_) {
        assert(!device.is_empty());
        lci::mr_t mr = lci::register_memory_x(buf, size).device(device)();
        lci::rmr_t my_rmr = lci::get_rmr(mr);

        std::vector<lci::rmr_t> rmrs(nranks);
        lci::allgather_x(&my_rmr, rmrs.data(), sizeof(lci::rmr_t))
            .device(setup_device)();

        heap_entry.device_entries.push_back({mr, std::move(rmrs)});
    }

    int heap_id = static_cast<int>(heap_table_.size());
    heap_table_.push_back(std::move(heap_entry));
    return heap_id;
}

void ep_comm_context_t::wait_drained() {
    KIWI_EP_ASSERT(
        proxy_threads_.empty(),
        "EP raw-network symmetric heap teardown requires joined proxy threads");
}

void ep_comm_context_t::deregister_symmetric(int heap_id) {
    assert(heap_id >= 0 && heap_id < static_cast<int>(heap_table_.size()));
    wait_drained();
    for (auto& entry : heap_table_[heap_id].device_entries) {
        lci::deregister_memory(&entry.mr);
    }
}

void ep_comm_context_t::spawn_proxy_threads(int num_threads) {
    if (num_threads == -1) {
        num_threads = num_queues_;
    }
    if (num_threads <= 0) {
        throw std::invalid_argument(
            "kiwi::ep::ep_comm_context_t::spawn_proxy_threads: num_threads=" +
            std::to_string(num_threads) + " must be positive");
    }
    assert(proxy_threads_.empty());
    stop_proxy_threads_.store(false, std::memory_order_relaxed);

    // Resolve the CPU set on the spawning thread, which has the context's
    // HIP device selected.
    const std::vector<int> proxy_cpus =
        kiwi::p2p::topology::proxy_thread_cpu_set();

    for (int i = 0; i < num_threads; i++) {
        proxy_threads_.emplace_back(&ep_comm_context_t::proxy_thread_func,
                                    this, i, num_threads, proxy_cpus);
    }
}

void ep_comm_context_t::join_proxy_threads() {
    stop_proxy_threads_.store(true, std::memory_order_relaxed);
    for (auto& t : proxy_threads_) {
        t.join();
    }
    proxy_threads_.clear();
}

void ep_comm_context_t::publish_active_queue_count(int active_count) {
    KIWI_EP_ASSERT(
        active_count > 0 && active_count <= num_queues_,
        "active queue count is outside the provisioned endpoint range");
    active_queue_request_.publish(active_count);
}

void ep_comm_context_t::proxy_thread_func(int thread_id,
                                               int num_proxy_threads,
                                               std::vector<int> cpus) {
    int device_idx = thread_id % num_devices_;
    kiwi::p2p::topology::bind_current_thread_to_cpus(cpus);

    internal::tl_device_idx = device_idx;
    internal::tl_device = devices_[device_idx];
    internal::tl_proxy_thread_idx = thread_id;

    std::vector<std::pair<int, ep_queue_t::HostPollState>>
        endpoint_poll_states;
    // Radix-fold ownership spreads active endpoint prefixes across proxies
    // without aliasing regularly spaced producer endpoints. Scan endpoint IDs
    // in order so this proxy's list remains sorted for the prefix break below.
    for (int q = 0; q < num_queues_; ++q) {
        if (internal::proxy_queue_owner(q, num_proxy_threads)
            != thread_id) {
            continue;
        }
        endpoint_poll_states.emplace_back(
            q, gc_context_->raw_queue(q)->host_poll_state());
    }
    // Poll every provisioned endpoint until a launch-driven shrink has drained
    // its retiring queues. This keeps the first launch safe before its control
    // callback is observed.
    internal::proxy_active_queue_state_t active_state(num_queues_);
    while (!stop_proxy_threads_.load(std::memory_order_relaxed)) {
        // A grow becomes visible immediately. A shrink leaves the current
        // polling range intact until this proxy drains its retiring queues.
        active_state.observe(active_queue_request_.load());

        bool had_ready_endpoint = false;
        bool retiring_queues_empty = true;
        for (auto& [q, poll_state] : endpoint_poll_states) {
            if (q >= active_state.polling_count()) {
                // endpoint_poll_states is ordered by increasing endpoint
                // index, so no later endpoint can be active.
                break;
            }
            poll_state.refresh();
            if (!poll_state.ready()) {
                continue;
            }

            had_ready_endpoint = true;
            gc_context_->progress(
                q, internal::kEpMaxCallbacksPerEndpointSweep);
            poll_state.refresh();
            internal::proxy_network_progress();
            if (active_state.is_retiring_queue(q) && poll_state.ready()) {
                retiring_queues_empty = false;
            }
        }

        if (active_state.has_retiring_queues() && retiring_queues_empty) {
            // Do not let a newer request be overwritten by completion of this
            // older shrink. The next loop observes the newer request and
            // grows immediately if needed.
            active_state.complete_shrink_if_current(
                active_queue_request_.load().generation);
        }

        if (!had_ready_endpoint) {
            internal::proxy_network_progress();
        }
    }

    bool made_progress = false;
    do {
        made_progress = false;
        for (auto& [q, poll_state] : endpoint_poll_states) {
            made_progress |=
                gc_context_->progress(
                    q, internal::kEpMaxCallbacksPerEndpointSweep)
                != 0;
            poll_state.refresh();
            internal::proxy_network_progress();
        }
    } while (made_progress);
    internal::raw_network_drain();

    internal::tl_proxy_thread_idx = -1;
    internal::tl_device_idx = -1;
}

ep_runtime_guard_t::ep_runtime_guard_t(std::string requested_device_name)
    : runtime_(std::move(requested_device_name)) {}

ep_runtime_guard_t::~ep_runtime_guard_t() = default;

int ep_current_proxy_thread_index() {
    return internal::tl_proxy_thread_idx;
}

} // namespace kiwi::ep
