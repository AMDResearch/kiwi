// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "comm/active_queue_state.hpp"
#include "comm/device_context.hip.hpp"
#include "kiwi/invoke/invoke.hpp"
#include "lci.hpp"

namespace kiwi::ep {

namespace internal {

class ep_runtime_ref_t {
public:
    explicit ep_runtime_ref_t(std::string requested_device_name = {});
    ~ep_runtime_ref_t();

    ep_runtime_ref_t(const ep_runtime_ref_t&) = delete;
    ep_runtime_ref_t& operator=(const ep_runtime_ref_t&) = delete;
};

// Thread-local EP proxy state. EP callbacks use these values so EP owns its
// complete comm/proxy stack independently from generic P2P.
inline thread_local lci::device_t tl_device;
inline thread_local int tl_device_idx = -1;
inline thread_local int tl_proxy_thread_idx = -1;

struct ep_heap_device_entry_t {
    lci::mr_t mr;
    std::vector<lci::rmr_t> rmrs;  // indexed by rank
};

struct ep_heap_entry_t {
    std::vector<ep_heap_device_entry_t> device_entries;
};

using ep_heap_table_t = std::vector<ep_heap_entry_t>;

} // namespace internal

class ep_comm_context_t {
public:
    explicit ep_comm_context_t(int num_endpoints = 1, int num_devices = 1,
                               std::string requested_device_name = {},
                               size_t queue_capacity = 2048);
    ~ep_comm_context_t();

    ep_comm_context_t(const ep_comm_context_t&) = delete;
    ep_comm_context_t& operator=(const ep_comm_context_t&) = delete;

    ep_comm_device_context_t get_device_context(uint8_t all2all_cb_id);

    int alloc_symmetric(void* buf, size_t size);
    // Must be called after join_proxy_threads(); raw-RDMA proxy devices drain
    // their own CQs while stopping and cannot be polled concurrently here.
    void deregister_symmetric(int heap_id);

    void spawn_proxy_threads(int num_threads = -1);
    void join_proxy_threads();
    void publish_active_queue_count(int active_count);

    template <typename Desc, typename Fn>
    uint8_t register_callback(Fn&& cb) {
        return gc_context_->register_callback(
            std::function<void(Desc)>(std::forward<Fn>(cb)));
    }

    lci::device_t get_device(int index = 0) const { return devices_[index]; }
    // Raw RCOMP proxy devices have LCI SHM disabled and no packet pool. Route
    // all high-level setup collectives through the runtime default device.
    lci::device_t get_setup_device() const {
        return lci::get_default_device();
    }
    int num_devices_count() const { return num_devices_; }

    const internal::ep_heap_table_t& heap_table() const { return heap_table_; }

private:
    void wait_drained();
    void proxy_thread_func(int thread_id, int num_proxy_threads,
                           int selected_cpu);

    internal::ep_runtime_ref_t runtime_;
    int num_queues_ = 0;
    int num_devices_ = 0;
    std::vector<lci::device_t> devices_;
    std::unique_ptr<kiwi::invoke::Context<ep_queue_t>> gc_context_;

    internal::ep_heap_table_t heap_table_;
    std::vector<std::thread> proxy_threads_;
    std::atomic<bool> stop_proxy_threads_{false};
    internal::active_queue_request_t active_queue_request_;

};

// Lightweight EP-owned LCI lifecycle guard for bindings/tests that need the LCI
// runtime alive without constructing a full ep_context_t.
class ep_runtime_guard_t {
public:
    explicit ep_runtime_guard_t(std::string requested_device_name = {});
    ~ep_runtime_guard_t();

    ep_runtime_guard_t(const ep_runtime_guard_t&) = delete;
    ep_runtime_guard_t& operator=(const ep_runtime_guard_t&) = delete;

private:
    internal::ep_runtime_ref_t runtime_;
};

int ep_current_proxy_thread_index();

} // namespace kiwi::ep
