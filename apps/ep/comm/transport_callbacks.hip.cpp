// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "comm/transport_descriptors.hip.hpp"
#include "comm/context.hip.hpp"
#include "comm/proxy_network.hip.hpp"
#include "ep_context.hip.hpp"

namespace kiwi::ep {
namespace internal {
namespace {

void post_raw_payload_and_readiness(
        int rank, void* local_buffer, size_t size, lci::mr_t mr,
        uintptr_t remote_disp, uintptr_t readiness_disp, lci::rmr_t rmr) {
    // Both PUTs use the device default endpoint, hence the same rank/QP. Do
    // not leave this callback until the payload is posted and its later
    // readiness signal is posted, preserving payload-before-signal ordering.
    lci::error_t error;
    do {
        error = lci::net_post_put_x(
                    rank, local_buffer, size, mr, remote_disp, rmr)
                    .device(internal::tl_device)();
        if (error.is_retry()) {
            internal::proxy_network_progress();
        }
    } while (error.is_retry());
    internal::proxy_network_record_posted_write(error);

    uint8_t ready = 1;
    do {
        error = lci::net_post_puts_x(
                    rank, &ready, sizeof(ready), readiness_disp, rmr)
                    .device(internal::tl_device)();
        if (error.is_retry()) {
            internal::proxy_network_progress();
        }
    } while (error.is_retry());
    internal::proxy_network_record_injected_write(error);
}

void post_put_callback(post_put_cmd_desc_t desc) {
    auto& heap = desc.ep_context->heap_table()[desc.heap_id];
    auto& entry = heap.device_entries[internal::tl_device_idx];
    lci::rmr_t rmr = entry.rmrs[desc.rank];
    lci::mr_t mr = entry.mr;

    const size_t readiness_index =
        desc.ep_context->config().payload_readiness_flag_index(
            lci::get_rank_me(), desc.readiness_slot);
    const uintptr_t readiness_disp =
        desc.readiness_flags_offset + readiness_index * sizeof(uint8_t);
    post_raw_payload_and_readiness(
        desc.rank, reinterpret_cast<void*>(desc.local_buffer), desc.size, mr,
        desc.remote_disp, readiness_disp, rmr);
}

void post_put_multicast_callback(post_put_cmd_desc_multicast_t desc) {
    auto& heap = desc.ep_context->heap_table()[desc.heap_id];
    auto& entry = heap.device_entries[internal::tl_device_idx];
    lci::mr_t mr = entry.mr;

    for (int i = 0; i < desc.n; i++) {
        lci::rmr_t rmr = entry.rmrs[desc.ranks[i]];
        const size_t readiness_index =
            desc.ep_context->config().payload_readiness_flag_index(
                lci::get_rank_me(), desc.readiness_slots[i]);
        const uintptr_t readiness_disp =
            desc.readiness_flags_offset + readiness_index * sizeof(uint8_t);
        post_raw_payload_and_readiness(
            desc.ranks[i], reinterpret_cast<void*>(desc.local_buffer),
            desc.size, mr, desc.remote_disps[i], readiness_disp, rmr);
    }
}

} // namespace

void register_post_put_callbacks(ep_comm_context_t& comm_ctx,
                                   uint8_t& post_put_cb_id,
                                   uint8_t& post_put_multicast_cb_id) {
    post_put_cb_id = comm_ctx.register_callback<post_put_cmd_desc_t>(
        post_put_callback);

    post_put_multicast_cb_id =
        comm_ctx.register_callback<post_put_cmd_desc_multicast_t>(
        post_put_multicast_callback);
}

} // namespace internal
} // namespace kiwi::ep
