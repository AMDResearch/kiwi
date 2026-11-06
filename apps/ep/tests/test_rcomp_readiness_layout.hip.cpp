// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <iostream>

#include "ep_context.hip.hpp"

namespace {

bool require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "test_rcomp_readiness_layout: " << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    bool passed = true;

    kiwi::ep::config_t cfg;
    cfg.world_size = 256;
    cfg.max_num_inp_token_per_rank = 256;
    cfg.rcomp_backend = kiwi::ep::rcomp_backend_t::HYBRID;

    const size_t flag_count = cfg.payload_readiness_flag_count();
    const size_t dispatch_flags_offset =
        cfg.dispatch_readiness_flags_offset();
    const size_t combine_flags_offset =
        cfg.combine_readiness_flags_offset();
    const kiwi::ep::ep_device_workspace_layout_t device_workspace_layout(cfg);

    passed &= require(
        flag_count == size_t{1} << 16,
        "maximum readiness layout must contain 65536 flags");
    passed &= require(
        kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE == 64,
        "readiness flags must be padded to a 64 B cache line");
    passed &= require(
        dispatch_flags_offset
            == kiwi::ep::align_up(
                cfg.combine_recv_offset() + cfg.combine_recv_size(),
                kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE),
        "dispatch readiness flags must follow the payload layout");
    passed &= require(
        combine_flags_offset
            == dispatch_flags_offset
                + flag_count * kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE,
        "combine readiness flags must follow dispatch readiness flags, "
        "one cache line per flag");
    passed &= require(
        cfg.protocol_epoch_offset()
            == kiwi::ep::align_up(
                combine_flags_offset
                    + flag_count * kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE,
                alignof(uint64_t)),
        "protocol epoch must follow hybrid readiness flags");
    passed &= require(
        cfg.total_heap_size()
            == cfg.barrier_alltoall_ingress_offset()
                + cfg.barrier_alltoall_region_size(),
        "hybrid heap must retain readiness and collective regions");
    passed &= require(
        device_workspace_layout.barrier_counter_offset
            < device_workspace_layout.all2all_counts_recv_offset
        && device_workspace_layout.all2all_counts_recv_offset
            < device_workspace_layout.get_num_bytes(),
        "regular device control state must stay outside payload readiness");

    kiwi::ep::config_t rdma_only = cfg;
    rdma_only.rcomp_backend = kiwi::ep::rcomp_backend_t::RDMA_ONLY;
    passed &= require(
        rdma_only.dispatch_readiness_flags_offset()
            == dispatch_flags_offset
        && rdma_only.combine_readiness_flags_offset()
            == combine_flags_offset,
        "RDMA-only and hybrid payload paths must share readiness layout");

    kiwi::ep::ep_gcontext_t gctx{};
    gctx.max_num_inp_token_per_rank = cfg.max_num_inp_token_per_rank;
    const size_t stride = kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE;
    passed &= require(gctx.payload_readiness_flag_index(/*source_rank=*/0,
                                                        /*source_local_slot=*/0)
                          == 0,
                      "flattened readiness index must start at zero");
    passed &= require(gctx.payload_readiness_flag_index(/*source_rank=*/0,
                                                        /*source_local_slot=*/255)
                          == 255 * stride,
                      "flattened readiness index must preserve local slots, "
                      "scaled by the cache-line stride");
    passed &= require(gctx.payload_readiness_flag_index(/*source_rank=*/1,
                                                        /*source_local_slot=*/0)
                          == 256 * stride,
                      "flattened readiness index must advance by rank "
                      "stride, scaled by the cache-line stride");
    passed &= require(gctx.payload_readiness_flag_index(
                          /*source_rank=*/255,
                          /*source_local_slot=*/255)
                          == (flag_count - 1) * stride,
        "maximum valid readiness index must be 65535 cache lines");
    passed &= require(
        gctx.payload_readiness_flag_index(/*source_rank=*/0,
                                          /*source_local_slot=*/1)
            - gctx.payload_readiness_flag_index(/*source_rank=*/0,
                                                /*source_local_slot=*/0)
            == stride,
        "adjacent readiness flags must be exactly one cache line apart");

    kiwi::ep::config_t oversized = cfg;
    oversized.world_size = 257;
    passed &= require(
        oversized.payload_readiness_flag_count() > flag_count,
        "readiness layout must not be limited by the removed 16-bit tag");
    gctx.max_num_inp_token_per_rank =
        oversized.max_num_inp_token_per_rank;
    passed &= require(
        gctx.payload_readiness_flag_index(/*source_rank=*/256,
                                          /*source_local_slot=*/0)
            == flag_count * stride,
        "readiness indices above 65535 cache lines must remain "
        "representable");
    return passed ? 0 : 1;
}
