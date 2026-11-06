// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "ep_context.hip.hpp"
#include "lci.hpp"

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

static void fill_tokens(float* h_tokens, int num_tokens, int hidden_dim) {
    for (int t = 0; t < num_tokens; t++)
        for (int d = 0; d < hidden_dim; d++)
            h_tokens[t * hidden_dim + d] = static_cast<float>(t * 1000 + d);
}

static void fill_routing(int32_t* h_indices, float* h_weights,
                          int num_tokens, int topk, int num_experts) {
    for (int t = 0; t < num_tokens; t++)
        for (int k = 0; k < topk; k++) {
            h_indices[t * topk + k] = (t * 7 + k * 13) % num_experts;
            h_weights[t * topk + k] = 1.0f / topk;
        }
}

__global__ void verify_recv_kernel(kiwi::ep::ep_gcontext_t gctx,
                                   int hidden_dim,
                                   const int32_t* recv_counts,
                                   uint64_t* error_count) {
    int rank_me = gctx.rank_me;
    int world_size = gctx.world_size;

    // dispatch() already waited for the EP all2all count vector.
    // Validate the received slots using the snapshotted per-rank receive counts.
    for (int r = 0; r < world_size; r++) {
        if (r == rank_me) continue;
        int32_t count = recv_counts[r];
        for (int32_t i = 0; i < count; i++) {
            char* slot = gctx.dispatch_recv_slot(r, (int)i);

            // Read src_token_id
            int32_t src_token_id = *gctx.dispatch_layout.slot.src_id(slot);

            // Verify hidden data matches pattern: token[t][d] = t * 1000 + d
            float* data = reinterpret_cast<float*>(gctx.dispatch_layout.slot.hidden(slot));
            for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
                float expected = static_cast<float>(src_token_id * 1000 + d);
                if (fabsf(data[d] - expected) > 1e-4f)
                    atomicAdd(error_count, 1ULL);
            }

            // Verify expert indices match expected pattern
            int32_t* indices = gctx.dispatch_layout.slot.indices(slot);
            int topk = gctx.topk;
            if (threadIdx.x == 0) {
                for (int k = 0; k < topk; k++) {
                    int32_t expert_id = indices[k];
                    int expected_expert = (src_token_id * 7 + k * 13)
                                        % (gctx.num_experts_per_rank * gctx.world_size);
                    if (expert_id != expected_expert)
                        atomicAdd(error_count, 1ULL);
                }
            }
        }
    }
}

static constexpr const char* kImplName = "mori-like";

static void test_heap_layout(const kiwi::ep::config_t& cfg) {
    assert(cfg.dispatch_readiness_flags_offset()
           == kiwi::ep::align_up(
               cfg.combine_recv_offset() + cfg.combine_recv_size(),
               kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE));
    assert(cfg.combine_readiness_flags_offset()
           == cfg.dispatch_readiness_flags_offset()
              + cfg.payload_readiness_flag_count()
                    * kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE);
    assert(cfg.protocol_epoch_offset()
           == kiwi::ep::align_up(
               cfg.combine_readiness_flags_offset()
                   + cfg.payload_readiness_flag_count()
                         * kiwi::ep::KIWI_EP_READINESS_FLAG_STRIDE,
               alignof(uint64_t)));

    kiwi::ep::config_t rdma_only = cfg;
    rdma_only.rcomp_backend = kiwi::ep::rcomp_backend_t::RDMA_ONLY;
    assert(rdma_only.dispatch_readiness_flags_offset()
           == cfg.dispatch_readiness_flags_offset());
    assert(rdma_only.combine_readiness_flags_offset()
           == cfg.combine_readiness_flags_offset());

    const size_t collective_region =
        cfg.count_alltoall_region_size();
    assert(cfg.count_alltoall_egress_offset()
           == cfg.protocol_epoch_offset() + sizeof(uint64_t));
    assert(cfg.count_alltoall_ingress_offset()
           == cfg.count_alltoall_egress_offset()
              + collective_region);
    assert(cfg.barrier_alltoall_egress_offset()
           == cfg.count_alltoall_ingress_offset()
              + collective_region);
    assert(cfg.barrier_alltoall_region_size()
           == 2 * collective_region);
    assert(cfg.barrier_alltoall_ingress_offset()
           == cfg.barrier_alltoall_egress_offset()
              + cfg.barrier_alltoall_region_size());
    assert(cfg.total_heap_size()
           == cfg.barrier_alltoall_ingress_offset()
              + cfg.barrier_alltoall_region_size());

    kiwi::ep::config_t mixed = cfg;
    mixed.dispatch_elem_size = 1;
    mixed.combine_elem_size = 2;
    mixed.dispatch_scale_dim = 4;
    mixed.dispatch_scale_type_size = 1;
    mixed.combine_scale_dim = 7;
    mixed.combine_scale_type_size = 2;
    assert(mixed.dispatch_slot_indices_offset()
           == static_cast<size_t>(mixed.model_hidden_dim));
    assert(mixed.combine_slot_weights_offset()
           == static_cast<size_t>(mixed.model_hidden_dim) * 2);
    const size_t dispatch_unaligned =
        static_cast<size_t>(mixed.model_hidden_dim)
        + static_cast<size_t>(mixed.topk) * sizeof(int32_t)
        + static_cast<size_t>(mixed.topk) * sizeof(float)
        + static_cast<size_t>(mixed.dispatch_scale_dim)
              * mixed.dispatch_scale_type_size
        + sizeof(int32_t);
    const size_t combine_unaligned =
        static_cast<size_t>(mixed.model_hidden_dim) * 2
        + static_cast<size_t>(mixed.topk) * sizeof(float);
    assert(mixed.combine_slot_payload_size() == combine_unaligned);
    assert(mixed.combine_slot_indices_offset() == combine_unaligned);
    assert(mixed.combine_slot_scales_offset() == combine_unaligned);
    assert(mixed.combine_slot_src_id_offset() == combine_unaligned);
    assert(mixed.dispatch_slot_stride()
           == kiwi::ep::align_up(
               dispatch_unaligned, kiwi::ep::EP_SLOT_ALIGNMENT));
    assert(mixed.combine_slot_stride()
           == kiwi::ep::align_up(
               combine_unaligned, kiwi::ep::EP_SLOT_ALIGNMENT));
    assert(mixed.dispatch_slot_stride() < mixed.combine_slot_stride());
    assert(mixed.dispatch_recv_region_size()
           == static_cast<size_t>(mixed.max_recv_slots_per_rank())
              * mixed.dispatch_slot_stride());
    assert(mixed.combine_recv_region_size()
           == static_cast<size_t>(mixed.max_recv_slots_per_rank())
              * mixed.combine_slot_stride());
    assert(mixed.dispatch_recv_offset()
           == mixed.combine_staging_offset() + mixed.combine_staging_size());

    kiwi::ep::config_t staging_cfg = mixed;
    staging_cfg.world_size = 2;
    staging_cfg.topk = 8; // topk > world_size
    staging_cfg.max_num_inp_token_per_rank = 7;
    assert(staging_cfg.max_dispatch_staging_slots()
           == staging_cfg.max_num_inp_token_per_rank);
    assert(staging_cfg.route_metadata_entries()
           == static_cast<size_t>(staging_cfg.max_num_inp_token_per_rank)
              * staging_cfg.topk);
    assert(staging_cfg.max_combine_staging_slots()
           == staging_cfg.world_size * staging_cfg.max_num_inp_token_per_rank);
    assert(staging_cfg.max_recv_slots_total()
           == staging_cfg.world_size * staging_cfg.max_num_inp_token_per_rank);
    assert(staging_cfg.dispatch_staging_size()
           == static_cast<size_t>(staging_cfg.max_num_inp_token_per_rank)
              * staging_cfg.dispatch_slot_stride());
    assert(staging_cfg.combine_staging_size()
           == static_cast<size_t>(staging_cfg.world_size
                                  * staging_cfg.max_num_inp_token_per_rank)
              * staging_cfg.combine_slot_stride());

    const auto dispatch_layout = mixed.dispatch_buffer_layout();
    const auto combine_layout = mixed.combine_buffer_layout();
    assert(dispatch_layout.slot.stride == mixed.dispatch_slot_stride());
    assert(dispatch_layout.staging_offset == mixed.dispatch_staging_offset());
    assert(dispatch_layout.recv_offset == mixed.dispatch_recv_offset());
    assert(dispatch_layout.recv_region_size == mixed.dispatch_recv_region_size());
    assert(dispatch_layout.remote_disp(/*rank_me=*/1, /*recv_slot_idx=*/3)
           == mixed.dispatch_recv_from_rank_offset(1)
              + 3 * mixed.dispatch_slot_stride());
    assert(combine_layout.slot.stride == mixed.combine_slot_stride());
    assert(combine_layout.staging_offset == mixed.combine_staging_offset());
    assert(combine_layout.recv_offset == mixed.combine_recv_offset());
    assert(combine_layout.recv_region_size == mixed.combine_recv_region_size());
    assert(combine_layout.remote_disp(/*rank_me=*/1, /*recv_slot_idx=*/3)
           == mixed.combine_recv_from_rank_offset(1)
              + 3 * mixed.combine_slot_stride());
}

static bool test_dispatch(const kiwi::ep::config_t& cfg,
                          int num_tokens, int rank) {
    printf("[Rank %d] Testing %s dispatch with %d tokens...\n",
           rank, kImplName, num_tokens);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    kiwi::ep::ep_context_t ctx(cfg);
    ctx.spawn_proxy_threads();
    auto gctx = ctx.get_device_context();
    ctx.reset(stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    int total_elems = num_tokens * cfg.model_hidden_dim;
    int total_routing = num_tokens * cfg.topk;

    std::vector<float> h_tokens(total_elems);
    std::vector<int32_t> h_indices(total_routing);
    std::vector<float> h_weights(total_routing);
    fill_tokens(h_tokens.data(), num_tokens, cfg.model_hidden_dim);
    fill_routing(h_indices.data(), h_weights.data(), num_tokens, cfg.topk, cfg.num_experts);

    float* d_tokens; int32_t* d_indices; float* d_weights;
    HIP_CHECK(hipMalloc(&d_tokens, total_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_indices, total_routing * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_weights, total_routing * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_tokens, h_tokens.data(), total_elems * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_indices, h_indices.data(), total_routing * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_weights.data(), total_routing * sizeof(float), hipMemcpyHostToDevice));

    uint64_t* d_error_count;
    HIP_CHECK(hipMallocManaged(&d_error_count, sizeof(uint64_t)));
    *d_error_count = 0;

    lci::barrier();

    kiwi::ep::dispatch(ctx, gctx,
                      static_cast<const void*>(d_tokens),
                      d_indices, d_weights,
                      nullptr /* d_scales */,
                      num_tokens, stream);

    verify_recv_kernel<<<1, 64, 0, stream>>>(
        gctx, cfg.model_hidden_dim,
        ctx.device_workspace().dispatch_recv_counts(), d_error_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<int32_t> h_recv_counts(cfg.world_size + 1);
    ctx.copy_dispatch_recv_counts_to_host(h_recv_counts.data(), stream);

    std::vector<int32_t> h_src_token_pos(h_recv_counts[cfg.world_size]);
    if (!h_src_token_pos.empty()) {
        HIP_CHECK(hipMemcpy(h_src_token_pos.data(), gctx.d_recv_src_token_pos,
                            h_src_token_pos.size() * sizeof(int32_t),
                            hipMemcpyDeviceToHost));

        int flat = 0;
        for (int src_rank = 0; src_rank < cfg.world_size; src_rank++) {
            int count = h_recv_counts[src_rank];
            for (int slot = 0; slot < count; slot++, flat++) {
                char* recv_slot = gctx.dispatch_recv_slot(src_rank, slot);
                int32_t src_token_id = -1;
                HIP_CHECK(hipMemcpy(&src_token_id, gctx.dispatch_layout.slot.src_id(recv_slot),
                                    sizeof(int32_t), hipMemcpyDeviceToHost));
                int32_t expected =
                    src_rank * (cfg.world_size * cfg.max_num_inp_token_per_rank)
                    + src_token_id;
                if (h_src_token_pos[flat] != expected) {
                    fprintf(stderr,
                            "[Rank %d] %s src_token_pos[%d] mismatch: got %d expected %d\n",
                            rank, kImplName, flat, h_src_token_pos[flat], expected);
                    (*d_error_count)++;
                }
            }
        }
    }

    lci::barrier();

    bool pass = (*d_error_count == 0);
    if (!pass)
        fprintf(stderr, "[Rank %d] %s: %lu errors\n",
                rank, kImplName, (unsigned long)*d_error_count);

    ctx.join_proxy_threads();

    HIP_CHECK(hipFree(d_tokens));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_error_count));
    HIP_CHECK(hipStreamDestroy(stream));

    printf("[Rank %d] %s dispatch: %s\n",
           rank, kImplName, pass ? "PASSED" : "FAILED");
    return pass;
}

int main() {
    // Select GPU device based on local rank
    {
        int num_gpus = 0;
        HIP_CHECK(hipGetDeviceCount(&num_gpus));
        const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!lr) lr = getenv("OMPI_COMM_WORLD_RANK");
        int local_rank = lr ? atoi(lr) : 0;
        HIP_CHECK(hipSetDevice(local_rank % num_gpus));
    }

    kiwi::ep::ep_runtime_guard_t lci_guard;

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    if (nranks < 2) {
        fprintf(stderr, "test_dispatch requires at least 2 ranks (got %d)\n", nranks);
        return 1;
    }

    printf("[Rank %d/%d] test_dispatch: started\n", rank, nranks);

    kiwi::ep::config_t cfg;
    cfg.world_size = nranks;
    cfg.num_experts = nranks * 8;
    cfg.model_hidden_dim = 64;
    cfg.topk = std::min(kiwi::ep::MAX_TOPK_SUPPORTED, nranks + 2);
    cfg.max_num_inp_token_per_rank = 64;
    cfg.num_blocks = 8;
    cfg.num_proxy_threads = 8;
    cfg.num_devices = 4;

    test_heap_layout(cfg);

    bool all_pass = true;
    for (int num_tokens : {1, 8, 32}) {
        all_pass = test_dispatch(cfg, num_tokens, rank) && all_pass;
    }

    // Keep all ranks inside the LCI runtime until every rank has completed the
    // rank-local EP context/GPU teardown performed by test_dispatch().  Without
    // this post-teardown rendezvous a fast rank can return from main and
    // finalize the process-wide LCI runtime while a peer is still releasing
    // dispatch resources, which showed up as an intermittent peer SIGSEGV after
    // the successful dispatch output had already been printed.
    lci::barrier();

    printf("[Rank %d] test_dispatch: %s\n", rank, all_pass ? "ALL PASSED" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
