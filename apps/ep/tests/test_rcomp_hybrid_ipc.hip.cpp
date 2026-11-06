// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "ep_context.hip.hpp"
#include "lci.hpp"

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(err));                    \
            abort();                                                      \
        }                                                                 \
    } while (0)

namespace {

constexpr int kIters = 8;
constexpr int kWordsPerPayload = 16;

__host__ __device__ uint64_t expected_word(int rank, int iter, int word) {
    return 0x6a09e667f3bcc909ULL
         ^ (static_cast<uint64_t>(rank) << 48)
         ^ (static_cast<uint64_t>(iter) << 24)
         ^ static_cast<uint64_t>(word);
}

__global__ void rcomp_hybrid_ipc_proof_kernel(
        kiwi::ep::ep_gcontext_t gctx, uint64_t* result) {
    const int lane = threadIdx.x % kiwi::ep::EP_WAVE_SIZE;
    const int peer = 1 - gctx.rank_me;

    if (lane == 0) {
        result[0] = 0;  // payload mismatches
        result[1] = 0;  // observed IPC readiness flags
    }
    __syncthreads();

    if (!gctx.rcomp_peer_uses_ipc(peer)) {
        if (lane == 0) {
            result[0] = ~uint64_t{0};
        }
        return;
    }

    for (int iter = 0; iter < kIters; ++iter) {
        uint64_t* peer_payload = reinterpret_cast<uint64_t*>(
            gctx.rcomp_peer_dispatch_recv_slot(peer, iter));
        for (int word = lane; word < kWordsPerPayload;
             word += kiwi::ep::EP_WAVE_SIZE) {
            kiwi::ep::ep_store_system_scope(
                &peer_payload[word],
                expected_word(gctx.rank_me, iter, word));
        }

        // The per-slot IPC flag is published only after every participating
        // lane has drained its payload stores and synchronized.
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if (lane == 0) {
            const size_t flag_index =
                gctx.payload_readiness_flag_index(gctx.rank_me, iter);
            kiwi::ep::payload_readiness_flag_publish(
                gctx.peer_dispatch_readiness_flags(peer) + flag_index);
        }
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
    }

    if (lane == 0) {
        uint64_t mismatches = 0;
        uint64_t observed_flags = 0;
        for (int iter = 0; iter < kIters; ++iter) {
            const size_t flag_index =
                gctx.payload_readiness_flag_index(peer, iter);
            uint8_t* local_flag =
                gctx.dispatch_readiness_flags() + flag_index;
            kiwi::ep::payload_readiness_flag_wait(local_flag);

            const uint64_t* local_payload =
                reinterpret_cast<const uint64_t*>(
                    gctx.dispatch_recv_slot(peer, iter));
            for (int word = 0; word < kWordsPerPayload; ++word) {
                const uint64_t got = kiwi::ep::ep_load<true>(
                    &local_payload[word]);
                const uint64_t expected =
                    expected_word(peer, iter, word);
                if (got != expected) {
                    ++mismatches;
                    if (mismatches <= 4) {
                        printf("[Rank %d] IPC proof iter=%d word=%d "
                               "got=0x%llx expected=0x%llx\n",
                               gctx.rank_me, iter, word,
                               static_cast<unsigned long long>(got),
                               static_cast<unsigned long long>(expected));
                    }
                }
            }
            kiwi::ep::payload_readiness_flag_reset(local_flag);
            ++observed_flags;
        }
        result[0] = mismatches;
        result[1] = observed_flags;
    }
}

} // namespace

int main() {
    int num_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    const char* local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!local_rank_env) local_rank_env = getenv("MPI_LOCALRANKID");
    if (!local_rank_env) local_rank_env = getenv("SLURM_LOCALID");
    if (!local_rank_env) local_rank_env = getenv("OMPI_COMM_WORLD_RANK");
    const int local_rank = local_rank_env ? atoi(local_rank_env) : 0;
    HIP_CHECK(hipSetDevice(local_rank % num_gpus));

    kiwi::ep::ep_runtime_guard_t lci_guard;
    const int rank = lci::get_rank_me();
    const int nranks = lci::get_rank_n();
    if (nranks != 2) {
        fprintf(stderr,
                "test_rcomp_hybrid_ipc requires exactly 2 ranks (got %d)\n",
                nranks);
        return 1;
    }

    kiwi::ep::config_t cfg;
    cfg.world_size = nranks;
    cfg.num_experts = nranks * 2;
    cfg.model_hidden_dim = 64;
    cfg.topk = 1;
    cfg.max_num_inp_token_per_rank = kIters;
    cfg.num_blocks = 1;
    cfg.num_proxy_threads = 1;
    cfg.rcomp_backend = kiwi::ep::rcomp_backend_t::HYBRID;

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    bool pass = true;
    {
        kiwi::ep::ep_context_t ctx(cfg);
        ctx.spawn_proxy_threads();
        const auto gctx = ctx.get_device_context();
        const int peer = 1 - rank;
        const auto& peer_is_ipc = ctx.rcomp_peer_is_ipc();
        if (peer_is_ipc.size() != static_cast<size_t>(nranks)
            || peer_is_ipc[static_cast<size_t>(peer)] == 0
            || gctx.rcomp_peer_is_ipc == nullptr) {
            fprintf(stderr,
                    "[Rank %d] hybrid IPC route was not initialized for peer %d\n",
                    rank, peer);
            pass = false;
        }

        uint64_t* result = nullptr;
        HIP_CHECK(hipMallocManaged(&result, 2 * sizeof(uint64_t)));
        result[0] = 0;
        result[1] = 0;

        ctx.reset(stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        lci::barrier();

        rcomp_hybrid_ipc_proof_kernel<<<1, kiwi::ep::EP_WAVE_SIZE, 0, stream>>>(
            gctx, result);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        lci::barrier();

        const bool proof_ok =
            result[0] == 0
            && result[1] == static_cast<uint64_t>(kIters);
        printf("[Rank %d/%d] rcomp_hybrid_ipc: %s "
               "(mismatches=%llu flags=%llu expected=%d)\n",
               rank, nranks, proof_ok ? "PASSED" : "FAILED",
               static_cast<unsigned long long>(result[0]),
               static_cast<unsigned long long>(result[1]), kIters);
        pass = proof_ok && pass;

        ctx.join_proxy_threads();
        HIP_CHECK(hipFree(result));
    }

    // Keep the global LCI runtime alive until both ranks have closed imported
    // mappings and released their EP contexts.
    lci::barrier();
    return pass ? 0 : 1;
}
