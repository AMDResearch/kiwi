// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#endif
#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_warp_sync_functions.h>
#endif
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

constexpr int TEST_ALLTOALL_ROUNDS = 4;
constexpr int TEST_BARRIER_ROUNDS = 8;

__host__ __device__ uint64_t test_alltoall_count(
        int round, int src_rank, int dest_rank) {
    switch (round) {
    case 0:
        return static_cast<uint64_t>(src_rank) * 100000ULL
             + static_cast<uint64_t>(dest_rank) + 17ULL;
    case 1:
        return ((src_rank + dest_rank) & 1) == 0
            ? 0ULL
            : static_cast<uint64_t>(src_rank + dest_rank + 1);
    case 2:
        return dest_rank == 0
            ? static_cast<uint64_t>((src_rank + 1) * 1009)
            : static_cast<uint64_t>(src_rank == dest_rank);
    default:
        return static_cast<uint32_t>(
            (static_cast<uint32_t>(src_rank + 1) * 2654435761U)
            ^ (static_cast<uint32_t>(dest_rank + 3) * 2246822519U));
    }
}

__device__ inline void ep_test_pause() {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__)
    __builtin_amdgcn_s_sleep(1);
#else
    asm volatile("" ::: "memory");
#endif
}

__device__ inline uint64_t ep_test_counter_load(const uint64_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_RELAXED);
}

// The production barrier is an execution rendezvous. Payload visibility is a
// separate caller responsibility, so this test exercises staggered repeated
// epochs without coupling the barrier to an unrelated communication PUT.
__device__ inline int ep_test_barrier_protocol(
        kiwi::ep::ep_gcontext_t& gctx, uint64_t& failure_detail) {
    const int lane = threadIdx.x % kiwi::ep::EP_WAVE_SIZE;
    int passed = 1;
    uint64_t detail = 0;
    auto handle = gctx.comm.get_handle();
    for (int round = 0; round < TEST_BARRIER_ROUNDS; ++round) {
        const int delay =
            ((round * 5 + gctx.rank_me * 3) % 11) * 256;
        for (int i = 0; i < delay; ++i) {
            ep_test_pause();
        }

        const uint32_t epoch = kiwi::ep::ep_protocol_epoch_begin(gctx);
        const auto ticket = kiwi::ep::post_barrier(gctx, handle);
        kiwi::ep::wait_barrier(gctx, ticket);

        if (lane == 0) {
            const uint64_t completed =
                ep_test_counter_load(gctx.device_workspace.barrier_counter());
            if (completed != epoch && passed != 0) {
                passed = 0;
                detail =
                    (static_cast<uint64_t>(round) << 48)
                    | completed;
            }

            for (int src_rank = 0; src_rank < gctx.world_size;
                 ++src_rank) {
                const int source_node =
                    src_rank / gctx.count_alltoall_local_size;
                const int source_lane =
                    src_rank % gctx.count_alltoall_local_size;
                const size_t cell =
                    static_cast<size_t>(source_node)
                        * gctx.count_alltoall_local_size
                    + gctx.count_alltoall_lane_idx;
                const uint64_t packed =
                    kiwi::ep::ep_load_system_scope<uint64_t>(
                        &gctx.alltoall_peer_ingress(
                            source_lane,
                            gctx.barrier_alltoall_ingress_epoch_offset(
                                epoch))[cell]);
                const uint32_t observed_epoch =
                    static_cast<uint32_t>(packed >> 32);
                if (observed_epoch != epoch && passed != 0) {
                    passed = 0;
                    detail =
                        (static_cast<uint64_t>(round) << 48)
                        | (static_cast<uint64_t>(src_rank) << 32)
                        | observed_epoch;
                }
            }
        }
        passed = __shfl(passed, 0);
        detail = __shfl(detail, 0);
    }

    failure_detail = detail;
    return passed;
}

} // namespace

__global__ void ep_collective_primitives_kernel(kiwi::ep::ep_gcontext_t gctx,
                                                uint64_t* out) {
    const int lane = threadIdx.x % kiwi::ep::EP_WAVE_SIZE;

    uint64_t all2all_ok = 1;
    uint64_t all2all_bad_src = 0;
    uint64_t all2all_bad_value = 0;
    uint64_t all2all_bad_round = 0;
    volatile uint64_t* const all2all_recv =
        reinterpret_cast<volatile uint64_t*>(
            gctx.device_workspace.all2all_counts_recv());
    uint32_t* const all2all_send = reinterpret_cast<uint32_t*>(
        gctx.device_workspace.dispatch_dest_token_counts_snapshot());
    {
        // Reuse one endpoint handle for each all-to-all and its reuse barrier
        // so graph-style rounds preserve endpoint state in one place.
        auto handle = gctx.comm.get_handle();
        for (int round = 0; round < TEST_ALLTOALL_ROUNDS; ++round) {
            const int delay =
                ((round * 3 + gctx.rank_me * 5) % 7) * 256;
            for (int i = 0; i < delay; ++i) {
                ep_test_pause();
            }
            if (lane == 0) {
                for (int dest = 0; dest < gctx.world_size; ++dest) {
                    all2all_send[dest] = static_cast<uint32_t>(
                        test_alltoall_count(round, gctx.rank_me, dest));
                }
            }
            __threadfence_system();
            const kiwi::ep::ep_all2all_ticket_t all2all_ticket =
                kiwi::ep::post_all2all(gctx, handle, all2all_send);
            kiwi::ep::wait_all2all(
                gctx, all2all_ticket,
                gctx.device_workspace.all2all_counts_recv());
            if (lane == 0) {
                for (int src = 0; src < gctx.world_size; ++src) {
                    const uint64_t got =
                        __atomic_load_n(&all2all_recv[src], __ATOMIC_ACQUIRE);
                    const uint64_t expected =
                        test_alltoall_count(round, src, gctx.rank_me);
                    if (got != expected && all2all_ok != 0ULL) {
                        all2all_ok = 0;
                        all2all_bad_src = static_cast<uint64_t>(src);
                        all2all_bad_value = got;
                        all2all_bad_round = static_cast<uint64_t>(round);
                    }
                }
                for (int dest = 0; dest < gctx.world_size; ++dest) {
                    const uint64_t got =
                        __atomic_load_n(&all2all_send[dest], __ATOMIC_ACQUIRE);
                    const uint64_t expected =
                        test_alltoall_count(round, gctx.rank_me, dest);
                    if (got != expected && all2all_ok != 0ULL) {
                        all2all_ok = 0;
                        all2all_bad_src = static_cast<uint64_t>(
                            gctx.world_size + dest);
                        all2all_bad_value = got;
                        all2all_bad_round = static_cast<uint64_t>(round);
                    }
                }
            }
            const auto reuse_barrier =
                kiwi::ep::post_barrier(gctx, handle);
            kiwi::ep::wait_barrier(gctx, reuse_barrier);
        }
    }
    all2all_ok = __shfl(all2all_ok, 0);
    all2all_bad_src = __shfl(all2all_bad_src, 0);
    all2all_bad_value = __shfl(all2all_bad_value, 0);
    all2all_bad_round = __shfl(all2all_bad_round, 0);

    uint64_t barrier_before = 0;
    if (lane == 0) {
        barrier_before =
            ep_test_counter_load(gctx.device_workspace.barrier_counter());
    }
    barrier_before = __shfl(barrier_before, 0);

    uint64_t barrier_protocol_detail = 0;
    const int barrier_protocol_ok =
        ep_test_barrier_protocol(gctx, barrier_protocol_detail);

    if (lane == 0) {
        const uint64_t barrier_after =
            ep_test_counter_load(gctx.device_workspace.barrier_counter());
        out[0] = barrier_before;
        out[1] = barrier_after;
        out[2] = barrier_after >= barrier_before + 1ULL ? 1ULL : 0ULL;
        out[3] = static_cast<uint64_t>(barrier_protocol_ok);
        out[4] = barrier_protocol_detail;
        out[5] = all2all_ok;
        out[6] = all2all_bad_src;
        out[7] = all2all_bad_value;
        out[8] = all2all_bad_round;
    }
}

int main() {
    int num_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) {
        fprintf(stderr, "test_ep_collective_primitives requires a HIP GPU\n");
        return 1;
    }
    const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!lr) lr = getenv("OMPI_COMM_WORLD_RANK");
    const int local_rank = lr ? atoi(lr) : 0;
    HIP_CHECK(hipSetDevice(local_rank % num_gpus));

    kiwi::ep::ep_runtime_guard_t lci_guard;

    const int rank = lci::get_rank_me();
    const int nranks = lci::get_rank_n();
    if (nranks < 2) {
        fprintf(stderr,
                "test_ep_collective_primitives requires at least 2 ranks (got %d)\n",
                nranks);
        return 1;
    }

    kiwi::ep::config_t cfg;
    cfg.world_size = nranks;
    cfg.num_experts = nranks * 2;
    cfg.model_hidden_dim = 8;
    cfg.topk = 1;
    cfg.max_num_inp_token_per_rank = 4;
    cfg.num_blocks = 1;
    cfg.num_proxy_threads = 1;
    cfg.num_devices = 1;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    kiwi::ep::ep_context_t ctx(cfg);
    ctx.spawn_proxy_threads();
    kiwi::ep::ep_gcontext_t gctx = ctx.get_device_context();
    ctx.reset(stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    uint64_t* d_out = nullptr;
    HIP_CHECK(hipMallocManaged(&d_out, 9 * sizeof(uint64_t)));
    for (int i = 0; i < 9; ++i) d_out[i] = 0;

    lci::barrier();
    ep_collective_primitives_kernel<<<1, kiwi::ep::EP_WAVE_SIZE, 0, stream>>>(
        gctx, d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    lci::barrier();

    bool pass = true;
    if (d_out[2] != 1ULL) {
        fprintf(stderr,
                "[Rank %d] barrier counter did not advance monotonically: before=%llu after=%llu\n",
                rank, static_cast<unsigned long long>(d_out[0]),
                static_cast<unsigned long long>(d_out[1]));
        pass = false;
    }
    if (d_out[5] != 1ULL) {
        const int bad_index = static_cast<int>(d_out[6]);
        if (bad_index < nranks) {
            const uint64_t expected_all2all = test_alltoall_count(
                static_cast<int>(d_out[8]), bad_index, rank);
            fprintf(stderr,
                    "[Rank %d] all2all count mismatch round=%llu from src=%d "
                    "got=%llu expected=%llu\n",
                    rank, static_cast<unsigned long long>(d_out[8]), bad_index,
                    static_cast<unsigned long long>(d_out[7]),
                    static_cast<unsigned long long>(expected_all2all));
        } else {
            const int bad_dest = bad_index - nranks;
            const uint64_t expected_send = test_alltoall_count(
                static_cast<int>(d_out[8]), rank, bad_dest);
            fprintf(stderr,
                    "[Rank %d] all2all mutated send count round=%llu dest=%d "
                    "got=%llu expected=%llu\n",
                    rank, static_cast<unsigned long long>(d_out[8]), bad_dest,
                    static_cast<unsigned long long>(d_out[7]),
                    static_cast<unsigned long long>(expected_send));
        }
        pass = false;
    }
    if (d_out[3] != 1ULL) {
        fprintf(stderr,
                "[Rank %d] execution barrier protocol failed: detail=0x%llx\n",
                rank, static_cast<unsigned long long>(d_out[4]));
        pass = false;
    }

    const uint64_t barrier_before = d_out[0];
    const uint64_t barrier_after = d_out[1];
    const uint64_t barrier_protocol_ok = d_out[3];
    const uint64_t barrier_protocol_detail = d_out[4];

    ctx.join_proxy_threads();
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipStreamDestroy(stream));

    printf("[Rank %d/%d] test_ep_collective_primitives hybrid: %s "
           "(barrier=%llu->%llu protocol_ok=%llu protocol_detail=%llu)\n",
           rank, nranks,
           pass ? "PASSED" : "FAILED",
           static_cast<unsigned long long>(barrier_before),
           static_cast<unsigned long long>(barrier_after),
           static_cast<unsigned long long>(barrier_protocol_ok),
           static_cast<unsigned long long>(barrier_protocol_detail));
    return pass ? 0 : 1;
}
