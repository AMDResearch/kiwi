// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <thread>
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

// Fill token data with a rank/round-specific pattern so multi-round tests
// catch stale payload reads instead of only proving no-hang.
static void fill_tokens(float* h_tokens, int rank, int round,
                        int num_tokens, int hidden_dim) {
    for (int t = 0; t < num_tokens; t++) {
        for (int d = 0; d < hidden_dim; d++) {
            h_tokens[t * hidden_dim + d] = static_cast<float>(
                rank * 100000 + round * 10000 + t * 1000 + d);
        }
    }
}

struct routing_coverage_t {
    int tokens_with_remote = 0;
    int tokens_all_local = 0;
    int tokens_with_duplicate_dest = 0;
    int tokens_with_multiple_remote_dests = 0;
};

static int expert_id_for_rank(int dest_rank, int expert_offset,
                              int num_experts_per_rank) {
    return dest_rank * num_experts_per_rank
         + (expert_offset % num_experts_per_rank);
}

// Deterministic routing pattern that explicitly covers WarpAccum and
// topk>world_size cases:
//   - token 0 mixes local, remote, and duplicate destinations (so the 1-token
//     case still exercises remote return slots)
//   - token 1 is all-local duplicate-destination
//   - token 2 is all remote_a duplicate-destination
//   - token 3+ cycles remote destinations, including multi-remote fanout when
//     world_size > 2
// Weights are non-uniform so output-weight summation cannot accidentally pass
// by treating duplicate top-k entries as identical halves.
static void fill_routing(int32_t* h_indices, float* h_weights,
                          int num_tokens, int topk, int num_experts,
                          int rank, int world_size, int round) {
    int num_experts_per_rank = num_experts / world_size;
    if (num_experts_per_rank < 2) {
        fprintf(stderr,
                "test_combine requires at least 2 experts per rank (got %d)\n",
                num_experts_per_rank);
        abort();
    }

    const int remote_a_rank = (rank + 1) % world_size;
    const int remote_b_rank = (world_size > 2) ? (rank + 2) % world_size
                                                : remote_a_rank;

    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < topk; k++) {
            int dest_rank = rank;
            // Keep each token-count case's coverage shape stable across
            // rounds (especially the 1-token case, which must always exercise
            // a remote return slot), while varying expert offsets and weights
            // below so multi-round tests still catch stale payloads.
            switch (t % 4) {
            case 0:
                if (k == 0) {
                    dest_rank = rank;
                } else if (k <= 2) {
                    dest_rank = remote_a_rank;
                } else {
                    dest_rank = (world_size > 2) ? remote_b_rank : rank;
                }
                break;
            case 1:
                dest_rank = rank;
                break;
            case 2:
                dest_rank = remote_a_rank;
                break;
            default:
                if (world_size > 2) {
                    dest_rank = (k % 3 == 0) ? remote_a_rank
                              : (k % 3 == 1) ? remote_b_rank
                                              : rank;
                } else {
                    dest_rank = (k % 2 == 0) ? remote_a_rank : rank;
                }
                break;
            }
            h_indices[t * topk + k] =
                expert_id_for_rank(dest_rank, t + k + round,
                                   num_experts_per_rank);
            h_weights[t * topk + k] =
                0.01f * static_cast<float>(t + 1)
              + static_cast<float>(k + 1) / static_cast<float>(topk + 1)
              + 0.001f * static_cast<float>(round + 1);
        }
    }
}

// CPU reference that mirrors current MORI-compatible WarpAccum semantics:
//   - Dedup by destination rank, including the local rank
//   - Ignore duplicate same-destination top-k entries that dispatch marks with
//     an out-of-range destination rank
//   - Sum one already-produced expert row per valid unique destination
// Expert computation is identity, so every returned hidden vector equals the
// original token hidden vector and weights are not applied to hidden data.
static void compute_expected(float* expected, const float* h_tokens,
                              const int32_t* h_indices,
                              int num_tokens, int topk, int num_experts,
                              int hidden_dim, int world_size) {
    int num_experts_per_rank = num_experts / world_size;
    memset(expected, 0, static_cast<size_t>(num_tokens) * hidden_dim * sizeof(float));

    for (int t = 0; t < num_tokens; t++) {
        int unique_dests[kiwi::ep::MAX_TOPK_SUPPORTED];
        int num_unique = 0;

        for (int k = 0; k < topk; k++) {
            int expert_id = h_indices[t * topk + k];
            int dest_rank = expert_id / num_experts_per_rank;

            int unique_idx = -1;
            for (int u = 0; u < num_unique; u++) {
                if (unique_dests[u] == dest_rank) {
                    unique_idx = u;
                    break;
                }
            }
            if (unique_idx == -1) {
                unique_dests[num_unique] = dest_rank;
                num_unique++;
            }
        }

        for (int u = 0; u < num_unique; u++) {
            for (int d = 0; d < hidden_dim; d++) {
                expected[t * hidden_dim + d] +=
                    h_tokens[t * hidden_dim + d];
            }
        }
    }
}

static void compute_expected_output_weights(float* expected,
                                             const int32_t* h_indices,
                                             const float* h_weights,
                                             int num_tokens, int topk,
                                             int num_experts, int world_size) {
    int num_experts_per_rank = num_experts / world_size;
    memset(expected, 0, static_cast<size_t>(num_tokens) * topk * sizeof(float));

    for (int t = 0; t < num_tokens; t++) {
        int unique_dests[kiwi::ep::MAX_TOPK_SUPPORTED];
        int num_unique = 0;

        for (int k = 0; k < topk; k++) {
            int expert_id = h_indices[t * topk + k];
            int dest_rank = expert_id / num_experts_per_rank;

            bool found = false;
            for (int u = 0; u < num_unique; u++) {
                if (unique_dests[u] == dest_rank) {
                    found = true;
                    break;
                }
            }
            if (found) {
                continue;
            }

            unique_dests[num_unique++] = dest_rank;
            // MORI returns the elementwise sum of the full weight vectors
            // associated with valid combine rows, not per-expert scalar
            // weights applied to hidden data.
            for (int kk = 0; kk < topk; kk++) {
                expected[t * topk + kk] += h_weights[t * topk + kk];
            }
        }
    }
}

static int validate_token_route_metadata(
        const int32_t* h_token_to_producer_rank,
        const int32_t* h_token_to_dispatch_recv_slot,
        const int32_t* h_indices,
        int num_tokens,
        int topk,
        int num_experts,
        int rank,
        int world_size,
        const int32_t* dest_token_counts,
        const char* impl_name,
        routing_coverage_t* coverage)
{
    constexpr int kMaxVerboseErrors = 10;

    int errors = 0;
    int num_experts_per_rank = num_experts / world_size;
    std::vector<int32_t> expected_dest_counts(world_size, 0);

    *coverage = {};

    for (int t = 0; t < num_tokens; t++) {
        int unique_dests[kiwi::ep::MAX_TOPK_SUPPORTED];
        int num_unique = 0;
        bool has_duplicate_dest = false;

        for (int k = 0; k < topk; k++) {
            int expert_id = h_indices[t * topk + k];
            int dest_rank = expert_id / num_experts_per_rank;
            const size_t route_idx = static_cast<size_t>(t) * topk + k;
            const int32_t producer_rank =
                h_token_to_producer_rank[route_idx];
            const int32_t dispatch_recv_slot =
                h_token_to_dispatch_recv_slot[route_idx];

            int unique_idx = -1;
            for (int u = 0; u < num_unique; u++) {
                if (unique_dests[u] == dest_rank) {
                    unique_idx = u;
                    break;
                }
            }

            if (unique_idx == -1) {
                if (producer_rank != dest_rank) {
                    if (errors < kMaxVerboseErrors) {
                        fprintf(stderr,
                                "[Rank %d] %s map producer_rank mismatch at "
                                "token %d k=%d: got %d, expected %d\n",
                                rank, impl_name, t, k, producer_rank,
                                dest_rank);
                    }
                    errors++;
                }

                if (dispatch_recv_slot < 0
                    || dispatch_recv_slot >= dest_token_counts[dest_rank]) {
                    if (errors < kMaxVerboseErrors) {
                        fprintf(stderr,
                                "[Rank %d] %s dispatch slot mismatch at token %d k=%d: "
                                "got slot %d, valid range [0, %d)\n",
                                rank, impl_name, t, k, dispatch_recv_slot,
                                dest_token_counts[dest_rank]);
                    }
                    errors++;
                }

                unique_idx = num_unique;
                unique_dests[num_unique] = dest_rank;
                expected_dest_counts[dest_rank]++;
                num_unique++;
            } else {
                has_duplicate_dest = true;
                if (producer_rank != world_size || dispatch_recv_slot != 0) {
                    if (errors < kMaxVerboseErrors) {
                        fprintf(stderr,
                                "[Rank %d] %s map duplicate sentinel mismatch "
                                "at token %d k=%d: got [%d, %d], expected "
                                "[%d, 0]\n",
                                rank, impl_name, t, k, producer_rank,
                                dispatch_recv_slot,
                                world_size);
                    }
                    errors++;
                }
            }
        }

        bool has_remote = false;
        int remote_unique = 0;
        for (int u = 0; u < num_unique; u++) {
            if (unique_dests[u] != rank) {
                has_remote = true;
                remote_unique++;
            }
        }
        if (!has_remote) {
            coverage->tokens_all_local++;
        }
        if (has_remote) {
            coverage->tokens_with_remote++;
        }
        if (has_duplicate_dest) {
            coverage->tokens_with_duplicate_dest++;
        }
        if (remote_unique > 1) {
            coverage->tokens_with_multiple_remote_dests++;
        }
    }

    for (int r = 0; r < world_size; r++) {
        if (dest_token_counts[r] != expected_dest_counts[r]) {
            if (errors < kMaxVerboseErrors) {
                fprintf(stderr,
                        "[Rank %d] %s dest_token_counts[%d] mismatch: got %d, "
                        "expected %d\n",
                        rank, impl_name, r, dest_token_counts[r],
                        expected_dest_counts[r]);
            }
            errors++;
        }
    }

    return errors;
}

static int validate_warpaccum_coverage(const routing_coverage_t& coverage,
                                       int rank,
                                       int world_size,
                                       const char* impl_name,
                                       bool require_full_coverage) {
    int errors = 0;
    if (require_full_coverage && coverage.tokens_all_local == 0) {
        fprintf(stderr,
                "[Rank %d] %s coverage missing all-local tokens\n",
                rank, impl_name);
        errors++;
    }
    if (coverage.tokens_with_remote == 0) {
        fprintf(stderr,
                "[Rank %d] %s coverage missing remote tokens\n",
                rank, impl_name);
        errors++;
    }
    if (coverage.tokens_with_duplicate_dest == 0) {
        fprintf(stderr,
                "[Rank %d] %s coverage missing duplicate-destination tokens\n",
                rank, impl_name);
        errors++;
    }
    if (require_full_coverage && world_size > 2
        && coverage.tokens_with_multiple_remote_dests == 0) {
        fprintf(stderr,
                "[Rank %d] %s coverage missing multi-dest remote tokens\n",
                rank, impl_name);
        errors++;
    }
    return errors;
}

static int validate_combine_return_slots(
        const kiwi::ep::ep_gcontext_t& gctx,
        const int32_t* h_token_to_producer_rank,
        const float* h_tokens,
        int hidden_dim,
        int num_tokens,
        int topk,
        int rank,
        const char* impl_name)
{
    constexpr int kMaxVerboseErrors = 10;

    int errors = 0;
    int checked = 0;
    int checked_remote = 0;

    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < topk; k++) {
            const int32_t producer_rank =
                h_token_to_producer_rank[static_cast<size_t>(t) * topk + k];
            if (producer_rank < 0 || producer_rank >= gctx.world_size) {
                continue;
            }

            bool already_checked = false;
            for (int prev_k = 0; prev_k < k; prev_k++) {
                const int32_t prev_producer_rank =
                    h_token_to_producer_rank[static_cast<size_t>(t) * topk
                                             + prev_k];
                if (prev_producer_rank == producer_rank) {
                    already_checked = true;
                    break;
                }
            }
            if (already_checked) {
                continue;
            }

            char* slot = gctx.combine_recv_slot(producer_rank, t);
            float hidden0 = 0.0f;
            HIP_CHECK(hipMemcpy(
                &hidden0, gctx.combine_layout.slot.hidden(slot),
                sizeof(float), hipMemcpyDeviceToHost));
            const float expected_hidden0 = h_tokens[t * hidden_dim];
            if (fabsf(hidden0 - expected_hidden0) > 1e-4f) {
                if (errors < kMaxVerboseErrors) {
                    fprintf(stderr,
                            "[Rank %d] %s combine return slot hidden[0] "
                            "mismatch for producer %d token %d: got %f "
                            "expected %f. This verifies source-token-addressed "
                            "routing without relying on retransmitted "
                            "metadata.\n",
                            rank, impl_name, producer_rank, t, hidden0,
                            expected_hidden0);
                }
                errors++;
            }

            checked++;
            if (producer_rank != rank) {
                checked_remote++;
            }
        }
    }

    if (checked == 0 || checked_remote == 0) {
        fprintf(stderr,
                "[Rank %d] %s combine return-slot assertion coverage "
                "insufficient: checked=%d remote=%d\n",
                rank, impl_name, checked, checked_remote);
        errors++;
    }

    return errors;
}

static constexpr const char* kImplName = "mori-like";

static const char* payload_mode_name() {
    return "rcomp";
}

static bool test_combine(const kiwi::ep::config_t& cfg,
                         int num_tokens, int rank, int rounds,
                         bool require_full_coverage) {
    const char* impl_name = kImplName;
    printf("[Rank %d] Testing %s combine/%s with %d tokens for %d rounds...\n",
           rank, impl_name, payload_mode_name(),
           num_tokens, rounds);

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

    float* d_tokens = nullptr;
    int32_t* d_indices = nullptr;
    float* d_weights = nullptr;
    float* d_output = nullptr;
    float* d_output_weights = nullptr;
    HIP_CHECK(hipMalloc(&d_tokens, total_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_indices, total_routing * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_weights, total_routing * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, total_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output_weights, total_routing * sizeof(float)));

    int errors = 0;
    const size_t route_capacity =
        static_cast<size_t>(cfg.max_num_inp_token_per_rank) * cfg.topk;

    for (int round = 0; round < rounds; round++) {
        ctx.reset(stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        fill_tokens(h_tokens.data(), rank, round, num_tokens,
                    cfg.model_hidden_dim);
        fill_routing(h_indices.data(), h_weights.data(), num_tokens,
                     cfg.topk, cfg.num_experts, rank, cfg.world_size, round);

        HIP_CHECK(hipMemcpyAsync(d_tokens, h_tokens.data(),
                                 total_elems * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_indices, h_indices.data(),
                                 total_routing * sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_weights, h_weights.data(),
                                 total_routing * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemsetAsync(d_output, 0,
                                 total_elems * sizeof(float), stream));
        HIP_CHECK(hipMemsetAsync(d_output_weights, 0,
                                 total_routing * sizeof(float), stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        lci::barrier();

        kiwi::ep::dispatch(ctx, gctx,
                          static_cast<const void*>(d_tokens), d_indices,
                          d_weights, nullptr, num_tokens, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        std::vector<int32_t> h_dest_token_counts(cfg.world_size);
        ctx.copy_dest_token_counts_to_host(h_dest_token_counts.data(), stream);
        std::vector<int32_t> h_token_to_producer_rank(route_capacity);
        std::vector<int32_t> h_token_to_dispatch_recv_slot(route_capacity);
        ctx.copy_token_route_metadata_to_host(
            h_token_to_producer_rank.data(),
            h_token_to_dispatch_recv_slot.data(),
            stream);

        routing_coverage_t coverage;
        int round_errors = validate_token_route_metadata(
            h_token_to_producer_rank.data(),
            h_token_to_dispatch_recv_slot.data(),
            h_indices.data(), num_tokens, cfg.topk, cfg.num_experts, rank,
            cfg.world_size, h_dest_token_counts.data(), impl_name, &coverage);
        round_errors += validate_warpaccum_coverage(coverage, rank,
                                                    cfg.world_size, impl_name,
                                                    require_full_coverage);
        printf("[Rank %d] %s/%s round %d routing coverage: "
               "local-only=%d remote=%d duplicate-dest=%d multi-dest=%d\n",
               rank, impl_name, payload_mode_name(),
               round, coverage.tokens_all_local, coverage.tokens_with_remote,
               coverage.tokens_with_duplicate_dest,
               coverage.tokens_with_multiple_remote_dests);

        // Deliberately skew rank arrival into combine() so faster ranks can
        // post before slower ranks enter the function.
        if (((rank + round) % 2) == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        kiwi::ep::combine(ctx, gctx,
                         gctx.d_recv_hidden,
                         (round % 2) == 0 ? nullptr : gctx.d_recv_weights,
                         nullptr,
                         d_indices,
                         static_cast<void*>(d_output),
                         d_output_weights,
                         num_tokens, stream);

        HIP_CHECK(hipStreamSynchronize(stream));
        lci::barrier();

        round_errors += validate_combine_return_slots(
            gctx, h_token_to_producer_rank.data(), h_tokens.data(),
            cfg.model_hidden_dim, num_tokens, cfg.topk, rank, impl_name);

        std::vector<float> h_output(total_elems);
        HIP_CHECK(hipMemcpy(h_output.data(), d_output,
                            total_elems * sizeof(float),
                            hipMemcpyDeviceToHost));
        std::vector<float> h_output_weights(total_routing);
        HIP_CHECK(hipMemcpy(h_output_weights.data(), d_output_weights,
                            total_routing * sizeof(float),
                            hipMemcpyDeviceToHost));

        std::vector<float> expected(total_elems);
        compute_expected(expected.data(), h_tokens.data(), h_indices.data(),
                         num_tokens, cfg.topk, cfg.num_experts,
                         cfg.model_hidden_dim, cfg.world_size);
        std::vector<float> expected_weights(total_routing);
        compute_expected_output_weights(expected_weights.data(),
                                        h_indices.data(), h_weights.data(),
                                        num_tokens, cfg.topk, cfg.num_experts,
                                        cfg.world_size);

        for (int i = 0; i < total_elems; i++) {
            float diff = fabsf(h_output[i] - expected[i]);
            float scale = fmaxf(fabsf(expected[i]), 1.0f);
            if (diff / scale > 1e-3f) {
                if (round_errors < 10) {
                    fprintf(stderr,
                            "[Rank %d] %s/%s round %d mismatch at "
                            "[%d/%d][%d/%d]: got %f, expected %f\n",
                            rank, impl_name,
                            payload_mode_name(),
                            round, i / cfg.model_hidden_dim, num_tokens,
                            i % cfg.model_hidden_dim, cfg.model_hidden_dim,
                            h_output[i], expected[i]);
                }
                round_errors++;
            }
        }
        for (int i = 0; i < total_routing; i++) {
            float diff = fabsf(h_output_weights[i] - expected_weights[i]);
            float scale = fmaxf(fabsf(expected_weights[i]), 1.0f);
            if (diff / scale > 1e-5f) {
                if (round_errors < 10) {
                    fprintf(stderr,
                            "[Rank %d] %s/%s round %d output weight "
                            "mismatch at token %d k=%d: got %f, "
                            "expected %f\n",
                            rank, impl_name,
                            payload_mode_name(),
                            round, i / cfg.topk, i % cfg.topk,
                            h_output_weights[i], expected_weights[i]);
                }
                round_errors++;
            }
        }

        errors += round_errors;
        printf("[Rank %d] %s/%s round %d: %s (%d errors)\n",
               rank, impl_name, payload_mode_name(),
               round, round_errors == 0 ? "PASSED" : "FAILED",
               round_errors);
    }

    // Keep every rank's EP context alive until peers have finished all local
    // validation for this test case.  The rcomp payload path intentionally
    // waits only for receiver-side remote completions in production; tests must
    // not let a faster rank tear down EP/LCI resources while a slower peer is
    // still completing the same combine case.
    lci::barrier();

    ctx.join_proxy_threads();

    HIP_CHECK(hipFree(d_tokens));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_output_weights));
    HIP_CHECK(hipStreamDestroy(stream));

    bool pass = (errors == 0);
    printf("[Rank %d] %s combine/%s: %s (%d errors)\n",
           rank, impl_name, payload_mode_name(),
           pass ? "PASSED" : "FAILED", errors);
    return pass;
}

int main(int argc, char** argv) {
    std::string rcomp_backend = "rdma-only";
    if (argc == 3 && std::string(argv[1]) == "--rcomp-backend") {
        rcomp_backend = argv[2];
    } else if (argc != 1) {
        fprintf(stderr,
                "usage: test_combine [--rcomp-backend rdma-only|hybrid]\n");
        return 1;
    }
    if (rcomp_backend != "rdma-only" && rcomp_backend != "hybrid") {
        fprintf(stderr, "unsupported --rcomp-backend '%s'\n",
                rcomp_backend.c_str());
        return 1;
    }

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
        fprintf(stderr, "test_combine requires at least 2 ranks (got %d)\n", nranks);
        return 1;
    }

    printf("[Rank %d/%d] test_combine: started\n", rank, nranks);

    kiwi::ep::config_t cfg;
    cfg.world_size = nranks;
    cfg.num_experts = nranks * 8;
    cfg.model_hidden_dim = 64;
    cfg.topk = std::min(kiwi::ep::MAX_TOPK_SUPPORTED, nranks + 2);
    cfg.max_num_inp_token_per_rank = 64;
    cfg.num_blocks = 4;
    cfg.rcomp_backend = rcomp_backend == "hybrid"
        ? kiwi::ep::rcomp_backend_t::HYBRID
        : kiwi::ep::rcomp_backend_t::RDMA_ONLY;

    constexpr int kRounds = 4;
    bool all_pass = true;
    for (int num_tokens : {1, 8, 32}) {
        const bool require_full_coverage = (num_tokens >= 8);
        all_pass = test_combine(cfg, num_tokens, rank, kRounds,
                                require_full_coverage) && all_pass;
    }

    // Mirror test_dispatch's teardown rendezvous: all rank-local EP contexts
    // have been destroyed by this point, but the process-wide LCI runtime is
    // still alive.  This prevents a fast rank from finalizing LCI while a peer
    // is still completing its final test teardown.
    lci::barrier();

    printf("[Rank %d] test_combine: %s\n", rank, all_pass ? "ALL PASSED" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
