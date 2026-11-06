// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <thread>
#include <string>
#include <vector>
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

static float token_value(int rank, int replay, int token, int dim) {
    return static_cast<float>(rank * 100000 + replay * 10000
                              + token * 100 + dim);
}

static float weight_value(int rank, int replay, int token, int k) {
    return 0.125f * static_cast<float>(k + 1)
         + 0.01f * static_cast<float>(token + 1)
         + 0.001f * static_cast<float>(rank + 1)
         + 0.0001f * static_cast<float>(replay + 1);
}

static int dest_rank_for(int src_rank, int replay, int token, int k,
                         int world_size) {
    const int pattern = (src_rank + replay + token) % 6;
    switch (pattern) {
    case 0:
        // Local-only duplicate-destination token.
        return src_rank;
    case 1:
        // Ring fan-out, including local for k==0 and remote ranks after that.
        return (src_rank + k) % world_size;
    case 2:
        // All entries target one remote rank; dispatch should deduplicate.
        return (src_rank + 1) % world_size;
    case 3:
        // Duplicate one remote destination, then route to a second destination.
        return (src_rank + 1 + (k >= 2 ? 1 : 0)) % world_size;
    case 4:
        // Replay-dependent hot-rank skew.  All sources can pile onto one rank.
        return replay % world_size;
    default:
        // Deterministic mixed pattern with possible duplicates when world_size
        // is small and multiple remote destinations when it is large.
        return (src_rank + token + replay + 2 * k) % world_size;
    }
}

static int unique_dest_count_for(int src_rank, int replay, int token,
                                 int topk, int world_size) {
    int unique[kiwi::ep::MAX_TOPK_SUPPORTED];
    int count = 0;
    for (int k = 0; k < topk; k++) {
        const int dest = dest_rank_for(src_rank, replay, token, k, world_size);
        bool seen = false;
        for (int i = 0; i < count; i++) {
            if (unique[i] == dest) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            unique[count++] = dest;
        }
    }
    return count;
}

static bool token_routes_to_rank(int src_rank, int replay, int token,
                                 int topk, int world_size, int target_rank) {
    for (int k = 0; k < topk; k++) {
        if (dest_rank_for(src_rank, replay, token, k, world_size)
                == target_rank) {
            return true;
        }
    }
    return false;
}

static int expected_recv_count_for_rank(int target_rank, int replay,
                                        int num_tokens, int topk,
                                        int world_size) {
    int count = 0;
    for (int src = 0; src < world_size; src++) {
        for (int token = 0; token < num_tokens; token++) {
            if (token_routes_to_rank(src, replay, token, topk, world_size,
                                     target_rank)) {
                count++;
            }
        }
    }
    return count;
}

static void fill_tokens(float* h_tokens, int rank, int replay,
                        int num_tokens, int hidden_dim) {
    for (int t = 0; t < num_tokens; t++) {
        for (int d = 0; d < hidden_dim; d++) {
            h_tokens[t * hidden_dim + d] = token_value(rank, replay, t, d);
        }
    }
}

static void fill_routing(int32_t* h_indices, float* h_weights,
                         int rank, int replay, int world_size,
                         int num_tokens, int topk,
                         int num_experts_per_rank) {
    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < topk; k++) {
            const int dest_rank = dest_rank_for(rank, replay, t, k, world_size);
            const int local_expert =
                (rank * 7 + replay * 5 + t * 3 + k) % num_experts_per_rank;
            h_indices[t * topk + k] =
                dest_rank * num_experts_per_rank + local_expert;
            h_weights[t * topk + k] = weight_value(rank, replay, t, k);
        }
    }
}

static void compute_expected_output(std::vector<float>& expected,
                                    std::vector<float>& expected_weights,
                                    int rank, int replay,
                                    int num_tokens, int hidden_dim, int topk,
                                    int world_size) {
    std::fill(expected.begin(), expected.end(), 0.0f);
    std::fill(expected_weights.begin(), expected_weights.end(), 0.0f);

    for (int t = 0; t < num_tokens; t++) {
        const int unique_dests =
            unique_dest_count_for(rank, replay, t, topk, world_size);
        for (int d = 0; d < hidden_dim; d++) {
            expected[t * hidden_dim + d] =
                token_value(rank, replay, t, d) * unique_dests;
        }
        for (int k = 0; k < topk; k++) {
            expected_weights[t * topk + k] =
                weight_value(rank, replay, t, k) * unique_dests;
        }
    }
}

static bool compare_vector(int rank, int replay, const char* name,
                           int row_width,
                           const std::vector<float>& got,
                           const std::vector<float>& expected,
                           float rel_tol) {
    int errors = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const float diff = fabsf(got[i] - expected[i]);
        const float scale = fmaxf(fabsf(expected[i]), 1.0f);
        if (diff / scale > rel_tol) {
            if (errors < 8) {
                fprintf(stderr,
                        "[Rank %d] replay %d %s mismatch at row %zu col %zu: "
                        "got=%f expected=%f\n",
                        rank, replay, name, i / row_width, i % row_width,
                        got[i], expected[i]);
            }
            errors++;
        }
    }
    return errors == 0;
}

struct captured_graph_round_t {
    captured_graph_round_t(int tokens, int hidden_dim, int topk)
        : num_tokens(tokens),
          total_elems(tokens * hidden_dim),
          total_routing(tokens * topk),
          h_tokens(static_cast<size_t>(total_elems)),
          h_indices(static_cast<size_t>(total_routing)),
          h_weights(static_cast<size_t>(total_routing)),
          expected(static_cast<size_t>(total_elems)),
          expected_weights(static_cast<size_t>(total_routing)),
          got(static_cast<size_t>(total_elems)),
          got_weights(static_cast<size_t>(total_routing)) {}

    int num_tokens;
    int total_elems;
    int total_routing;
    std::vector<float> h_tokens;
    std::vector<int32_t> h_indices;
    std::vector<float> h_weights;
    std::vector<float> expected;
    std::vector<float> expected_weights;
    std::vector<float> got;
    std::vector<float> got_weights;
    float* d_tokens = nullptr;
    int32_t* d_indices = nullptr;
    float* d_weights = nullptr;
    float* d_output = nullptr;
    float* d_output_weights = nullptr;
    int32_t* d_recv_count_snapshot = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
};

static size_t nonzero_count(size_t count) {
    return count == 0 ? 1 : count;
}

static void allocate_graph_round(captured_graph_round_t& round) {
    HIP_CHECK(hipMalloc(
        &round.d_tokens,
        nonzero_count(static_cast<size_t>(round.total_elems))
        * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &round.d_indices,
        nonzero_count(static_cast<size_t>(round.total_routing))
        * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(
        &round.d_weights,
        nonzero_count(static_cast<size_t>(round.total_routing))
        * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &round.d_output,
        nonzero_count(static_cast<size_t>(round.total_elems))
        * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &round.d_output_weights,
        nonzero_count(static_cast<size_t>(round.total_routing))
        * sizeof(float)));
    HIP_CHECK(hipMalloc(&round.d_recv_count_snapshot, sizeof(int32_t)));
}

static void destroy_graph_round(captured_graph_round_t& round) {
    if (round.graph_exec) HIP_CHECK(hipGraphExecDestroy(round.graph_exec));
    if (round.graph) HIP_CHECK(hipGraphDestroy(round.graph));
    if (round.d_recv_count_snapshot) HIP_CHECK(hipFree(round.d_recv_count_snapshot));
    if (round.d_output_weights) HIP_CHECK(hipFree(round.d_output_weights));
    if (round.d_output) HIP_CHECK(hipFree(round.d_output));
    if (round.d_weights) HIP_CHECK(hipFree(round.d_weights));
    if (round.d_indices) HIP_CHECK(hipFree(round.d_indices));
    if (round.d_tokens) HIP_CHECK(hipFree(round.d_tokens));
}

static void capture_graph_round(kiwi::ep::ep_context_t& ctx,
                                kiwi::ep::ep_gcontext_t& gctx,
                                captured_graph_round_t& round,
                                hipStream_t stream) {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    kiwi::ep::dispatch_enqueue(
        ctx, gctx, static_cast<const void*>(round.d_tokens), round.d_indices,
        round.d_weights, nullptr, round.num_tokens, stream);
    kiwi::ep::combine_enqueue(
        ctx, gctx, gctx.d_recv_hidden, nullptr, nullptr, round.d_indices,
        static_cast<void*>(round.d_output), round.d_output_weights,
        round.num_tokens, stream);
    HIP_CHECK(hipStreamEndCapture(stream, &round.graph));
    HIP_CHECK(hipGraphInstantiate(&round.graph_exec, round.graph,
                                  nullptr, nullptr, 0));
}

static void prepare_graph_round(captured_graph_round_t& round,
                                const kiwi::ep::config_t& cfg,
                                int rank, int replay, int world_size,
                                hipStream_t stream) {
    fill_tokens(round.h_tokens.data(), rank, replay, round.num_tokens,
                cfg.model_hidden_dim);
    fill_routing(round.h_indices.data(), round.h_weights.data(), rank, replay,
                 world_size, round.num_tokens, cfg.topk,
                 cfg.num_experts_per_rank());
    compute_expected_output(round.expected, round.expected_weights, rank,
                            replay, round.num_tokens, cfg.model_hidden_dim,
                            cfg.topk, world_size);

    if (round.total_elems > 0) {
        HIP_CHECK(hipMemcpyAsync(round.d_tokens, round.h_tokens.data(),
                                 static_cast<size_t>(round.total_elems)
                                 * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemsetAsync(round.d_output, 0xA5,
                                 static_cast<size_t>(round.total_elems)
                                 * sizeof(float), stream));
    }
    if (round.total_routing > 0) {
        HIP_CHECK(hipMemcpyAsync(round.d_indices, round.h_indices.data(),
                                 static_cast<size_t>(round.total_routing)
                                 * sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(round.d_weights, round.h_weights.data(),
                                 static_cast<size_t>(round.total_routing)
                                 * sizeof(float),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemsetAsync(round.d_output_weights, 0x5A,
                                 static_cast<size_t>(round.total_routing)
                                 * sizeof(float), stream));
    }
}

static void launch_graph_round(const kiwi::ep::ep_gcontext_t& gctx,
                               captured_graph_round_t& round,
                               hipStream_t stream) {
    HIP_CHECK(hipGraphLaunch(round.graph_exec, stream));
    // Preserve each graph's recv count before the next alternating graph
    // reuses the shared EP workspace.
    HIP_CHECK(hipMemcpyAsync(round.d_recv_count_snapshot,
                             gctx.device_workspace.recv_count(),
                             sizeof(int32_t), hipMemcpyDeviceToDevice,
                             stream));
}

static bool validate_graph_round(const kiwi::ep::config_t& cfg,
                                 captured_graph_round_t& round,
                                 int rank, int replay, int world_size) {
    int32_t got_recv_count = 0;
    HIP_CHECK(hipMemcpy(&got_recv_count, round.d_recv_count_snapshot,
                        sizeof(got_recv_count), hipMemcpyDeviceToHost));
    const int32_t expected_recv_count = expected_recv_count_for_rank(
        rank, replay, round.num_tokens, cfg.topk, world_size);
    if (got_recv_count != expected_recv_count) {
        fprintf(stderr,
                "[Rank %d] graph replay %d tokens=%d recv_count "
                "mismatch got=%d expected=%d\n",
                rank, replay, round.num_tokens, got_recv_count,
                expected_recv_count);
        return false;
    }
    if (round.num_tokens == 0) {
        return true;
    }

    HIP_CHECK(hipMemcpy(round.got.data(), round.d_output,
                        static_cast<size_t>(round.total_elems) * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(round.got_weights.data(), round.d_output_weights,
                        static_cast<size_t>(round.total_routing)
                        * sizeof(float), hipMemcpyDeviceToHost));
    return compare_vector(rank, replay, "output",
                          cfg.model_hidden_dim, round.got, round.expected,
                          1e-6f)
        && compare_vector(rank, replay, "output_weights",
                          cfg.topk, round.got_weights,
                          round.expected_weights, 1e-5f);
}

struct graph_replay_t {
    captured_graph_round_t* round;
    int replay;
};

static bool replay_graph_sequence(
        const kiwi::ep::ep_gcontext_t& gctx,
        std::initializer_list<graph_replay_t> replays,
        const kiwi::ep::config_t& cfg, int rank, int world_size,
        hipStream_t stream) {
    for (const auto& replay : replays) {
        prepare_graph_round(
            *replay.round, cfg, rank, replay.replay, world_size, stream);
        if (rank == (replay.replay % std::min(world_size, 4))) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        launch_graph_round(gctx, *replay.round, stream);
    }

    // Multiple captured token buckets deliberately replay back-to-back on one
    // stream with no host or inter-rank synchronization between them.
    HIP_CHECK(hipStreamSynchronize(stream));

    bool ok = true;
    for (const auto& replay : replays) {
        ok = validate_graph_round(
                 cfg, *replay.round, rank, replay.replay, world_size)
            && ok;
    }
    return ok;
}

static int run_automatic_policy_graph_regression(int rank, int nranks) {
    constexpr int kOneToken = 1;
    constexpr int kFallbackRdmaTokens = 3;
    constexpr int kFallbackHybridTokens = 4;
    constexpr int kLargeNumTokens = 33;

    if (nranks == 8) {
        fprintf(stderr,
                "test_combine_graph_replay/auto requires a non-ws8 topology "
                "to validate the ws16-fallback 3-to-4-token backend change\n");
        return 1;
    }

    kiwi::ep::config_t cfg;
    cfg.world_size = nranks;
    cfg.num_experts = nranks * 8;
    cfg.model_hidden_dim = 192;
    cfg.topk = std::min(kiwi::ep::MAX_TOPK_SUPPORTED, nranks + 2);
    cfg.max_num_inp_token_per_rank = kLargeNumTokens;
    cfg.num_blocks = 0;
    cfg.rcomp_backend = kiwi::ep::rcomp_backend_t::AUTO;

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    kiwi::ep::ep_context_t ctx(cfg);
    ctx.spawn_proxy_threads();
    auto gctx = ctx.get_device_context();

    captured_graph_round_t one(
        kOneToken, cfg.model_hidden_dim, cfg.topk);
    captured_graph_round_t fallback_rdma(
        kFallbackRdmaTokens, cfg.model_hidden_dim, cfg.topk);
    captured_graph_round_t fallback_hybrid(
        kFallbackHybridTokens, cfg.model_hidden_dim, cfg.topk);
    captured_graph_round_t large(
        kLargeNumTokens, cfg.model_hidden_dim, cfg.topk);
    captured_graph_round_t zero(0, cfg.model_hidden_dim, cfg.topk);
    allocate_graph_round(one);
    allocate_graph_round(fallback_rdma);
    allocate_graph_round(fallback_hybrid);
    allocate_graph_round(large);
    allocate_graph_round(zero);

    // Capture does not execute kernels, so the first actual launch below
    // deliberately shrinks from the provisioned maximum to one token.
    lci::barrier();
    ctx.reset(stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    lci::barrier();
    capture_graph_round(ctx, gctx, one, stream);
    capture_graph_round(ctx, gctx, fallback_rdma, stream);
    capture_graph_round(ctx, gctx, fallback_hybrid, stream);
    capture_graph_round(ctx, gctx, large, stream);
    capture_graph_round(ctx, gctx, zero, stream);

    bool ok = true;
    ok = replay_graph_sequence(
             gctx, {{&one, 0}}, cfg, rank, nranks, stream)
        && ok;
    ok = replay_graph_sequence(
             gctx, {{&fallback_rdma, 1}}, cfg, rank, nranks, stream)
        && ok;
    // Three to four tokens keeps the 32-block grid fixed while exercising the
    // ws16-fallback RCOMP specialization transition.
    ok = replay_graph_sequence(
             gctx, {{&fallback_hybrid, 2}}, cfg, rank, nranks, stream)
        && ok;
    ok = replay_graph_sequence(
             gctx, {{&large, 3}}, cfg, rank, nranks, stream)
        && ok;
    // Zero tokens retains the one-token automatic launch policy and still
    // executes dispatch's collective signal phase.
    ok = replay_graph_sequence(
             gctx, {{&zero, 4}}, cfg, rank, nranks, stream)
        && ok;

    ok = replay_graph_sequence(
             gctx, {{&one, 5}, {&large, 6}}, cfg, rank, nranks, stream)
        && ok;
    ok = replay_graph_sequence(
             gctx, {{&large, 7}, {&one, 8}}, cfg, rank, nranks, stream)
        && ok;

    lci::barrier();
    if (rank == 0) {
        printf("test_combine_graph_replay/auto: %s "
               "(tokens=1,3,4,33,zero; pairs=2)\n",
               ok ? "PASSED" : "FAILED");
    }

    destroy_graph_round(zero);
    destroy_graph_round(large);
    destroy_graph_round(fallback_hybrid);
    destroy_graph_round(fallback_rdma);
    destroy_graph_round(one);
    ctx.join_proxy_threads();
    HIP_CHECK(hipStreamDestroy(stream));

    // Keep LCI alive until every rank has torn down the EP context and graphs.
    lci::barrier();
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string rcomp_backend = "rdma-only";
    if (argc == 3 && std::string(argv[1]) == "--rcomp-backend") {
        rcomp_backend = argv[2];
    } else if (argc != 1) {
        fprintf(stderr,
                "usage: test_combine_graph_replay "
                "[--rcomp-backend auto|rdma-only|hybrid]\n");
        return 1;
    }
    if (rcomp_backend != "auto" && rcomp_backend != "rdma-only"
        && rcomp_backend != "hybrid") {
        fprintf(stderr, "unsupported --rcomp-backend '%s'\n",
                rcomp_backend.c_str());
        return 1;
    }
    {
        int num_gpus = 0;
        HIP_CHECK(hipGetDeviceCount(&num_gpus));
        const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!lr) lr = getenv("MPI_LOCALRANKID");
        if (!lr) lr = getenv("SLURM_LOCALID");
        if (!lr) lr = getenv("OMPI_COMM_WORLD_RANK");
        int local_rank = lr ? atoi(lr) : 0;
        HIP_CHECK(hipSetDevice(local_rank % num_gpus));
    }

    kiwi::ep::ep_runtime_guard_t lci_guard;
    const int rank = lci::get_rank_me();
    const int nranks = lci::get_rank_n();
    if (nranks < 2) {
        fprintf(stderr,
                "test_combine_graph_replay requires at least 2 ranks (got %d)\n",
                nranks);
        return 1;
    }
    if (rcomp_backend == "auto") {
        return run_automatic_policy_graph_regression(rank, nranks);
    }

    kiwi::ep::config_t cfg;
    cfg.world_size = nranks;
    cfg.num_experts = nranks * 8;
    cfg.model_hidden_dim = 192;
    cfg.topk = std::min(kiwi::ep::MAX_TOPK_SUPPORTED, nranks + 2);
    cfg.max_num_inp_token_per_rank = 16;
    cfg.num_blocks = 4;
    cfg.rcomp_backend = rcomp_backend == "hybrid"
        ? kiwi::ep::rcomp_backend_t::HYBRID
        : kiwi::ep::rcomp_backend_t::RDMA_ONLY;

    constexpr int kNumTokens = 12;
    constexpr int kReplays = 16;
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    kiwi::ep::ep_context_t ctx(cfg);
    ctx.spawn_proxy_threads();
    auto gctx = ctx.get_device_context();

    captured_graph_round_t round(
        kNumTokens, cfg.model_hidden_dim, cfg.topk);
    allocate_graph_round(round);

    // Capture only graph-safe per-round enqueue work. reset()/MPI barriers are
    // deliberately outside the graph; the replays below have no host-side
    // inter-rank barrier, use replay-varying inputs/routing, poison outputs
    // before launch, and add host-side rank skew. On rcomp, the combine-exit
    // guard must preserve dispatch/combine alternation across those replays.
    lci::barrier();
    ctx.reset(stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    lci::barrier();
    capture_graph_round(ctx, gctx, round, stream);

    bool ok = true;
    for (int replay = 0; replay < kReplays; replay++) {
        if (!replay_graph_sequence(
                gctx, {{&round, replay}}, cfg, rank, nranks, stream)) {
            ok = false;
            break;
        }
    }

    lci::barrier();
    if (rank == 0) {
        printf("test_combine_graph_replay/%s: %s "
               "(tokens=%d hidden=%d topk=%d replays=%d)\n",
               "rcomp",
               ok ? "PASSED" : "FAILED", kNumTokens, cfg.model_hidden_dim,
               cfg.topk, kReplays);
    }

    destroy_graph_round(round);
    ctx.join_proxy_threads();
    HIP_CHECK(hipStreamDestroy(stream));

    // Keep LCI alive on all ranks until every peer has finished destroying the
    // graph-replay EP context and GPU resources.  This is test teardown
    // synchronization only; production rcomp combine remains free to rely on
    // receiver-side completions without adding local-completion waits.
    lci::barrier();

    return ok ? 0 : 1;
}
