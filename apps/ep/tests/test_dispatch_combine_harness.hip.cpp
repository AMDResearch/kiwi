// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <unistd.h>
#include <CLI/CLI.hpp>
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

// Dummy kernel to ramp GPU clock
__global__ void gpu_warmup_kernel(int niters) {
    uint64_t x = threadIdx.x;
    for (int i = 0; i < niters; i++) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    if (x == 0) asm volatile("");
}

static float token_value(int rank, int token, int hidden_idx) {
    return static_cast<float>(rank * 1000000 + token * 1000 + hidden_idx);
}

// Fill token data with a rank-specific pattern so correctness checks can
// distinguish a payload delivered from the wrong source rank.
static void fill_tokens(float* h_tokens, int num_tokens, int hidden_dim,
                        int rank) {
    for (int t = 0; t < num_tokens; t++) {
        for (int d = 0; d < hidden_dim; d++) {
            h_tokens[t * hidden_dim + d] = token_value(rank, t, d);
        }
    }
}

// Deterministic routing
static void fill_routing(int32_t* h_indices, float* h_weights,
                          int num_tokens, int topk, int num_experts,
                          int rank, int nranks,
                          const std::string& routing) {
    if (routing == "self-only") {
        assert(nranks > 0);
        assert(num_experts % nranks == 0);
        const int experts_per_rank = num_experts / nranks;
        assert(experts_per_rank > 0);
        const int first_self_expert = rank * experts_per_rank;
        for (int t = 0; t < num_tokens; t++) {
            for (int k = 0; k < topk; k++) {
                h_indices[t * topk + k] =
                    first_self_expert + ((t * topk + k) % experts_per_rank);
                h_weights[t * topk + k] = 1.0f / topk;
            }
        }
        return;
    }

    if (routing == "all-ranks") {
        assert(nranks > 0);
        assert(num_experts % nranks == 0);
        const int experts_per_rank = num_experts / nranks;
        assert(experts_per_rank > 0);
        for (int t = 0; t < num_tokens; t++) {
            for (int k = 0; k < topk; k++) {
                const int dest_rank = k % nranks;
                h_indices[t * topk + k] =
                    dest_rank * experts_per_rank + (t % experts_per_rank);
                h_weights[t * topk + k] = 1.0f / topk;
            }
        }
        return;
    }

    for (int t = 0; t < num_tokens; t++) {
        for (int k = 0; k < topk; k++) {
            h_indices[t * topk + k] = (t * 7 + k * 13) % num_experts;
            h_weights[t * topk + k] = 1.0f / topk;
        }
    }
}

struct harness_config_t {
    int num_tokens = 64;
    int hidden_dim = 128;
    int num_experts = 8;
    int topk = 2;
    int num_blocks = 4;
    int dispatch_elem_size = 4;
    int combine_elem_size = 4;
    int warmup_iters = 5;
    int measure_iters = 20;
    bool run_combine = false;
    bool debug_timing = false;
    bool verify_results = false;
    std::string rcomp_backend = "rdma-only";
    std::string routing = "deterministic";
    int assert_ipc_peer_count = -1;
    int assert_rdma_peer_count = -1;
};

static double elapsed_us(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

static void print_phase_debug_timing(const char* phase_name,
                                     int rank, int iter,
                                     const kiwi::ep::ep_context_t& ctx,
                                     double host_phase_us,
                                     const char* const* labels,
                                     int num_labels) {
    const uint64_t* t = ctx.debug_time_buf();
    const int32_t* m = ctx.debug_time_meta();
    if (t == nullptr || m == nullptr || t[0] == 0) {
        return;
    }

    int wall_clock_rate_khz = 0;
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipError_t attr_err = hipDeviceGetAttribute(
        &wall_clock_rate_khz, hipDeviceAttributeWallClockRate, device);
    if (attr_err != hipSuccess) {
        wall_clock_rate_khz = 0;
    }
    const double tick_us = wall_clock_rate_khz > 0
        ? 1000.0 / static_cast<double>(wall_clock_rate_khz)
        : 0.0;

    printf("# %s-debug rank=%d iter=%d host_%s_us=%.1f "
           "wall_clock_rate_khz=%d meta={rank:%d,world:%d,tokens:%d,"
           "field3:%d,field4:%d,blocks:%d,remote_recv:%d,self_recv:%d}\n",
           phase_name, rank, iter, phase_name, host_phase_us,
           wall_clock_rate_khz,
           m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);

    int previous_idx = 0;
    for (int i = 1; i < num_labels; i++) {
        if (labels[i] == nullptr || t[i] == 0) continue;
        const uint64_t delta_ticks = t[i] - t[previous_idx];
        const uint64_t cumulative_ticks = t[i] - t[0];
        printf("# %s-debug rank=%d iter=%d phase=%s "
               "delta_ticks=%llu cumulative_ticks=%llu",
               phase_name, rank, iter, labels[i],
               static_cast<unsigned long long>(delta_ticks),
               static_cast<unsigned long long>(cumulative_ticks));
        if (tick_us > 0.0) {
            printf(" delta_us=%.3f cumulative_us=%.3f",
                   static_cast<double>(delta_ticks) * tick_us,
                   static_cast<double>(cumulative_ticks) * tick_us);
        }
        printf("\n");
        previous_idx = i;
    }
    fflush(stdout);
}

static void print_dispatch_debug_timing(int rank, int iter,
                                        const kiwi::ep::ep_context_t& ctx,
                                        double host_dispatch_us) {
    static const char* rcomp_labels[] = {
        "entry",
        "count_prescan_done",
        "count_all2all_started",
        "rcomp_rdma_posted",
        "rcomp_ipc_copies_complete",
        "incoming_count_all2all_wait_start",
        "incoming_counts_observed",
        "rank_slot_maps_built",
        "unpack_barrier_released",
        "unpack_done",
    };
    print_phase_debug_timing(
        "dispatch", rank, iter, ctx, host_dispatch_us,
        rcomp_labels,
        static_cast<int>(
            sizeof(rcomp_labels) / sizeof(rcomp_labels[0])));
}

static void print_combine_debug_timing(int rank, int iter,
                                       const kiwi::ep::ep_context_t& ctx,
                                       double host_combine_us) {
    static const char* rcomp_labels[] = {
        "entry",
        "rcomp_combine_work_start",
        "block0_pack_post_done",
        "local_pack_barrier_wait_start",
        "combine_exit_barrier_posted_after_pack",
        "all_blocks_local_pack_ready",
        "accumulation_start",
        "block0_accumulation_done",
        "all_blocks_accumulation_done",
        "wait_for_combine_exit_barrier",
        "combine_exit_barrier_complete",
    };
    print_phase_debug_timing(
        "combine", rank, iter, ctx, host_combine_us,
        rcomp_labels,
        static_cast<int>(
            sizeof(rcomp_labels) / sizeof(rcomp_labels[0])));
}

static std::vector<int> unique_destinations_for_token(
        const int32_t* indices, int token, int topk, int num_experts,
        int nranks) {
    const int experts_per_rank = num_experts / nranks;
    std::vector<int> destinations;
    for (int k = 0; k < topk; ++k) {
        const int expert_id = indices[token * topk + k];
        const int destination =
            expert_id >= 0 ? expert_id / experts_per_rank : -1;
        if (destination < 0 || destination >= nranks) {
            continue;
        }
        if (std::find(destinations.begin(), destinations.end(), destination)
            == destinations.end()) {
            destinations.push_back(destination);
        }
    }
    return destinations;
}

static bool float_matches(float got, float expected) {
    const float diff = fabsf(got - expected);
    return diff <= 1e-4f * std::max(1.0f, fabsf(expected));
}

static int verify_dispatch_results(
        const kiwi::ep::ep_context_t& ctx,
        const kiwi::ep::ep_gcontext_t& gctx,
        const harness_config_t& bcfg, int rank, int nranks,
        hipStream_t stream) {
    constexpr int kMaxVerboseErrors = 10;
    int errors = 0;
    int printed_errors = 0;
    const int total_routing = bcfg.num_tokens * bcfg.topk;
    std::vector<int32_t> recv_counts(static_cast<size_t>(nranks) + 1);
    ctx.copy_dispatch_recv_counts_to_host(recv_counts.data(), stream);

    std::vector<float> received_hidden(static_cast<size_t>(bcfg.hidden_dim));
    std::vector<int32_t> received_indices(static_cast<size_t>(bcfg.topk));
    for (int source_rank = 0; source_rank < nranks; ++source_rank) {
        std::vector<int32_t> source_indices(
            static_cast<size_t>(total_routing));
        std::vector<float> source_weights(static_cast<size_t>(total_routing));
        fill_routing(source_indices.data(), source_weights.data(),
                     bcfg.num_tokens, bcfg.topk, bcfg.num_experts,
                     source_rank, nranks, bcfg.routing);

        std::vector<uint8_t> expected_token(
            static_cast<size_t>(bcfg.num_tokens), 0);
        std::vector<uint8_t> seen_token(
            static_cast<size_t>(bcfg.num_tokens), 0);
        int expected_count = 0;
        for (int token = 0; token < bcfg.num_tokens; ++token) {
            const auto destinations = unique_destinations_for_token(
                source_indices.data(), token, bcfg.topk, bcfg.num_experts,
                nranks);
            if (std::find(destinations.begin(), destinations.end(), rank)
                != destinations.end()) {
                expected_token[static_cast<size_t>(token)] = 1;
                ++expected_count;
            }
        }

        const int received_count = recv_counts[static_cast<size_t>(source_rank)];
        if (received_count != expected_count) {
            if (printed_errors++ < kMaxVerboseErrors) {
                fprintf(stderr,
                        "RESULT_CHECK dispatch rank=%d source=%d recv_count "
                        "got=%d expected=%d\n",
                        rank, source_rank, received_count, expected_count);
            }
            ++errors;
        }

        const int slots_to_check = std::max(
            0, std::min(received_count,
                         gctx.max_num_inp_token_per_rank));
        for (int slot_idx = 0; slot_idx < slots_to_check; ++slot_idx) {
            char* slot = gctx.dispatch_recv_slot(source_rank, slot_idx);
            int32_t source_token = -1;
            HIP_CHECK(hipMemcpy(
                &source_token, gctx.dispatch_layout.slot.src_id(slot),
                sizeof(source_token), hipMemcpyDeviceToHost));
            if (source_token < 0 || source_token >= bcfg.num_tokens
                || expected_token[static_cast<size_t>(source_token)] == 0
                || seen_token[static_cast<size_t>(source_token)] != 0) {
                if (printed_errors++ < kMaxVerboseErrors) {
                    fprintf(stderr,
                            "RESULT_CHECK dispatch rank=%d source=%d slot=%d "
                            "invalid_or_duplicate_src_token=%d\n",
                            rank, source_rank, slot_idx, source_token);
                }
                ++errors;
                continue;
            }
            seen_token[static_cast<size_t>(source_token)] = 1;

            HIP_CHECK(hipMemcpy(
                received_hidden.data(),
                gctx.dispatch_layout.slot.hidden(slot),
                received_hidden.size() * sizeof(float),
                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(
                received_indices.data(),
                gctx.dispatch_layout.slot.indices(slot),
                received_indices.size() * sizeof(int32_t),
                hipMemcpyDeviceToHost));

            for (int d = 0; d < bcfg.hidden_dim; ++d) {
                const float expected =
                    token_value(source_rank, source_token, d);
                if (!float_matches(received_hidden[static_cast<size_t>(d)],
                                   expected)) {
                    if (printed_errors++ < kMaxVerboseErrors) {
                        fprintf(stderr,
                                "RESULT_CHECK dispatch rank=%d source=%d "
                                "token=%d hidden=%d got=%f expected=%f\n",
                                rank, source_rank, source_token, d,
                                received_hidden[static_cast<size_t>(d)],
                                expected);
                    }
                    ++errors;
                    break;
                }
            }
            for (int k = 0; k < bcfg.topk; ++k) {
                const int32_t expected =
                    source_indices[static_cast<size_t>(source_token)
                                   * bcfg.topk + k];
                if (received_indices[static_cast<size_t>(k)] != expected) {
                    if (printed_errors++ < kMaxVerboseErrors) {
                        fprintf(stderr,
                                "RESULT_CHECK dispatch rank=%d source=%d "
                                "token=%d topk=%d got=%d expected=%d\n",
                                rank, source_rank, source_token, k,
                                received_indices[static_cast<size_t>(k)],
                                expected);
                    }
                    ++errors;
                    break;
                }
            }
        }

        for (int token = 0; token < bcfg.num_tokens; ++token) {
            if (expected_token[static_cast<size_t>(token)] != 0
                && seen_token[static_cast<size_t>(token)] == 0) {
                if (printed_errors++ < kMaxVerboseErrors) {
                    fprintf(stderr,
                            "RESULT_CHECK dispatch rank=%d source=%d "
                            "missing_token=%d\n",
                            rank, source_rank, token);
                }
                ++errors;
            }
        }
    }

    return errors;
}

static int verify_combine_results(
        const kiwi::ep::ep_gcontext_t& gctx, const float* h_tokens,
        const int32_t* h_indices, const float* d_output,
        const harness_config_t& bcfg, int rank, int nranks) {
    constexpr int kMaxVerboseErrors = 10;
    int errors = 0;
    int printed_errors = 0;
    std::vector<float> returned_hidden(static_cast<size_t>(bcfg.hidden_dim));
    std::vector<float> output(
        static_cast<size_t>(bcfg.num_tokens) * bcfg.hidden_dim);

    for (int token = 0; token < bcfg.num_tokens; ++token) {
        const auto destinations = unique_destinations_for_token(
            h_indices, token, bcfg.topk, bcfg.num_experts, nranks);
        for (int destination : destinations) {
            // Combine slots are source-token-addressed and intentionally omit
            // the dispatch slot's src_id metadata.
            char* slot = gctx.combine_recv_slot(destination, token);
            HIP_CHECK(hipMemcpy(
                returned_hidden.data(), gctx.combine_layout.slot.hidden(slot),
                returned_hidden.size() * sizeof(float),
                hipMemcpyDeviceToHost));
            for (int d = 0; d < bcfg.hidden_dim; ++d) {
                const float expected = h_tokens[
                    static_cast<size_t>(token) * bcfg.hidden_dim + d];
                if (!float_matches(returned_hidden[static_cast<size_t>(d)],
                                   expected)) {
                    if (printed_errors++ < kMaxVerboseErrors) {
                        fprintf(stderr,
                                "RESULT_CHECK combine rank=%d destination=%d "
                                "token=%d hidden=%d got=%f expected=%f\n",
                                rank, destination, token, d,
                                returned_hidden[static_cast<size_t>(d)],
                                expected);
                    }
                    ++errors;
                    break;
                }
            }
        }
    }

    HIP_CHECK(hipMemcpy(output.data(), d_output,
                        output.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    for (int token = 0; token < bcfg.num_tokens; ++token) {
        const int num_destinations = static_cast<int>(
            unique_destinations_for_token(
                h_indices, token, bcfg.topk, bcfg.num_experts, nranks).size());
        for (int d = 0; d < bcfg.hidden_dim; ++d) {
            const float expected =
                static_cast<float>(num_destinations)
                * h_tokens[static_cast<size_t>(token) * bcfg.hidden_dim + d];
            const float got =
                output[static_cast<size_t>(token) * bcfg.hidden_dim + d];
            if (!float_matches(got, expected)) {
                if (printed_errors++ < kMaxVerboseErrors) {
                    fprintf(stderr,
                            "RESULT_CHECK combine_output rank=%d token=%d "
                            "hidden=%d got=%f expected=%f destinations=%d\n",
                            rank, token, d, got, expected, num_destinations);
                }
                ++errors;
                break;
            }
        }
    }

    return errors;
}

static bool run_harness(const harness_config_t& bcfg) {
    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();

    if (bcfg.verify_results) {
        if (!bcfg.run_combine) {
            fprintf(stderr,
                    "--verify-results requires --combine so it can validate "
                    "both dispatch and combine\n");
            return false;
        }
        if (bcfg.dispatch_elem_size != 4 || bcfg.combine_elem_size != 4) {
            fprintf(stderr,
                    "--verify-results currently requires 4-byte dispatch and "
                    "combine elements\n");
            return false;
        }
        if (bcfg.num_experts % nranks != 0) {
            fprintf(stderr,
                    "--verify-results requires --experts to be divisible by "
                    "the rank count (%d %% %d != 0)\n",
                    bcfg.num_experts, nranks);
            return false;
        }
    }
    if (bcfg.routing == "all-ranks" && bcfg.topk < nranks) {
        fprintf(stderr,
                "--routing all-ranks requires --topk >= rank count "
                "(%d < %d)\n",
                bcfg.topk, nranks);
        return false;
    }

    kiwi::ep::config_t cfg;
    cfg.world_size = nranks;
    cfg.num_experts = bcfg.num_experts;
    cfg.model_hidden_dim = bcfg.hidden_dim;
    cfg.topk = bcfg.topk;
    cfg.dispatch_elem_size = bcfg.dispatch_elem_size;
    cfg.combine_elem_size = bcfg.combine_elem_size;
    cfg.max_num_inp_token_per_rank = bcfg.num_tokens * 2;  // headroom
    cfg.num_blocks = bcfg.num_blocks;
    if (bcfg.rcomp_backend == "rdma-only") {
        cfg.rcomp_backend = kiwi::ep::rcomp_backend_t::RDMA_ONLY;
    } else if (bcfg.rcomp_backend == "hybrid") {
        cfg.rcomp_backend = kiwi::ep::rcomp_backend_t::HYBRID;
    } else {
        fprintf(stderr, "Unsupported --rcomp-backend=%s\n",
                bcfg.rcomp_backend.c_str());
        std::abort();
    }
    if (bcfg.assert_ipc_peer_count >= 0
        || bcfg.assert_rdma_peer_count >= 0) {
        if (bcfg.rcomp_backend != "hybrid") {
            fprintf(stderr,
                    "--assert-ipc-peer-count/--assert-rdma-peer-count require "
                    "--rcomp-backend hybrid\n");
            std::abort();
        }
    }

    if (rank == 0) {
        printf("kiwi-ep protocol: rcomp\n");
    }

    int total_elems = bcfg.num_tokens * bcfg.hidden_dim;
    int total_routing = bcfg.num_tokens * bcfg.topk;

    // Allocate token buffer as float for fill_tokens, then cast to void* for dispatch
    std::vector<float> h_tokens(total_elems);
    std::vector<int32_t> h_indices(total_routing);
    std::vector<float> h_weights(total_routing);

    fill_tokens(h_tokens.data(), bcfg.num_tokens, bcfg.hidden_dim, rank);
    fill_routing(h_indices.data(), h_weights.data(), bcfg.num_tokens,
                 bcfg.topk, bcfg.num_experts, rank, nranks, bcfg.routing);

    float* d_tokens;
    int32_t* d_indices;
    float* d_weights;
    float* d_output = nullptr;
    HIP_CHECK(hipMalloc(&d_tokens, total_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_indices, total_routing * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_weights, total_routing * sizeof(float)));
    if (bcfg.run_combine) {
        HIP_CHECK(hipMalloc(&d_output, total_elems * sizeof(float)));
    }

    HIP_CHECK(hipMemcpy(d_tokens, h_tokens.data(),
                        total_elems * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_indices, h_indices.data(),
                        total_routing * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, h_weights.data(),
                        total_routing * sizeof(float), hipMemcpyHostToDevice));

    // Create a non-blocking stream (avoids default-stream implicit sync overhead)
    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Use host wall-clock for both phases so the numbers include stream
    // synchronization and proxy-thread-driven progress, matching the Python
    // Python harness methodology.
    double total_dispatch_us = 0;
    double total_combine_us = 0;
    int dispatch_verification_errors = 0;
    int combine_verification_errors = 0;

    // Create one context for all iterations (avoids exhausting LCI packet pool)
    kiwi::ep::ep_context_t ctx(cfg);
    ctx.set_debug_timing_enabled(bcfg.debug_timing);
    ctx.spawn_proxy_threads(cfg.num_proxy_threads);
    auto gctx = ctx.get_device_context();
    if (bcfg.assert_ipc_peer_count >= 0
        || bcfg.assert_rdma_peer_count >= 0) {
        const auto& peer_is_ipc = ctx.rcomp_peer_is_ipc();
        const auto& peer_is_local = ctx.rcomp_peer_is_local();
        if (peer_is_ipc.size() != static_cast<size_t>(nranks)
            || peer_is_local.size() != static_cast<size_t>(nranks)) {
            fprintf(stderr,
                    "rank %d hybrid route table size mismatch: ipc=%zu "
                    "local=%zu world=%d\n",
                    rank, peer_is_ipc.size(), peer_is_local.size(), nranks);
            std::abort();
        }

        int ipc_peer_count = 0;
        int rdma_peer_count = 0;
        int local_nonself_peer_count = 0;
        int local_non_ipc_peer_count = 0;
        std::string routes;
        for (int peer = 0; peer < nranks; ++peer) {
            if (!routes.empty()) {
                routes += ",";
            }
            routes += std::to_string(peer);
            routes += ":";
            if (peer == rank) {
                routes += "self";
                continue;
            }
            if (peer_is_local[static_cast<size_t>(peer)] != 0) {
                ++local_nonself_peer_count;
            }
            if (peer_is_ipc[static_cast<size_t>(peer)] != 0) {
                ++ipc_peer_count;
                routes += "ipc";
            } else {
                ++rdma_peer_count;
                routes += "rdma";
                if (peer_is_local[static_cast<size_t>(peer)] != 0) {
                    ++local_non_ipc_peer_count;
                }
            }
        }

        char hostname[256] = {};
        if (gethostname(hostname, sizeof(hostname)) != 0) {
            perror("gethostname");
            std::abort();
        }
        hostname[sizeof(hostname) - 1] = '\0';
        printf("RCOMP_HYBRID_ROUTE rank=%d hostname=%s world=%d "
               "local_nonself_peer_count=%d local_non_ipc_peer_count=%d "
               "ipc_peer_count=%d expected_ipc_peer_count=%d "
               "rdma_peer_count=%d expected_rdma_peer_count=%d "
               "rcomp_backend=hybrid ipc_resources_provisioned=%d routes=%s\n",
               rank, hostname, nranks, local_nonself_peer_count,
               local_non_ipc_peer_count, ipc_peer_count,
               bcfg.assert_ipc_peer_count, rdma_peer_count,
               bcfg.assert_rdma_peer_count,
               gctx.rcomp_peer_is_ipc != nullptr,
               routes.c_str());
        fflush(stdout);
        if ((bcfg.assert_ipc_peer_count >= 0
             && ipc_peer_count != bcfg.assert_ipc_peer_count)
            || (bcfg.assert_rdma_peer_count >= 0
                && rdma_peer_count != bcfg.assert_rdma_peer_count)
            || local_non_ipc_peer_count != 0
            || gctx.rcomp_peer_is_ipc == nullptr) {
            fprintf(stderr,
                    "rank %d hybrid routing mismatch: expected ipc=%d rdma=%d; "
                    "got ipc=%d rdma=%d local_non_ipc=%d "
                    "ipc_resources_provisioned=%d\n",
                    rank, bcfg.assert_ipc_peer_count,
                    bcfg.assert_rdma_peer_count, ipc_peer_count,
                    rdma_peer_count, local_non_ipc_peer_count,
                    gctx.rcomp_peer_is_ipc != nullptr);
            std::abort();
        }
    }

    for (int iter = 0; iter < bcfg.warmup_iters + bcfg.measure_iters; iter++) {
        ctx.reset(stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        lci::barrier();

        HIP_CHECK(hipStreamSynchronize(stream));
        auto wc_start = std::chrono::steady_clock::now();
        kiwi::ep::dispatch(ctx, gctx,
                          static_cast<const void*>(d_tokens),
                          d_indices, d_weights, nullptr /*d_scales*/,
                          bcfg.num_tokens, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        auto wc_end = std::chrono::steady_clock::now();

        double dispatch_us = elapsed_us(wc_start, wc_end);
        if (bcfg.debug_timing && iter >= bcfg.warmup_iters) {
            print_dispatch_debug_timing(rank, iter, ctx, dispatch_us);
        }
        if (bcfg.verify_results) {
            const int errors = verify_dispatch_results(
                ctx, gctx, bcfg, rank, nranks, stream);
            dispatch_verification_errors += errors;
            printf("RESULT_CHECK dispatch rank=%d iter=%d status=%s errors=%d\n",
                   rank, iter, errors == 0 ? "PASS" : "FAIL", errors);
            fflush(stdout);
        }

        double combine_us = 0;
        if (bcfg.run_combine) {
            HIP_CHECK(hipStreamSynchronize(stream));
            auto combine_start = std::chrono::steady_clock::now();
            kiwi::ep::combine(ctx, gctx,
                             gctx.d_recv_hidden,    // expert output (identity)
                             nullptr,               // d_expert_weights
                             nullptr,               // d_expert_scales
                             d_indices,             // d_topk_ids
                             static_cast<void*>(d_output),
                             nullptr,               // d_output_weights
                             bcfg.num_tokens, stream);
            HIP_CHECK(hipStreamSynchronize(stream));
            auto combine_end = std::chrono::steady_clock::now();
            combine_us = elapsed_us(combine_start, combine_end);

            if (bcfg.debug_timing && iter >= bcfg.warmup_iters) {
                print_combine_debug_timing(rank, iter, ctx, combine_us);
            }
            if (bcfg.verify_results) {
                const int errors = verify_combine_results(
                    gctx, h_tokens.data(), h_indices.data(), d_output, bcfg,
                    rank, nranks);
                combine_verification_errors += errors;
                printf("RESULT_CHECK combine rank=%d iter=%d status=%s "
                       "errors=%d\n",
                       rank, iter, errors == 0 ? "PASS" : "FAIL", errors);
                fflush(stdout);
            }
        }

        if (iter >= bcfg.warmup_iters) {
            total_dispatch_us += dispatch_us;
            total_combine_us += combine_us;
        }

        lci::barrier();
    }

    ctx.join_proxy_threads();

    double avg_dispatch_us = total_dispatch_us / bcfg.measure_iters;
    double avg_combine_us = total_combine_us / bcfg.measure_iters;

    // Compute data volume for throughput (approximate upper bound).
    // With per-token dedup, each token sends to min(topk, world_size-1)
    // unique remote ranks.  Each slot is cfg.dispatch_slot_stride() bytes.
    double dests_per_token = std::min(bcfg.topk, nranks - 1);
    double tokens_sent = bcfg.num_tokens * dests_per_token;
    double bytes_sent = tokens_sent * cfg.dispatch_slot_stride();
    double dispatch_gbps = bytes_sent / (avg_dispatch_us * 1e-6) / 1e9;

    if (rank == 0) {
        if (bcfg.run_combine) {
            printf("%-14s %10.1f %10.1f %10.1f %10.2f\n",
                   "mori-like",
                   avg_dispatch_us, avg_combine_us,
                   avg_dispatch_us + avg_combine_us,
                   dispatch_gbps);
        } else {
            printf("%-14s %10.1f %10.2f\n",
                   "mori-like",
                   avg_dispatch_us, dispatch_gbps);
        }
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_tokens));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_weights));
    if (d_output) HIP_CHECK(hipFree(d_output));

    const bool pass = dispatch_verification_errors == 0
        && combine_verification_errors == 0;
    if (bcfg.verify_results) {
        printf("RESULT_CHECK summary rank=%d status=%s dispatch_errors=%d "
               "combine_errors=%d\n",
               rank, pass ? "PASS" : "FAIL", dispatch_verification_errors,
               combine_verification_errors);
        fflush(stdout);
    }
    return pass;
}

int main(int argc, char** argv) {
    harness_config_t bcfg;
    CLI::App app{"kiwi-ep native dispatch/combine test harness"};
    app.add_option("--tokens", bcfg.num_tokens, "Tokens per rank")->default_val(64);
    app.add_option("--hidden", bcfg.hidden_dim, "Hidden dimension")->default_val(128);
    app.add_option("--experts", bcfg.num_experts, "Total experts")->default_val(8);
    app.add_option("--topk", bcfg.topk, "Experts per token")->default_val(2);
    app.add_option("--num-blocks", bcfg.num_blocks, "Kernel grid block count; queues are derived as num_blocks * EP_WAVES_PER_BLOCK")->default_val(4);
    app.add_option("--dispatch-elem-size", bcfg.dispatch_elem_size, "Dispatch element size in bytes (1/2/4)")->default_val(4);
    app.add_option("--combine-elem-size", bcfg.combine_elem_size, "Combine element size in bytes (1/2/4)")->default_val(4);
    app.add_option("--warmup", bcfg.warmup_iters, "Warmup iterations")->default_val(5);
    app.add_option("--iters", bcfg.measure_iters, "Timed smoke iterations")->default_val(20);
    app.add_option("--rcomp-backend", bcfg.rcomp_backend, "RCOMP payload backend: rdma-only or hybrid")->default_val("rdma-only")->check(CLI::IsMember({"rdma-only", "hybrid"}));
    app.add_option("--routing", bcfg.routing,
                   "Routing pattern: deterministic, self-only, or all-ranks")
        ->default_val("deterministic")
        ->check(CLI::IsMember(
            {"deterministic", "self-only", "all-ranks"}));
    app.add_flag("--combine", bcfg.run_combine,
                 "Also exercise combine phase");
    app.add_flag("--verify-results", bcfg.verify_results,
                 "Validate dispatch slots, combine return slots, and output");
    app.add_flag("--debug-timeline", bcfg.debug_timing,
                 "Print gated combine_kernel phase timeline from wall_clock64()");
    app.add_option("--assert-ipc-peer-count", bcfg.assert_ipc_peer_count,
                   "Assert the hybrid rcomp IPC peer count (excludes self)")
        ->default_val(-1);
    app.add_option("--assert-rdma-peer-count", bcfg.assert_rdma_peer_count,
                   "Assert the hybrid rcomp RDMA peer count (excludes self)")
        ->default_val(-1);
    CLI11_PARSE(app, argc, argv);

    if (bcfg.dispatch_elem_size != 1 && bcfg.dispatch_elem_size != 2 && bcfg.dispatch_elem_size != 4) {
        fprintf(stderr, "--dispatch-elem-size must be one of 1, 2, or 4 (got %d)\n",
                bcfg.dispatch_elem_size);
        return 1;
    }
    if (bcfg.combine_elem_size != 1 && bcfg.combine_elem_size != 2 && bcfg.combine_elem_size != 4) {
        fprintf(stderr, "--combine-elem-size must be one of 1, 2, or 4 (got %d)\n",
                bcfg.combine_elem_size);
        return 1;
    }
    if (bcfg.run_combine && bcfg.dispatch_elem_size != bcfg.combine_elem_size) {
        fprintf(stderr,
                "--combine in this C++ test harness reuses dispatch output as expert output, "
                "so dispatch/combine element sizes must match.\n");
        return 1;
    }

    // Select GPU device based on local rank (one GPU per rank)
    {
        int num_gpus = 0;
        HIP_CHECK(hipGetDeviceCount(&num_gpus));
        // Use local rank for multi-node; fall back to global rank for single-node
        const char* local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!local_rank_str) local_rank_str = getenv("MPI_LOCALRANKID");
        if (!local_rank_str) local_rank_str = getenv("SLURM_LOCALID");
        int local_rank = local_rank_str ? atoi(local_rank_str) : -1;
        if (local_rank < 0) {
            // Pre-LCI init: parse rank from env
            const char* rank_str = getenv("OMPI_COMM_WORLD_RANK");
            local_rank = rank_str ? atoi(rank_str) : 0;
        }
        HIP_CHECK(hipSetDevice(local_rank % num_gpus));
    }

    kiwi::ep::ep_runtime_guard_t lci_guard;

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    if (nranks < 2) {
        fprintf(stderr, "test_dispatch_combine_harness requires at least 2 ranks (got %d)\n", nranks);
        return 1;
    }

    // Warm GPU clock
    gpu_warmup_kernel<<<1, 64>>>(2000000);
    HIP_CHECK(hipDeviceSynchronize());

    if (rank == 0) {
        printf("# kiwi-ep native dispatch/combine test harness\n");
        printf("# mode=%s\n", bcfg.run_combine ? "dispatch+combine" : "dispatch-only");
        printf("# ranks=%d, tokens=%d, hidden=%d, experts=%d, topk=%d, num_blocks=%d, dispatch_elem_size=%d, combine_elem_size=%d\n",
               nranks, bcfg.num_tokens, bcfg.hidden_dim,
               bcfg.num_experts, bcfg.topk, bcfg.num_blocks,
               bcfg.dispatch_elem_size, bcfg.combine_elem_size);
        printf("# warmup=%d, iters=%d, routing=%s\n",
               bcfg.warmup_iters, bcfg.measure_iters,
               bcfg.routing.c_str());
        if (bcfg.run_combine) {
            printf("%-14s %10s %10s %10s %10s\n",
                   "impl", "disp(us)", "comb(us)", "total(us)", "disp(GB/s)");
            printf("%-14s %10s %10s %10s %10s\n",
                   "----------", "--------", "--------", "---------", "---------");
        } else {
            printf("%-14s %10s %10s\n", "impl", "disp(us)", "disp(GB/s)");
            printf("%-14s %10s %10s\n", "----------", "--------", "---------");
        }
    }

    return run_harness(bcfg) ? 0 : 1;
}
