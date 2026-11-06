// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_device_functions.h>
#endif
#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include "ep_context.hip.hpp"
#include "lci.hpp"

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,  \
                         __LINE__, hipGetErrorString(err));               \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace {

struct config_t {
    int warmup_iters = 10000;
    int measure_iters = 100000;
    int num_blocks = 1;
    int num_proxy_threads = 1;
    int hip_device = -1;
    std::string lci_device_name;
    bool skip_lci = false;
    bool skip_kiwi = false;
};

struct metric_t {
    double sum = 0.0;
    double min = 0.0;
    double max = 0.0;
};

struct local_stats_t {
    double avg_us = 0.0;
    double min_us = 0.0;
    double p50_us = 0.0;
    double max_us = 0.0;
};

int env_int_or(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value) return fallback;
    return static_cast<int>(parsed);
}

int choose_local_rank() {
    int local_rank = env_int_or("OMPI_COMM_WORLD_LOCAL_RANK", -1);
    if (local_rank < 0) local_rank = env_int_or("MPI_LOCALRANKID", -1);
    if (local_rank < 0) local_rank = env_int_or("SLURM_LOCALID", -1);
    if (local_rank < 0) local_rank = env_int_or("LOCAL_RANK", -1);
    if (local_rank < 0) local_rank = env_int_or("OMPI_COMM_WORLD_RANK", 0);
    return local_rank;
}

int select_hip_device(int requested_device) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "bench_ep_collectives requires at least one HIP GPU\n");
        std::abort();
    }
    int device = requested_device;
    if (device < 0) {
        device = choose_local_rank() % device_count;
    }
    if (device < 0 || device >= device_count) {
        std::fprintf(stderr, "Invalid HIP device %d (device_count=%d)\n",
                     device, device_count);
        std::abort();
    }
    HIP_CHECK(hipSetDevice(device));
    return device;
}

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count());
}

local_stats_t compute_stats(std::vector<double> samples) {
    local_stats_t stats;
    if (samples.empty()) return stats;
    std::sort(samples.begin(), samples.end());
    stats.min_us = samples.front();
    stats.max_us = samples.back();
    stats.p50_us = samples[samples.size() / 2];
    stats.avg_us = std::accumulate(samples.begin(), samples.end(), 0.0)
        / static_cast<double>(samples.size());
    return stats;
}

void metric_reduce_op(const void* left, const void* right, void* dst,
                      size_t n) {
    const auto* l = static_cast<const metric_t*>(left);
    const auto* r = static_cast<const metric_t*>(right);
    auto* d = static_cast<metric_t*>(dst);
    for (size_t i = 0; i < n; ++i) {
        d[i].sum = l[i].sum + r[i].sum;
        d[i].min = std::min(l[i].min, r[i].min);
        d[i].max = std::max(l[i].max, r[i].max);
    }
}

void summarize(const char* op, const local_stats_t& stats,
               lci::device_t device, int rank, int nranks, int iters,
               const char* algorithm, const char* effective_algorithm) {
    metric_t local{stats.avg_us, stats.avg_us, stats.avg_us};
    metric_t global{};
    lci::allreduce_x(&local, &global, 1, sizeof(metric_t), metric_reduce_op)
        .device(device)();

    std::printf(
        "RESULT op=%s algorithm=%s effective_algorithm=%s rank=%d iters=%d avg_us=%.3f p50_us=%.3f min_us=%.3f max_us=%.3f\n",
        op, algorithm, effective_algorithm, rank, iters, stats.avg_us,
        stats.p50_us, stats.min_us, stats.max_us);
    if (rank == 0) {
        std::printf(
            "SUMMARY op=%s algorithm=%s effective_algorithm=%s ranks=%d iters=%d avg_rank_avg_us=%.3f min_rank_avg_us=%.3f max_rank_avg_us=%.3f\n",
            op, algorithm, effective_algorithm, nranks, iters,
            global.sum / static_cast<double>(nranks), global.min, global.max);
    }
}

__host__ __device__ uint64_t all2all_value(int src_rank, int dest_rank) {
    return static_cast<uint64_t>(src_rank) * 100000ULL
         + static_cast<uint64_t>(dest_rank) + 17ULL;
}

local_stats_t bench_lci_barrier(lci::device_t device, int warmup, int iters) {
    lci::barrier_x().device(device)();
    for (int i = 0; i < warmup; ++i) {
        lci::barrier_x().device(device)();
    }
    lci::barrier_x().device(device)();

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const uint64_t t0 = now_ns();
        lci::barrier_x().device(device)();
        const uint64_t t1 = now_ns();
        samples.push_back(static_cast<double>(t1 - t0) / 1000.0);
    }
    lci::barrier_x().device(device)();
    return compute_stats(std::move(samples));
}

local_stats_t bench_lci_alltoall(lci::device_t device, int rank, int nranks,
                                 int warmup, int iters) {
    std::vector<uint64_t> send(static_cast<size_t>(nranks));
    std::vector<uint64_t> recv(static_cast<size_t>(nranks));
    for (int dest = 0; dest < nranks; ++dest) {
        send[static_cast<size_t>(dest)] = all2all_value(rank, dest);
    }

    auto check_recv = [&](const char* phase, int iter) {
        for (int src = 0; src < nranks; ++src) {
            const uint64_t expected = all2all_value(src, rank);
            const uint64_t got = recv[static_cast<size_t>(src)];
            if (got != expected) {
                std::fprintf(stderr,
                             "LCI alltoall mismatch rank=%d %s=%d src=%d got=%llu expected=%llu\n",
                             rank, phase, iter, src,
                             static_cast<unsigned long long>(got),
                             static_cast<unsigned long long>(expected));
                std::abort();
            }
        }
    };

    lci::barrier_x().device(device)();
    for (int i = 0; i < warmup; ++i) {
        std::fill(recv.begin(), recv.end(), 0);
        lci::alltoall_x(send.data(), recv.data(), sizeof(uint64_t))
            .device(device)();
        check_recv("warmup", i);
    }
    lci::barrier_x().device(device)();

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        std::fill(recv.begin(), recv.end(), 0);
        const uint64_t t0 = now_ns();
        lci::alltoall_x(send.data(), recv.data(), sizeof(uint64_t))
            .device(device)();
        const uint64_t t1 = now_ns();
        check_recv("iter", i);
        samples.push_back(static_cast<double>(t1 - t0) / 1000.0);
    }
    lci::barrier_x().device(device)();
    return compute_stats(std::move(samples));
}

__device__ inline void kiwi_ep_barrier_once(
        const kiwi::ep::ep_gcontext_t& gctx,
        kiwi::ep::ep_comm_handle_t& handle) {
    (void)kiwi::ep::ep_protocol_epoch_begin(gctx);
    auto ticket = kiwi::ep::post_barrier(gctx, handle);
    kiwi::ep::wait_barrier(gctx, ticket);
}

__device__ inline void kiwi_ep_barrier_current_epoch(
        const kiwi::ep::ep_gcontext_t& gctx,
        kiwi::ep::ep_comm_handle_t& handle) {
    auto ticket = kiwi::ep::post_barrier(gctx, handle);
    kiwi::ep::wait_barrier(gctx, ticket);
}

__global__ void kiwi_barrier_bench_kernel(kiwi::ep::ep_gcontext_t gctx,
                                          int warmup, int iters,
                                          uint64_t* ticks_out) {
    const int lane = threadIdx.x % kiwi::ep::EP_WAVE_SIZE;
    auto barrier_handle = gctx.comm.get_handle();
    for (int i = 0; i < warmup; ++i) {
        kiwi_ep_barrier_once(gctx, barrier_handle);
    }
    for (int i = 0; i < iters; ++i) {
        const uint64_t t0 = static_cast<uint64_t>(wall_clock64());
        kiwi_ep_barrier_once(gctx, barrier_handle);
        const uint64_t t1 = static_cast<uint64_t>(wall_clock64());
        if (lane == 0) ticks_out[i] = t1 - t0;
    }
}

__global__ void kiwi_all2all_bench_kernel(kiwi::ep::ep_gcontext_t gctx,
                                          int warmup, int iters,
                                          uint64_t* ticks_out,
                                          uint64_t* bad_value_out,
                                          uint64_t* bad_src_out,
                                          int* error_out) {
    const int lane = threadIdx.x % kiwi::ep::EP_WAVE_SIZE;
    uint32_t* const all2all_send = reinterpret_cast<uint32_t*>(
        gctx.device_workspace.dispatch_dest_token_counts_snapshot());
    if (lane == 0) {
        for (int dest = 0; dest < gctx.world_size; ++dest) {
            all2all_send[dest] = static_cast<uint32_t>(
                all2all_value(gctx.rank_me, dest));
        }
    }
    __threadfence_system();
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();

    volatile uint64_t* const all2all_recv =
        reinterpret_cast<volatile uint64_t*>(
            gctx.device_workspace.all2all_counts_recv());

    auto check_recv = [&](volatile uint64_t* recv) {
        if (lane == 0) {
            for (int src = 0; src < gctx.world_size; ++src) {
                const uint64_t got =
                    __atomic_load_n(&recv[src], __ATOMIC_ACQUIRE);
                const uint64_t expected = all2all_value(src, gctx.rank_me);
                if (got != expected) {
                    *error_out = 1;
                    *bad_value_out = got;
                    *bad_src_out = static_cast<uint64_t>(src);
                    break;
                }
            }
        }
    };

    auto handle = gctx.comm.get_handle();
    // warmup
    for (int i = 0; i < warmup; ++i) {
        const kiwi::ep::ep_all2all_ticket_t all2all_ticket =
            kiwi::ep::post_all2all(gctx, handle, all2all_send);
        kiwi::ep::wait_all2all(
            gctx, all2all_ticket,
            gctx.device_workspace.all2all_counts_recv());
        check_recv(all2all_recv);
        // Keep the next epoch from overtaking a peer that has not consumed
        // this receive-side transpose yet.
        kiwi_ep_barrier_current_epoch(gctx, handle);
    }
    // measure
    for (int i = 0; i < iters; ++i) {
        const uint64_t t0 = static_cast<uint64_t>(wall_clock64());
        const kiwi::ep::ep_all2all_ticket_t all2all_ticket =
            kiwi::ep::post_all2all(gctx, handle, all2all_send);
        kiwi::ep::wait_all2all(
            gctx, all2all_ticket,
            gctx.device_workspace.all2all_counts_recv());
        const uint64_t t1 = static_cast<uint64_t>(wall_clock64());
        if (lane == 0) {
            ticks_out[i] = t1 - t0;
        }
        check_recv(all2all_recv);
        // Keep the reuse barrier outside the measured all-to-all interval.
        kiwi_ep_barrier_current_epoch(gctx, handle);
    }
}

local_stats_t stats_from_ticks(const uint64_t* ticks, int iters,
                               double tick_us) {
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        samples.push_back(static_cast<double>(ticks[i]) * tick_us);
    }
    return compute_stats(std::move(samples));
}

double wall_clock_tick_us(int device) {
    int wall_clock_rate_khz = 0;
    hipError_t err = hipDeviceGetAttribute(&wall_clock_rate_khz,
                                           hipDeviceAttributeWallClockRate,
                                           device);
    if (err != hipSuccess || wall_clock_rate_khz <= 0) {
        std::fprintf(stderr,
                     "hipDeviceAttributeWallClockRate unavailable; cannot convert GPU ticks\n");
        std::abort();
    }
    return 1000.0 / static_cast<double>(wall_clock_rate_khz);
}

local_stats_t bench_kiwi_barrier(kiwi::ep::ep_gcontext_t gctx, hipStream_t stream,
                                 int device, int warmup, int iters) {
    uint64_t* ticks = nullptr;
    HIP_CHECK(hipMallocManaged(&ticks, static_cast<size_t>(iters) * sizeof(uint64_t)));
    HIP_CHECK(hipMemsetAsync(ticks, 0, static_cast<size_t>(iters) * sizeof(uint64_t), stream));
    kiwi_barrier_bench_kernel<<<1, kiwi::ep::EP_WAVE_SIZE, 0, stream>>>(
        gctx, warmup, iters, ticks);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    local_stats_t stats = stats_from_ticks(ticks, iters, wall_clock_tick_us(device));
    HIP_CHECK(hipFree(ticks));
    return stats;
}

local_stats_t bench_kiwi_all2all(kiwi::ep::ep_gcontext_t gctx,
                                 hipStream_t stream, int device,
                                 int rank, int warmup, int iters) {
    uint64_t* ticks = nullptr;
    uint64_t* bad_value = nullptr;
    uint64_t* bad_src = nullptr;
    int* error = nullptr;
    HIP_CHECK(hipMallocManaged(&ticks, static_cast<size_t>(iters) * sizeof(uint64_t)));
    HIP_CHECK(hipMallocManaged(&bad_value, sizeof(uint64_t)));
    HIP_CHECK(hipMallocManaged(&bad_src, sizeof(uint64_t)));
    HIP_CHECK(hipMallocManaged(&error, sizeof(int)));
    HIP_CHECK(hipMemsetAsync(ticks, 0, static_cast<size_t>(iters) * sizeof(uint64_t), stream));
    HIP_CHECK(hipMemsetAsync(bad_value, 0, sizeof(uint64_t), stream));
    HIP_CHECK(hipMemsetAsync(bad_src, 0, sizeof(uint64_t), stream));
    HIP_CHECK(hipMemsetAsync(error, 0, sizeof(int), stream));
    kiwi_all2all_bench_kernel<<<1, kiwi::ep::EP_WAVE_SIZE, 0, stream>>>(
        gctx, warmup, iters, ticks, bad_value, bad_src, error);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    if (*error != 0) {
        const int src = static_cast<int>(*bad_src);
        const uint64_t expected = all2all_value(src, rank);
        std::fprintf(stderr,
                     "KIWI all2all mismatch rank=%d src=%d got=%llu expected=%llu\n",
                     rank, src, static_cast<unsigned long long>(*bad_value),
                     static_cast<unsigned long long>(expected));
        std::abort();
    }
    local_stats_t stats = stats_from_ticks(ticks, iters, wall_clock_tick_us(device));
    HIP_CHECK(hipFree(error));
    HIP_CHECK(hipFree(bad_src));
    HIP_CHECK(hipFree(bad_value));
    HIP_CHECK(hipFree(ticks));
    return stats;
}

} // namespace

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{
        "Standalone LCI vs GPU-initiated KIWI/EP barrier/all2all microbenchmark"};
    app.add_option("--warmup", cfg.warmup_iters, "Warmup iterations")->default_val(10000);
    app.add_option("--iters", cfg.measure_iters, "Measured iterations")->default_val(100000);
    app.add_option("--num-blocks", cfg.num_blocks, "EP kernel grid block count for KIWI path; queues are derived as num_blocks * EP_WAVES_PER_BLOCK")->default_val(1);
    app.add_option("--num-proxy-threads", cfg.num_proxy_threads, "CPU proxy threads for KIWI path")->default_val(1);
    app.add_option("--hip-device", cfg.hip_device, "HIP device override; defaults to local rank modulo visible GPUs")->default_val(-1);
    app.add_option("--lci-device-name", cfg.lci_device_name, "Optional LCI device/HCA name used during runtime init");
    app.add_flag("--skip-lci", cfg.skip_lci, "Skip direct CPU-side LCI benchmarks");
    app.add_flag("--skip-kiwi", cfg.skip_kiwi, "Skip GPU-initiated KIWI/EP benchmarks");
    CLI11_PARSE(app, argc, argv);

    const int hip_device = select_hip_device(cfg.hip_device);

    // Initializes LCI once through the EP runtime guard so the subsequent EP context
    // shares the same runtime.  This guard's device is not on the timed path.
    kiwi::ep::ep_runtime_guard_t lci_runtime_guard(cfg.lci_device_name);
    const int rank = lci::get_rank_me();
    const int nranks = lci::get_rank_n();

    lci::device_t direct_device = lci::alloc_device();

    if (rank == 0) {
        std::printf(
            "# bench_ep_collectives ranks=%d hip_device_per_local_rank warmup=%d iters=%d num_blocks=%d num_proxy_threads=%d protocol=rcomp count_alltoall=hybrid\n",
            nranks, cfg.warmup_iters, cfg.measure_iters, cfg.num_blocks,
            cfg.num_proxy_threads);
    }
    std::printf("# rank=%d selected_hip_device=%d\n", rank, hip_device);

    if (!cfg.skip_lci) {
        summarize("lci_barrier", bench_lci_barrier(direct_device,
                  cfg.warmup_iters, cfg.measure_iters), direct_device,
                  rank, nranks, cfg.measure_iters, "na", "na");
        summarize("lci_alltoall", bench_lci_alltoall(
                  direct_device, rank, nranks, cfg.warmup_iters,
                  cfg.measure_iters), direct_device, rank, nranks,
                  cfg.measure_iters, "na", "na");
    }

    if (!cfg.skip_kiwi) {
        kiwi::ep::config_t ep_cfg;
        ep_cfg.world_size = nranks;
        ep_cfg.num_experts = nranks;
        ep_cfg.model_hidden_dim = 8;
        ep_cfg.topk = 1;
        ep_cfg.max_num_inp_token_per_rank = 1;
        ep_cfg.num_blocks = cfg.num_blocks;
        ep_cfg.num_proxy_threads = cfg.num_proxy_threads;
        ep_cfg.device_name = cfg.lci_device_name;

        kiwi::ep::ep_context_t ep_ctx(ep_cfg);
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        ep_ctx.spawn_proxy_threads(cfg.num_proxy_threads);
        kiwi::ep::ep_gcontext_t gctx = ep_ctx.get_device_context();
        ep_ctx.reset(stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        lci::barrier_x().device(direct_device)();

        summarize("kiwi_ep_barrier", bench_kiwi_barrier(
                  gctx, stream, hip_device, cfg.warmup_iters,
                  cfg.measure_iters), direct_device, rank, nranks,
                  cfg.measure_iters, "na", "na");
        summarize("kiwi_ep_all2all", bench_kiwi_all2all(
                  gctx, stream, hip_device, rank, cfg.warmup_iters,
                  cfg.measure_iters), direct_device, rank, nranks,
                  cfg.measure_iters, "hybrid", "hybrid");

        lci::barrier_x().device(direct_device)();
        ep_ctx.join_proxy_threads();
        HIP_CHECK(hipStreamDestroy(stream));
    }

    lci::barrier_x().device(direct_device)();
    lci::free_device(&direct_device);
    return 0;
}
