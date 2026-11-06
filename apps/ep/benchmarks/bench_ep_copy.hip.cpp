// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#include <hip/amd_detail/amd_device_functions.h>
#endif

#include <CLI/CLI.hpp>
#include <mpi.h>

#include "bench_ep_copy_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kWaveSize = 64;
constexpr int kThreadsPerBlock = 1024;
constexpr int kWavesPerBlock = kThreadsPerBlock / kWaveSize;
constexpr int kDeviceVerifyThreads = 256;

static_assert(kWavesPerBlock == 16);

enum class wave_sync_t {
    none,
    block,
    atomic,
};

struct config_t {
    std::string memory_target = "local";
    int tokens = 334;
    int row_bytes = 7168;
    int blocks = 4;
    int waves_per_token = 1;
    std::string src_access = "normal";
    std::string dst_access = "normal";
    std::string waitcnt = "on";
    std::string wave_sync = "block";
    int warmup = 100;
    int iters = 1000;
    int hip_device = -1;
};

enum class verify_failure_t : uint32_t {
    none,
    ready,
    arrival,
    payload,
};

struct device_verify_result_t {
    uint32_t failure = static_cast<uint32_t>(verify_failure_t::none);
    uint32_t token = 0;
    uint32_t byte = 0;
    uint32_t got = 0;
    uint32_t expected = 0;
};

wave_sync_t parse_wave_sync(const std::string& value) {
    if (value == "none") return wave_sync_t::none;
    if (value == "block") return wave_sync_t::block;
    if (value == "atomic") return wave_sync_t::atomic;
    std::fprintf(stderr, "unsupported --wave-sync value '%s'\n", value.c_str());
    std::abort();
}

void reset_iteration(uint32_t* local_ready, uint32_t* local_arrivals,
                     uint64_t* group_starts, uint64_t* group_ends,
                     int tokens, int total_groups, hipStream_t stream) {
    HIP_CHECK(hipMemsetAsync(
        group_starts, 0xff,
        static_cast<size_t>(total_groups) * sizeof(uint64_t), stream));
    HIP_CHECK(hipMemsetAsync(
        group_ends, 0,
        static_cast<size_t>(total_groups) * sizeof(uint64_t), stream));
    if (local_ready != nullptr) {
        HIP_CHECK(hipMemsetAsync(
            local_ready, 0, static_cast<size_t>(tokens) * sizeof(uint32_t),
            stream));
    }
    if (local_arrivals != nullptr) {
        HIP_CHECK(hipMemsetAsync(
            local_arrivals, 0,
            static_cast<size_t>(tokens) * sizeof(uint32_t), stream));
    }
}

__host__ __device__ uint8_t pattern_byte(
        int source_rank, int token, int byte) {
    uint64_t value = static_cast<uint64_t>(source_rank + 1)
                   * 0x9e3779b185ebca87ULL;
    value ^= static_cast<uint64_t>(token + 1) * 0xc2b2ae3d27d4eb4fULL;
    value ^= static_cast<uint64_t>(byte + 1) * 0x165667b19e3779f9ULL;
    value ^= value >> 29;
    return static_cast<uint8_t>(value);
}

void fill_source(std::vector<uint8_t>& source, int source_rank,
                 int tokens, int row_bytes) {
    for (int token = 0; token < tokens; ++token) {
        for (int byte = 0; byte < row_bytes; ++byte) {
            source[static_cast<size_t>(token) * row_bytes + byte] =
                pattern_byte(source_rank, token, byte);
        }
    }
}

using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));

__device__ inline uint4 system_load_u4(const uint4* src) {
    const unsigned long long address =
        reinterpret_cast<unsigned long long>(src);
    u32x4_t raw;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n\t"
                 "s_waitcnt vmcnt(0)"
                 : "=v"(raw) : "v"(address) : "memory");
    uint4 value;
    value.x = raw[0];
    value.y = raw[1];
    value.z = raw[2];
    value.w = raw[3];
    return value;
}

__device__ inline void system_store_u4(uint4* dst, uint4 value) {
    const unsigned long long address =
        reinterpret_cast<unsigned long long>(dst);
    const u32x4_t raw = {value.x, value.y, value.z, value.w};
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1"
                 :: "v"(address), "v"(raw) : "memory");
}

template <bool SystemAccess>
__device__ inline uint4 payload_load(const uint4* src) {
    if constexpr (SystemAccess) {
        return system_load_u4(src);
    } else {
        return *src;
    }
}

template <bool SystemAccess>
__device__ inline void payload_store(uint4* dst, uint4 value) {
    if constexpr (SystemAccess) {
        system_store_u4(dst, value);
    } else {
        *dst = value;
    }
}

__device__ inline void publish_ready(uint32_t* ready) {
    __hip_atomic_store(ready, 1u, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ inline void record_verify_failure(
        device_verify_result_t* result, verify_failure_t failure,
        uint32_t token, uint32_t byte, uint32_t got, uint32_t expected) {
    if (atomicCAS(
            &result->failure,
            static_cast<uint32_t>(verify_failure_t::none),
            static_cast<uint32_t>(failure))
        == static_cast<uint32_t>(verify_failure_t::none)) {
        result->token = token;
        result->byte = byte;
        result->got = got;
        result->expected = expected;
    }
}

__global__ void verify_device_output_kernel(
        const uint8_t* dst, const uint32_t* ready,
        const uint32_t* arrivals, int expected_source_rank,
        int tokens, int row_bytes, int waves_per_token,
        device_verify_result_t* result) {
    const int lane = threadIdx.x % kWaveSize;
    const int wave =
        (blockIdx.x * blockDim.x + threadIdx.x) / kWaveSize;
    const int token = wave;
    if (token >= tokens) return;

    if (ready != nullptr) {
        const uint32_t observed = __hip_atomic_load(
            ready + token, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (observed != 1) {
            record_verify_failure(
                result, verify_failure_t::ready,
                static_cast<uint32_t>(token), 0, observed, 1);
            return;
        }
    }

    const uint8_t* row = dst + static_cast<size_t>(token) * row_bytes;
    for (int byte = lane; byte < row_bytes; byte += kWaveSize) {
        const uint8_t got = row[byte];
        const uint8_t expected =
            pattern_byte(expected_source_rank, token, byte);
        if (got != expected) {
            record_verify_failure(
                result, verify_failure_t::payload,
                static_cast<uint32_t>(token),
                static_cast<uint32_t>(byte), got, expected);
            return;
        }
    }

    if (arrivals != nullptr && lane == 0) {
        const uint32_t got = arrivals[token];
        const uint32_t expected = static_cast<uint32_t>(waves_per_token);
        if (got != expected) {
            record_verify_failure(
                result, verify_failure_t::arrival,
                static_cast<uint32_t>(token), 0, got, expected);
        }
    }
}

template <bool SrcSystem, bool DstSystem, bool Waitcnt,
          wave_sync_t WaveSync>
__global__ void ep_copy_kernel(const char* src, char* dst, int tokens,
                               int row_bytes, int waves_per_token,
                               int group_iterations, uint32_t* ready,
                               uint32_t* local_arrivals,
                               uint64_t* group_starts,
                               uint64_t* group_ends) {
    const int lane = threadIdx.x % kWaveSize;
    const int wave_in_block = threadIdx.x / kWaveSize;
    const int groups_per_block = kWavesPerBlock / waves_per_token;
    const int group_in_block = wave_in_block / waves_per_token;
    const int wave_in_group = wave_in_block % waves_per_token;
    const int group_id = blockIdx.x * groups_per_block + group_in_block;
    const int total_groups = gridDim.x * groups_per_block;
    const bool group_active = group_id < tokens;

    if (group_active && lane == 0) {
        atomicMin(
            reinterpret_cast<unsigned long long*>(group_starts + group_id),
            static_cast<unsigned long long>(wall_clock64()));
    }

    for (int iteration = 0; iteration < group_iterations; ++iteration) {
        const int token = group_id + iteration * total_groups;
        const bool active = token < tokens;

        if (active) {
            const int begin = row_bytes * wave_in_group / waves_per_token;
            const int end = row_bytes * (wave_in_group + 1)
                          / waves_per_token;
            const uint4* src4 = reinterpret_cast<const uint4*>(
                src + static_cast<size_t>(token) * row_bytes + begin);
            uint4* dst4 = reinterpret_cast<uint4*>(
                dst + static_cast<size_t>(token) * row_bytes + begin);
            for (int element = lane; element < (end - begin) / 16;
                 element += kWaveSize) {
                payload_store<DstSystem>(
                    dst4 + element, payload_load<SrcSystem>(src4 + element));
            }
            if constexpr (Waitcnt) {
                __builtin_amdgcn_s_waitcnt(0);
            }
        }

        if constexpr (WaveSync == wave_sync_t::block) {
            __syncthreads();
            if (active && lane == 0 && wave_in_group == 0) {
                publish_ready(ready + token);
            }
        } else if constexpr (WaveSync == wave_sync_t::atomic) {
            if (active && lane == 0) {
                const uint32_t previous = __hip_atomic_fetch_add(
                    local_arrivals + token, 1u, __ATOMIC_ACQ_REL,
                    __HIP_MEMORY_SCOPE_AGENT);
                if (previous + 1 == static_cast<uint32_t>(waves_per_token)) {
                    publish_ready(ready + token);
                }
            }
        }
    }

    if (group_active && lane == 0) {
        atomicMax(
            reinterpret_cast<unsigned long long*>(group_ends + group_id),
            static_cast<unsigned long long>(wall_clock64()));
    }
}

template <bool SrcSystem, bool DstSystem, bool Waitcnt>
void launch_by_wave_sync(wave_sync_t wave_sync, const config_t& cfg,
                         const char* src, char* dst, int group_iterations,
                         uint32_t* ready, uint32_t* local_arrivals,
                         uint64_t* group_starts, uint64_t* group_ends,
                         hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>(cfg.blocks));
    const dim3 block(kThreadsPerBlock);
    switch (wave_sync) {
    case wave_sync_t::none:
        ep_copy_kernel<SrcSystem, DstSystem, Waitcnt, wave_sync_t::none>
            <<<grid, block, 0, stream>>>(
                src, dst, cfg.tokens, cfg.row_bytes, cfg.waves_per_token,
                group_iterations, nullptr, nullptr,
                group_starts, group_ends);
        break;
    case wave_sync_t::block:
        ep_copy_kernel<SrcSystem, DstSystem, Waitcnt, wave_sync_t::block>
            <<<grid, block, 0, stream>>>(
                src, dst, cfg.tokens, cfg.row_bytes, cfg.waves_per_token,
                group_iterations, ready, nullptr,
                group_starts, group_ends);
        break;
    case wave_sync_t::atomic:
        ep_copy_kernel<SrcSystem, DstSystem, Waitcnt, wave_sync_t::atomic>
            <<<grid, block, 0, stream>>>(
                src, dst, cfg.tokens, cfg.row_bytes, cfg.waves_per_token,
                group_iterations, ready, local_arrivals,
                group_starts, group_ends);
        break;
    }
    HIP_CHECK(hipGetLastError());
}

template <bool SrcSystem, bool DstSystem>
void launch_by_waitcnt(bool waitcnt, wave_sync_t wave_sync,
                       const config_t& cfg, const char* src, char* dst,
                       int group_iterations, uint32_t* ready,
                       uint32_t* local_arrivals, uint64_t* group_starts,
                       uint64_t* group_ends, hipStream_t stream) {
    if (waitcnt) {
        launch_by_wave_sync<SrcSystem, DstSystem, true>(
            wave_sync, cfg, src, dst, group_iterations, ready,
            local_arrivals, group_starts, group_ends, stream);
    } else {
        launch_by_wave_sync<SrcSystem, DstSystem, false>(
            wave_sync, cfg, src, dst, group_iterations, ready,
            local_arrivals, group_starts, group_ends, stream);
    }
}

void launch_kernel(bool src_system, bool dst_system, bool waitcnt,
                   wave_sync_t wave_sync, const config_t& cfg,
                   const char* src, char* dst, int group_iterations,
                   uint32_t* ready, uint32_t* local_arrivals,
                   uint64_t* group_starts, uint64_t* group_ends,
                   hipStream_t stream) {
    if (src_system) {
        if (dst_system) {
            launch_by_waitcnt<true, true>(
                waitcnt, wave_sync, cfg, src, dst, group_iterations, ready,
                local_arrivals, group_starts, group_ends, stream);
        } else {
            launch_by_waitcnt<true, false>(
                waitcnt, wave_sync, cfg, src, dst, group_iterations, ready,
                local_arrivals, group_starts, group_ends, stream);
        }
    } else if (dst_system) {
        launch_by_waitcnt<false, true>(
            waitcnt, wave_sync, cfg, src, dst, group_iterations, ready,
            local_arrivals, group_starts, group_ends, stream);
    } else {
        launch_by_waitcnt<false, false>(
            waitcnt, wave_sync, cfg, src, dst, group_iterations, ready,
            local_arrivals, group_starts, group_ends, stream);
    }
}

bool verify_device_output(
        const char* dst, const uint32_t* ready,
        const uint32_t* arrivals, int expected_source_rank,
        int tokens, int row_bytes, int waves_per_token,
        hipStream_t stream) {
    device_verify_result_t initial{};
    device_verify_result_t result{};
    device_verify_result_t* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, sizeof(*d_result)));
    HIP_CHECK(hipMemcpyAsync(
        d_result, &initial, sizeof(initial), hipMemcpyHostToDevice, stream));

    constexpr int kVerifyWavesPerBlock =
        kDeviceVerifyThreads / kWaveSize;
    const unsigned blocks = static_cast<unsigned>(
        (tokens + kVerifyWavesPerBlock - 1) / kVerifyWavesPerBlock);
    verify_device_output_kernel<<<blocks, kDeviceVerifyThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(dst), ready, arrivals,
        expected_source_rank, tokens, row_bytes, waves_per_token, d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyAsync(
        &result, d_result, sizeof(result), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(d_result));

    const verify_failure_t failure =
        static_cast<verify_failure_t>(result.failure);
    if (failure == verify_failure_t::ready) {
        std::fprintf(stderr,
                     "ready flag mismatch token=%u got=%u expected=1\n",
                     result.token, result.got);
        return false;
    }
    if (failure == verify_failure_t::arrival) {
        std::fprintf(stderr,
                     "arrival counter mismatch token=%u got=%u expected=%u\n",
                     result.token, result.got, result.expected);
        return false;
    }
    if (failure == verify_failure_t::payload) {
        std::fprintf(
            stderr,
            "payload mismatch token=%u byte=%u got=%u expected=%u\n",
            result.token, result.byte, result.got, result.expected);
        return false;
    }
    return failure == verify_failure_t::none;
}

double wall_clock_tick_us(int device) {
    int rate_khz = 0;
    HIP_CHECK(hipDeviceGetAttribute(
        &rate_khz, hipDeviceAttributeWallClockRate, device));
    if (rate_khz <= 0) {
        std::fprintf(stderr,
                     "hipDeviceAttributeWallClockRate returned %d kHz\n",
                     rate_khz);
        std::abort();
    }
    return 1000.0 / static_cast<double>(rate_khz);
}

}  // namespace

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{
        "Standalone EP payload-copy microbenchmark with local and HIP-IPC peer memory targets"};
    app.add_option("--memory-target", cfg.memory_target,
                   "Memory target: local or peer")
        ->default_val("local")
        ->check(CLI::IsMember({"local", "peer"}));
    app.add_option("--tokens", cfg.tokens, "Number of payload rows")
        ->default_val(334);
    app.add_option("--row-bytes", cfg.row_bytes,
                   "Bytes per row (supported: 0, 7168, 14336)")
        ->default_val(7168)
        ->check(CLI::IsMember({0, 7168, 14336}));
    app.add_option("--blocks", cfg.blocks, "1024-thread blocks")
        ->default_val(4);
    app.add_option("--waves-per-token", cfg.waves_per_token,
                   "Waves assigned to one token")
        ->default_val(1)
        ->check(CLI::IsMember({1, 2, 4, 8, 16}));
    app.add_option("--src-access", cfg.src_access,
                   "Payload load access: normal or system")
        ->default_val("normal")
        ->check(CLI::IsMember({"normal", "system"}));
    app.add_option("--dst-access", cfg.dst_access,
                   "Payload store access: normal or system")
        ->default_val("normal")
        ->check(CLI::IsMember({"normal", "system"}));
    app.add_option("--waitcnt", cfg.waitcnt,
                   "Drain payload VMEM operations before synchronization: on or off")
        ->default_val("on")
        ->check(CLI::IsMember({"on", "off"}));
    app.add_option("--wave-sync", cfg.wave_sync,
                   "Per-token synchronization: none, block, or atomic")
        ->default_val("block")
        ->check(CLI::IsMember({"none", "block", "atomic"}));
    app.add_option("--warmup", cfg.warmup, "Warmup kernel launches")
        ->default_val(100);
    app.add_option("--iters", cfg.iters, "Timed kernel launches")
        ->default_val(1000);
    app.add_option("--hip-device", cfg.hip_device,
                   "HIP device override (-1 selects local rank modulo visible GPUs)")
        ->default_val(-1);
    CLI11_PARSE(app, argc, argv);

    if (cfg.tokens <= 0 || cfg.blocks <= 0 || cfg.warmup < 0 || cfg.iters <= 0) {
        std::fprintf(stderr,
                     "--tokens and --blocks must be positive; --warmup must be nonnegative; "
                     "--iters must be positive\n");
        return 1;
    }
    if (cfg.row_bytes != 0 && static_cast<size_t>(cfg.tokens)
        > std::numeric_limits<size_t>::max()
            / static_cast<size_t>(cfg.row_bytes)) {
        std::fprintf(stderr, "payload size overflows size_t\n");
        return 1;
    }

    const wave_sync_t wave_sync = parse_wave_sync(cfg.wave_sync);
    const bool src_system = cfg.src_access == "system";
    const bool dst_system = cfg.dst_access == "system";
    if (src_system && !dst_system) {
        std::fprintf(
            stderr,
            "--src-access system requires --dst-access system "
            "(supported pairs: normal->normal, normal->system, "
            "system->system)\n");
        return 1;
    }
    const bool waitcnt = cfg.waitcnt == "on";
    const int groups_per_block = kWavesPerBlock / cfg.waves_per_token;
    if (cfg.blocks > std::numeric_limits<int>::max() / groups_per_block) {
        std::fprintf(stderr, "too many blocks for wave-group accounting\n");
        return 1;
    }
    const int total_groups = cfg.blocks * groups_per_block;
    const int group_iterations =
        (cfg.tokens + total_groups - 1) / total_groups;

    const ep_copy_utils::bootstrap_t bootstrap =
        ep_copy_utils::bootstrap(
            cfg.memory_target, cfg.hip_device, &argc, &argv);
    const bool mpi_initialized = bootstrap.mpi_initialized;
    const int rank = bootstrap.rank;
    const int peer_rank = bootstrap.peer_rank;
    const int device = bootstrap.device;

    const size_t payload_bytes =
        static_cast<size_t>(cfg.tokens) * static_cast<size_t>(cfg.row_bytes);
    const size_t payload_allocation_bytes = std::max<size_t>(payload_bytes, 1);
    const size_t ready_bytes =
        static_cast<size_t>(cfg.tokens) * sizeof(uint32_t);
    const bool map_peer_payload = payload_bytes > 0;

    char* d_src = nullptr;
    char* d_dst = nullptr;
    uint32_t* local_ready = nullptr;
    uint32_t* local_arrivals = nullptr;
    uint64_t* d_group_starts = nullptr;
    uint64_t* d_group_ends = nullptr;
    HIP_CHECK(hipMalloc(&d_src, payload_allocation_bytes));
    HIP_CHECK(hipMalloc(&d_dst, payload_allocation_bytes));
    if (wave_sync != wave_sync_t::none) {
        HIP_CHECK(hipMalloc(&local_ready, ready_bytes));
        // Materialize readiness storage before exporting its IPC handle.
        HIP_CHECK(hipMemset(local_ready, 0, ready_bytes));
    }
    HIP_CHECK(hipMalloc(
        &d_group_starts, static_cast<size_t>(total_groups) * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(
        &d_group_ends, static_cast<size_t>(total_groups) * sizeof(uint64_t)));
    if (wave_sync == wave_sync_t::atomic) {
        HIP_CHECK(hipMalloc(&local_arrivals, ready_bytes));
    }

    std::vector<uint8_t> source(payload_bytes);
    fill_source(source, rank, cfg.tokens, cfg.row_bytes);
    if (payload_bytes > 0) {
        HIP_CHECK(hipMemcpy(d_src, source.data(), payload_bytes,
                            hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemset(d_dst, 0, payload_allocation_bytes));

    ep_copy_utils::peer_mapping_t peer_mapping;
    if (mpi_initialized
        && (map_peer_payload || local_ready != nullptr)) {
        peer_mapping = ep_copy_utils::open_peer_mappings(
            map_peer_payload ? d_dst : nullptr, local_ready, peer_rank);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    const char* target_src = d_src;
    // Zero-byte payloads do not dereference dst, so d_dst remains the
    // local placeholder used for kernel address arithmetic.
    char* target_dst =
        mpi_initialized && map_peer_payload ? peer_mapping.dst : d_dst;
    uint32_t* peer_ready = mpi_initialized ? peer_mapping.ready : nullptr;
    uint32_t* ready_to_publish =
        mpi_initialized ? peer_ready : local_ready;

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    auto prepare_launch = [&] {
        if (mpi_initialized) MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        reset_iteration(
            local_ready, local_arrivals, d_group_starts, d_group_ends,
            cfg.tokens, total_groups, stream);
        if (mpi_initialized) {
            HIP_CHECK(hipStreamSynchronize(stream));
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
    };

    for (int iteration = 0; iteration < cfg.warmup; ++iteration) {
        prepare_launch();
        launch_kernel(src_system, dst_system, waitcnt, wave_sync, cfg,
                      target_src, target_dst, group_iterations,
                      ready_to_publish, local_arrivals, d_group_starts,
                      d_group_ends, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    const double tick_us = wall_clock_tick_us(device);
    std::vector<uint64_t> group_starts(static_cast<size_t>(total_groups));
    std::vector<uint64_t> group_ends(static_cast<size_t>(total_groups));
    double total_kernel_us = 0.0;
    double total_group_iteration_cycles = 0.0;
    for (int iteration = 0; iteration < cfg.iters; ++iteration) {
        prepare_launch();
        HIP_CHECK(hipEventRecord(start, stream));
        launch_kernel(src_system, dst_system, waitcnt, wave_sync, cfg,
                      target_src, target_dst, group_iterations,
                      ready_to_publish, local_arrivals, d_group_starts,
                      d_group_ends, stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        total_kernel_us += static_cast<double>(elapsed_ms) * 1000.0;

        HIP_CHECK(hipMemcpyAsync(
            group_starts.data(), d_group_starts,
            group_starts.size() * sizeof(uint64_t), hipMemcpyDeviceToHost,
            stream));
        HIP_CHECK(hipMemcpyAsync(
            group_ends.data(), d_group_ends,
            group_ends.size() * sizeof(uint64_t), hipMemcpyDeviceToHost,
            stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        double max_group_iteration_cycles = 0.0;
        const int active_groups = std::min(cfg.tokens, total_groups);
        for (int group = 0; group < active_groups; ++group) {
            const int tokens_done =
                1 + (cfg.tokens - 1 - group) / total_groups;
            const uint64_t elapsed =
                group_ends[static_cast<size_t>(group)]
                - group_starts[static_cast<size_t>(group)];
            max_group_iteration_cycles = std::max(
                max_group_iteration_cycles,
                static_cast<double>(elapsed) / tokens_done);
        }
        total_group_iteration_cycles += max_group_iteration_cycles;
    }

    if (mpi_initialized) MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    HIP_CHECK(hipDeviceSynchronize());

    const int expected_source_rank = mpi_initialized ? peer_rank : rank;
    int verified = verify_device_output(
        d_dst, local_ready, local_arrivals, expected_source_rank,
        cfg.tokens, cfg.row_bytes, cfg.waves_per_token, stream)
            ? 1
            : 0;
    if (mpi_initialized) {
        int all_verified = 0;
        MPI_CHECK(MPI_Allreduce(
            &verified, &all_verified, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        verified = all_verified;
    }
    if (verified == 0) {
        if (mpi_initialized) MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    double kernel_us = total_kernel_us / static_cast<double>(cfg.iters);
    double group_iteration_us =
        total_group_iteration_cycles / static_cast<double>(cfg.iters)
        * tick_us;
    const int ranks = mpi_initialized ? 2 : 1;
    if (mpi_initialized) {
        double kernel_us_sum = 0.0;
        double group_iteration_us_sum = 0.0;
        MPI_CHECK(MPI_Reduce(
            &kernel_us, &kernel_us_sum, 1, MPI_DOUBLE, MPI_SUM, 0,
            MPI_COMM_WORLD));
        MPI_CHECK(MPI_Reduce(
            &group_iteration_us, &group_iteration_us_sum, 1, MPI_DOUBLE,
            MPI_SUM, 0,
            MPI_COMM_WORLD));
        if (rank == 0) {
            kernel_us = kernel_us_sum / static_cast<double>(ranks);
            group_iteration_us =
                group_iteration_us_sum / static_cast<double>(ranks);
        }
    }
    const double bandwidth_gbps =
        kernel_us > 0.0
            ? static_cast<double>(payload_bytes) / kernel_us / 1000.0
            : 0.0;
    const char* sync_verify =
        wave_sync == wave_sync_t::none ? "not_applicable" : "pass";
    const char* metric_scope =
        mpi_initialized ? "rank_average" : "rank_local";
    if (!mpi_initialized || rank == 0) {
        std::printf(
            "RESULT memory_target=%s ranks=%d rank=%d peer_rank=%d hip_device=%d "
            "tokens=%d row_bytes=%d blocks=%d threads_per_block=%d "
            "waves_per_block=%d waves_per_token=%d wave_groups=%d "
            "group_iterations=%d src_access=%s dst_access=%s waitcnt=%s "
            "wave_sync=%s warmup=%d iters=%d kernel_us=%.3f "
            "group_iteration_us=%.3f group_timing=max_normalized_loop "
            "payload_GBps=%.3f metric_scope=%s verify=pass sync_verify=%s\n",
            cfg.memory_target.c_str(), ranks, rank, peer_rank, device, cfg.tokens,
            cfg.row_bytes, cfg.blocks, kThreadsPerBlock, kWavesPerBlock,
            cfg.waves_per_token, total_groups, group_iterations,
            cfg.src_access.c_str(), cfg.dst_access.c_str(),
            cfg.waitcnt.c_str(), cfg.wave_sync.c_str(), cfg.warmup, cfg.iters,
            kernel_us, group_iteration_us, bandwidth_gbps, metric_scope,
            sync_verify);
    }

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipStreamDestroy(stream));

    if (mpi_initialized) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        ep_copy_utils::close_peer_mappings(peer_mapping);
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
    HIP_CHECK(hipFree(d_group_ends));
    HIP_CHECK(hipFree(d_group_starts));
    if (local_arrivals != nullptr) HIP_CHECK(hipFree(local_arrivals));
    if (local_ready != nullptr) HIP_CHECK(hipFree(local_ready));
    HIP_CHECK(hipFree(d_dst));
    HIP_CHECK(hipFree(d_src));

    ep_copy_utils::shutdown(bootstrap);
    return 0;
}
