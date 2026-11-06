// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Standalone HIP host-mapped polling/signal publication reproducer.
//
// This benchmark intentionally has no KIWI P2P, invocation, LCI, MPI, or project
// headers.  It isolates the smallest HIP-only shape that reproduced a
// node-to-node difference in polling behavior: one GPU wave publishes HIP mapped
// host-memory signal slots while a broad GPU grid polls those slots, followed
// by a second ack-like publish/poll phase.
//
// Default reproducer shape:
//   16 blocks x 256 threads, 1 count slot, 1 ack slot, 10 us pre-publish delay.
//
// It can be built standalone, for example:
//   hipcc -O3 --offload-arch=gfx942 bench_host_mapped_grid_poll.hip.cpp \
//     -o bench_host_mapped_grid_poll

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err__ = (call);                                        \
        if (err__ != hipSuccess) {                                        \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,  \
                         __LINE__, hipGetErrorString(err__));             \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace {

enum time_idx_t {
    T_ENTRY = 0,
    T_COUNT_PUBLISH_START = 1,
    T_COUNT_PUBLISH_DONE = 2,
    T_COUNTS_OBSERVED = 3,
    T_ACK_PUBLISH_START = 4,
    T_ACK_PUBLISH_DONE = 5,
    T_ACKS_OBSERVED = 6,
    T_NUM_TIMES = 7,
};

__device__ inline void repro_sleep() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#endif
}

__device__ inline void repro_delay_ticks(uint64_t ticks) {
    if (ticks == 0) return;
    const uint64_t start = static_cast<uint64_t>(wall_clock64());
    while (static_cast<uint64_t>(wall_clock64()) - start < ticks) {
        repro_sleep();
    }
}

__device__ inline void publish_slots_serial_lane0(volatile uint64_t* slots,
                                                  int num_slots,
                                                  uint64_t epoch,
                                                  int lane) {
    if (lane == 0) {
        for (int s = 0; s < num_slots; ++s) {
            slots[s] = epoch;
        }
    }
    __threadfence_system();
}

__device__ inline void poll_slots(volatile const uint64_t* slots,
                                  int num_slots,
                                  uint64_t epoch) {
    for (int s = 0; s < num_slots; ++s) {
        while (slots[s] != epoch) {
            repro_sleep();
        }
    }
}

__global__ void hip_grid_poll_signal_kernel(volatile uint64_t* signals,
                                            int* block_counts,
                                            uint64_t* times,
                                            int num_slots,
                                            int active_blocks,
                                            uint64_t epoch,
                                            uint64_t pre_publish_delay_ticks) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const bool first_wave = tid < 64;
    if (bid >= active_blocks) return;

    volatile uint64_t* count_slots = signals;
    volatile uint64_t* ack_slots = signals + num_slots;

    if (bid == 0 && tid == 0) {
        times[T_ENTRY] = static_cast<uint64_t>(wall_clock64());
    }

    // Count phase: delay publisher so the rest of the grid reaches the polling
    // loop first, then publish the first slot set from block 0 wave 0.
    if (bid == 0 && first_wave) {
        repro_delay_ticks(pre_publish_delay_ticks);
        if (lane == 0) {
            times[T_COUNT_PUBLISH_START] = static_cast<uint64_t>(wall_clock64());
        }
        publish_slots_serial_lane0(count_slots, num_slots, epoch, lane);
        if (lane == 0) {
            times[T_COUNT_PUBLISH_DONE] = static_cast<uint64_t>(wall_clock64());
        }
    }

    // Full-grid polling is intentional: every active GPU thread polls every
    // signal slot to stress host-mapped memory visibility under broad polling.
    poll_slots(count_slots, num_slots, epoch);
    __syncthreads();
    if (tid == 0) atomicAdd(&block_counts[0], 1);
    if (bid == 0 && tid == 0) {
        while (atomicAdd(&block_counts[0], 0) < active_blocks) {
            repro_sleep();
        }
        times[T_COUNTS_OBSERVED] = static_cast<uint64_t>(wall_clock64());
    }

    // Ack phase: repeat the publish/poll pattern after the grid has observed
    // count completion.  This phase is kept because it amplified the node split.
    if (bid == 0 && first_wave) {
        if (lane == 0) {
            times[T_ACK_PUBLISH_START] = static_cast<uint64_t>(wall_clock64());
        }
        publish_slots_serial_lane0(ack_slots, num_slots, epoch, lane);
        if (lane == 0) {
            times[T_ACK_PUBLISH_DONE] = static_cast<uint64_t>(wall_clock64());
        }
    }

    poll_slots(ack_slots, num_slots, epoch);
    __syncthreads();
    if (tid == 0) atomicAdd(&block_counts[1], 1);
    if (bid == 0 && tid == 0) {
        while (atomicAdd(&block_counts[1], 0) < active_blocks) {
            repro_sleep();
        }
        times[T_ACKS_OBSERVED] = static_cast<uint64_t>(wall_clock64());
    }
}

struct config_t {
    int device = 0;
    int warmup = 5;
    int iters = 20;
    int grid_blocks = 16;
    int block_threads = 256;
    int slots = 1;
    double pre_publish_delay_us = 10.0;
};

void usage(const char* argv0) {
    std::printf(
        "Usage: %s [options]\n\n"
        "HIP host-mapped full-grid polling/signal reproducer. Defaults are the\n"
        "simplified repro: 16x256 threads, 1 count slot, 1 ack slot,\n"
        "10 us pre-publish delay. Ack phase is always enabled.\n\n"
        "Options:\n"
        "  --device N                  HIP device ordinal (default: 0)\n"
        "  --warmup N                  warmup launches (default: 5)\n"
        "  --iters N                   measured launches (default: 20)\n"
        "  --grid-blocks N             active GPU blocks (default: 16)\n"
        "  --block-threads N           threads per block, >=64 (default: 256)\n"
        "  --slots N                   count slots and ack slots (default: 1)\n"
        "  --pre-publish-delay-us US   delay before count publish; not included\n"
        "                              in count_publish_us (default: 10)\n"
        "  --help                      show this help\n",
        argv0);
}

bool parse_int_arg(const char* text, int* out) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    *out = static_cast<int>(value);
    return true;
}

bool parse_double_arg(const char* text, double* out) {
    char* end = nullptr;
    double value = std::strtod(text, &end);
    if (end == text || *end != '\0') return false;
    *out = value;
    return true;
}

bool parse_args(int argc, char** argv, config_t* cfg) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(arg, "--device") == 0) {
            const char* v = need_value(arg);
            if (!v || !parse_int_arg(v, &cfg->device)) return false;
        } else if (std::strcmp(arg, "--warmup") == 0) {
            const char* v = need_value(arg);
            if (!v || !parse_int_arg(v, &cfg->warmup)) return false;
        } else if (std::strcmp(arg, "--iters") == 0) {
            const char* v = need_value(arg);
            if (!v || !parse_int_arg(v, &cfg->iters)) return false;
        } else if (std::strcmp(arg, "--grid-blocks") == 0) {
            const char* v = need_value(arg);
            if (!v || !parse_int_arg(v, &cfg->grid_blocks)) return false;
        } else if (std::strcmp(arg, "--block-threads") == 0) {
            const char* v = need_value(arg);
            if (!v || !parse_int_arg(v, &cfg->block_threads)) return false;
        } else if (std::strcmp(arg, "--slots") == 0) {
            const char* v = need_value(arg);
            if (!v || !parse_int_arg(v, &cfg->slots)) return false;
        } else if (std::strcmp(arg, "--pre-publish-delay-us") == 0) {
            const char* v = need_value(arg);
            if (!v || !parse_double_arg(v, &cfg->pre_publish_delay_us)) return false;
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", arg);
            return false;
        }
    }
    return true;
}

double wall_tick_us() {
    int device = 0;
    (void)hipGetDevice(&device);
    int wall_clock_rate_khz = 0;
    hipError_t err = hipDeviceGetAttribute(
        &wall_clock_rate_khz, hipDeviceAttributeWallClockRate, device);
    if (err != hipSuccess || wall_clock_rate_khz <= 0) return 0.0;
    return 1000.0 / static_cast<double>(wall_clock_rate_khz);
}

void print_csv_line(int round, float host_us, const uint64_t* times,
                    double tick_us) {
    const auto delta = [&](int a, int b) -> double {
        if (times[a] == 0 || times[b] == 0 || times[b] < times[a]
            || tick_us <= 0.0) {
            return -1.0;
        }
        return static_cast<double>(times[b] - times[a]) * tick_us;
    };

    std::printf("round=%d,host_us=%.3f,count_publish_us=%.3f,"
                "count_poll_us=%.3f,ack_publish_us=%.3f,"
                "ack_poll_us=%.3f,kernel_total_us=%.3f\n",
                round, host_us,
                delta(T_COUNT_PUBLISH_START, T_COUNT_PUBLISH_DONE),
                delta(T_COUNT_PUBLISH_DONE, T_COUNTS_OBSERVED),
                delta(T_ACK_PUBLISH_START, T_ACK_PUBLISH_DONE),
                delta(T_ACK_PUBLISH_DONE, T_ACKS_OBSERVED),
                delta(T_ENTRY, T_ACKS_OBSERVED));
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    config_t cfg;
    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 1;
    }

    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "No HIP devices visible\n");
        return 1;
    }
    if (cfg.device < 0 || cfg.device >= device_count) {
        std::fprintf(stderr, "Invalid --device %d; visible devices: %d\n",
                     cfg.device, device_count);
        return 1;
    }
    if (cfg.warmup < 0 || cfg.iters <= 0 || cfg.grid_blocks <= 0
        || cfg.block_threads < 64 || cfg.slots <= 0
        || cfg.pre_publish_delay_us < 0.0) {
        std::fprintf(stderr, "Invalid arguments: require warmup>=0, iters>0, "
                     "grid-blocks>0, block-threads>=64, slots>0, "
                     "pre-publish-delay-us>=0\n");
        return 1;
    }
    HIP_CHECK(hipSetDevice(cfg.device));

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, cfg.device));

    const size_t signal_bytes =
        static_cast<size_t>(2 * cfg.slots) * sizeof(uint64_t);
    uint64_t* h_signals = nullptr;
    uint64_t* d_signals_raw = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_signals), signal_bytes,
                            hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&d_signals_raw),
                                      h_signals, 0));
    volatile uint64_t* d_signals = d_signals_raw;

    int* d_block_counts = nullptr;
    HIP_CHECK(hipMalloc(&d_block_counts, 2 * sizeof(int)));
    uint64_t* d_times = nullptr;
    HIP_CHECK(hipMallocManaged(&d_times, T_NUM_TIMES * sizeof(uint64_t)));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const double tick_us = wall_tick_us();
    const uint64_t pre_publish_delay_ticks = tick_us > 0.0
        ? static_cast<uint64_t>(cfg.pre_publish_delay_us / tick_us)
        : 0;

    std::printf("# bench_host_mapped_grid_poll: device=%d name='%s' "
                "visible_devices=%d warmup=%d iters=%d grid_blocks=%d "
                "block_threads=%d slots=%d count_slots=%d ack_slots=%d "
                "signal_memory=host-mapped include_ack=1 poll_mode=full "
                "publish_mode=serial pre_publish_delay_us=%.3f "
                "wall_tick_us=%.9f\n",
                cfg.device, prop.name, device_count, cfg.warmup, cfg.iters,
                cfg.grid_blocks, cfg.block_threads, cfg.slots, cfg.slots,
                cfg.slots, cfg.pre_publish_delay_us, tick_us);
    std::printf("# CSV fields: round,host_us,count_publish_us,count_poll_us,"
                "ack_publish_us,ack_poll_us,kernel_total_us\n");
    std::fflush(stdout);

    const auto clear_state = [&]() {
        std::memset(h_signals, 0, signal_bytes);
        HIP_CHECK(hipMemset(d_block_counts, 0, 2 * sizeof(int)));
        HIP_CHECK(hipMemset(d_times, 0, T_NUM_TIMES * sizeof(uint64_t)));
    };

    for (int i = 0; i < cfg.warmup; ++i) {
        clear_state();
        const uint64_t epoch = static_cast<uint64_t>(i + 1);
        hip_grid_poll_signal_kernel<<<cfg.grid_blocks, cfg.block_threads>>>(
            d_signals, d_block_counts, d_times, cfg.slots, cfg.grid_blocks,
            epoch, pre_publish_delay_ticks);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }

    for (int i = 0; i < cfg.iters; ++i) {
        clear_state();
        const uint64_t epoch = static_cast<uint64_t>(cfg.warmup + i + 1);
        HIP_CHECK(hipEventRecord(start));
        hip_grid_poll_signal_kernel<<<cfg.grid_blocks, cfg.block_threads>>>(
            d_signals, d_block_counts, d_times, cfg.slots, cfg.grid_blocks,
            epoch, pre_publish_delay_ticks);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float host_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&host_ms, start, stop));
        HIP_CHECK(hipDeviceSynchronize());
        print_csv_line(i, static_cast<float>(host_ms * 1000.0f), d_times,
                       tick_us);
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_times));
    HIP_CHECK(hipFree(d_block_counts));
    HIP_CHECK(hipHostFree(h_signals));
    return 0;
}
