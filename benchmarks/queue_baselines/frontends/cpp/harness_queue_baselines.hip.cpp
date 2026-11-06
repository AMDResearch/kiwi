// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file harness_queue_baselines.hip.cpp
 * @brief C++ frontend of the cross-frontend kiwi-spsc queue-baseline harness.
 *
 * The kiwi-spsc kernels of the standalone bench_queue_baselines
 * (benchmarks/queue_baselines/bench_queue_baselines.hip.cpp), copied
 * unchanged and launched as it launches them, run through the harness's host
 * code (common/qb_harness.hpp) like the Python frontends. Same modes (latency, push-cost, throughput,
 * idle-poll), options and RESULT lines as `bench_queue_baselines -q
 * kiwi-spsc`, with queue=kiwi-spsc-cpp.
 *
 * Every producer wave of a multi-wave run must be resident at once: kernels
 * carry __launch_bounds__(kMaxBlockThreads), and a launch fails if its grid
 * cannot be resident. --check-overlap records wave spans and requires all
 * waves to overlap (a diagnostic that perturbs the measurement).
 */

#include "qb_harness.hpp"
#include "adapters/kiwi.hpp"

#include <CLI/CLI.hpp>

namespace qb {

/// Largest workgroup the harness launches (--waves-per-block <= 4). Every
/// kernel is bounded to it: without a bound the compiler assumes 1024-thread
/// workgroups, and a stack array it promotes to LDS is then reserved for 1024
/// threads per block (32 KiB for the RCCL GIN push), which caps residency.
constexpr int kMaxWavesPerBlock = 4;
constexpr int kMaxBlockThreads = 64 * kMaxWavesPerBlock;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Wait until all of this wave's outstanding memory operations (including
/// posted stores to host memory) are acknowledged.
__device__ inline void drain_memory_ops() {
    __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

/// Wave start/end ticks of a --check-overlap run (PushCostLaunch::spans,
/// ThroughputLaunch::spans). A device global rather than a kernel argument, so
/// the measured kernels keep main's exact signature and code.
__device__ uint64_t* g_wave_spans;

/// With kOn, records when a producer wave starts (construction) and finishes
/// (destruction) its work in g_wave_spans. Without, compiles to nothing.
template <bool kOn>
struct WaveSpan {
    __device__ explicit WaveSpan(int) {}
};

template <>
struct WaveSpan<true> {
    __device__ explicit WaveSpan(int wave) : slot_(g_wave_spans + 2 * wave) {
        if (lane_id() == 0) slot_[0] = wall_clock64();
    }
    __device__ ~WaveSpan() {
        if (lane_id() == 0) slot_[1] = wall_clock64();
    }
    uint64_t* slot_;
};

/// Fails (returns false) if `blocks` workgroups of `threads` threads of
/// `kernel` cannot all be resident on the device at once.
template <typename Kernel>
bool grid_fits(Kernel kernel, int blocks, int threads, const char* name) {
    int per_cu = 0, cus = 0;
    QB_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_cu, reinterpret_cast<const void*>(kernel), threads, 0));
    QB_HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
    if (static_cast<long>(per_cu) * cus < blocks) {
        std::fprintf(stderr,
                     "ERROR: %s: %d workgroups of %d threads, but only %d per CU x %d CUs "
                     "can be resident\n",
                     name, blocks, threads, per_cu, cus);
        return false;
    }
    return true;
}

template <typename A>
__global__ __launch_bounds__(kMaxBlockThreads) void latency_kernel(typename A::Handle h, volatile uint64_t* ack,
                               long iters, uint64_t* latency_ticks) {
    uint64_t t_prev = wall_clock64();
    for (long i = 0; i < iters; ++i) {
        A::push(h, make_msg(0, static_cast<uint32_t>(i)));
        if constexpr (!kSync<A>) {
            // As in the KIWI latency benchmark: let the publish stores
            // complete before polling the ack. Otherwise the first ack poll
            // races ahead of the posted write and every queue's RTT grows by
            // ~0.85 us on MI300X.
            drain_memory_ops();
            const uint64_t expected = static_cast<uint64_t>(i) + 1;
            while (*ack < expected) {}
        }
        const uint64_t t_now = wall_clock64();
        if (lane_id() == 0) latency_ticks[i] = t_now - t_prev;
        t_prev = t_now;
    }
}

constexpr int kCalibrationIters = 256;

/// clock64() is s_memtime (shader clock, up to 2.1 GHz on MI300X, varies with
/// DVFS); wall_clock64() is s_memrealtime (constant clock, 100 MHz on MI300X).
/// Both are scalar memory ops whose result arrives later; resolving it
/// immediately keeps its return latency out of whatever the bracket measures.
template <bool kWall = false>
__device__ inline uint64_t timestamp() {
    const uint64_t t = kWall ? wall_clock64() : clock64();
    __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    return t;
}

/// Cost of an empty timing bracket (minimum over kCalibrationIters).
template <bool kWall = false>
__device__ inline uint64_t timer_bracket_cost() {
    uint64_t best = ~0ULL;
    for (int i = 0; i < kCalibrationIters; ++i) {
        drain_memory_ops();
        const uint64_t t0 = timestamp<kWall>();
        const uint64_t t1 = timestamp<kWall>();
        best = t1 - t0 < best ? t1 - t0 : best;
    }
    return best;
}

/// GPU-side cost of push() with `waves` concurrent producer waves.
///
/// Isolated: every push starts with nothing in flight (drain), and each push
/// is timed; this is the stall a wave sees for a push between compute.
/// Burst: pushes are issued back to back and each wave reports its whole loop
/// (ending with a drain) divided by the number of pushes.
template <typename A, bool kWall, bool kSpans>
__global__ __launch_bounds__(kMaxBlockThreads) void push_cost_kernel(const typename A::Handle* handles, int waves,
                                 int waves_per_block, long warmup, long iters,
                                 bool burst, uint64_t* cycles, uint64_t* calibration) {
    const int wave = blockIdx.x * waves_per_block + threadIdx.x / 64;
    if (wave >= waves) return;
    typename A::Handle h = handles[wave];
    A::globalize(h);
    WaveSpan<kSpans> span(wave);
    if (wave == 0) {
        const uint64_t c = timer_bracket_cost<kWall>();
        if (lane_id() == 0) *calibration = c;
    }
    const uint32_t producer = static_cast<uint32_t>(wave);
    if (!burst) {
        for (long i = 0; i < warmup + iters; ++i) {
            const Msg m = make_msg(producer, static_cast<uint32_t>(i));
            // Without draining, back-to-back posted stores to host memory
            // saturate vmcnt and the bracket measures store bandwidth.
            drain_memory_ops();
            const uint64_t t0 = timestamp<kWall>();
            A::push(h, m);
            const uint64_t t1 = timestamp<kWall>();
            if (i >= warmup && lane_id() == 0) cycles[wave * iters + (i - warmup)] = t1 - t0;
        }
    } else {
        for (long i = 0; i < warmup; ++i) A::push(h, make_msg(producer, static_cast<uint32_t>(i)));
        drain_memory_ops();
        const uint64_t t0 = timestamp<kWall>();
        for (long i = warmup; i < warmup + iters; ++i)
            A::push(h, make_msg(producer, static_cast<uint32_t>(i)));
        drain_memory_ops();
        const uint64_t t1 = timestamp<kWall>();
        if (lane_id() == 0) cycles[wave] = t1 - t0;
    }
}

template <typename A, bool kSpans>
__global__ __launch_bounds__(kMaxBlockThreads) void throughput_kernel(const typename A::Handle* handles, int waves,
                                  int waves_per_block, long msgs_per_wave) {
    const int wave = blockIdx.x * waves_per_block + threadIdx.x / 64;
    if (wave >= waves) return;
    typename A::Handle h = handles[wave];
    A::globalize(h);  // same global_* codegen as a by-value kernel argument
    WaveSpan<kSpans> span(wave);
    for (long i = 0; i < msgs_per_wave; ++i)
        A::push(h, make_msg(static_cast<uint32_t>(wave), static_cast<uint32_t>(i)));
}

/// Launches the kernels above for the harness's host code (qb_harness.hpp).
template <typename A>
struct HipLauncher {
    /// The kernel instantiation for a launch: with wave spans (--check-overlap)
    /// or without.
    static auto* push_cost_for(bool wall, bool spans) {
        return spans ? (wall ? push_cost_kernel<A, true, true> : push_cost_kernel<A, false, true>)
                     : (wall ? push_cost_kernel<A, true, false>
                             : push_cost_kernel<A, false, false>);
    }
    static auto* throughput_for(bool spans) {
        return spans ? throughput_kernel<A, true> : throughput_kernel<A, false>;
    }

    /// Points g_wave_spans at a --check-overlap run's span buffer.
    static void bind_spans(uint64_t* spans) {
        if (spans) QB_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_wave_spans), &spans, sizeof(spans)));
    }

    bool fits(const Options& o, int blocks) {
        const int threads = 64 * o.waves_per_block;
        if (o.mode == "push-cost")
            return grid_fits(push_cost_for(o.timer == "wall", o.check_overlap), blocks, threads,
                             "push_cost_kernel");
        return grid_fits(throughput_for(o.check_overlap), blocks, threads, "throughput_kernel");
    }
    void latency(const LatencyLaunch<A>& l) {
        latency_kernel<A><<<1, 64, 0, l.stream>>>(l.handle, l.ack, l.iters, l.latency_ticks);
    }
    void push_cost(const PushCostLaunch<A>& l) {
        bind_spans(l.spans);
        auto* kernel = push_cost_for(l.wall, l.spans);
        kernel<<<l.blocks, 64 * l.waves_per_block>>>(l.handles, l.waves, l.waves_per_block,
                                                     l.warmup, l.iters, l.burst, l.cycles,
                                                     l.calibration);
    }
    void throughput(const ThroughputLaunch<A>& l) {
        bind_spans(l.spans);
        auto* kernel = throughput_for(l.spans);
        kernel<<<l.blocks, 64 * l.waves_per_block>>>(l.handles, l.waves, l.waves_per_block,
                                                     l.msgs_per_wave);
    }
};

}  // namespace qb

int main(int argc, char** argv) {
    using namespace qb;
    using Adapter = KiwiSpscAdapter;
    CLI::App app{"GPU->CPU handoff microbenchmark: kiwi-spsc-cpp (kiwi-spsc host side)"};
    Options o;
    o.name = "kiwi-spsc-cpp";
    app.add_option("-m,--mode", o.mode, "latency | push-cost | throughput | idle-poll")
        ->check(CLI::IsMember({"latency", "push-cost", "throughput", "idle-poll"}));
    app.add_option("-c,--capacity", o.capacity, "Slots per queue; power of two")
        ->check(CLI::PositiveNumber);
    app.add_option("--queues", o.queues, "Queues (0 = one per producer wave)");
    app.add_option("-w,--waves", o.waves, "Producer waves (throughput, push-cost)")
        ->check(CLI::PositiveNumber);
    app.add_option("--waves-per-block", o.waves_per_block, "Waves per workgroup (throughput)")
        ->check(CLI::Range(1, kMaxWavesPerBlock));
    app.add_flag("--check-overlap", o.check_overlap,
                 "push-cost/throughput: record wave spans and require all waves to overlap "
                 "(diagnostic; perturbs the measurement)");
    app.add_option("-t,--threads", o.threads, "Pinned proxy threads (throughput)")
        ->check(CLI::PositiveNumber);
    app.add_option("--batch", o.batch, "Max pops per queue visit (throughput)")
        ->check(CLI::PositiveNumber);
    app.add_option("--idle-pause", o.idle_pause,
                   "x86 pause instructions after an empty polling round (throughput)")
        ->check(CLI::NonNegativeNumber);
    app.add_option("--ack-mem", o.ack_mem,
                   "latency: ack flag in 'device' (fine-grained VRAM, polled locally) "
                   "or 'host' memory (each poll is a PCIe read)")
        ->check(CLI::IsMember({"device", "host"}));
    app.add_option("--timer", o.timer,
                   "push-cost: 'shader' clock64() cycles or 'wall' wall_clock64() "
                   "(constant clock, reported in ns)")
        ->check(CLI::IsMember({"shader", "wall"}));
    app.add_flag("--burst", o.burst,
                 "push-cost: time back-to-back pushes (loop time / pushes per wave)");
    app.add_option("-n,--iters", o.iters, "Measured iterations (messages per wave)")
        ->check(CLI::PositiveNumber);
    app.add_option("--warmup", o.warmup, "Warmup iterations (latency, push-cost)");
    app.add_option("--idle-rounds", o.idle_rounds, "Polling rounds (idle-poll)")
        ->check(CLI::PositiveNumber);
    CLI11_PARSE(app, argc, argv);

    if (o.capacity & (o.capacity - 1)) {
        std::fprintf(stderr, "ERROR: capacity must be a power of two\n");
        return 2;
    }

    // As bench_queue_baselines' main() and run<KiwiSpscAdapter>().
    check_numa_affinity();
    gpu_clock_warmup();
    HipLauncher<Adapter> launcher;
    if (o.mode == "latency") return run_latency<Adapter>(o, launcher);
    if (o.mode == "push-cost") return run_push_cost<Adapter>(o, launcher);
    if (o.mode == "throughput") return run_throughput<Adapter>(o, launcher);
    return run_idle_poll<Adapter>(o);
}
