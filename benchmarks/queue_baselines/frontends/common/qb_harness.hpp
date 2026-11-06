// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file qb_harness.hpp
 * @brief Host side of the cross-frontend kiwi-spsc queue-baseline harness.
 *
 * Options, validation, proxy threads, acknowledgement path, statistics and
 * output for the latency, push-cost, throughput and idle-poll modes: a copy
 * of the host code of the standalone bench_queue_baselines
 * (benchmarks/queue_baselines/bench_queue_baselines.hip.cpp) with the kernel
 * launches taken from a launcher, so every frontend of the harness runs the
 * same host code: the C++ kernels (cpp/) and the Python kernels (triton/, ...)
 * through kiwi_queue_bench_ext.
 */

#pragma once

#include "qb_common.hpp"

#include <bench_common.hpp>

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace qb {

struct Options {
    std::string queue = "kiwi-spsc";
    std::string name;  // queue name in the output; empty = the adapter's kName
    std::string mode = "latency";
    int capacity = 1024;
    int queues = 0;  // 0 => one queue per producer wave
    int waves = 1;
    int waves_per_block = 1;
    int threads = 1;
    int batch = 32;
    int idle_pause = 0;  // x86 pause instructions after an empty polling round
    bool burst = false;  // push-cost: back-to-back pushes instead of isolated
    std::string timer = "shader";  // push-cost: shader clock or 100 MHz wall clock
    long bg_max = 1 << 20;  // loaded-latency: cap on messages per background wave
    double bg_rate = 0;     // loaded-latency: total background Mops (0 = unthrottled)
    std::string ack_mem = "device";  // latency modes: where the host writes the ack
    long iters = 100000;
    long warmup = 1000;
    long idle_rounds = 1000000;
    bool check_overlap = false;  // push-cost/throughput: record wave spans (diagnostic)
};

/// Synchronous adapters (e.g. hostcall) return from push() only after the host
/// handled the message; they have no host pop() and no separate ack.
template <typename A, typename = void>
struct is_synchronous : std::false_type {};
template <typename A>
struct is_synchronous<A, std::void_t<decltype(A::kSynchronous)>>
    : std::bool_constant<A::kSynchronous> {};
template <typename A>
constexpr bool kSync = is_synchronous<A>::value;

/// Unbounded adapters (raw-store) have no capacity check; only modes with a
/// single outstanding message or a never-wrapping ring are valid.
template <typename A, typename = void>
struct is_unbounded : std::false_type {};
template <typename A>
struct is_unbounded<A, std::void_t<decltype(A::kUnbounded)>>
    : std::bool_constant<A::kUnbounded> {};
template <typename A>
constexpr bool kUnbounded = is_unbounded<A>::value;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Per-producer validation state. Each producer maps to exactly one queue and
/// each queue to exactly one consumer thread, so no state is shared; every
/// producer's state is cache-line aligned to avoid false sharing between
/// consumer threads.
class Validator {
public:
    Validator(int producers, int num_queues, long msgs_per_producer, bool fifo)
        : num_queues_(num_queues), msgs_(msgs_per_producer), fifo_(fifo),
          producers_(producers) {
        for (auto& p : producers_) p.seen.assign((msgs_per_producer + 63) / 64, 0);
    }

    /// Returns false on the first error for this message.
    bool check(int queue, const Msg& m) {
        if (m.producer >= producers_.size() ||
            static_cast<int>(m.producer % num_queues_) != queue ||
            m.seq >= static_cast<uint64_t>(msgs_) ||
            m.tag != msg_tag(m.producer, m.seq)) {
            report(queue, m, "corrupt message");
            return false;
        }
        Producer& p = producers_[m.producer];
        uint64_t& word = p.seen[m.seq / 64];
        const uint64_t bit = 1ULL << (m.seq % 64);
        if (word & bit) {
            report(queue, m, "duplicate message");
            return false;
        }
        word |= bit;
        if (fifo_ && m.seq != p.next) {
            report(queue, m, "FIFO order violation");
            return false;
        }
        p.next = m.seq + 1;
        ++p.received;
        return true;
    }

    /// Completeness against per-producer counts known only after the run.
    bool complete(const std::vector<long>& expected) const {
        for (size_t i = 0; i < producers_.size(); ++i) {
            if (producers_[i].received != expected[i]) {
                std::fprintf(stderr, "ERROR: producer %zu delivered %ld of %ld\n", i,
                             producers_[i].received, expected[i]);
                return false;
            }
        }
        return errors_.load() == 0;
    }

    bool complete() const {
        for (size_t i = 0; i < producers_.size(); ++i) {
            if (producers_[i].received != msgs_) {
                std::fprintf(stderr, "ERROR: producer %zu delivered %ld of %ld\n",
                             i, producers_[i].received, msgs_);
                return false;
            }
        }
        return errors_.load() == 0;
    }

private:
    struct alignas(64) Producer {
        uint64_t next = 0;
        long received = 0;
        std::vector<uint64_t> seen;
    };

    void report(int queue, const Msg& m, const char* what) {
        if (errors_.fetch_add(1) < 10)
            std::fprintf(stderr,
                         "ERROR: %s on queue %d: producer=%u seq=%u tag=%016" PRIx64 "\n",
                         what, queue, m.producer, m.seq, m.tag);
    }

    int num_queues_;
    long msgs_;
    bool fifo_;
    std::vector<Producer> producers_;
    std::atomic<long> errors_{0};
};

// ---------------------------------------------------------------------------
// Kernel launches
// ---------------------------------------------------------------------------
//
// The run_* functions below are the host side of each mode. They take a
// launcher that starts the mode's kernel where bench_queue_baselines launches
// it, so every frontend's kernels are measured by the same host code:
//
//   struct Launcher {
//     bool fits(const Options& o, int blocks);     // push-cost, throughput
//     void latency(const LatencyLaunch<A>&);       // 1 block of 64, on l.stream
//     void push_cost(const PushCostLaunch<A>&);    // l.blocks x 64*waves_per_block
//     void throughput(const ThroughputLaunch<A>&); // l.blocks x 64*waves_per_block
//   };
//
// Each kernel's contract is the matching kernel in bench_queue_baselines.hip.cpp
// (copied to cpp/). fits() runs before a multi-wave mode sets anything up; it
// returns false, after saying why, if o.mode's kernel cannot run as `blocks`
// blocks (the C++ launcher requires every block to be resident at once).

/// One wave: push, wait for the host ack, record the cycle time per message.
template <typename A>
struct LatencyLaunch {
    typename A::Handle handle;  // by value
    volatile uint64_t* ack;
    long iters;                 // warmup included
    uint64_t* latency_ticks;    // [iters], wall_clock64 ticks
    hipStream_t stream;
};

/// GPU-side cost of push(): per-sample (isolated) or per-wave loop (burst).
template <typename A>
struct PushCostLaunch {
    const typename A::Handle* handles;  // [waves], device memory
    int blocks;
    int waves;
    int waves_per_block;
    long warmup;
    long iters;
    bool burst;
    bool wall;              // wall_clock64 (true) or clock64 timestamps
    uint64_t* cycles;       // [burst ? waves : waves * iters]
    uint64_t* calibration;  // empty-bracket cost, written by wave 0
    uint64_t* spans;        // --check-overlap: wave spans (alloc_wave_spans), else null
};

/// W waves push msgs_per_wave messages each.
template <typename A>
struct ThroughputLaunch {
    const typename A::Handle* handles;  // [waves], device memory
    int blocks;
    int waves;
    int waves_per_block;
    long msgs_per_wave;
    uint64_t* spans;  // --check-overlap: wave spans (alloc_wave_spans), else null
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Queue name printed in the output.
template <typename A>
const char* queue_name(const Options& o) {
    return o.name.empty() ? A::kName : o.name.c_str();
}

struct Percentiles {
    double min, p50, p90, p99, p999, max, avg;
};

inline Percentiles percentiles(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    auto at = [&](double q) { return v[std::min(n - 1, static_cast<size_t>(q * n))]; };
    double sum = 0;
    for (double x : v) sum += x;
    return {v.front(), at(0.50), at(0.90), at(0.99), at(0.999), v.back(), sum / n};
}

template <typename A>
typename A::Handle* upload_handles(A& adapter, int waves, int num_queues) {
    std::vector<typename A::Handle> host(waves);
    for (int w = 0; w < waves; ++w) host[w] = adapter.handle(w % num_queues);
    typename A::Handle* dev = nullptr;
    QB_HIP_CHECK(hipMalloc(&dev, waves * sizeof(typename A::Handle)));
    QB_HIP_CHECK(hipMemcpy(dev, host.data(), waves * sizeof(typename A::Handle),
                           hipMemcpyHostToDevice));
    return dev;
}

/// Pinned proxy threads that drain and validate `num_queues` queues fed by
/// `o.waves` producers of `msgs_per_wave` messages (wave w -> queue w % Q;
/// thread t owns queues q with q % threads == t). Synchronous adapters are
/// drained by the runtime, so the pool has no threads for them.
template <typename A>
class ConsumerPool {
public:
    ConsumerPool(A& adapter, Validator& validator, const Options& o, int num_queues,
                 long msgs_per_wave = -1)
        : threads_(kSync<A> ? 0 : o.threads) {
        if constexpr (kSync<A>) A::sink = [&validator](const Msg& m) { validator.check(0, m); };
        const long per_wave = msgs_per_wave < 0 ? o.iters : msgs_per_wave;
        std::vector<long> expected(num_queues, 0);
        for (int w = 0; w < o.waves; ++w) expected[w % num_queues] += per_wave;
        const std::vector<int> cpus = worker_cpu_plan(o.threads);
        for (int t = 0; t < threads_; ++t) {
            workers_.emplace_back([&adapter, &validator, &o, this, t, num_queues, expected,
                                   cpu = cpus[t]] {
                pin_current_thread_to_cpu(cpu);
                std::vector<int> mine;
                long remaining = 0;
                for (int q = t; q < num_queues; q += threads_) {
                    mine.push_back(q);
                    remaining += expected[q];
                }
                ready_.fetch_add(1);
                while (!go_.load(std::memory_order_acquire)) {}
                while (remaining > 0) {
                    bool progressed = false;
                    for (int q : mine) {
                        Msg m;
                        for (int b = 0; b < o.batch && adapter.pop(q, m); ++b) {
                            if (!validator.check(q, m)) ok_ = false;
                            --remaining;
                            progressed = true;
                        }
                    }
                    // Back off after a round with no messages. Spinning on
                    // the line the GPU is writing slows the GPU's posted
                    // writes, so the best setting differs per queue; see
                    // README.
                    if (!progressed)
                        for (int k = 0; k < o.idle_pause; ++k) __builtin_ia32_pause();
                }
            });
        }
        while (ready_.load() < threads_) {}
    }

    ~ConsumerPool() {
        go_.store(true, std::memory_order_release);
        for (auto& w : workers_)
            if (w.joinable()) w.join();
    }

    void start() { go_.store(true, std::memory_order_release); }
    void join() {
        for (auto& w : workers_) w.join();
    }
    bool ok() const { return ok_.load(); }

private:
    int threads_;
    std::vector<std::thread> workers_;
    std::atomic<int> ready_{0};
    std::atomic<bool> go_{false};
    std::atomic<bool> ok_{true};
};

/// Ack flag for the latency modes. "device": fine-grained GPU memory that the
/// host writes over PCIe and the GPU polls locally. "host": coherent host
/// memory, so every GPU poll is a PCIe read; the RTT is then quantized by the
/// poll round trip (~1.3 us on MI300X), and which side of a poll the host's
/// ack lands on depends on each queue's timing, which adds a queue-dependent
/// bimodal artifact.
inline volatile uint64_t* alloc_ack(const std::string& where) {
    void* p = nullptr;
    if (where == "host")
        QB_HIP_CHECK(hipHostMalloc(&p, sizeof(uint64_t), hipHostMallocCoherent));
    else
        QB_HIP_CHECK(hipExtMallocWithFlags(&p, sizeof(uint64_t), hipDeviceMallocFinegrained));
    auto* ack = static_cast<volatile uint64_t*>(p);
    *ack = 0;
    return ack;
}

/// Publish the ack. A CPU store to fine-grained VRAM goes through the
/// write-combining buffer and can sit there for ~1 ms unless flushed.
inline void write_ack(volatile uint64_t* ack, uint64_t value) {
    std::atomic_thread_fence(std::memory_order_release);
    *ack = value;
    _mm_sfence();
}

inline void free_ack(volatile uint64_t* ack, const std::string& where) {
    if (where == "host")
        QB_HIP_CHECK(hipHostFree((void*)ack));
    else
        QB_HIP_CHECK(hipFree((void*)ack));
}

/// Overlap of the producer waves from their recorded spans: `overlap` is the
/// number of waves still running when the last wave started (all of them if
/// every wave was resident at once), `spread_us` the time from the first to
/// the last wave start.
struct Overlap {
    int overlap;
    double spread_us;
};

/// Device buffer for a --check-overlap run's wave spans, or null without
/// --check-overlap: wave w writes wall_clock64() ticks to [2w] when it starts
/// and to [2w + 1] when it finishes.
inline uint64_t* alloc_wave_spans(bool on, int waves) {
    if (!on) return nullptr;
    uint64_t* d = nullptr;
    QB_HIP_CHECK(hipMalloc(&d, 2 * static_cast<size_t>(waves) * sizeof(uint64_t)));
    return d;
}

inline Overlap wave_overlap(const uint64_t* d_spans, int waves) {
    std::vector<uint64_t> s(2 * static_cast<size_t>(waves));
    QB_HIP_CHECK(hipMemcpy(s.data(), d_spans, s.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
    uint64_t first = ~uint64_t{0}, last = 0;
    for (int w = 0; w < waves; ++w) {
        first = std::min(first, s[2 * w]);
        last = std::max(last, s[2 * w]);
    }
    int overlap = 0;
    for (int w = 0; w < waves; ++w) overlap += s[2 * w + 1] >= last;
    int wall_khz = 0;
    QB_HIP_CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    return {overlap, static_cast<double>(last - first) * 1e3 / wall_khz};
}

inline void print_header(const Options& o, const char* name, int num_queues,
                  int descriptor_bytes, const std::string& notes) {
    std::printf("# queue=%s mode=%s capacity=%d queues=%d waves=%d threads=%d "
                "descriptor_bytes=%d\n# %s\n",
                name, o.mode.c_str(), o.capacity, num_queues, o.waves, o.threads,
                descriptor_bytes, notes.c_str());
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

template <typename A, typename Launcher>
int run_latency(const Options& o, Launcher& launcher) {
    const int num_queues = 1;
    A adapter(num_queues, o.capacity);
    print_header(o, queue_name<A>(o), num_queues, A::kDescriptorBytes, A::notes());
    // Single producer: pass the handle by value as a kernel argument (as the
    // KIWI latency benchmark does), so queue pointers compile to global_* ops.
    const typename A::Handle handle = adapter.handle(0);

    const long total = o.warmup + o.iters;
    volatile uint64_t* ack = alloc_ack(o.ack_mem);
    uint64_t* d_ticks = nullptr;
    QB_HIP_CHECK(hipMalloc(&d_ticks, total * sizeof(uint64_t)));

    int wall_khz = 0;
    QB_HIP_CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));

    pin_current_thread_to_cpu(worker_cpu_plan(1)[0]);
    Validator validator(1, num_queues, total, A::kFifo);
    if constexpr (kSync<A>) A::sink = [&](const Msg& m) { validator.check(0, m); };

    hipStream_t stream;
    QB_HIP_CHECK(hipStreamCreate(&stream));
    launcher.latency(LatencyLaunch<A>{handle, ack, total, d_ticks, stream});
    QB_HIP_CHECK(hipGetLastError());

    bool ok = true;
    if constexpr (kSync<A>) {
        // The runtime's listener thread validates inside the host function;
        // the call's own completion is the acknowledgement.
        QB_HIP_CHECK(hipStreamSynchronize(stream));
    } else {
        for (long i = 0; i < total; ++i) {
            Msg m;
            while (!adapter.pop(0, m)) {}
            if (!validator.check(0, m)) ok = false;
            write_ack(ack, static_cast<uint64_t>(i) + 1);
        }
        QB_HIP_CHECK(hipStreamSynchronize(stream));
    }
    ok = ok && validator.complete();

    std::vector<uint64_t> ticks(total);
    QB_HIP_CHECK(hipMemcpy(ticks.data(), d_ticks, total * sizeof(uint64_t),
                           hipMemcpyDeviceToHost));
    std::vector<double> ns;
    ns.reserve(o.iters);
    for (long i = o.warmup; i < total; ++i)
        ns.push_back(static_cast<double>(ticks[i]) * 1e6 / wall_khz);
    const Percentiles p = percentiles(ns);

    std::printf("RESULT queue=%s mode=latency ack_mem=%s capacity=%d iters=%ld valid=%d "
                "rtt_min_ns=%.0f rtt_p50_ns=%.0f rtt_p90_ns=%.0f rtt_p99_ns=%.0f "
                "rtt_p999_ns=%.0f rtt_max_ns=%.0f rtt_avg_ns=%.1f\n",
                queue_name<A>(o), o.ack_mem.c_str(), o.capacity, o.iters, ok ? 1 : 0, p.min, p.p50, p.p90,
                p.p99, p.p999, p.max, p.avg);

    QB_HIP_CHECK(hipStreamDestroy(stream));
    QB_HIP_CHECK(hipFree(d_ticks));
    free_ack(ack, o.ack_mem);
    return ok ? 0 : 1;
}

/// Checks shared by the multi-wave modes; returns the queue count or -1.
template <typename A>
int multi_wave_queues(const Options& o) {
    const int num_queues = kSync<A> ? 1 : (o.queues > 0 ? o.queues : o.waves);
    if (!A::kMultiProducer && num_queues != o.waves) {
        std::fprintf(stderr, "ERROR: %s is single-producer; use --queues 0 or %d\n",
                     queue_name<A>(o), o.waves);
        return -1;
    }
    if (!kSync<A> && (num_queues > o.waves || o.threads > num_queues)) {
        std::fprintf(stderr, "ERROR: need threads <= queues <= waves\n");
        return -1;
    }
    return num_queues;
}

template <typename A, typename Launcher>
int run_push_cost(const Options& o, Launcher& launcher) {
    const int num_queues = multi_wave_queues<A>(o);
    if (num_queues < 0) return 2;
    const int blocks = (o.waves + o.waves_per_block - 1) / o.waves_per_block;
    const bool wall = o.timer == "wall";
    if (!launcher.fits(o, blocks)) return 2;
    A adapter(num_queues, o.capacity);
    print_header(o, queue_name<A>(o), num_queues, A::kDescriptorBytes, A::notes());
    auto* handles = upload_handles(adapter, o.waves, num_queues);

    const long total = o.warmup + o.iters;
    const size_t samples = o.burst ? o.waves : static_cast<size_t>(o.waves) * o.iters;
    uint64_t* d_cycles = nullptr;
    uint64_t* d_calibration = nullptr;
    QB_HIP_CHECK(hipMalloc(&d_cycles, samples * sizeof(uint64_t)));
    QB_HIP_CHECK(hipMalloc(&d_calibration, sizeof(uint64_t)));

    Validator validator(o.waves, num_queues, total, A::kFifo);
    ConsumerPool<A> pool(adapter, validator, o, num_queues, total);
    uint64_t* d_spans = alloc_wave_spans(o.check_overlap, o.waves);
    pool.start();
    launcher.push_cost(PushCostLaunch<A>{handles, blocks, o.waves, o.waves_per_block, o.warmup,
                                         o.iters, o.burst, wall, d_cycles, d_calibration,
                                         d_spans});
    QB_HIP_CHECK(hipGetLastError());
    pool.join();
    QB_HIP_CHECK(hipDeviceSynchronize());
    const Overlap ov = o.check_overlap ? wave_overlap(d_spans, o.waves) : Overlap{-1, -1.0};
    if (d_spans) QB_HIP_CHECK(hipFree(d_spans));
    const bool valid =
        pool.ok() && validator.complete() && (!o.check_overlap || ov.overlap == o.waves);

    std::vector<uint64_t> cycles(samples);
    uint64_t calibration = 0;
    QB_HIP_CHECK(hipMemcpy(cycles.data(), d_cycles, samples * sizeof(uint64_t),
                           hipMemcpyDeviceToHost));
    QB_HIP_CHECK(hipMemcpy(&calibration, d_calibration, sizeof(uint64_t),
                           hipMemcpyDeviceToHost));

    std::vector<double> c;
    c.reserve(samples);
    long dropped = 0;
    if (o.burst) {
        // Per-wave loop time / pushes; the bracket cost is negligible.
        for (uint64_t v : cycles) c.push_back(static_cast<double>(v) / o.iters);
    } else {
        // A sample below the empty-bracket cost (or a wrapped t1 < t0) is a
        // timer artifact; drop and report it rather than skew the mean.
        for (uint64_t v : cycles) {
            const int64_t d = static_cast<int64_t>(v) - static_cast<int64_t>(calibration);
            if (d < 0) {
                ++dropped;
                continue;
            }
            c.push_back(static_cast<double>(d));
        }
    }
    const Percentiles p = percentiles(c);

    // Shader-clock results stay in cycles; wall-clock ticks are reported in ns.
    double unit = 1.0;
    if (wall) {
        int wall_khz = 0;
        QB_HIP_CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
        unit = 1e6 / wall_khz;
    }
    const char* u = wall ? "ns" : "cycles";
    std::printf("RESULT queue=%s mode=push-cost pattern=%s capacity=%d queues=%d "
                "waves=%d threads=%d idle_pause=%d iters=%ld valid=%d timer=%s "
                "timer_overhead_%s=%.0f push_min_%s=%.0f push_p50_%s=%.0f push_p90_%s=%.0f "
                "push_p99_%s=%.0f push_avg_%s=%.1f dropped_samples=%ld overlap_waves=%d "
                "start_spread_us=%.1f\n",
                queue_name<A>(o), o.burst ? "burst" : "isolated", o.capacity, num_queues, o.waves,
                o.threads, o.idle_pause, o.iters, valid ? 1 : 0, o.timer.c_str(), u,
                calibration * unit, u, p.min * unit, u, p.p50 * unit, u, p.p90 * unit, u,
                p.p99 * unit, u, p.avg * unit, dropped, ov.overlap, ov.spread_us);

    QB_HIP_CHECK(hipFree(d_cycles));
    QB_HIP_CHECK(hipFree(d_calibration));
    QB_HIP_CHECK(hipFree(handles));
    return valid ? 0 : 1;
}

template <typename A, typename Launcher>
int run_throughput(const Options& o, Launcher& launcher) {
    // A synchronous adapter has a single runtime-owned consumer.
    const int num_queues = multi_wave_queues<A>(o);
    if (num_queues < 0) return 2;
    const int blocks = (o.waves + o.waves_per_block - 1) / o.waves_per_block;
    if (!launcher.fits(o, blocks)) return 2;
    A adapter(num_queues, o.capacity);
    print_header(o, queue_name<A>(o), num_queues, A::kDescriptorBytes, A::notes());
    auto* handles = upload_handles(adapter, o.waves, num_queues);

    Validator validator(o.waves, num_queues, o.iters, A::kFifo);
    ConsumerPool<A> pool(adapter, validator, o, num_queues);
    uint64_t* d_spans = alloc_wave_spans(o.check_overlap, o.waves);
    hipEvent_t ev0, ev1;
    QB_HIP_CHECK(hipEventCreate(&ev0));
    QB_HIP_CHECK(hipEventCreate(&ev1));
    pool.start();
    const auto t0 = std::chrono::steady_clock::now();
    QB_HIP_CHECK(hipEventRecord(ev0));
    launcher.throughput(
        ThroughputLaunch<A>{handles, blocks, o.waves, o.waves_per_block, o.iters, d_spans});
    QB_HIP_CHECK(hipGetLastError());
    QB_HIP_CHECK(hipEventRecord(ev1));
    pool.join();
    if constexpr (kSync<A>) QB_HIP_CHECK(hipEventSynchronize(ev1));
    const auto t1 = std::chrono::steady_clock::now();
    QB_HIP_CHECK(hipEventSynchronize(ev1));
    float kernel_ms = 0;
    QB_HIP_CHECK(hipEventElapsedTime(&kernel_ms, ev0, ev1));
    const Overlap ov = o.check_overlap ? wave_overlap(d_spans, o.waves) : Overlap{-1, -1.0};
    if (d_spans) QB_HIP_CHECK(hipFree(d_spans));
    const bool valid =
        pool.ok() && validator.complete() && (!o.check_overlap || ov.overlap == o.waves);

    const double total_msgs = static_cast<double>(o.waves) * o.iters;
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("RESULT queue=%s mode=throughput capacity=%d queues=%d waves=%d "
                "threads=%d batch=%d idle_pause=%d msgs_per_wave=%ld valid=%d wall_ms=%.3f "
                "kernel_ms=%.3f mops=%.3f overlap_waves=%d start_spread_us=%.1f\n",
                queue_name<A>(o), o.capacity, num_queues, o.waves, o.threads, o.batch,
                o.idle_pause, o.iters, valid ? 1 : 0, wall_s * 1e3, kernel_ms,
                total_msgs / wall_s / 1e6, ov.overlap, ov.spread_us);

    QB_HIP_CHECK(hipEventDestroy(ev0));
    QB_HIP_CHECK(hipEventDestroy(ev1));
    QB_HIP_CHECK(hipFree(handles));
    return valid ? 0 : 1;
}

template <typename A>
int run_idle_poll(const Options& o) {
    const int num_queues = o.queues > 0 ? o.queues : 1;
    A adapter(num_queues, o.capacity);
    print_header(o, queue_name<A>(o), num_queues, A::kDescriptorBytes, A::notes());
    pin_current_thread_to_cpu(worker_cpu_plan(1)[0]);

    long spurious = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (long r = 0; r < o.idle_rounds; ++r) {
        for (int q = 0; q < num_queues; ++q) {
            Msg m;
            if (adapter.pop(q, m)) ++spurious;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    std::printf("RESULT queue=%s mode=idle-poll capacity=%d queues=%d rounds=%ld "
                "valid=%d ns_per_queue_poll=%.2f\n",
                queue_name<A>(o), o.capacity, num_queues, o.idle_rounds,
                spurious == 0 ? 1 : 0, ns / (static_cast<double>(o.idle_rounds) * num_queues));
    return spurious == 0 ? 0 : 1;
}

}  // namespace qb
