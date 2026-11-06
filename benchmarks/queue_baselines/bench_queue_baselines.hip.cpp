// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file bench_queue_baselines.hip.cpp
 * @brief GPU->CPU handoff microbenchmarks: KIWI-Queue vs. other reverse-offload queues.
 *
 * One harness drives every queue through the adapter contract in
 * qb_common.hpp, so all queues share the same kernels, host proxy loop,
 * CPU pinning, acknowledgement path, timers, and validation.
 *
 * Modes:
 *   latency     one wave: push -> host pop -> host ack -> wave sees ack (RTT)
 *   push-cost   one wave: GPU cycles spent inside push() with an active consumer
 *   throughput  W waves x M messages, P pinned proxy threads, Q queues
 *   idle-poll   host cost of polling Q empty queues
 *
 * Every popped message is validated (producer, seq range, tag, no duplicates,
 * exact count); FIFO order is also checked for adapters that promise it.
 *
 * Every producer wave of a multi-wave run must be resident at once: kernels
 * carry __launch_bounds__(kMaxBlockThreads), so the compiler sizes per-block
 * resources (e.g. a stack array it promotes to LDS) for the blocks actually
 * launched, and a launch fails if its grid cannot be resident. With
 * --check-overlap, push-cost and throughput runs also record when each wave
 * starts and ends and report the waves still running when the last one
 * started (overlap_waves), which must then equal the wave count for the run
 * to be valid. The recording is a separate kernel instantiation, off by
 * default: it changes the code around the timed push and moves the measured
 * cost (e.g. MSCCL++ at W=1024: 7.3 -> 5.8 us), so measurements run without it.
 */

#include "qb_common.hpp"
#include "adapters/kiwi.hpp"
#include "adapters/llvm_rpc.hpp"
#include "adapters/mscclpp_fifo.hpp"
#include "adapters/rccl_gin_proxy.hpp"
#include "adapters/nvshmem_proxy.hpp"
#include "adapters/raw_store.hpp"
#include "adapters/rocm_hostcall.hpp"
#include "adapters/rocshmem_ro.hpp"
#include "adapters/uccl_ep_fifo.hpp"

#include <bench_common.hpp>
#include <kiwi/queue/detail/host_memory.hpp>
#include <CLI/CLI.hpp>

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace qb {

/// Largest workgroup the harness launches (--waves-per-block <= 4). Every
/// kernel is bounded to it: without a bound the compiler assumes 1024-thread
/// workgroups, and a stack array it promotes to LDS is then reserved for 1024
/// threads per block (32 KiB for the RCCL GIN push), which caps residency.
constexpr int kMaxWavesPerBlock = 4;
constexpr int kMaxBlockThreads = 64 * kMaxWavesPerBlock;

struct Options {
    std::string queue = "kiwi-spsc";
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
// Kernels
// ---------------------------------------------------------------------------

/// Wait until all of this wave's outstanding memory operations (including
/// posted stores to host memory) are acknowledged.
__device__ inline void drain_memory_ops() {
    __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
}

/// Wave start/end ticks of a --check-overlap run: [2w] start, [2w + 1] end.
/// A device global rather than a kernel argument, so the measured kernels keep
/// main's exact signature and code.
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

/// Overlap of the producer waves from their recorded spans: `overlap` is the
/// number of waves still running when the last wave started (all of them if
/// every wave was resident at once), `spread_us` the time from the first to
/// the last wave start.
struct Overlap {
    int overlap;
    double spread_us;
};

/// Device buffer for g_wave_spans (2 entries per wave), or null without
/// --check-overlap.
inline uint64_t* alloc_wave_spans(bool on, int waves) {
    if (!on) return nullptr;
    uint64_t* d = nullptr;
    QB_HIP_CHECK(hipMalloc(&d, 2 * static_cast<size_t>(waves) * sizeof(uint64_t)));
    QB_HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_wave_spans), &d, sizeof(d)));
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

/// Latency under load: wave 0 is the probe (push, wait for its ack), and
/// waves 1..W push back to back until the probe finishes. Background waves
/// stop at `bg_max` messages; each reports how many it pushed.
template <typename A>
__global__ __launch_bounds__(kMaxBlockThreads) void loaded_latency_kernel(const typename A::Handle* handles, int waves,
                                      int waves_per_block, volatile uint64_t* ack,
                                      long probe_iters, uint64_t* latency_ticks, int* stop,
                                      long bg_max, uint64_t gap_cycles, long* counts) {
    const int wave = blockIdx.x * waves_per_block + threadIdx.x / 64;
    if (wave >= waves) return;
    typename A::Handle h = handles[wave];
    A::globalize(h);
    if (wave == 0) {
        uint64_t t_prev = wall_clock64();
        for (long i = 0; i < probe_iters; ++i) {
            A::push(h, make_msg(0, static_cast<uint32_t>(i)));
            drain_memory_ops();
            const uint64_t expected = static_cast<uint64_t>(i) + 1;
            while (*ack < expected) {}
            const uint64_t t_now = wall_clock64();
            if (lane_id() == 0) latency_ticks[i] = t_now - t_prev;
            t_prev = t_now;
        }
        if (lane_id() == 0) __hip_atomic_store(stop, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else {
        // Throttled waves issue one push every gap_cycles (open loop: a push
        // that overruns its slot is not made up later).
        long n = 0;
        uint64_t next = clock64();
        while (n < bg_max) {
            if ((n & 15) == 0 &&
                __hip_atomic_load(stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                break;
            if (gap_cycles) {
                while (clock64() < next) __builtin_amdgcn_s_sleep(1);
                next = clock64() + gap_cycles;
            }
            A::push(h, make_msg(static_cast<uint32_t>(wave), static_cast<uint32_t>(n)));
            ++n;
        }
        if (lane_id() == 0) counts[wave] = n;
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

/// Floor measurements for the push-cost bracket (one wave, isolated):
///   0 empty bracket, 1 LDS stage round trip (as the KIWI adapter does),
///   2 one wave-wide 64B store (8 lanes x 8B) to device memory,
///   3 ... to coherent host memory (hipHostMallocCoherent),
///   4 ... to KIWI's uncached extended-scope host memory (host_malloc_uc).
constexpr int kFloorVariants = 5;

__global__ __launch_bounds__(kMaxBlockThreads) void floor_kernel(uint64_t* dev, uint64_t* host_coherent, uint64_t* host_uc,
                             long iters, uint64_t* cycles) {
    __shared__ uint64_t row[8];
    const int lane = lane_id();
    for (int v = 0; v < kFloorVariants; ++v) {
        uint64_t* target = v == 2 ? dev : v == 3 ? host_coherent : host_uc;
        for (long i = 0; i < iters; ++i) {
            const uint64_t value = static_cast<uint64_t>(i) * 8 + lane;
            drain_memory_ops();
            const uint64_t t0 = timestamp();
            if (v == 1) {
                if (lane < 7) row[lane] = value;
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
                const uint64_t x = lane < 8 ? row[lane & 7] : 0;
                __asm__ volatile("" ::"v"(x));
            } else if (v >= 2) {
                if (lane < 8) target[(i & 63) * 8 + lane] = value;
            }
            const uint64_t t1 = timestamp();
            if (lane == 0) cycles[v * iters + i] = t1 - t0;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct Percentiles {
    double min, p50, p90, p99, p999, max, avg;
};

Percentiles percentiles(std::vector<double> v) {
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
volatile uint64_t* alloc_ack(const std::string& where) {
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

void free_ack(volatile uint64_t* ack, const std::string& where) {
    if (where == "host")
        QB_HIP_CHECK(hipHostFree((void*)ack));
    else
        QB_HIP_CHECK(hipFree((void*)ack));
}

void print_header(const Options& o, const char* name, int num_queues,
                  int descriptor_bytes, const std::string& notes) {
    std::printf("# queue=%s mode=%s capacity=%d queues=%d waves=%d threads=%d "
                "descriptor_bytes=%d\n# %s\n",
                name, o.mode.c_str(), o.capacity, num_queues, o.waves, o.threads,
                descriptor_bytes, notes.c_str());
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

template <typename A>
int run_latency(const Options& o) {
    const int num_queues = 1;
    A adapter(num_queues, o.capacity);
    print_header(o, A::kName, num_queues, A::kDescriptorBytes, A::notes());
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
    latency_kernel<A><<<1, 64, 0, stream>>>(handle, ack, total, d_ticks);
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
                A::kName, o.ack_mem.c_str(), o.capacity, o.iters, ok ? 1 : 0, p.min, p.p50, p.p90,
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
                     A::kName, o.waves);
        return -1;
    }
    if (!kSync<A> && (num_queues > o.waves || o.threads > num_queues)) {
        std::fprintf(stderr, "ERROR: need threads <= queues <= waves\n");
        return -1;
    }
    return num_queues;
}

template <typename A>
int run_push_cost(const Options& o) {
    const int num_queues = multi_wave_queues<A>(o);
    if (num_queues < 0) return 2;
    const int blocks = (o.waves + o.waves_per_block - 1) / o.waves_per_block;
    const bool wall = o.timer == "wall";
    auto* kernel = o.check_overlap ? (wall ? push_cost_kernel<A, true, true>
                                           : push_cost_kernel<A, false, true>)
                                    : (wall ? push_cost_kernel<A, true, false>
                                           : push_cost_kernel<A, false, false>);
    if (!grid_fits(kernel, blocks, 64 * o.waves_per_block, "push_cost_kernel")) return 2;
    A adapter(num_queues, o.capacity);
    print_header(o, A::kName, num_queues, A::kDescriptorBytes, A::notes());
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
    kernel<<<blocks, 64 * o.waves_per_block>>>(handles, o.waves, o.waves_per_block, o.warmup,
                                               o.iters, o.burst, d_cycles, d_calibration);
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
                A::kName, o.burst ? "burst" : "isolated", o.capacity, num_queues, o.waves,
                o.threads, o.idle_pause, o.iters, valid ? 1 : 0, o.timer.c_str(), u,
                calibration * unit, u, p.min * unit, u, p.p50 * unit, u, p.p90 * unit, u,
                p.p99 * unit, u, p.avg * unit, dropped, ov.overlap, ov.spread_us);

    QB_HIP_CHECK(hipFree(d_cycles));
    QB_HIP_CHECK(hipFree(d_calibration));
    QB_HIP_CHECK(hipFree(handles));
    return valid ? 0 : 1;
}

template <typename A>
int run_throughput(const Options& o) {
    // A synchronous adapter has a single runtime-owned consumer.
    const int num_queues = multi_wave_queues<A>(o);
    if (num_queues < 0) return 2;
    const int blocks = (o.waves + o.waves_per_block - 1) / o.waves_per_block;
    auto* kernel = o.check_overlap ? throughput_kernel<A, true> : throughput_kernel<A, false>;
    if (!grid_fits(kernel, blocks, 64 * o.waves_per_block, "throughput_kernel")) return 2;
    A adapter(num_queues, o.capacity);
    print_header(o, A::kName, num_queues, A::kDescriptorBytes, A::notes());
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
    kernel<<<blocks, 64 * o.waves_per_block>>>(handles, o.waves, o.waves_per_block, o.iters);
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
                A::kName, o.capacity, num_queues, o.waves, o.threads, o.batch,
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
    print_header(o, A::kName, num_queues, A::kDescriptorBytes, A::notes());
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
                A::kName, o.capacity, num_queues, o.idle_rounds,
                spurious == 0 ? 1 : 0, ns / (static_cast<double>(o.idle_rounds) * num_queues));
    return spurious == 0 ? 0 : 1;
}

/// Probe RTT (wave 0) with --waves background producers pushing concurrently
/// through the same proxy threads. --queues 0 gives every wave its own queue.
template <typename A>
int run_loaded_latency(const Options& o) {
    const int total_waves = o.waves + 1;
    const int num_queues = o.queues > 0 ? o.queues : total_waves;
    if (!A::kMultiProducer && num_queues != total_waves) {
        std::fprintf(stderr, "ERROR: %s is single-producer; use --queues 0\n", A::kName);
        return 2;
    }
    if (num_queues > total_waves || o.threads > num_queues) {
        std::fprintf(stderr, "ERROR: need threads <= queues <= waves + 1\n");
        return 2;
    }
    const int blocks = (total_waves + o.waves_per_block - 1) / o.waves_per_block;
    if (!grid_fits(loaded_latency_kernel<A>, blocks, 64 * o.waves_per_block,
                   "loaded_latency_kernel"))
        return 2;
    A adapter(num_queues, o.capacity);
    print_header(o, A::kName, num_queues, A::kDescriptorBytes, A::notes());
    auto* handles = upload_handles(adapter, total_waves, num_queues);

    const long probe_iters = o.warmup + o.iters;
    volatile uint64_t* ack = alloc_ack(o.ack_mem);
    uint64_t* d_ticks = nullptr;
    int* d_stop = nullptr;
    long* d_counts = nullptr;
    QB_HIP_CHECK(hipMalloc(&d_ticks, probe_iters * sizeof(uint64_t)));
    QB_HIP_CHECK(hipMalloc(&d_stop, sizeof(int)));
    QB_HIP_CHECK(hipMemset(d_stop, 0, sizeof(int)));
    QB_HIP_CHECK(hipMalloc(&d_counts, total_waves * sizeof(long)));
    QB_HIP_CHECK(hipMemset(d_counts, 0, total_waves * sizeof(long)));

    Validator validator(total_waves, num_queues, std::max(probe_iters, o.bg_max), A::kFifo);

    // Proxy threads: drain every queue, validate, and ack the probe. They
    // stop once the kernel is done and every counted message was consumed.
    const std::vector<int> cpus = worker_cpu_plan(o.threads);
    std::atomic<bool> go{false}, ok{true}, targets_ready{false};
    std::vector<long> target(o.threads, 0);
    std::vector<std::thread> workers;
    for (int t = 0; t < o.threads; ++t) {
        workers.emplace_back([&, t] {
            pin_current_thread_to_cpu(cpus[t]);
            std::vector<int> mine;
            for (int q = t; q < num_queues; q += o.threads) mine.push_back(q);
            long received = 0;
            while (!go.load(std::memory_order_acquire)) {}
            while (!(targets_ready.load(std::memory_order_acquire) && received >= target[t])) {
                bool progressed = false;
                for (int q : mine) {
                    Msg m;
                    for (int b = 0; b < o.batch && adapter.pop(q, m); ++b) {
                        if (!validator.check(q, m)) ok = false;
                        if (m.producer == 0) write_ack(ack, static_cast<uint64_t>(m.seq) + 1);
                        ++received;
                        progressed = true;
                    }
                }
                if (!progressed)
                    for (int k = 0; k < o.idle_pause; ++k) __builtin_ia32_pause();
            }
        });
    }

    int wall_khz = 0;
    QB_HIP_CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    // Per-wave gap for the requested total rate, in shader cycles (clock64).
    int clock_khz = 0;
    QB_HIP_CHECK(hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeClockRate, 0));
    const uint64_t gap_cycles =
        o.bg_rate > 0 ? static_cast<uint64_t>(o.waves * (clock_khz * 1e3) / (o.bg_rate * 1e6))
                      : 0;
    go.store(true, std::memory_order_release);
    loaded_latency_kernel<A><<<blocks, 64 * o.waves_per_block>>>(
        handles, total_waves, o.waves_per_block, ack, probe_iters, d_ticks, d_stop, o.bg_max,
        gap_cycles, d_counts);
    QB_HIP_CHECK(hipGetLastError());
    QB_HIP_CHECK(hipDeviceSynchronize());

    std::vector<long> counts(total_waves);
    QB_HIP_CHECK(hipMemcpy(counts.data(), d_counts, total_waves * sizeof(long),
                           hipMemcpyDeviceToHost));
    counts[0] = probe_iters;
    for (int w = 0; w < total_waves; ++w) target[(w % num_queues) % o.threads] += counts[w];
    targets_ready.store(true, std::memory_order_release);
    for (auto& w : workers) w.join();
    const bool valid = ok && validator.complete(counts);

    std::vector<uint64_t> ticks(probe_iters);
    QB_HIP_CHECK(hipMemcpy(ticks.data(), d_ticks, probe_iters * sizeof(uint64_t),
                           hipMemcpyDeviceToHost));
    std::vector<double> ns;
    double probe_ns = 0;
    for (long i = o.warmup; i < probe_iters; ++i) {
        ns.push_back(static_cast<double>(ticks[i]) * 1e6 / wall_khz);
        probe_ns += ns.back();
    }
    const Percentiles p = percentiles(ns);
    long bg_total = 0, bg_capped = 0;
    for (int w = 1; w < total_waves; ++w) {
        bg_total += counts[w];
        bg_capped += counts[w] >= o.bg_max;
    }
    // Background rate over the whole probe run (warmup included).
    double all_ns = 0;
    for (long i = 0; i < probe_iters; ++i) all_ns += static_cast<double>(ticks[i]) * 1e6 / wall_khz;

    std::printf("RESULT queue=%s mode=loaded-latency ack_mem=%s capacity=%d queues=%d bg_waves=%d "
                "threads=%d idle_pause=%d bg_target_mops=%.3f iters=%ld valid=%d rtt_min_ns=%.0f "
                "rtt_p50_ns=%.0f rtt_p90_ns=%.0f rtt_p99_ns=%.0f rtt_p999_ns=%.0f "
                "rtt_avg_ns=%.1f bg_mops=%.3f bg_waves_capped=%ld\n",
                A::kName, o.ack_mem.c_str(), o.capacity, num_queues, o.waves, o.threads,
                o.idle_pause, o.bg_rate,
                o.iters,
                valid ? 1 : 0, p.min, p.p50, p.p90, p.p99, p.p999, p.avg,
                all_ns > 0 ? bg_total / all_ns * 1e3 : 0.0, bg_capped);

    QB_HIP_CHECK(hipFree(d_ticks));
    QB_HIP_CHECK(hipFree(d_stop));
    QB_HIP_CHECK(hipFree(d_counts));
    QB_HIP_CHECK(hipFree(handles));
    free_ack(ack, o.ack_mem);
    return valid ? 0 : 1;
}

int run_floor(const Options& o) {
    uint64_t* dev = nullptr;
    uint64_t* host_coherent = nullptr;
    const size_t bytes = 64 * 64;
    QB_HIP_CHECK(hipMalloc(&dev, bytes));
    QB_HIP_CHECK(hipHostMalloc((void**)&host_coherent, bytes, hipHostMallocCoherent));
    auto* host_uc = kiwi::queue::detail::host_malloc_uc<uint64_t>(bytes / sizeof(uint64_t));
    uint64_t* d_cycles = nullptr;
    QB_HIP_CHECK(hipMalloc(&d_cycles, kFloorVariants * o.iters * sizeof(uint64_t)));
    floor_kernel<<<1, 64>>>(dev, host_coherent, host_uc, o.iters, d_cycles);
    QB_HIP_CHECK(hipGetLastError());
    QB_HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint64_t> cycles(kFloorVariants * o.iters);
    QB_HIP_CHECK(hipMemcpy(cycles.data(), d_cycles, cycles.size() * sizeof(uint64_t),
                           hipMemcpyDeviceToHost));
    const char* names[kFloorVariants] = {"empty", "lds_stage", "store64_device",
                                         "store64_host_coherent", "store64_host_uc"};
    for (int v = 0; v < kFloorVariants; ++v) {
        std::vector<double> c(cycles.begin() + v * o.iters, cycles.begin() + (v + 1) * o.iters);
        const Percentiles p = percentiles(c);
        std::printf("RESULT mode=floor op=%s iters=%ld valid=1 min_cycles=%.0f "
                    "p50_cycles=%.0f p99_cycles=%.0f (bracket not subtracted)\n",
                    names[v], o.iters, p.min, p.p50, p.p99);
    }
    QB_HIP_CHECK(hipFree(d_cycles));
    QB_HIP_CHECK(hipFree(dev));
    QB_HIP_CHECK(hipHostFree(host_coherent));
    kiwi::queue::detail::host_free_uc(host_uc);
    return 0;
}

template <typename A>
int run(const Options& o) {
    if (kUnbounded<A> && (o.mode == "throughput" || o.mode == "loaded-latency")) {
        std::fprintf(stderr, "%s has no capacity check: mode %s does not apply\n", A::kName,
                     o.mode.c_str());
        return 2;
    }
    if (kUnbounded<A> && o.mode == "push-cost" &&
        static_cast<long>(o.capacity) < o.warmup + o.iters) {
        std::fprintf(stderr, "%s: push-cost needs capacity >= warmup + iters\n", A::kName);
        return 2;
    }
    if (kSync<A> && (o.mode == "push-cost" || o.mode == "idle-poll" ||
                     o.mode == "loaded-latency")) {
        std::fprintf(stderr, "%s is synchronous: mode %s does not apply\n", A::kName,
                     o.mode.c_str());
        return 2;
    }
    if (o.mode == "latency") return run_latency<A>(o);
    if (o.mode == "push-cost") return run_push_cost<A>(o);
    if (o.mode == "throughput") return run_throughput<A>(o);
    if (o.mode == "loaded-latency") return run_loaded_latency<A>(o);
    if (o.mode == "idle-poll") return run_idle_poll<A>(o);
    std::fprintf(stderr, "ERROR: unknown mode %s\n", o.mode.c_str());
    return 2;
}

}  // namespace qb

int main(int argc, char** argv) {
    using namespace qb;
    CLI::App app{"GPU->CPU handoff microbenchmarks: KIWI-Queue vs. reverse-offload baselines"};
    Options o;
    const std::vector<std::string> queues = {"kiwi-spsc", "kiwi-mpsc", "kiwi-invoke", "mscclpp-fifo",
                                             "llvm-rpc", "rocm-hostcall", "rocshmem-ro",
                                             "uccl-ep-fifo", "nvshmem-proxy",
                                             "rccl-gin-proxy", "raw-store"};
    app.add_option("-q,--queue", o.queue, "Queue implementation")
        ->check(CLI::IsMember(queues));
    app.add_option("-m,--mode", o.mode,
                   "latency | loaded-latency | push-cost | throughput | idle-poll | floor")
        ->check(CLI::IsMember({"latency", "loaded-latency", "push-cost", "throughput",
                               "idle-poll", "floor"}));
    app.add_option("-c,--capacity", o.capacity, "Slots (ports for llvm-rpc) per queue; power of two")
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
    app.add_option("--bg-max", o.bg_max,
                   "loaded-latency: max messages per background wave")
        ->check(CLI::PositiveNumber);
    app.add_option("--ack-mem", o.ack_mem,
                   "latency modes: ack flag in 'device' (fine-grained VRAM, polled locally) "
                   "or 'host' memory (each poll is a PCIe read)")
        ->check(CLI::IsMember({"device", "host"}));
    app.add_option("--bg-rate", o.bg_rate,
                   "loaded-latency: total background Mops, throttled on the GPU (0 = unthrottled)")
        ->check(CLI::NonNegativeNumber);
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

    check_numa_affinity();
    gpu_clock_warmup();
    if (o.mode == "floor") return run_floor(o);

    if (o.queue == "kiwi-spsc") return run<KiwiSpscAdapter>(o);
    if (o.queue == "kiwi-mpsc") return run<KiwiMpscAdapter>(o);
    if (o.queue == "kiwi-invoke") return run<KiwiInvokeAdapter>(o);
    if (o.queue == "mscclpp-fifo") return run<MscclppFifoAdapter>(o);
    if (o.queue == "llvm-rpc") return run<LlvmRpcAdapter>(o);
    if (o.queue == "rocm-hostcall") return run<RocmHostcallAdapter>(o);
    if (o.queue == "rocshmem-ro") return run<RocshmemRoAdapter>(o);
    if (o.queue == "uccl-ep-fifo") return run<UcclEpFifoAdapter>(o);
    if (o.queue == "nvshmem-proxy") return run<NvshmemProxyAdapter>(o);
    if (o.queue == "rccl-gin-proxy") return run<RcclGinProxyAdapter>(o);
    if (o.queue == "raw-store") return run<RawStoreAdapter>(o);
    return 2;
}
