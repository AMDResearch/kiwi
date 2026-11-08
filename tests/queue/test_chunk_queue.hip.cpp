// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_chunk_queue.hip.cpp
 * @brief Device-to-host queues (SPSC and MPSC): chunk format, every push
 *        path, wraparound, consumer index publication, backpressure, late
 *        chunks, and many-producer MPSC stress
 *
 * Every stream checks each popped message's length, metadata, payload and
 * that pop() wrote exactly `bytes` bytes of the caller's buffer,
 * exactly-once delivery, and per-producer-wave FIFO order.
 */

#include <kiwi/queue/queue.hpp>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace kiwi::queue;
using detail::Chunk;

#define HIP_OK(expr) ASSERT_EQ((expr), hipSuccess)

namespace {

constexpr int kMaxWords = static_cast<int>(kMaxMessageBytes / 8);
constexpr int kSeqBits = 20;

/// Header-only, partial and whole words, 1 to 16 lines, and the KIWI-EP /
/// kiwi-p2p descriptor sizes.
constexpr int kSizes[] = {0, 1, 4, 8, 13, 24, 32, 33, 48, 56, 64, 100, 168, 255, 448, 512};
constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

/// Payload word k of the message with key `key`.
__host__ __device__ inline uint64_t word(uint64_t key, int k) {
    uint64_t z = key * 0x9e3779b97f4a7c15ull + static_cast<uint64_t>(k) * 0xbf58476d1ce4e5b9ull + 1;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

__host__ __device__ inline uint64_t msg_key(uint32_t wave, uint32_t seq) {
    return (static_cast<uint64_t>(wave) << 32) | seq;
}

/// Message (wave, seq): a size from kSizes; in a ring too small for it, the
/// largest size that fits. With path = seq % 4, every size meets every path
/// in 256 messages.
__host__ __device__ inline int msg_size(uint32_t wave, uint32_t seq, size_t cap) {
    const int bytes = kSizes[(seq * 5 + wave * 3 + seq / kNumSizes) % kNumSizes];
    if (message_lines(bytes) <= cap) return bytes;
    int fit = 0;
    for (int s : kSizes) fit = message_lines(s) <= cap ? s : fit;
    return fit;
}

enum Path : int { kReg, kShared, kGlobal, kLanes };

template <int B, typename H>
__device__ bool push_reg(H& h, uint64_t key, uint32_t meta) {
    if constexpr (B == 0) {
        return h.push_lanes(0, 0, meta);  // no zero-size type
    } else {
        struct {
            uint8_t b[B];
        } m;
        uint64_t w[(B + 7) / 8];
#pragma unroll
        for (int k = 0; k < (B + 7) / 8; ++k) w[k] = word(key, k);
        __builtin_memcpy(m.b, w, B);
        return h.push_reg(m, meta);
    }
}

template <typename H, size_t... I>
__device__ bool push_reg_any(H& h, int bytes, uint64_t key, uint32_t meta,
                             std::index_sequence<I...>) {
    bool ok = false;
    ((bytes == kSizes[I] ? (ok = push_reg<kSizes[I]>(h, key, meta), 0) : 0), ...);
    return ok;
}

/// One push attempt of message `key` through `path`. `lds` and `global` are
/// this wave's 64-word staging rows; lane l stages word 63 - l, so the push
/// reads words written by other lanes.
template <typename H>
__device__ bool push_message(H& h, Path path, int bytes, uint64_t key, uint32_t meta,
                             uint64_t* lds, uint64_t* global) {
    const int lane = static_cast<int>(__lane_id());
    const int k = kMaxWords - 1 - lane;
    switch (path) {
        case kReg:
            return push_reg_any(h, bytes, key, meta, std::make_index_sequence<kNumSizes>{});
        case kShared:
            lds[k] = word(key, k);
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            return h.push_shared(lds, bytes, meta);
        case kGlobal:
            global[k] = word(key, k);
            __threadfence_block();
            __builtin_amdgcn_wave_barrier();
            return h.push_global(global, bytes, meta);
        default:
            return h.push_lanes(word(key, lane), bytes, meta);
    }
}

/// Wave `first_wave + local wave` pushes messages first_seq .. first_seq +
/// msgs - 1.
template <typename H>
__global__ void producer_kernel(H h, uint32_t first_wave, uint32_t first_seq, int msgs,
                                size_t cap, uint64_t* rows) {
    __shared__ uint64_t lds[4][kMaxWords];
    const uint32_t local = threadIdx.x / 64;
    const uint32_t wave = first_wave + blockIdx.x * (blockDim.x / 64) + local;
    for (uint32_t seq = first_seq; seq < first_seq + msgs; ++seq) {
        const auto path = static_cast<Path>(seq % 4);
        while (!push_message(h, path, msg_size(wave, seq, cap), msg_key(wave, seq),
                             (wave << kSeqBits) | seq, lds[local], rows + wave * kMaxWords)) {
        }
        __builtin_amdgcn_wave_barrier();  // the next message reuses the rows
    }
}

struct Popped {
    uint32_t wave, seq;
};

/// Pops `count` messages through HostPollState, like a KIWI-EP proxy, and
/// checks every message's contents. Aborts if no message arrives for 20 s
/// (a kernel may still wait for credit).
template <typename Q>
void consume(Q& q, size_t cap, int64_t count, std::vector<Popped>& popped, uint64_t& bad) {
    auto ps = q.host_poll_state();
    alignas(8) uint8_t buf[kMaxMessageBytes + 8];
    uint64_t expect[kMaxWords];
    auto last = std::chrono::steady_clock::now();
    for (int64_t n = 0; n < count;) {
        std::memset(buf, 0xab, sizeof(buf));
        uint32_t meta = 0;
        size_t bytes = 0;
        if (!(ps.ready() && q.pop(buf, &meta, &bytes))) {
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(20)) {
                ADD_FAILURE() << "timed out with " << count - n << " messages outstanding";
                std::abort();
            }
            continue;
        }
        ps.refresh();
        last = std::chrono::steady_clock::now();
        ++n;
        const Popped p{meta >> kSeqBits, meta & ((1u << kSeqBits) - 1)};
        popped.push_back(p);
        for (int k = 0; k < kMaxWords; ++k) expect[k] = word(msg_key(p.wave, p.seq), k);
        bool good = bytes == static_cast<size_t>(msg_size(p.wave, p.seq, cap)) &&
                    std::memcmp(buf, expect, bytes) == 0;
        for (size_t b = bytes; b < sizeof(buf); ++b) good = good && buf[b] == 0xab;
        bad += !good;
    }
}

struct Stream {
    int blocks, waves_per_block, msgs;  ///< producer waves, messages per wave per launch
    size_t capacity;                    ///< ring lines
    int launches;
    bool prefill;                       ///< whole launch in the ring before popping
    bool overlap = false;               ///< launches run at once on their own streams (MPSC)
};

/// Runs `r.launches` launches, each with a fresh device_handle(), and checks
/// exactly-once delivery and per-wave FIFO order. Sequential launches
/// continue the same waves' sequences; overlapping launches are new waves.
template <typename Q>
void run(const Stream& r) {
    SCOPED_TRACE(::testing::Message() << r.blocks * r.waves_per_block << " waves x " << r.msgs
                                      << " msgs x " << r.launches << " launches, " << r.capacity
                                      << " lines" << (r.overlap ? ", overlapping" : ""));
    const int waves = r.blocks * r.waves_per_block;
    const int launch_waves = r.overlap ? waves * r.launches : waves;
    const int seqs = r.overlap ? r.msgs : r.msgs * r.launches;
    Q q(r.capacity);
    uint64_t* rows = nullptr;
    HIP_OK(hipMalloc(&rows, launch_waves * kMaxWords * sizeof(uint64_t)));
    std::vector<hipStream_t> streams(r.overlap ? r.launches : 1);
    for (auto& s : streams) HIP_OK(hipStreamCreate(&s));
    std::vector<Popped> popped;
    uint64_t bad = 0;
    for (int launch = 0; launch < r.launches; ++launch) {
        const uint32_t first_wave = r.overlap ? launch * waves : 0;
        const uint32_t first_seq = r.overlap ? 0 : launch * r.msgs;
        producer_kernel<<<r.blocks, 64 * r.waves_per_block, 0, streams[launch % streams.size()]>>>(
            q.device_handle(), first_wave, first_seq, r.msgs, r.capacity, rows);
        HIP_OK(hipGetLastError());
        if (r.overlap) continue;
        if (r.prefill) HIP_OK(hipStreamSynchronize(streams[0]));
        consume(q, r.capacity, static_cast<int64_t>(waves) * r.msgs, popped, bad);
        HIP_OK(hipStreamSynchronize(streams[0]));
    }
    if (r.overlap) {
        consume(q, r.capacity, static_cast<int64_t>(launch_waves) * r.msgs, popped, bad);
    }
    for (auto& s : streams) {
        HIP_OK(hipStreamSynchronize(s));
        HIP_OK(hipStreamDestroy(s));
    }
    HIP_OK(hipFree(rows));

    uint8_t buf[kMaxMessageBytes];
    EXPECT_FALSE(q.pop(buf));
    EXPECT_FALSE(q.host_poll_state().ready());

    std::vector<int> count(static_cast<size_t>(launch_waves) * seqs, 0);
    std::vector<int64_t> last(launch_waves, -1);
    uint64_t errors = bad;
    for (const Popped& p : popped) {
        if (p.wave >= static_cast<uint32_t>(launch_waves) || p.seq >= static_cast<uint32_t>(seqs)) {
            ++errors;
            continue;
        }
        ++count[p.wave * seqs + p.seq];
        errors += static_cast<int64_t>(p.seq) <= last[p.wave];
        last[p.wave] = p.seq;
    }
    for (int c : count) errors += c != 1;
    EXPECT_EQ(errors, 0u) << "lost, duplicated, reordered or corrupted messages";
}

template <typename H>
__global__ void write_chunk0_kernel(H h, uint32_t bytes) {
    if (__lane_id() == 0) h.chunks[0] = detail::make_chunk(0, detail::make_signal(0, bytes, 1));
}

template <typename H>
__global__ void read_published_kernel(H h, uint64_t* out) {
    if (__lane_id() == 0) *out = detail::load_published(h.consumer_index);
}

template <typename H>
uint64_t published_index(const H& h) {
    uint64_t* d_out = nullptr;
    uint64_t value = ~uint64_t{0};
    EXPECT_EQ(hipMalloc(&d_out, sizeof(value)), hipSuccess);
    read_published_kernel<<<1, 64>>>(h, d_out);
    EXPECT_EQ(hipMemcpy(&value, d_out, sizeof(value), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(hipFree(d_out), hipSuccess);
    return value;
}

}  // namespace

template <typename Q>
class ChunkQueue : public ::testing::Test {};
using Queues = ::testing::Types<SPSCD2HQueue, MPSCD2HQueue>;
TYPED_TEST_SUITE(ChunkQueue, Queues);

TEST(ChunkLayout, SignalWordAndFootprint) {
    constexpr uint64_t s = detail::make_signal(0xab, 512, 0xdeadbeef);
    static_assert(detail::signal_epoch(s) == 0xab && detail::signal_bytes(s) == 512);
    static_assert(detail::signal_metadata(s) == 0xdeadbeef);
    // Both 4-byte signals carry the epoch in their low byte.
    static_assert((s & 0xff) == 0xab && ((s >> 32) & 0xff) == 0xab);
    static_assert(detail::make_signal(0, 0, 0) == 0);
    static_assert(sizeof(Chunk) == 16 && alignof(Chunk) == 16);
    // {data lo, sig 0, data hi, sig 1}
    constexpr Chunk c = detail::make_chunk(0x1111222233334444, s);
    static_assert(c[0] == 0x33334444 && c[1] == static_cast<uint32_t>(s));
    static_assert(c[2] == 0x11112222 && c[3] == static_cast<uint32_t>(s >> 32));
    static_assert(detail::chunk_data(c) == 0x1111222233334444 && detail::chunk_signal(c) == s);
    // One 64-byte line up to 32 B, then one line per 32 B.
    const size_t bytes[] = {0, 1, 8, 9, 32, 33, 48, 168, 448, 512};
    const size_t lines[] = {1, 1, 1, 1, 1, 2, 2, 6, 14, 16};
    for (size_t i = 0; i < sizeof(bytes) / sizeof(bytes[0]); ++i) {
        EXPECT_EQ(message_lines(bytes[i]), lines[i]) << bytes[i];
    }
}

// A visible chunk 0 whose length exceeds the maximum is a corrupted ring:
// pop() throws and consumes nothing.
TYPED_TEST(ChunkQueue, OversizedHeaderThrows) {
    TypeParam q(4);
    write_chunk0_kernel<<<1, 64>>>(q.device_handle(), kMaxMessageBytes + 8);
    HIP_OK(hipDeviceSynchronize());
    uint8_t buf[kMaxMessageBytes];
    EXPECT_THROW(q.pop(buf), queue_error);
    EXPECT_THROW(q.pop(buf), queue_error);
}

// Capacity must be a power of two of at most kMaxCapacityLines (so lines in
// flight fit the producer's 32-bit credit check); both throw before
// allocating the ring.
TYPED_TEST(ChunkQueue, CapacityLimits) {
    EXPECT_THROW(TypeParam(3), queue_error);
    EXPECT_THROW(TypeParam(size_t{1} << 32), queue_error);
    static_assert(kMaxCapacityLines == size_t{1} << 31);
}

// Every size through every push path from one wave, in rings from 1 line
// (every message a new lap; the 8-bit epoch wraps every 256) to 16 lines (a
// 512-byte message is a whole lap) and 4096 lines. Each launch takes a fresh
// device_handle() that continues where the previous launch stopped.
TYPED_TEST(ChunkQueue, EverySizeEveryPathEveryRing) {
    for (size_t cap : {1, 2, 8, 16, 32}) run<TypeParam>({1, 1, 600, cap, 3, false});
    run<TypeParam>({1, 1, 3000, 4096, 2, false});
    run<TypeParam>({1, 1, 3000, 1 << 14, 1, /*prefill=*/true});
}

// Many producer waves, one consumer, small rings and many laps: producers
// wait for credit and claims wrap the ring at every line offset. A 1-line
// ring makes every claim wait for the previous message's pop.
TEST(MpscD2HStress, ManyProducersOneConsumer) {
    const Stream runs[] = {
        {16, 1, 400, 1, 1, false},       // 1-line ring, one-line messages only
        {8, 2, 400, 2, 1, false},
        {16, 1, 1000, 16, 1, false},     // 16 waves, 1 per block
        {8, 4, 1000, 16, 2, false},      // 32 waves, 4 per block, relaunched
        {8, 4, 1000, 64, 1, false},      // 64-line ring
        {304, 1, 100, 32, 1, false},     // a wave on every MI300X CU
        {64, 4, 40, 1024, 1, false},     // 256 waves
        {8, 1, 1000, 16, 4, false, true},  // four launches running at once
    };
    for (const Stream& r : runs) run<MPSCD2HQueue>(r);
}

// The consumer publishes its position after every pop, multi-line messages
// included, so the published index always equals it, and a failed pop
// publishes nothing. A new device handle starts there on both queues.
TYPED_TEST(ChunkQueue, PublishesConsumerIndexOnEveryPop) {
    constexpr int kMsgs = 40;  // 1- to 16-line messages, at most 640 lines
    TypeParam q(1024);
    uint64_t* rows = nullptr;
    HIP_OK(hipMalloc(&rows, kMaxWords * sizeof(uint64_t)));
    const auto h = q.device_handle();
    producer_kernel<<<1, 64>>>(h, 0, 0, kMsgs, 16, rows);
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipFree(rows));
    EXPECT_EQ(published_index(h), 0u);
    uint8_t buf[kMaxMessageBytes];
    uint64_t position = 0;
    for (int i = 0; i < kMsgs; ++i) {
        size_t bytes = 0;
        ASSERT_TRUE(q.pop(buf, nullptr, &bytes));
        position += message_lines(bytes);
        EXPECT_EQ(published_index(h), position) << i;
    }
    EXPECT_GT(position, static_cast<uint64_t>(kMsgs));  // multi-line messages too
    EXPECT_FALSE(q.pop(buf));
    EXPECT_EQ(published_index(h), position);
    EXPECT_EQ(q.device_handle().consumer_index_cached, position);
}

namespace {

/// Pushes `bytes`-byte messages until a push is refused, retries it twice,
/// and reports how many were accepted and whether a refusal wrote anything.
__global__ void fill_kernel(SPSCD2HQueue::DeviceHandle h, int bytes, uint64_t* out) {
    uint64_t accepted = 0;
    while (h.push_lanes(word(accepted, __lane_id()), bytes, static_cast<uint32_t>(accepted))) {
        ++accepted;
    }
    const uint64_t before = h.producer_index;
    const bool retried = h.push_lanes(0, bytes, 0) || h.push_lanes(0, bytes, 0);
    if (__lane_id() == 0) {
        out[0] = accepted;
        out[1] = retried || h.producer_index != before;
    }
}

/// Every wave pushes `msgs` one-line messages (metadata: wave << 16 | i).
__global__ void mpsc_fill_kernel(MPSCD2HQueue::DeviceHandle h, int msgs) {
    const uint32_t wave = blockIdx.x;
    for (int i = 0; i < msgs; ++i) {
        const uint32_t meta = (wave << 16) | i;
        while (!h.push_lanes(word(meta, __lane_id()), 8, meta)) {}
    }
}

}  // namespace

// Without a consumer an SPSC C-line ring accepts floor(C / lines) messages
// and then refuses without writing; after the host drains it the next
// launch fills it again.
TEST(ChunkQueueBackpressure, SpscFullRingRefusesAndRecovers) {
    uint64_t* d_out;
    HIP_OK(hipMalloc(&d_out, 2 * sizeof(uint64_t)));
    for (int bytes : {8, 48, 168}) {
        SCOPED_TRACE(bytes);
        const uint64_t fit = 16 / message_lines(bytes);
        SPSCD2HQueue q(16);
        for (int launch = 0; launch < 2; ++launch) {
            fill_kernel<<<1, 64>>>(q.device_handle(), bytes, d_out);
            uint64_t out[2];
            HIP_OK(hipMemcpy(out, d_out, sizeof(out), hipMemcpyDeviceToHost));
            EXPECT_EQ(out[0], fit);
            EXPECT_EQ(out[1], 0u);
            uint64_t buf[kMaxWords];
            uint32_t meta = 0;
            for (uint64_t i = 0; i < fit; ++i) {
                ASSERT_TRUE(q.pop(buf, &meta)) << i;
                EXPECT_EQ(meta, i);
                EXPECT_EQ(buf[0], word(i, 0));
            }
            EXPECT_FALSE(q.pop(buf));  // the refused pushes wrote nothing
        }
    }
    HIP_OK(hipFree(d_out));
}

// Four MPSC producers fill a 16-line ring and then wait for credit: the
// kernel stays running with exactly 16 messages in the ring, and finishes
// once the host pops. Waiting producers never overwrite unread lines.
TEST(ChunkQueueBackpressure, MpscFullRingWaitsForConsumer) {
    constexpr int kWaves = 4, kMsgs = 50;
    MPSCD2HQueue q(16);
    hipStream_t s;
    HIP_OK(hipStreamCreate(&s));
    mpsc_fill_kernel<<<kWaves, 64, 0, s>>>(q.device_handle(), kMsgs);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_EQ(hipStreamQuery(s), hipErrorNotReady);
    std::vector<int> next(kWaves, 0);
    uint64_t buf[kMaxWords];
    uint32_t meta = 0;
    size_t bytes = 0;
    int errors = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    for (int n = 0; n < kWaves * kMsgs;) {
        if (!q.pop(buf, &meta, &bytes)) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "timed out at " << n;
            continue;
        }
        ++n;
        const uint32_t wave = meta >> 16, i = meta & 0xffff;
        errors += wave >= static_cast<uint32_t>(kWaves) || static_cast<int>(i) != next[wave]++ ||
                  bytes != 8 || buf[0] != word(meta, 0);
    }
    HIP_OK(hipStreamSynchronize(s));
    HIP_OK(hipStreamDestroy(s));
    EXPECT_EQ(errors, 0);
    EXPECT_FALSE(q.pop(buf));
}

namespace {

constexpr int kFaultMsgs = 200;
constexpr int kFaultBytes = 48;  // two lines: chunks 0-5 payload, 6-7 padding

/// Pushes two-line messages like push_lanes(), except that chunk `victim` is
/// stored 20 us after the rest of its message. With `late_half` 0 or 1, only
/// that 8-byte half of the victim chunk (data lo + sig 0, or data hi + sig 1)
/// is late; the other half is stored with the rest of the message. Claims
/// lines and waits for credit like the queues do.
template <typename H>
__global__ void fault_kernel(H h, int victim, int late_half) {
    constexpr uint64_t lines = message_lines(kFaultBytes);
    const int lane = __lane_id();
    const uint64_t cap = uint64_t{1} << h.log2_capacity;
    for (int i = 0; i < kFaultMsgs; ++i) {
        uint64_t base = 0;
        if constexpr (std::is_same_v<H, SPSCD2HQueue::DeviceHandle>) {
            base = h.producer_index;
            h.producer_index += lines;
        } else {
            if (lane == 0) {
                base = __hip_atomic_fetch_add(h.producer_index, lines, __ATOMIC_RELAXED,
                                              __HIP_MEMORY_SCOPE_AGENT);
            }
            base = detail::broadcast_lane0(base);
        }
        if (lane == 0) {
            while (base + lines - detail::load_published(h.consumer_index) > cap) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __builtin_amdgcn_wave_barrier();
        const uint64_t line = base + lane / 4;
        const auto epoch = static_cast<uint8_t>(line >> h.log2_capacity);
        const uint64_t data = lane * 8 < kFaultBytes ? word(i, lane) : 0;
        const uint64_t signal = detail::make_signal(epoch, lane == 0 ? kFaultBytes : 0,
                                                    lane == 0 ? i : 0);
        auto p = h.chunks + (line & (cap - 1)) * 4 + lane % 4;
        const Chunk c = detail::make_chunk(data, signal);
        const auto halves = (detail::global_ptr<uint64_t>)p;  // address-space cast
        const uint64_t half[2] = {c[0] | uint64_t{c[1]} << 32, c[2] | uint64_t{c[3]} << 32};
        if (lane < static_cast<int>(lines * 4) && lane != victim) *p = c;
        if (lane == victim && late_half >= 0) halves[1 - late_half] = half[1 - late_half];
        __asm__ volatile("s_waitcnt vmcnt(0)" ::: "memory");
        const uint64_t t0 = wall_clock64();
        while (wall_clock64() - t0 < 2000) __builtin_amdgcn_s_sleep(8);  // 20 us
        if (lane == victim) {
            if (late_half >= 0) {
                halves[late_half] = half[late_half];
            } else {
                *p = c;
            }
        }
        __asm__ volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
}

}  // namespace

// The late chunk is in chunk 0's line, in the second line, or the second
// line's padding. The 16-line ring wraps every 8 messages, so the late chunk
// holds the previous lap. The host sees chunk 0, finds the late chunk stale,
// and waits: it never consumes a torn message. A late chunk 0 (or its sig 0
// half) hides the whole message until it lands. A late half chunk is the
// tear LL8 guards against: each 4-byte signal covers only its own half,
// including chunk 0's sig 1, which carries most of the metadata.
TYPED_TEST(ChunkQueue, LateChunkIsNeverConsumed) {
    const std::pair<int, int> cases[] = {{0, -1}, {1, -1}, {4, -1}, {7, -1},
                                         {0, 0},  {0, 1},  {1, 0},  {5, 1}, {6, 0}};
    for (auto [victim, late_half] : cases) {
        SCOPED_TRACE(::testing::Message() << "victim " << victim << " late_half " << late_half);
        TypeParam q(16);
        auto ps = q.host_poll_state();
        fault_kernel<<<1, 64>>>(q.device_handle(), victim, late_half);
        int torn = 0, detected = 0;
        alignas(8) uint8_t buf[kMaxMessageBytes];
        uint64_t expect[kFaultBytes / 8];
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        for (int i = 0; i < kFaultMsgs; ++i) {
            size_t bytes = 0;
            uint32_t meta = 0;
            bool waited = false;
            while (!(ps.ready() && q.pop(buf, &meta, &bytes))) {
                waited = waited || ps.ready();  // chunk 0 visible, another chunk not yet
                if (std::chrono::steady_clock::now() > deadline) {
                    ADD_FAILURE() << "timed out at message " << i;
                    std::abort();
                }
            }
            ps.refresh();
            detected += waited;
            for (int k = 0; k < kFaultBytes / 8; ++k) expect[k] = word(i, k);
            torn += bytes != kFaultBytes || meta != static_cast<uint32_t>(i) ||
                    std::memcmp(buf, expect, kFaultBytes) != 0;
        }
        HIP_OK(hipDeviceSynchronize());
        EXPECT_EQ(torn, 0);
        if (!(victim == 0 && late_half != 1)) EXPECT_GT(detected, 0);
    }
}
