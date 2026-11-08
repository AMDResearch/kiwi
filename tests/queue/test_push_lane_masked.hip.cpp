// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file test_push_lane_masked.hip.cpp
 * @brief Lane-masked device-to-host pushes (push_lane_masked()) on both queues
 *
 * Producer waves call push_lane_masked() from the whole wave with per-lane
 * active masks (all, none, alternating, single lanes, prefixes, suffixes,
 * random); inactive lanes pass lengths that would trap if they were read.
 * Messages are 13 to 512 bytes, also of sizes that are not a multiple of 8,
 * optionally mixed with collective pushes on the same queue, into rings from
 * 1 line up; a call may fill the whole ring. SPSC producers retry calls
 * that return false, and their handles persist across launches as saved
 * handles. The host pops while the producers run and checks every
 * message's length, payload and metadata, exactly-once delivery, the order
 * within each producer wave (program order across steps, lane order within a
 * call), and that an SPSC handle's producer index ends where the consumer
 * stopped. Further tests check that a short SPSC call writes nothing and
 * returns false, and the traps.
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

#define HIP_OK(expr) ASSERT_EQ((expr), hipSuccess)

namespace {

constexpr int kLanes = 64;

__host__ __device__ inline uint64_t mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/// Payload word k of the message with metadata `meta`.
__host__ __device__ inline uint64_t word(uint32_t meta, int k) {
    return mix((static_cast<uint64_t>(meta) << 8) | static_cast<uint64_t>(k));
}

/// A message of 8 * W + R bytes: words 0 .. W - 1, then the first R bytes of
/// word W (so sizeof is not a multiple of 8 unless R is 0).
template <int W, int R>
struct Msg {
    uint64_t w[W];
    uint8_t tail[R];
} __attribute__((packed));

template <int W>
struct Msg<W, 0> {
    uint64_t w[W];
};

/// The message with metadata `meta`: its payload bytes are those of words
/// word(meta, 0), word(meta, 1), ...
template <int W, int R>
__host__ __device__ inline Msg<W, R> make_msg(uint32_t meta) {
    Msg<W, R> m;
#pragma unroll
    for (int k = 0; k < W; ++k) m.w[k] = word(meta, k);
    if constexpr (R > 0) {
        const uint64_t last = word(meta, W);
#pragma unroll
        for (int j = 0; j < R; ++j) m.tail[j] = static_cast<uint8_t>(last >> (8 * j));
    }
    return m;
}

/// Metadata of the message lane `lane` pushes at step `step` of wave
/// `wave`: wave (9 bits) | step (16) | lane (6). A collective push is lane 0.
__host__ __device__ inline uint32_t tag(uint32_t wave, uint32_t step, uint32_t lane) {
    return wave << 22 | step << 6 | lane;
}

struct Tag {
    uint32_t wave, step, lane;
};

inline Tag untag(uint32_t meta) { return {meta >> 22, (meta >> 6) & 0xffff, meta & 63}; }

constexpr int kMaxWaves = 512;
constexpr int kMaxSteps = 1 << 16;

enum Kind : int { kWave, kMasked, kCollective, kNumKinds };

__host__ __device__ inline uint64_t step_hash(uint32_t wave, uint32_t step) {
    return mix(static_cast<uint64_t>(wave) << 32 | step);
}

/// What wave `wave` does at step `step`: a lane-masked push with every lane
/// active, one with lane_pred() as the active mask, or one collective push
/// (a masked step unless `mixed`).
__host__ __device__ inline Kind step_kind(uint32_t wave, uint32_t step, bool mixed) {
    const auto k = static_cast<Kind>(step_hash(wave, step) % kNumKinds);
    return k == kCollective && !mixed ? kMasked : k;
}

/// The active mask of a masked step: random lanes, a prefix, one lane, the
/// odd lanes, or a suffix (prefixes and suffixes may be empty or the whole
/// wave).
__host__ __device__ inline bool lane_pred(uint32_t wave, uint32_t step, uint32_t lane) {
    const uint64_t h = step_hash(wave, step);
    const uint32_t n = static_cast<uint32_t>((h >> 16) % 65);
    switch ((h >> 8) % 5) {
        case 0: return (mix(h ^ lane) & 1) != 0;
        case 1: return lane < n;
        case 2: return lane == n % 64;
        case 3: return (lane & 1) != 0;
        default: return lane >= n;
    }
}

/// Active lanes of a lane-masked step (kWave: every lane; kMasked:
/// lane_pred()), cut to the lowest `cap` of them so that one line per lane
/// fits a call into a `cap`-line ring.
__host__ __device__ inline uint64_t active_mask(uint32_t wave, uint32_t step, Kind k, size_t cap) {
    uint64_t mask = 0;
    uint64_t n = 0;
    for (uint32_t lane = 0; lane < static_cast<uint32_t>(kLanes) && n < cap; ++lane) {
        if (k == kWave || lane_pred(wave, step, lane)) {
            mask |= uint64_t{1} << lane;
            ++n;
        }
    }
    return mask;
}

/// Lines each active lane of a call with active lanes `mask` may take, so
/// that the call fits a `cap`-line ring.
__host__ __device__ inline uint64_t lane_budget(uint64_t mask, size_t cap) {
    const int n = __builtin_popcountll(mask);
    return n == 0 ? 0 : cap / n;
}

/// Length of message `meta`: up to `max_bytes`, and at most `lines` lines
/// (message_lines(b) <= lines for b <= 32 * lines).
__host__ __device__ inline uint32_t msg_bytes(uint32_t meta, uint32_t max_bytes, uint64_t lines) {
    const uint32_t b = static_cast<uint32_t>(mix(~static_cast<uint64_t>(meta)) % (max_bytes + 1));
    const uint64_t fit = 32 * lines;
    return b > fit ? static_cast<uint32_t>(fit) : b;
}

/// Index of message (wave, step, lane) in the per-message arrays.
__host__ __device__ inline size_t msg_index(uint32_t wave, uint32_t step, uint32_t lane,
                                            int total_steps) {
    return (static_cast<size_t>(wave) * total_steps + step) * kLanes + lane;
}

/// The whole wave calls push_lane_masked() until it succeeds (SPSC returns
/// false when the ring is short); the calling lane pushes its message of
/// step `step` if it is in active_mask(). An inactive lane passes a length
/// past any message, which must be ignored.
template <int W, int R, typename P>
__device__ void push_lane(P& p, uint32_t wave, uint32_t step, size_t cap, Kind k) {
    const uint32_t lane = __lane_id();
    const uint32_t meta = tag(wave, step, lane);
    const Msg<W, R> m = make_msg<W, R>(meta);
    const uint64_t mask = active_mask(wave, step, k, cap);
    const bool active = (mask >> lane) & 1;
    const uint32_t bytes = active ? msg_bytes(meta, sizeof(m), lane_budget(mask, cap)) : ~lane;
    while (!p.push_lane_masked(m, bytes, meta, active)) {}
}

/// The whole wave pushes its collective message of step `step`.
template <typename P>
__device__ void push_collective(P& p, uint32_t wave, uint32_t step, size_t cap) {
    const uint32_t meta = tag(wave, step, 0);
    const uint32_t bytes = msg_bytes(meta, kMaxMessageBytes, cap);
    while (!p.push_lanes(word(meta, static_cast<int>(__lane_id())), bytes, meta)) {}
}

/// Wave w runs steps first_step .. first_step + steps - 1. SPSC: a launch
/// with `resume` starts from the handle the previous launch saved in `saved`
/// instead of `h`; every launch saves its handle there.
template <typename Q, int W, int R, bool Mixed>
__global__ void producer_kernel(typename Q::DeviceHandle h, typename Q::DeviceHandle* saved,
                                bool resume, uint32_t first_step, int steps, size_t cap) {
    constexpr bool kSpsc = std::is_same_v<Q, SPSCD2HQueue>;
    const uint32_t wave = blockIdx.x * (blockDim.x / kLanes) + threadIdx.x / kLanes;
    if (kSpsc && resume) h = detail::wave_uniform(*saved);
    for (uint32_t step = first_step; step < first_step + steps; ++step) {
        const Kind k = step_kind(wave, step, Mixed);
        if (k == kCollective) {
            if constexpr (Mixed) push_collective(h, wave, step, cap);
        } else {
            push_lane<W, R>(h, wave, step, cap, k);
        }
    }
    if (kSpsc && __lane_id() == 0) *saved = h;
}

struct Run {
    int blocks, waves_per_block;  ///< producer waves (SPSC: one)
    int steps;                    ///< steps per wave per launch
    size_t capacity;              ///< ring lines
    int launches;                 ///< sequential launches, each drained before the next
    int delay_ms;                 ///< consumer start delay per launch (fills small rings)
};

/// Expected messages and the consumer-side checks.
struct Checker {
    int total_steps;
    size_t cap;
    uint32_t lane_max_bytes;
    bool mixed;
    std::vector<uint8_t> status;  ///< per message index: 0 not pushed, 1 expected, 2 popped
    std::vector<int64_t> last_step;
    std::vector<int> last_lane;
    uint64_t position = 0;  ///< consumer position (lines)
    uint64_t errors = 0;

    Checker(int waves, int total_steps_, size_t cap_, uint32_t lane_max_bytes_, bool mixed_)
        : total_steps(total_steps_), cap(cap_), lane_max_bytes(lane_max_bytes_), mixed(mixed_),
          status(static_cast<size_t>(waves) * total_steps * kLanes, 0), last_step(waves, -1),
          last_lane(waves, -1) {}

    int waves() const { return static_cast<int>(last_step.size()); }

    /// Marks steps [first, first + steps) of every wave expected; returns
    /// how many messages they push.
    int64_t expect(uint32_t first, int steps) {
        int64_t n = 0;
        for (uint32_t w = 0; w < static_cast<uint32_t>(waves()); ++w) {
            for (uint32_t s = first; s < first + steps; ++s) {
                const Kind k = step_kind(w, s, mixed);
                const uint64_t mask = k == kCollective ? 1 : active_mask(w, s, k, cap);
                for (uint32_t lane = 0; lane < kLanes; ++lane) {
                    if (((mask >> lane) & 1) == 0) continue;
                    status[msg_index(w, s, lane, total_steps)] = 1;
                    ++n;
                }
            }
        }
        return n;
    }

    void check(uint32_t meta, size_t bytes, const uint8_t* buf, size_t buf_size) {
        const Tag t = untag(meta);
        if (t.wave >= static_cast<uint32_t>(waves()) ||
            t.step >= static_cast<uint32_t>(total_steps)) {
            ++errors;
            return;
        }
        const size_t i = msg_index(t.wave, t.step, t.lane, total_steps);
        errors += status[i] != 1;  // unexpected or duplicated
        status[i] = 2;
        const Kind k = step_kind(t.wave, t.step, mixed);
        const uint32_t want =
            k == kCollective ? msg_bytes(meta, kMaxMessageBytes, cap)
                             : msg_bytes(meta, lane_max_bytes,
                                         lane_budget(active_mask(t.wave, t.step, k, cap), cap));
        uint64_t expect[kMaxMessageBytes / 8];
        for (int k = 0; k < static_cast<int>(kMaxMessageBytes / 8); ++k) expect[k] = word(meta, k);
        bool good = bytes == want && std::memcmp(buf, expect, bytes) == 0;
        for (size_t b = bytes; b < buf_size; ++b) good = good && buf[b] == 0xab;
        errors += !good;
        // Program order across steps, lane order within one call.
        if (static_cast<int64_t>(t.step) < last_step[t.wave]) ++errors;
        if (static_cast<int64_t>(t.step) > last_step[t.wave]) {
            last_step[t.wave] = t.step;
            last_lane[t.wave] = -1;
        }
        errors += static_cast<int>(t.lane) <= last_lane[t.wave];
        last_lane[t.wave] = static_cast<int>(t.lane);
        position += message_lines(bytes);
    }
};

/// Pops `count` messages through HostPollState and checks them. Aborts if no
/// message arrives for 20 s (the producers could not finish).
template <typename Q>
void consume(Q& q, int64_t count, Checker& c) {
    auto ps = q.host_poll_state();
    alignas(8) uint8_t buf[kMaxMessageBytes + 8];
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
        c.check(meta, bytes, buf, sizeof(buf));
    }
}

/// Runs `r.launches` sequential launches of lane-masked pushes of up to
/// 8 * W + R bytes and checks every message. Odd SPSC launches resume from
/// the handle the previous launch saved, without a new device_handle();
/// after every SPSC launch the saved handle's producer index must be the
/// consumer's position.
template <typename Q, int W, int R, bool Mixed>
void run(const Run& r) {
    constexpr bool kSpsc = std::is_same_v<Q, SPSCD2HQueue>;
    const int waves = r.blocks * r.waves_per_block;
    const int total_steps = r.steps * r.launches;
    SCOPED_TRACE(::testing::Message() << (kSpsc ? "SPSC " : "MPSC ") << waves << " waves x "
                                      << r.steps << " steps x " << r.launches << " launches, "
                                      << r.capacity << " lines, " << sizeof(Msg<W, R>)
                                      << " B lane-masked" << (Mixed ? ", mixed with collective" : ""));
    ASSERT_TRUE(!kSpsc || waves == 1);
    ASSERT_LE(waves, kMaxWaves);
    ASSERT_LE(total_steps, kMaxSteps);
    Q q(r.capacity);
    Checker c(waves, total_steps, r.capacity, sizeof(Msg<W, R>), Mixed);
    typename Q::DeviceHandle* saved = nullptr;
    HIP_OK(hipMalloc(&saved, sizeof(*saved)));
    hipStream_t s;
    HIP_OK(hipStreamCreate(&s));
    for (int launch = 0; launch < r.launches; ++launch) {
        const uint32_t first = static_cast<uint32_t>(launch * r.steps);
        const int64_t count = c.expect(first, r.steps);
        const bool resume = kSpsc && launch % 2 == 1;
        const auto h = resume ? typename Q::DeviceHandle{} : q.device_handle();
        producer_kernel<Q, W, R, Mixed><<<r.blocks, kLanes * r.waves_per_block, 0, s>>>(
            h, saved, resume, first, r.steps, r.capacity);
        HIP_OK(hipGetLastError());
        if (r.delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(r.delay_ms));
        consume(q, count, c);
        HIP_OK(hipStreamSynchronize(s));
        if constexpr (kSpsc) {
            typename Q::DeviceHandle end{};
            HIP_OK(hipMemcpy(&end, saved, sizeof(end), hipMemcpyDeviceToHost));
            EXPECT_EQ(end.producer_index, c.position) << "launch " << launch;
        }
    }
    HIP_OK(hipStreamDestroy(s));
    uint8_t buf[kMaxMessageBytes];
    EXPECT_FALSE(q.pop(buf));

    HIP_OK(hipFree(saved));
    uint64_t lost = 0;
    for (uint8_t st : c.status) lost += st == 1;
    EXPECT_EQ(c.errors, 0u) << "unexpected, duplicated, reordered or corrupted messages";
    EXPECT_EQ(lost, 0u) << "messages never popped";
}

/// One wave on a fresh 64-line queue: collective pushes of 0, 33, 512 and 8
/// bytes (1, 2, 16, 1 lines), a push_lane_masked() with lanes 1, 5 and 63
/// active, of 8, 100 and 0 bytes (1, 4, 1 lines), then a collective 8-byte
/// push. Inactive lanes pass lengths past their message and the ring.
/// Lane 0 stores the handle and the push_lane_masked() result at the end.
template <typename H>
__global__ void order_kernel(H h, H* out, bool* pushed) {
    const uint32_t lane = __lane_id();
    auto collective = [&](uint32_t i, uint32_t bytes) {
        while (!h.push_lanes(word(i, lane), bytes, i)) {}
    };
    collective(0, 0);
    collective(1, 33);
    collective(2, 512);
    collective(3, 8);
    const uint32_t meta = 4 << 8 | lane;
    const Msg<16, 0> m = make_msg<16, 0>(meta);
    const bool ok = h.push_lane_masked(
        m, lane == 1 ? 8 : lane == 5 ? 100 : lane == 63 ? 0 : 4096 + lane, meta,
        lane == 1 || lane == 5 || lane == 63);
    collective(5, 8);
    if (lane == 0) {
        *out = h;
        *pushed = ok;
    }
}

/// Lanes [0, n) push one-line 8-byte messages with metadata tag(0, step,
/// lane); returns the push_lane_masked() result.
__device__ bool push_ones(SPSCD2HQueue::DeviceHandle& h, uint32_t step, uint32_t n) {
    const uint32_t lane = __lane_id();
    const uint32_t meta = tag(0, step, lane);
    return h.push_lane_masked(make_msg<1, 0>(meta), 8, meta, lane < n);
}

/// Lines of calls 0-3 of short_kernel.
constexpr uint32_t kShortCalls[] = {10, 10, 6, 1};

/// First launch: calls 0-3 once each into a fresh 16-line ring with no
/// consumer; lane 0 records each result. Calls 0 and 2 fit (16 lines), so
/// calls 1 and 3 must find the ring short. A resumed launch retries calls 1
/// and 3 from the saved handle until they succeed. Both save the handle.
__global__ void short_kernel(SPSCD2HQueue::DeviceHandle h, SPSCD2HQueue::DeviceHandle* saved,
                             bool resume, bool* results) {
    if (resume) {
        h = detail::wave_uniform(*saved);
        while (!push_ones(h, 1, kShortCalls[1])) {}
        while (!push_ones(h, 3, kShortCalls[3])) {}
    } else {
        for (uint32_t step = 0; step < 4; ++step) {
            const bool ok = push_ones(h, step, kShortCalls[step]);
            if (__lane_id() == 0) results[step] = ok;
        }
    }
    if (__lane_id() == 0) *saved = h;
}

/// What a trap kernel does (wrong).
enum Fault : int {
    kPartialWave,  ///< lanes 0-31 call push_lane_masked()
    kBytesPastT,   ///< lane 7 passes bytes = sizeof(T) + 1, still one line
    kCall,         ///< lanes [0, lanes) push `bytes` bytes each
};

template <typename H>
__global__ void fault_kernel(H h, Fault f, uint32_t lanes, uint32_t bytes) {
    const uint32_t lane = __lane_id();
    const Msg<64, 0> m = make_msg<64, 0>(lane);  // up to 512 B, 16 lines
    switch (f) {
        case kPartialWave:
            if (lane < 32) h.push_lane_masked(m, 8, 0, true);
            break;
        case kBytesPastT: {
            // 9 bytes take one line, so only the bytes > sizeof(T) check can
            // trap (a 513-byte message would also make the call too long).
            const Msg<1, 0> one = make_msg<1, 0>(lane);
            static_assert(message_lines(sizeof(one) + 1) == 1);
            h.push_lane_masked(one, lane == 7 ? sizeof(one) + 1 : 8, 0, lane == 7);
            break;
        }
        default:
            while (!h.push_lane_masked(m, bytes, 0, lane < lanes)) {}
            break;
    }
}

// The death tests call push_fault() only in the host pass (the device pass
// lacks gtest's death-test macros), so instantiate the kernels here, or the
// device code would lack them and every launch would fail.
template __global__ void fault_kernel(SPSCD2HQueue::DeviceHandle, Fault, uint32_t, uint32_t);
template __global__ void fault_kernel(MPSCD2HQueue::DeviceHandle, Fault, uint32_t, uint32_t);

/// Runs fault_kernel on a fresh `capacity`-line queue; exits normally only if
/// it did not trap.
template <typename Q>
void push_fault(size_t capacity, Fault f, uint32_t lanes = 0, uint32_t bytes = 0) {
    Q q(capacity);
    fault_kernel<<<1, kLanes>>>(q.device_handle(), f, lanes, bytes);
    if (hipDeviceSynchronize() != hipSuccess) std::abort();
    std::exit(0);
}

}  // namespace

template <typename Q>
class PushLaneMasked : public ::testing::Test {};
using Queues = ::testing::Types<SPSCD2HQueue, MPSCD2HQueue>;
TYPED_TEST_SUITE(PushLaneMasked, Queues);

template <typename Q>
class PushLaneMaskedDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(PushLaneMaskedDeathTest, Queues);

// push_lane_masked() is wave-collective: on SPSC a call from part of the wave
// would advance the register producer index in those lanes only, and on MPSC
// the credit cache. Debug builds trap; the check is debug-only, as in the
// other pushes.
TYPED_TEST(PushLaneMaskedDeathTest, PartialWaveTrapsInDebug) {
    // (The device pass parses host code without gtest's death-test macros.)
#if defined(NDEBUG) || defined(__HIP_DEVICE_COMPILE__) || !GTEST_HAS_DEATH_TEST
    GTEST_SKIP() << "assert_full_wave() checks only in debug builds";
#else
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(push_fault<TypeParam>(16, kPartialWave), "");
#endif
}

// An active lane's bytes > sizeof(T) traps in every build.
TYPED_TEST(PushLaneMaskedDeathTest, BytesPastMessageTraps) {
#if defined(__HIP_DEVICE_COMPILE__) || !GTEST_HAS_DEATH_TEST
    GTEST_SKIP() << "needs death tests";
#else
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(push_fault<TypeParam>(16, kBytesPastT), "");
#endif
}

// A call whose active lanes need capacity + 1 lines traps in every build, on
// an empty ring too; a call of exactly `capacity` lines goes through. The
// largest call, 64 x 16 lines, fills a 1024-line ring exactly, so no call can
// be too long from 1024 lines on.
TYPED_TEST(PushLaneMaskedDeathTest, CallLongerThanRingTraps) {
#if defined(__HIP_DEVICE_COMPILE__) || !GTEST_HAS_DEATH_TEST
    GTEST_SKIP() << "needs death tests";
#else
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    static_assert(message_lines(64) == 2 && message_lines(160) == 5 &&
                  message_lines(288) == 9 && message_lines(512) == 16);
    EXPECT_DEATH(push_fault<TypeParam>(1, kCall, 1, 64), "");       // 2 lines
    EXPECT_DEATH(push_fault<TypeParam>(16, kCall, 17, 8), "");      // 17 lines
    EXPECT_DEATH(push_fault<TypeParam>(64, kCall, 13, 160), "");    // 65 lines
    EXPECT_DEATH(push_fault<TypeParam>(512, kCall, 57, 288), "");   // 513 lines
    EXPECT_EXIT(push_fault<TypeParam>(1, kCall, 1, 32), ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(push_fault<TypeParam>(16, kCall, 16, 8), ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(push_fault<TypeParam>(64, kCall, 64, 8), ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(push_fault<TypeParam>(512, kCall, 32, 512), ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(push_fault<TypeParam>(1024, kCall, 64, 512), ::testing::ExitedWithCode(0), "");
#endif
}

// Exact pop order, lengths, metadata and payloads of collective pushes and a
// masked push_lane_masked(); SPSC: the handle advanced by exactly the lines
// pushed.
TYPED_TEST(PushLaneMasked, LaneOrderWithActiveMask) {
    TypeParam q(64);
    using Handle = typename TypeParam::DeviceHandle;
    Handle* d_out = nullptr;
    bool* d_pushed = nullptr;
    HIP_OK(hipMalloc(&d_out, sizeof(Handle)));
    HIP_OK(hipMalloc(&d_pushed, sizeof(bool)));
    order_kernel<<<1, kLanes>>>(q.device_handle(), d_out, d_pushed);
    HIP_OK(hipDeviceSynchronize());
    Handle out{};
    bool pushed = false;
    HIP_OK(hipMemcpy(&out, d_out, sizeof(out), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(&pushed, d_pushed, sizeof(pushed), hipMemcpyDeviceToHost));
    HIP_OK(hipFree(d_out));
    HIP_OK(hipFree(d_pushed));
    EXPECT_TRUE(pushed);
    if constexpr (std::is_same_v<TypeParam, SPSCD2HQueue>) {
        EXPECT_EQ(out.producer_index, 27u);  // 1 + 2 + 16 + 1, 1 + 4 + 1, 1
    }

    // Popped in claim order with the pushed lengths, metadata and payloads.
    const std::pair<uint32_t, size_t> want[] = {{0, 0},           {1, 33},          {2, 512},
                                                {3, 8},           {4 << 8 | 1, 8},  {4 << 8 | 5, 100},
                                                {4 << 8 | 63, 0}, {5, 8}};
    alignas(8) uint8_t buf[kMaxMessageBytes];
    for (const auto& [meta, bytes] : want) {
        SCOPED_TRACE(meta);
        uint32_t m = 0;
        size_t b = 0;
        ASSERT_TRUE(q.pop(buf, &m, &b));
        EXPECT_EQ(m, meta);
        ASSERT_EQ(b, bytes);
        for (size_t k = 0; k < bytes / 8; ++k) {
            uint64_t w;
            std::memcpy(&w, buf + 8 * k, 8);
            EXPECT_EQ(w, word(meta, static_cast<int>(k)));
        }
    }
    EXPECT_FALSE(q.pop(buf));
}

// SPSC: a call the ring lacks credit for returns false and writes nothing
// (the earlier messages it would overwrite pop intact), and the producer
// index does not move. Retried from the saved handle, it waits while the
// consumer is away and succeeds once the consumer frees the lines.
TEST(PushLaneMaskedSpsc, ShortCallReturnsFalseAndWritesNothing) {
    SPSCD2HQueue q(16);
    SPSCD2HQueue::DeviceHandle* saved = nullptr;
    bool* d_results = nullptr;
    HIP_OK(hipMalloc(&saved, sizeof(*saved)));
    HIP_OK(hipMalloc(&d_results, 4 * sizeof(bool)));
    short_kernel<<<1, kLanes>>>(q.device_handle(), saved, false, d_results);
    HIP_OK(hipDeviceSynchronize());
    bool results[4] = {};
    SPSCD2HQueue::DeviceHandle h{};
    HIP_OK(hipMemcpy(results, d_results, sizeof(results), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(&h, saved, sizeof(h), hipMemcpyDeviceToHost));
    EXPECT_TRUE(results[0]);
    EXPECT_FALSE(results[1]);
    EXPECT_TRUE(results[2]);
    EXPECT_FALSE(results[3]);
    EXPECT_EQ(h.producer_index, 16u);

    // The retries wait while the ring is full and nobody pops.
    hipStream_t s;
    HIP_OK(hipStreamCreate(&s));
    short_kernel<<<1, kLanes, 0, s>>>(SPSCD2HQueue::DeviceHandle{}, saved, true, d_results);
    HIP_OK(hipGetLastError());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(hipStreamQuery(s), hipErrorNotReady);

    // Calls 0 and 2, then the retried 1 and 3, in lane order.
    alignas(8) uint8_t buf[kMaxMessageBytes];
    for (uint32_t step : {0u, 2u, 1u, 3u}) {
        for (uint32_t lane = 0; lane < kShortCalls[step]; ++lane) {
            SCOPED_TRACE(::testing::Message() << "call " << step << " lane " << lane);
            uint32_t meta = 0;
            size_t bytes = 0;
            const auto start = std::chrono::steady_clock::now();
            while (!q.pop(buf, &meta, &bytes)) {
                ASSERT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(20));
            }
            EXPECT_EQ(meta, tag(0, step, lane));
            EXPECT_EQ(bytes, 8u);
            uint64_t w = 0;
            std::memcpy(&w, buf, 8);
            EXPECT_EQ(w, word(meta, 0));
        }
    }
    HIP_OK(hipStreamSynchronize(s));
    HIP_OK(hipStreamDestroy(s));
    HIP_OK(hipMemcpy(&h, saved, sizeof(h), hipMemcpyDeviceToHost));
    EXPECT_EQ(h.producer_index, 27u);
    EXPECT_FALSE(q.pop(buf));
    HIP_OK(hipFree(saved));
    HIP_OK(hipFree(d_results));
}

// One wave into a large ring, every lane active or an active mask (none,
// alternating, one lane, prefixes, suffixes, random). Messages of up to 64,
// 512, 101 and 13 bytes; at 512 bytes each lane writes 16 lines.
TYPED_TEST(PushLaneMasked, ActiveMasks) {
    run<TypeParam, 8, 0, false>({1, 1, 2000, 4096, 2, 0});
    run<TypeParam, 64, 0, false>({1, 1, 300, 4096, 1, 0});
    run<TypeParam, 12, 5, false>({1, 1, 1000, 4096, 2, 0});
    run<TypeParam, 1, 5, false>({1, 1, 2000, 4096, 2, 0});
}

// Rings of 1 to 2048 lines wrap many times, with calls of up to the whole
// ring and multi-line messages. The consumer starts 50 ms late, so the
// producer fills the ring: SPSC calls return false and are retried, MPSC
// calls wait for credit for their whole claim.
TYPED_TEST(PushLaneMasked, WrapAroundWithBackpressure) {
    for (size_t cap : {1, 2, 4, 16, 64}) run<TypeParam, 8, 0, false>({1, 1, 400, cap, 2, 50});
    for (size_t cap : {16, 32, 1024, 2048}) run<TypeParam, 64, 0, false>({1, 1, 100, cap, 2, 50});
    run<TypeParam, 12, 5, false>({1, 1, 300, 4, 2, 50});
}

// Collective and lane-masked pushes interleaved on one queue, in both orders.
// SPSC: odd launches resume from the handle the previous launch saved.
TYPED_TEST(PushLaneMasked, MixedWithCollective) {
    for (size_t cap : {1, 16, 64}) run<TypeParam, 8, 0, true>({1, 1, 600, cap, 3, 50});
    run<TypeParam, 8, 0, true>({1, 1, 2000, 4096, 3, 0});
    run<TypeParam, 64, 0, true>({1, 1, 200, 16, 2, 50});
}

// Many MPSC producer waves contend for one index with lane-masked claims
// (one fetch-add per wave and call), alone or mixed with collective claims.
TEST(PushLaneMaskedMpsc, ManyWavesContend) {
    run<MPSCD2HQueue, 8, 0, true>({8, 4, 200, 16, 1, 50});    // 32 waves, 16-line ring
    run<MPSCD2HQueue, 8, 0, false>({8, 4, 200, 64, 2, 0});    // 32 waves, 64-line ring
    run<MPSCD2HQueue, 8, 0, true>({304, 1, 40, 1024, 1, 0});  // a wave on every MI300X CU
    run<MPSCD2HQueue, 8, 0, true>({64, 4, 30, 256, 1, 0});    // 256 waves
    run<MPSCD2HQueue, 64, 0, true>({16, 2, 30, 64, 1, 0});    // 16-line messages
    run<MPSCD2HQueue, 12, 5, true>({16, 2, 60, 64, 1, 0});    // 101-byte messages
}
