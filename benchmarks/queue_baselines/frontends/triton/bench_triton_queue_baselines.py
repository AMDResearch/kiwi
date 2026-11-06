# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton frontend of the kiwi-spsc queue-baseline harness (queue=kiwi-spsc-triton).

Same modes, options, host code and RESULT lines as the harness's C++ frontend
(kiwi_queue_bench): only the producer kernels are Triton. They mirror the C++
latency_kernel, push_cost_kernel and throughput_kernel, and push with
kiwi_triton.push, the Triton copy of while (!handle.push_lanes(...)) {}.
A Triton program is one wavefront, so --waves-per-block must be 1.
"""

import sys

try:
    import torch
    import triton
    import triton.language as tl
    import kiwi_triton as kiwi
    import kiwi_queue_bench
    from triton_bench_ops import drain_memory_ops, launch_empty, timestamp, u64_ptr
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"SKIP: benchmark dependency unavailable: {exc}")
    sys.exit(77)

# One 16-byte message: lane 0 carries word0, lane 1 the tag (qb::Msg).
MSG_BYTES = tl.constexpr(16)
NO_METADATA = tl.constexpr(0)  # a raw push (KiwiSpscAdapter passes metadata 0)
CALIBRATION_ITERS = tl.constexpr(256)  # kCalibrationIters
HANDLE_SIZE = tl.constexpr(kiwi.HANDLE_SIZE)


@triton.jit
def msg_word(producer, seq):
    """KiwiSpscAdapter::push's per-lane word for make_msg(producer, seq):
    lane 0 carries word0 and every other lane the tag (push_lanes ignores the
    words of lanes past the 16-byte message)."""
    lane = tl.arange(0, 64)
    # word0 = seq << 32 | producer and tag = ~word0 & INT64_MAX, built from
    # 32-bit halves (as hipcc splits them) so the loop-invariant producer half
    # is hoisted out of the push loop.
    producer = tl.cast(producer, tl.uint32)
    seq = tl.cast(seq, tl.uint32)
    lo = tl.where(lane == 0, producer, ~producer)
    hi = tl.where(lane == 0, seq, ~seq & 0x7FFFFFFF)
    return (hi.to(tl.uint64) << 32) | lo.to(tl.uint64)


@triton.jit
def push(chunks, consumer_index, producer, cached, log2_capacity, word):
    """KiwiSpscAdapter::push: while (!h.push_lanes(word, 16, 0)) {}."""
    return kiwi.push(chunks, consumer_index, producer, cached, log2_capacity, word,
                     tl.full((), NO_METADATA, tl.uint32), MSG_BYTES)


@triton.jit(do_not_specialize=["chunks", "consumer_index", "producer", "cached",
                                "log2_capacity", "ack", "iters", "latency_ticks"])
def latency_kernel(chunks, consumer_index, producer, cached, log2_capacity, ack, iters,
                   latency_ticks):
    """latency_kernel<KiwiSpscAdapter>: the handle is passed by value."""
    chunks = tl.cast(chunks, tl.uint64)
    consumer_index = tl.cast(consumer_index, tl.uint64)
    producer = tl.cast(producer, tl.uint64)
    cached = tl.cast(cached, tl.uint64)
    log2_capacity = tl.cast(log2_capacity, tl.uint64)
    ack_ptr = u64_ptr(ack)
    ticks = u64_ptr(latency_ticks)
    t_prev = timestamp(True)
    for i in range(iters):
        producer, cached = push(chunks, consumer_index, producer, cached, log2_capacity,
                                msg_word(0, i))
        # Let the publish stores complete before polling the ack.
        drain_memory_ops()
        expected = tl.cast(i, tl.uint64) + 1
        while tl.load(ack_ptr, cache_modifier=".cv", volatile=True) < expected:
            pass
        t_now = timestamp(True)
        tl.store(ticks + i, t_now - t_prev)
        t_prev = t_now


@triton.jit(do_not_specialize=["handles", "waves", "warmup", "iters", "cycles", "calibration"])
def push_cost_kernel(handles, waves, warmup, iters, cycles, calibration,
                     BURST: tl.constexpr, WALL: tl.constexpr):
    """push_cost_kernel<KiwiSpscAdapter, WALL>, one wave per program."""
    wave = tl.program_id(0)
    if wave < waves:
        chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_handle(
            tl.cast(handles, tl.uint64) + tl.cast(wave, tl.uint64) * HANDLE_SIZE)
        if wave == 0:
            best = tl.full((), 0xFFFFFFFFFFFFFFFF, tl.uint64)
            for _ in range(CALIBRATION_ITERS):
                drain_memory_ops()
                t0 = timestamp(WALL)
                t1 = timestamp(WALL)
                best = tl.minimum(best, t1 - t0)
            tl.store(u64_ptr(calibration), best)
        out = u64_ptr(cycles)
        if not BURST:
            for i in range(warmup + iters):
                word = msg_word(wave, i)
                # Without draining, back-to-back posted stores saturate vmcnt.
                drain_memory_ops()
                t0 = timestamp(WALL)
                producer, cached = push(chunks, consumer_index, producer, cached,
                                        log2_capacity, word)
                t1 = timestamp(WALL)
                if i >= warmup:
                    tl.store(out + tl.cast(wave, tl.int64) * iters + (i - warmup), t1 - t0)
        else:
            for i in range(warmup):
                producer, cached = push(chunks, consumer_index, producer, cached,
                                        log2_capacity, msg_word(wave, i))
            drain_memory_ops()
            t0 = timestamp(WALL)
            for i in range(warmup, warmup + iters):
                producer, cached = push(chunks, consumer_index, producer, cached,
                                        log2_capacity, msg_word(wave, i))
            drain_memory_ops()
            t1 = timestamp(WALL)
            tl.store(out + wave, t1 - t0)


@triton.jit(do_not_specialize=["handles", "waves", "msgs_per_wave"])
def throughput_kernel(handles, waves, msgs_per_wave):
    """throughput_kernel<KiwiSpscAdapter>, one wave per program."""
    wave = tl.program_id(0)
    if wave < waves:
        chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_handle(
            tl.cast(handles, tl.uint64) + tl.cast(wave, tl.uint64) * HANDLE_SIZE)
        for i in range(msgs_per_wave):
            producer, cached = push(chunks, consumer_index, producer, cached,
                                    log2_capacity, msg_word(wave, i))


class TritonLauncher:
    """Launches the kernels above for kiwi_queue_bench."""

    def __init__(self):
        # The C++ frontend has no JIT step: compile, check and load every
        # kernel variant before the run, with launches that do no work (no
        # iterations or no waves).
        dummy = torch.zeros(1, dtype=torch.int64, device="cuda")
        p = dummy.data_ptr()
        launch_empty(latency_kernel, (1,), p, p, 0, 0, 0, p, 0, p)
        for burst in (False, True):
            for wall in (False, True):
                launch_empty(push_cost_kernel, (1,), p, 0, 0, 0, p, p, BURST=burst, WALL=wall)
        launch_empty(throughput_kernel, (1,), p, 0, 0)

    @staticmethod
    def check_options(args):
        """Called by kiwi_queue_bench before the harness starts."""
        if args.waves_per_block != 1:
            return "a Triton program is one wave; use --waves-per-block 1"
        return None

    def latency(self, chunks, consumer_index, producer, cached, log2_capacity, ack, iters,
                latency_ticks, stream):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            kiwi.launch(latency_kernel, (1,), chunks, consumer_index, producer, cached,
                        log2_capacity, ack, iters, latency_ticks)

    def push_cost(self, handles, blocks, waves, waves_per_block, warmup, iters, burst, wall,
                  cycles, calibration):
        kiwi.launch(push_cost_kernel, (waves,), handles, waves, warmup, iters, cycles,
                    calibration, BURST=burst, WALL=wall)

    def throughput(self, handles, blocks, waves, waves_per_block, msgs_per_wave):
        kiwi.launch(throughput_kernel, (waves,), handles, waves, msgs_per_wave)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no HIP device is available")
        sys.exit(77)
    torch.cuda.init()
    kiwi_queue_bench.main(TritonLauncher(), "kiwi-spsc-triton")
