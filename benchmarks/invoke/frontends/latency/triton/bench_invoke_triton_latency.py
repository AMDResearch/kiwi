# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton frontend of the invoke round-trip latency harness.

Same options, host code and output as the harness's C++ frontend
(kiwi_invoke_latency_bench); ping_pong_latency_kernel mirrors the C++
ping_pong_latency_handle_kernel line for line.
Each kernel variant is compiled and checked before its first launch, the
untimed warmup kernel, which also loads it.
"""

import torch
import triton
import triton.language as tl

import kiwi_triton as kiwi
import kiwi_invoke_latency_bench
from triton_bench_ops import require_no_scratch, wait_vmcnt0, wall_clock64


@triton.jit(do_not_specialize=["ping_cb", "ack_flag", "num_iterations", "latencies",
                                "warmup_iterations", "latency_stats"])
def ping_pong_latency_kernel(device_context, ping_cb, ack_flag, num_iterations, latencies,
                             warmup_iterations, latency_stats, PAYLOAD_SIZE: tl.constexpr):
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(
        device_context, 0)
    ack = ack_flag.to(tl.uint64).to(tl.pointer_type(tl.uint64))
    times = latencies.to(tl.uint64).to(tl.pointer_type(tl.uint64))

    prev_ack = tl.full((), 0, tl.uint64)

    # Full cycle time: end of the previous iteration to the end of this one.
    t_prev = tl.full((), 0, tl.uint64)
    if latency_stats != 0:
        t_prev = wall_clock64()

    for i in range(num_iterations):
        # FixedPayload<PAYLOAD_SIZE>{.value = prev_ack}
        producer, cached = kiwi.invoke_u32(chunks, consumer_index, producer, cached,
                                           log2_capacity, ping_cb, prev_ack, PAYLOAD_SIZE)
        wait_vmcnt0()

        # Wait for CPU acknowledgment (all lanes wait in lockstep).
        current_ack = tl.load(ack, cache_modifier=".cv", volatile=True)
        while current_ack == prev_ack:
            current_ack = tl.load(ack, cache_modifier=".cv", volatile=True)
        prev_ack = current_ack

        if latency_stats != 0:
            t_now = wall_clock64()
            if i >= warmup_iterations:
                tl.store(times + (i - warmup_iterations), t_now - t_prev)
            t_prev = t_now


_checked = set()


def launch(ctx, ping_cb, ack_flag, num_iterations, latencies, warmup_iterations,
           latency_stats, stream, payload_size):
    args = (ctx, ping_cb, ack_flag, num_iterations, latencies, warmup_iterations,
            int(latency_stats))
    if payload_size not in _checked:
        require_no_scratch(kiwi.warmup(ping_pong_latency_kernel, (1,), *args,
                                       PAYLOAD_SIZE=payload_size))
        _checked.add(payload_size)
    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        kiwi.launch(ping_pong_latency_kernel, (1,), *args, PAYLOAD_SIZE=payload_size)


if __name__ == "__main__":
    torch.cuda.init()
    kiwi_invoke_latency_bench.main(launch, "invoke Round-Trip Latency Benchmark (Triton kernel)")
