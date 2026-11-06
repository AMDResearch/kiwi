# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""FlyDSL frontend of the invoke round-trip latency harness.

Same options, host code and output as the harness's C++ frontend
(kiwi_invoke_latency_bench); ping_pong_latency_kernel mirrors the C++
ping_pong_latency_handle_kernel line for line.
"""

import torch  # noqa: F401  (load the HIP runtime FlyDSL uses before the extension)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
import kiwi_flydsl as kiwi
import kiwi_invoke_latency_bench

from flydsl_bench_ops import load_volatile_u64, store_u64, wait_vmcnt0, wall_clock64


@flyc.kernel
def ping_pong_latency_kernel(ctx: kiwi.DeviceContext, ping_cb: fx.Int32, ack_flag: fx.Int64,
                             num_iterations: fx.Int32, latencies: fx.Int64,
                             warmup_iterations: fx.Int32, latency_stats: fx.Int32,
                             payload_size: fx.Constexpr[int]):
    handle = ctx.get_device_handle(0)
    record = latency_stats != 0 and fx.thread_idx.x == 0

    prev_ack = fx.Uint64(0)

    # Full cycle time: end of the previous iteration to the end of this one.
    t_prev = fx.Uint64(0)
    if record:
        t_prev = wall_clock64()

    for i in range(num_iterations):
        # FixedPayload<payload_size>: uint32 value, then zero padding.
        handle = handle.invoke(ping_cb, fx.Uint32(prev_ack),
                               *[fx.Uint32(0) for _ in range_constexpr((payload_size - 4) // 4)])
        wait_vmcnt0()

        # Wait for CPU acknowledgment (all lanes wait in lockstep).
        current_ack = load_volatile_u64(ack_flag)
        while current_ack == prev_ack:
            current_ack = load_volatile_u64(ack_flag)
        prev_ack = current_ack

        if record:
            t_now = wall_clock64()
            if i >= warmup_iterations:
                store_u64(fx.Int64(latencies) + fx.Int64(i - warmup_iterations) * 8,
                          t_now - t_prev)
            t_prev = t_now


@flyc.jit
def launch_ping_pong(ctx: kiwi.DeviceContext, ping_cb: fx.Int32, ack_flag: fx.Int64,
                     num_iterations: fx.Int32, latencies: fx.Int64,
                     warmup_iterations: fx.Int32, latency_stats: fx.Int32,
                     payload_size: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
    ping_pong_latency_kernel(ctx, ping_cb, ack_flag, num_iterations, latencies,
                             warmup_iterations, latency_stats, payload_size).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def launch(ctx, ping_cb, ack_flag, num_iterations, latencies, warmup_iterations,
           latency_stats, stream, payload_size):
    launch_ping_pong(kiwi.Endpoints(ctx), ping_cb, ack_flag, num_iterations, latencies,
                     warmup_iterations, int(latency_stats), payload_size,
                     stream=fx.Stream(stream))


if __name__ == "__main__":
    kiwi_invoke_latency_bench.main(launch, "invoke Round-Trip Latency Benchmark (FlyDSL kernel)")
