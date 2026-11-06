# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""FlyDSL frontend of the invoke callback throughput harness.

Same options, host code and output as the harness's C++ frontend
(kiwi_invoke_throughput_bench); throughput_invoke_kernel mirrors the C++
kernel line for line.
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
import kiwi_flydsl as kiwi
import kiwi_invoke_throughput_bench


@flyc.kernel
def throughput_invoke_kernel(ctx: kiwi.DeviceContext, callback_id: fx.Int32,
                             num_msgs_per_endpoint: fx.Int32, num_endpoints: fx.Int32,
                             waves_per_block: fx.Int32):
    # One wavefront owns one endpoint; extra endpoint wavefronts are packed
    # into each block.
    wave_in_block = fx.Int32(fx.thread_idx.x) // 64
    endpoint = fx.Int32(fx.block_idx.x) * waves_per_block + wave_in_block
    if endpoint < num_endpoints:
        handle = ctx.get_device_handle(endpoint)
        for i in range(num_msgs_per_endpoint):
            value = fx.Uint32(endpoint * num_msgs_per_endpoint + i)
            handle = handle.invoke(callback_id, value)


@flyc.jit
def launch_throughput(ctx: kiwi.DeviceContext, callback_id: fx.Int32,
                      num_msgs_per_endpoint: fx.Int32, num_endpoints: fx.Int32,
                      waves_per_block: fx.Constexpr[int], launch_blocks: fx.Constexpr[int],
                      stream: fx.Stream = fx.Stream(None)):
    throughput_invoke_kernel(ctx, callback_id, num_msgs_per_endpoint, num_endpoints,
                             fx.Int32(waves_per_block)).launch(
        grid=(launch_blocks, 1, 1), block=(64 * waves_per_block, 1, 1), stream=stream)


def launch(ctx, callback_id, num_msgs_per_endpoint, num_endpoints, waves_per_block,
           launch_blocks):
    launch_throughput(kiwi.Endpoints(ctx), callback_id, num_msgs_per_endpoint,
                      num_endpoints, waves_per_block, launch_blocks)


def precompile(waves_per_block, launch_blocks):
    """Compile for this launch shape before the timed region. With
    num_endpoints = 0 every wave exits before touching the context, so the
    dummy address is never dereferenced."""
    dummy = torch.zeros(1, dtype=torch.int64, device="cuda")
    launch(dummy.data_ptr(), 0, 0, 0, waves_per_block, launch_blocks)
    torch.cuda.synchronize()


if __name__ == "__main__":
    kiwi_invoke_throughput_bench.main(launch, "invoke Throughput Benchmark (FlyDSL kernel)",
                                      precompile)
