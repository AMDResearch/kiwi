# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton frontend of the invoke callback throughput harness.

Same options, host code and output as the harness's C++ frontend
(kiwi_invoke_throughput_bench); throughput_invoke_kernel mirrors the C++
kernel line for line. A Triton program here is one wavefront, so at most 256
endpoints are supported (the C++ kernel packs several endpoint wavefronts into
a block above that).
"""

import torch
import triton
import triton.language as tl

import kiwi_triton as kiwi
import kiwi_invoke_throughput_bench
from triton_bench_ops import launch_empty


@triton.jit(do_not_specialize=["callback_id", "num_msgs_per_endpoint", "num_endpoints",
                                "waves_per_block"])
def throughput_invoke_kernel(device_context, callback_id, num_msgs_per_endpoint, num_endpoints,
                             waves_per_block):
    # One wavefront (num_warps=1) per program owns one endpoint.
    endpoint = tl.program_id(0) * waves_per_block
    if endpoint >= num_endpoints:
        return

    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(
        device_context, endpoint)

    for i in range(num_msgs_per_endpoint):
        value = endpoint * num_msgs_per_endpoint + i
        producer, cached = kiwi.invoke_u32(chunks, consumer_index, producer, cached,
                                           log2_capacity, callback_id, value, 4)


def launch(ctx, callback_id, num_msgs_per_endpoint, num_endpoints, waves_per_block,
           launch_blocks):
    kiwi.launch(throughput_invoke_kernel, (launch_blocks,), ctx, callback_id,
                num_msgs_per_endpoint, num_endpoints, waves_per_block)


def precompile(waves_per_block, launch_blocks):
    """Compile, check and load the kernel before the timed region (the C++
    benchmark has no JIT step). With num_endpoints = 0 every program exits
    before touching the context, so the dummy address is never dereferenced."""
    if waves_per_block != 1:
        raise SystemExit("the Triton kernel supports at most 256 endpoints")
    dummy = torch.zeros(1, dtype=torch.int64, device="cuda")
    launch_empty(throughput_invoke_kernel, (launch_blocks,), dummy.data_ptr(), 0, 0, 0,
                 waves_per_block)


if __name__ == "__main__":
    torch.cuda.init()
    kiwi_invoke_throughput_bench.main(launch, "invoke Throughput Benchmark (Triton kernel)",
                                      precompile)
