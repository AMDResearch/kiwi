# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A Triton kernel invokes a C++ callback that prints its arguments.

The C++ side is examples/common's kiwi_print_ext, shared with the FlyDSL
example. Run from the repository root (see README.md).
"""

import argparse

import torch
import triton
import triton.language as tl
import kiwi_triton as kiwi
import kiwi_print_ext as ext


@triton.jit
def print_kernel(ctx, cb, n):
    lane = tl.arange(0, 64)
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(ctx, 0)
    for i in range(n):
        # Arguments of the C++ callback (int32_t, float, uint64_t), packed as
        # ArgumentSerializer packs them: lane c holds bytes [8c, 8c + 8).
        step = i.to(tl.uint32).to(tl.uint64)
        value = (i.to(tl.float32) * 0.5).to(tl.uint32, bitcast=True).to(tl.uint64)
        tag = i.to(tl.uint64) + 0xC0FFEE00
        word = tl.where(lane == 0, step | (value << 32), tl.where(lane == 1, tag, 0))
        producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity,
                                       cb, word.to(tl.uint64), 16)
    # Persist queue progress so a later launch continues where this one ended.
    kiwi.save_device_handle(ctx, 0, producer, cached)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=8, help="number of invokes")
    args = parser.parse_args()

    printer = ext.PrintContext(queue_capacity=1024)
    # One 64-lane wavefront owns endpoint 0: kiwi.launch runs it with num_warps=1.
    kiwi.launch(print_kernel, (1,), printer.device_context, printer.callback_id, args.n)

    executed = 0
    while executed < args.n:
        executed += printer.progress()
    torch.cuda.synchronize()
    print(f"Triton kernel invoked the C++ callback {executed} times")


if __name__ == "__main__":
    main()
