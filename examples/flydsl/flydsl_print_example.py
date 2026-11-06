# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""A FlyDSL kernel invokes a C++ callback that prints its arguments.

Run from the build directory's examples/flydsl/ (see README.md).
"""

import argparse

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
import kiwi_flydsl as kiwi
import kiwi_print_ext as ext


@flyc.kernel
def producer(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32):
    h = ctx.get_device_handle(0)
    for i in range(n):
        # Argument types match the C++ callback: (int32_t, float, uint64_t).
        h = h.invoke(cb, i, fx.Float32(fx.Int32(i)) * 0.5, fx.Uint64(i) + 0xC0FFEE00)


@flyc.jit
def launch(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32,
           stream: fx.Stream = fx.Stream(None)):
    producer(ctx, cb, n).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=8, help="number of invokes")
    args = parser.parse_args()

    printer = ext.PrintContext(queue_capacity=1024)
    launch(kiwi.Endpoints(printer.device_context), printer.callback_id, args.n)

    executed = 0
    while executed < args.n:
        executed += printer.progress()
    torch.cuda.synchronize()
    print(f"FlyDSL kernel invoked the C++ callback {executed} times")


if __name__ == "__main__":
    main()
