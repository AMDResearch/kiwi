# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Driver for the KIWI readiness harness with the protocol kernel written in a Python GPU language.

main() takes the same options as bench_kiwi_readiness and the harness's C++
frontend and runs the harness's host code (kiwi_readiness_bench_ext); a
frontend (flydsl/, ...) supplies only the kernel launcher:

    launch(ctx, rank, peer, source, target, source_heap, target_heap, cb_id,
           iters, depth)
        Launch the persistent protocol kernel (one block of 64 threads)
        asynchronously. `ctx` is the invoke DeviceContext address of the P2P
        context; the readiness descriptor is two u64 (source, remote offset)
        and four i32 (source heap, target heap, rank, padding).

Addresses are ints. Run with exactly two ranks. Import the frontend (and with
it the HIP runtime it loads) before calling main().
"""

import argparse
import sys


def main(launch, backend, title, argv=None):
    import kiwi_readiness_bench_ext as ext
    d = ext.DEFAULTS
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--warmup", type=int, default=d["warmup"])
    parser.add_argument("--iters", type=int, default=d["iters"])
    parser.add_argument("--depth", type=int, default=d["depth"],
                        help="Forward writes per acknowledged batch; use 1 for latency")
    parser.add_argument("--main-core", type=int, default=d["main_core"],
                        help="Pin the main/control thread to this CPU (-1: inherit)")
    parser.add_argument("--proxy-core", type=int, default=d["proxy_core"],
                        help="Pin the KIWI proxy thread to this CPU (-1: inherit)")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iters <= 0 or args.depth <= 0:
        parser.error("--warmup must be >= 0; --iters and --depth must be positive")

    # One untimed round first: the kernel is compiled and loaded on its first
    # launch, and the C++ kernel has no JIT step.
    sys.exit(ext.run(launch, args.warmup, args.iters, args.depth, args.main_core,
                     args.proxy_core, backend, 1))
