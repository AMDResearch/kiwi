# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Driver for the invoke round-trip latency harness with a kernel written in a Python GPU language.

main() takes the same options as bench_invoke_latency and the harness's C++
frontend and runs the harness's host code (kiwi_invoke_latency_bench_ext); a
frontend (flydsl/, ...) supplies only the kernel launcher:

    launch(ctx, ping_cb, ack_flag, num_iterations, latencies,
           warmup_iterations, latency_stats, stream, payload_size)
        Launch the ping-pong kernel (one wavefront) asynchronously on `stream`.

Addresses and the stream are ints. Import the frontend (and with it the HIP
runtime it loads) before calling main().
"""

import argparse


def main(launch, title, argv=None):
    import kiwi_invoke_latency_bench_ext as ext
    d = ext.DEFAULTS
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("-c", "--capacity", type=int, default=d["capacity"],
                        help="Queue capacity (64-byte lines)")
    parser.add_argument("-n", "--iterations", type=int, default=d["iterations"],
                        help="Number of iterations")
    parser.add_argument("-w", "--warmup", type=int, default=d["warmup"],
                        help="Warmup iterations for GPU clock ramp (separate kernel)")
    parser.add_argument("-p", "--payload-size", type=int, default=d["payload_size"],
                        choices=ext.PAYLOAD_SIZES, help="Payload size in bytes")
    parser.add_argument("-s", "--stats", action="store_true",
                        help="Collect and report detailed latency statistics (percentiles)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device-ack", action="store_true",
                        help="Use fine-grained device memory for ack_flag (GPU VRAM, CPU-writable)")
    args = parser.parse_args(argv)
    if args.capacity <= 0 or args.iterations <= 0:
        parser.error("--capacity and --iterations must be positive")

    ext.run(launch, args.capacity, args.iterations, args.warmup, args.payload_size, args.stats,
            args.device_ack, title)
