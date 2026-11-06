# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Driver for the invoke callback throughput harness with a kernel written in a Python GPU language.

main() takes the same options as bench_invoke_throughput and the harness's
C++ frontend and runs the harness's host code
(kiwi_invoke_throughput_bench_ext); a frontend (flydsl/, ...) supplies only
the kernel launcher:

    launch(ctx, callback_id, num_msgs_per_endpoint, num_endpoints,
           waves_per_block, launch_blocks)
        Launch launch_blocks blocks of 64 * waves_per_block threads
        asynchronously on the null stream.

Addresses are ints. Import the frontend (and with it the HIP runtime it
loads) before calling main().
"""

import argparse
import sys


def main(launch, title, precompile=None, argv=None):
    """`precompile(waves_per_block, launch_blocks)`, if given, runs before the
    timed region so that a JIT compile and module load are not measured."""
    import kiwi_invoke_throughput_bench_ext as ext
    d = ext.DEFAULTS
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("-c", "--capacity", type=int, default=d["capacity"],
                        help="Queue capacity (64-byte lines)")
    parser.add_argument("-m", "--messages", type=int, default=d["messages"],
                        help="Total number of messages")
    parser.add_argument("-e", "--endpoints", type=int, default=d["endpoints"],
                        help="Number of endpoints (parallel queues)")
    parser.add_argument("-t", "--cpu-threads", type=int, default=d["cpu_threads"],
                        help="Number of CPU consumer threads")
    parser.add_argument("--poll-mode", default=d["poll_mode"], choices=ext.POLL_MODES,
                        help="Consumer polling: direct or host-poll-state")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args(argv)
    if min(args.capacity, args.messages, args.endpoints, args.cpu_threads) <= 0:
        parser.error("--capacity, --messages, --endpoints and --cpu-threads must be positive")
    if args.endpoints > ext.MAX_ENDPOINTS:
        print(f"endpoint count must not exceed {ext.MAX_ENDPOINTS}", file=sys.stderr)
        sys.exit(1)

    if precompile is not None:
        precompile(*ext.launch_shape(args.endpoints))

    processed = ext.run(launch, args.capacity, args.messages, args.endpoints, args.cpu_threads,
                        args.poll_mode, args.verbose, title)
    expected = args.messages // args.endpoints * args.endpoints
    if processed != expected:
        print(f"processed {processed} of {expected} messages", file=sys.stderr)
        sys.exit(1)
