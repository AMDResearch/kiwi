# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Driver for the kiwi-spsc queue-baseline harness with a producer written in a Python GPU language.

main() takes the options of the harness's C++ frontend (those of
`bench_queue_baselines -q kiwi-spsc` for its modes) and runs the harness's
host code for the KIWI SPSC queue (kiwi_queue_bench_ext): the same proxy
threads, ack path, timers, validation and RESULT lines. A frontend
(triton/, ...) supplies `launcher`, an object whose methods launch kernels
with the contracts of the C++ kernels (cpp/, copied from
bench_queue_baselines.hip.cpp):

    latency(chunks, consumer_index, producer_index, consumer_index_cached,
            log2_capacity, ack, iters, latency_ticks, stream)
        latency_kernel: one wavefront on `stream`, the queue handle by value.
    push_cost(handles, blocks, waves, waves_per_block, warmup, iters, burst,
              wall, cycles, calibration)
        push_cost_kernel: `blocks` blocks of 64 * waves_per_block threads.
    throughput(handles, blocks, waves, waves_per_block, msgs_per_wave)
        throughput_kernel: `blocks` blocks of 64 * waves_per_block threads.

Addresses and the stream are ints; `handles` is a device array of
SPSCD2HQueue::DeviceHandle, one per wave. The launcher may also define
`check_options(args)`, returning an error message for options it does not
support; it is called before the harness starts. Import the frontend (and with it the
HIP runtime it loads) before calling main().
"""

import argparse
import sys


def main(launcher, name, argv=None):
    parser = argparse.ArgumentParser(
        description=f"GPU->CPU handoff microbenchmark: {name} (kiwi-spsc host side)")
    parser.add_argument("-m", "--mode", default="latency",
                        choices=("latency", "push-cost", "throughput", "idle-poll"))
    parser.add_argument("-c", "--capacity", type=int, default=1024,
                        help="Slots per queue; power of two")
    parser.add_argument("--queues", type=int, default=0,
                        help="Queues (0 = one per producer wave)")
    parser.add_argument("-w", "--waves", type=int, default=1,
                        help="Producer waves (throughput, push-cost)")
    parser.add_argument("--waves-per-block", type=int, default=1,
                        help="Waves per workgroup (throughput, push-cost)")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Pinned proxy threads (throughput)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Max pops per queue visit (throughput)")
    parser.add_argument("--idle-pause", type=int, default=0,
                        help="x86 pause instructions after an empty polling round (throughput)")
    parser.add_argument("--ack-mem", default="device", choices=("device", "host"),
                        help="latency: ack flag in 'device' (fine-grained VRAM) or 'host' memory")
    parser.add_argument("--timer", default="shader", choices=("shader", "wall"),
                        help="push-cost: 'shader' clock64() cycles or 'wall' wall_clock64() ns")
    parser.add_argument("--burst", action="store_true",
                        help="push-cost: time back-to-back pushes (loop time / pushes per wave)")
    parser.add_argument("-n", "--iters", type=int, default=100000,
                        help="Measured iterations (messages per wave)")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="Warmup iterations (latency, push-cost)")
    parser.add_argument("--idle-rounds", type=int, default=1000000,
                        help="Polling rounds (idle-poll)")
    parser.add_argument("--check-overlap", action="store_true",
                        help="Not supported: only the C++ kernels record wave spans")
    args = parser.parse_args(argv)
    if args.check_overlap:
        parser.error("--check-overlap is not supported: only the C++ kernels record wave spans")
    for opt in ("capacity", "waves", "waves_per_block", "threads", "batch", "iters",
                "idle_rounds"):
        if getattr(args, opt) <= 0:
            parser.error(f"--{opt.replace('_', '-')} must be positive")
    if args.capacity & (args.capacity - 1):
        parser.error("capacity must be a power of two")
    # Frontend-specific limits, checked before the harness starts its threads.
    check = getattr(launcher, "check_options", None)
    problem = check(args) if check else None
    if problem:
        parser.error(problem)

    import kiwi_queue_bench_ext as ext
    sys.exit(ext.run(launcher, name, args.mode, args.capacity, args.queues, args.waves,
                     args.waves_per_block, args.threads, args.batch, args.idle_pause,
                     args.burst, args.timer, args.ack_mem, args.iters, args.warmup,
                     args.idle_rounds))
