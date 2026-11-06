#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Run queue-baseline sweeps; prints one RESULT line per run to stdout.
#
# Usage: run_sweep.sh <bench_queue_baselines binary> <profile> [reps]
#   quick       one-rep smoke test of every mode
#   latency     raw-store floor, unloaded RTT, RTT under fixed background load
#   push        GPU push cost vs producer waves (isolated and burst)
#   throughput  Mops vs waves / proxy threads / idle backoff, and idle-poll cost
#   paper       latency + push + throughput
# Pin the process on the GPU's NUMA node, e.g.
#   HIP_VISIBLE_DEVICES=0 taskset -c 8-23 run_sweep.sh <bin> paper 3
set -uo pipefail

BIN=${1:?path to bench_queue_baselines}
PROFILE=${2:-quick}
REPS=${3:-3}
QUEUES="kiwi-spsc kiwi-mpsc kiwi-invoke mscclpp-fifo uccl-ep-fifo nvshmem-proxy rccl-gin-proxy rocshmem-ro llvm-rpc"

run() { "$BIN" "$@" 2>/dev/null | grep '^RESULT' || echo "RESULT failed args=$*"; }

# Queues each library shares by default (waves -> queues): NVSHMEM one
# channel per PE, MSCCL++ one FIFO per ProxyService, UCCL-EP 32 FIFOs, RCCL GIN
# contexts x peers (4 here), rocSHMEM 32 contexts, LLVM RPC one client
# per device; kiwi-mpsc fully shared.
default_queues() {
    local q=$1 w=$2 n
    case $q in
        uccl-ep-fifo|rocshmem-ro) n=32 ;;
        rccl-gin-proxy) n=4 ;;
        *) n=1 ;;
    esac
    (( n > w )) && n=$w
    echo "$n"
}

layouts_for() {  # one queue per wave (0), then the library default if different
    local q=$1 w=$2 d
    [[ $q == kiwi-spsc || $q == kiwi-invoke || $q == raw-store ]] && { echo 0; return; }
    d=$(default_queues "$q" "$w")
    (( d == w )) && echo 0 || echo "0 $d"
}

latency_profile() {
    # Each process launch settles into its own timing mode (about +/-0.4 us on
    # MI300X), so latency points use many short launches; report the median
    # and range across launches.
    local lat_n=$1 launches=$2
    for rep in $(seq "$launches"); do
        for am in host device; do
            for q in raw-store $QUEUES rocm-hostcall; do
                n=$lat_n; [[ $q == rocm-hostcall ]] && n=5000
                run -q "$q" -m latency -n "$n" --warmup 2000 --ack-mem "$am"
            done
        done
        # Probe RTT with 63 background waves at fixed offered loads that
        # every queue except rocSHMEM RO and LLVM RPC can sustain. The ack
        # lives in device memory: a host-memory ack makes the RTT depend on
        # poll phase (see README).
        for rate in 1 4 12; do
            for q in $QUEUES; do
                for nq in $(layouts_for "$q" 64); do
                    t=4; (( nq > 0 && nq < t )) && t=$nq
                    run -q "$q" -m loaded-latency -w 63 --queues "$nq" -t "$t" \
                        -n 10000 --bg-rate "$rate" --ack-mem device
                done
            done
        done
    done
}

push_profile() {
    local reps=$1 iters=1000 warm=100
    for rep in $(seq "$reps"); do
        for w in 1 8 64 256 1024; do
            for q in raw-store $QUEUES; do
                for nq in $(layouts_for "$q" "$w"); do
                    eff=$nq; (( eff == 0 )) && eff=$w
                    t=4; (( t > eff )) && t=$eff
                    # Rings hold every message of the run, so a push never
                    # waits on a full queue and proxies cannot throttle
                    # producers. LLVM RPC reuses its 4 KiB ports instead.
                    need=$(( (w / eff) * (iters + warm) )); cap=1
                    while (( cap < need )); do cap=$(( cap * 2 )); done
                    [[ $q == llvm-rpc ]] && cap=64
                    for pat in "" --burst; do
                        run -q "$q" -m push-cost -w "$w" --queues "$nq" -t "$t" \
                            -n "$iters" --warmup "$warm" -c "$cap" $pat
                    done
                done
            done
        done
    done
    run -m floor -n 10000
}

throughput_profile() {
    local reps=$1 waves=$2 threads=$3 tput_n=$4
    for rep in $(seq "$reps"); do
        for q in $QUEUES; do
            run -q "$q" -m idle-poll --queues 64 --idle-rounds 20000
            for w in $waves; do
                for t in $threads; do
                    (( t > w )) && continue
                    for p in 0 100 1000; do
                        run -q "$q" -m throughput -w "$w" -t "$t" -n "$tput_n" --idle-pause "$p"
                        [[ $q == kiwi-spsc || $q == kiwi-invoke ]] && continue
                        d=$(default_queues "$q" "$w")
                        (( d == w || d < t )) && continue
                        run -q "$q" -m throughput -w "$w" --queues "$d" -t "$t" -n "$tput_n" \
                            --idle-pause "$p"
                    done
                done
            done
        done
        for w in $waves; do run -q rocm-hostcall -m throughput -w "$w" -n 2000; done
    done
}

case $PROFILE in
    quick)
        latency_profile 20000 1
        for q in raw-store $QUEUES; do
            c=32768; [[ $q == llvm-rpc ]] && c=4096  # rpc.h caps ports at 16384
            run -q "$q" -m push-cost -n 20000 -c "$c"
        done
        throughput_profile 1 "1 64" "1 4" 20000 ;;
    latency) latency_profile 20000 "${3:-10}" ;;
    push) push_profile "$REPS" ;;
    throughput) throughput_profile "$REPS" "1 8 64 256" "1 4 8" 20000 ;;
    paper)
        latency_profile 20000 10
        push_profile "$REPS"
        throughput_profile "$REPS" "1 8 64 256" "1 4 8" 20000 ;;
    *) echo "unknown profile $PROFILE" >&2; exit 2 ;;
esac
