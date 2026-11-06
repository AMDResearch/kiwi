#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# SLURM wrapper for run_sweep.sh on a single exclusive MI300X node.
#
# Submit from the repository root on the login node, for example:
#   sbatch --account=<account> --partition=<partition> --nodes=1 --gres=gpu:8 \
#     --exclusive --time=06:00:00 --output=qb-%j.out \
#     benchmarks/queue_baselines/slurm_sweep.sh paper 3
#
# Builds in node-local scratch (QB_SCRATCH, default ${TMPDIR:-/tmp}/$USER; set it
# to the site's node-local scratch), runs on GPU 0 pinned to CPUs 8-23 (QB_GPU,
# QB_CPUS; choose CPUs on the GPU's NUMA node), and writes
# results/<job>/ under the submission directory.
set -euo pipefail

PROFILE=${1:-paper}
REPS=${2:-3}
SRC=${SLURM_SUBMIT_DIR:-$PWD}
JOB=${SLURM_JOB_ID:-manual-$(date +%s)}
WORK=${QB_SCRATCH:-${TMPDIR:-/tmp}/$USER}/qb-$JOB
OUT=$SRC/benchmarks/queue_baselines/results/$JOB
GPU=${QB_GPU:-0}
CPUS=${QB_CPUS:-8-23}

mkdir -p "$WORK" "$OUT"
rsync -a --exclude build --exclude .git --exclude results "$SRC/" "$WORK/src/"

{
    echo "job=$JOB host=$(hostname) date=$(date -Is)"
    echo "profile=$PROFILE reps=$REPS gpu=$GPU cpus=$CPUS"
    echo "commit=${QB_COMMIT:-$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)}"
    echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm-*)"
    echo "kernel=$(uname -r)"
    lscpu | grep -E 'Model name|NUMA node[0-9]'
    rocm-smi --showproductname --showbus 2>/dev/null | grep -E 'GPU\[' | head -16
} > "$OUT/env.txt"

cmake -S "$WORK/src" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release \
    -DHIP_ARCHITECTURES=gfx942 -DKIWI_BUILD_BENCHMARKS=ON \
    -DKIWI_BUILD_QUEUE_BASELINES=ON > "$OUT/configure.log" 2>&1
cmake --build "$WORK/build" -j 32 --target bench_queue_baselines > "$OUT/build.log" 2>&1

BIN=$WORK/build/benchmarks/queue_baselines/bench_queue_baselines
HIP_VISIBLE_DEVICES=$GPU taskset -c "$CPUS" \
    "$WORK/src/benchmarks/queue_baselines/run_sweep.sh" "$BIN" "$PROFILE" "$REPS" \
    > "$OUT/results.txt" 2> "$OUT/sweep.err"

echo "runs=$(grep -c '^RESULT' "$OUT/results.txt") invalid=$(grep -vc 'valid=1' "$OUT/results.txt")" \
    | tee "$OUT/summary.txt"
rm -rf "$WORK"
