#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Run a reproducible KIWI EP graph benchmark matrix.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/ep_bench/common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/ep_bench/run_kiwi_ep_benchmark.sh [options]

Options:
  --build-dir DIR     Build directory containing kiwi_ep_ext (default: build/ep-bench)
  --output-dir DIR    Log/provenance directory (default: ep-bench-logs/kiwi-graph-<timestamp>)
  --dry-run           Print commands without executing
  -h, --help          Show this help

Matrix environment knobs:
  EP_WORLD_SIZES      Space-separated MPI ranks (default: "8")
  MAX_TOKENS_LIST     Space-separated max tokens per rank (default: "1 8 64")
  HIDDEN_DIMS         Space-separated hidden dims (default: "7168")
  DISPATCH_DTYPES     Space-separated dispatch dtypes (default: "bf16")
  COMBINE_DTYPES      Space-separated combine dtypes (default: "bf16")
  NUM_EXPERTS_TOTAL   Total MORI-style experts used to derive per-rank default (default: 256)
  NUM_EXPERTS_PER_RANK Override experts per rank for every world size
  NUM_EXPERTS_PER_TOKEN, WARMUP, ITERS, GRAPH_REPLAY_ITERS

Runtime knobs:
  NUM_PROXY_THREADS (default: 4), RCOMP_BACKEND (default: auto, using the
  topology-aware threshold: hybrid at 8+ tokens for ws8 and at 4+ for the
  ws16-derived fallback),
  KIWI_REPO, MPIRUN, PYTHON, MASTER_ADDR, MASTER_PORT_BASE, ALLOW_RUN_AS_ROOT,
  EXTRA_BENCH_ARGS, LD_LIBRARY_PATH, PYTHONPATH.
USAGE
}

BUILD_DIR=${BUILD_DIR:-}
DRY_RUN=${DRY_RUN:-0}
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --build-dir) shift; BUILD_DIR=${1:?missing --build-dir value} ;;
    --output-dir) shift; OUTPUT_DIR=${1:?missing --output-dir value} ;;
    --dry-run) DRY_RUN=1 ;;
    *) ep_bench_die "unknown argument: $1" ;;
  esac
  shift
done

repo=$(ep_bench_repo_root)
BUILD_DIR=${BUILD_DIR:-$repo/build/ep-bench}
out=$(ep_bench_output_dir kiwi-graph-$(_ep_bench_timestamp))
PYTHON=${PYTHON:-python3}
MPIRUN=${MPIRUN:-mpirun}
# Allow MPIRUN to include stable launcher flags such as
# "mpirun --map-by ppr:4:socket --bind-to socket --rank-by slot".
# This mirrors the other wrapper knobs that intentionally use simple
# whitespace splitting for site-local launch configuration.
# shellcheck disable=SC2206
mpirun_cmd=($MPIRUN)
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29540}
NUM_EXPERTS_TOTAL=${NUM_EXPERTS_TOTAL:-256}
NUM_EXPERTS_PER_TOKEN=${NUM_EXPERTS_PER_TOKEN:-8}
WARMUP=${WARMUP:-1}
ITERS=${ITERS:-10}
GRAPH_REPLAY_ITERS=${GRAPH_REPLAY_ITERS:-10}
NUM_PROXY_THREADS=${NUM_PROXY_THREADS:-4}
RCOMP_BACKEND=${RCOMP_BACKEND:-auto}

ep_bench_split_words "${EP_WORLD_SIZES:-8}" world_sizes
ep_bench_split_words "${MAX_TOKENS_LIST:-1 8 64}" token_list
ep_bench_split_words "${HIDDEN_DIMS:-7168}" hidden_dims
ep_bench_split_words "${DISPATCH_DTYPES:-bf16}" dispatch_dtypes
ep_bench_split_words "${COMBINE_DTYPES:-bf16}" combine_dtypes

bench_py=$repo/apps/ep/benchmarks/python/bench_kiwi_ep.py
kiwi_ext_dir=${KIWI_EXT_DIR:-$BUILD_DIR/apps/ep/python}
bench_pythonpath=$(ep_bench_make_pythonpath "$kiwi_ext_dir")

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -f "$bench_py" ]] || ep_bench_die "missing benchmark script: $bench_py"
  [[ -d "$kiwi_ext_dir" ]] || ep_bench_die "missing kiwi_ep_ext directory: $kiwi_ext_dir (run run_kiwi_ep_smoke_correctness.sh first or set KIWI_EXT_DIR)"
fi

"$SCRIPT_DIR/collect_env.sh" --output-dir "$out/provenance" --label kiwi-graph >/dev/null || true

port=$MASTER_PORT_BASE
summary=$out/matrix.tsv
printf 'world_size\tmax_tokens\tmodel_hidden_dim\tdispatch_hidden_elems\tcombine_hidden_elems\tdispatch_dtype\tcombine_dtype\tdispatch_scale_dim\tdispatch_scale_dtype\tcombine_scale_dim\tcombine_scale_dtype\tlog\n' >"$summary"

bench_hidden_elems_for_dtype() {
  local dtype=$1
  local model_hidden_dim=$2
  case "$dtype" in
    fp4)
      (( model_hidden_dim % 2 == 0 )) || ep_bench_die "FP4 requires even model hidden dim (got $model_hidden_dim)"
      printf '%s\n' "$((model_hidden_dim / 2))"
      ;;
    *)
      printf '%s\n' "$model_hidden_dim"
      ;;
  esac
}

bench_scale_dim_for_dtype() {
  local dtype=$1
  local model_hidden_dim=$2
  case "$dtype" in
    fp8|fp8_e4m3_fnuz|fp8_e4m3)
      (( model_hidden_dim % 128 == 0 )) || ep_bench_die "FP8 block scales require hidden dim divisible by 128 (got $model_hidden_dim)"
      printf '%s\n' "$((model_hidden_dim / 128))"
      ;;
    fp4)
      printf '4\n'
      ;;
    *)
      printf '0\n'
      ;;
  esac
}

bench_scale_dtype_for_dtype() {
  local dtype=$1
  case "$dtype" in
    fp8|fp8_e4m3_fnuz|fp8_e4m3) printf 'float32\n' ;;
    fp4) printf 'e8m0\n' ;;
    *) printf 'none\n' ;;
  esac
}

for ws in "${world_sizes[@]}"; do
  if [[ -n "${NUM_EXPERTS_PER_RANK:-}" ]]; then
    experts_per_rank=$NUM_EXPERTS_PER_RANK
  else
    (( NUM_EXPERTS_TOTAL % ws == 0 )) || ep_bench_die "NUM_EXPERTS_TOTAL=$NUM_EXPERTS_TOTAL is not divisible by world size $ws"
    experts_per_rank=$((NUM_EXPERTS_TOTAL / ws))
  fi
  for tokens in "${token_list[@]}"; do
    for hidden in "${hidden_dims[@]}"; do
      for dispatch_dtype in "${dispatch_dtypes[@]}"; do
        for combine_dtype in "${combine_dtypes[@]}"; do
          dispatch_hidden_elems=$(bench_hidden_elems_for_dtype "$dispatch_dtype" "$hidden")
          combine_hidden_elems=$(bench_hidden_elems_for_dtype "$combine_dtype" "$hidden")
          dispatch_scale_dim=$(bench_scale_dim_for_dtype "$dispatch_dtype" "$hidden")
          dispatch_scale_dtype=$(bench_scale_dtype_for_dtype "$dispatch_dtype")
          combine_scale_dim=$(bench_scale_dim_for_dtype "$combine_dtype" "$hidden")
          combine_scale_dtype=$(bench_scale_dtype_for_dtype "$combine_dtype")
          log=$out/kiwi_ws${ws}_tok${tokens}_h${hidden}_dispatch-${dispatch_dtype}_combine-${combine_dtype}.log
          cmd=(
            env
            MASTER_ADDR="$MASTER_ADDR"
            MASTER_PORT="$port"
            PYTHONPATH="$bench_pythonpath"
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
          )
          cmd+=(
            "${mpirun_cmd[@]}"
          )
          mapfile -d '' -t root_args < <(ep_bench_mpirun_root_args)
          cmd+=("${root_args[@]}" -np "$ws")
          mapfile -d '' -t exports < <(ep_bench_mpirun_env_exports)
          cmd+=("${exports[@]}")
          cmd+=(
            "$PYTHON" "$bench_py"
            --cmd bench
            --timing graph
            --dispatch-dtype "$dispatch_dtype"
            --combine-dtype "$combine_dtype"
            --dispatch-scale-dim "$dispatch_scale_dim"
            --dispatch-scale-dtype "$dispatch_scale_dtype"
            --combine-scale-dim "$combine_scale_dim"
            --combine-scale-dtype "$combine_scale_dtype"
            --num-proxy-threads "$NUM_PROXY_THREADS"
            --rcomp-backend "$RCOMP_BACKEND"
            --max-tokens "$tokens"
            --hidden-dim "$hidden"
            --world-size "$ws"
            --num-experts-per-rank "$experts_per_rank"
            --num-experts-per-token "$NUM_EXPERTS_PER_TOKEN"
            --warmup "$WARMUP"
            --iters "$ITERS"
            --graph-replay-iters "$GRAPH_REPLAY_ITERS"
            --torch-dist auto
          )
          if [[ -n "${EXTRA_BENCH_ARGS:-}" ]]; then
            # shellcheck disable=SC2206
            extra_bench=($EXTRA_BENCH_ARGS)
            cmd+=("${extra_bench[@]}")
          fi
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$ws" "$tokens" "$hidden" "$dispatch_hidden_elems" "$combine_hidden_elems" \
            "$dispatch_dtype" "$combine_dtype" \
            "$dispatch_scale_dim" "$dispatch_scale_dtype" \
            "$combine_scale_dim" "$combine_scale_dtype" "$log" >>"$summary"
          if [[ "${DRY_RUN:-0}" == "1" ]]; then
            ep_bench_print_cmd "${cmd[@]}"
          else
            ep_bench_print_cmd "${cmd[@]}" | tee "$log"
            "${cmd[@]}" 2>&1 | tee -a "$log"
          fi
          port=$((port + 1))
        done
      done
    done
  done
done

printf 'Wrote KIWI EP benchmark logs under %s\n' "$out"
