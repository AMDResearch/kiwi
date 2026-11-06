#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Run official MORI EP compatibility/reference tests when MORI dependencies exist.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/ep_bench/common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/ep_bench/run_mori_ep_compat.sh [options]

Options:
  --build-dir DIR      Build directory containing kiwi_ep_ext (default: build/ep-bench)
  --mori DIR           Official MORI source checkout (or set MORI_SOURCE_DIR)
  --profile NAME       Downstream sync profile; may repeat. Known profiles:
                         vllm-fp8-token-sync, vllm-fp8-block-sync,
                         sglang-fp8-block-sync
  --world-size N       MPI ranks (default: WORLD_SIZE or 2; profiles default to 8)
  --kernel-type NAME   MORI kernel type for test_official_mori_compat.py (default: InterNodeV1)
  --api-mode MODE      sync or split (default: sync)
  --dispatch-dtype NAME
                       Dispatch payload dtype: float32, bfloat16, fp8, fp4 (default: float32)
  --num-tokens N       Forwarded test token count
  --num-tokens-list S  Forwarded test token-count sweep
  --max-num-inp-token-per-rank N
                       Forwarded MORI max input tokens per rank
  --max-token-type-size N
                       Forwarded MORI max token type size for combine payloads
  --hidden-dim N       Forwarded logical hidden dim
  --topk N             Forwarded experts per token
  --num-experts-per-rank N
                       Forwarded experts per rank
  --scale-dim N        Forwarded dispatch scale dim
  --scale-dtype NAME   Forwarded dispatch scale dtype
  --downstream-sync-only
                       Exercise only dispatch(...) + combine(..., None, topk_ids)
  --test-arg ARG       Forward one raw argument to test_official_mori_compat.py
  --dry-run            Print command without executing
  -h, --help           Show this help

If official MORI is not configured/importable, this wrapper prints SKIP and
exits 77. A missing KIWI extension is treated as configuration error because the
compatibility test compares MORI against the built KIWI Python binding.
USAGE
}

BUILD_DIR=${BUILD_DIR:-}
WORLD_SIZE=${WORLD_SIZE:-2}
KERNEL_TYPE=${KERNEL_TYPE:-InterNodeV1}
API_MODE=${API_MODE:-sync}
DISPATCH_DTYPE=${DISPATCH_DTYPE:-float32}
GPU_PER_NODE=${GPU_PER_NODE:-}
WORLD_SIZE_EXPLICIT=0
KERNEL_TYPE_EXPLICIT=0
API_MODE_EXPLICIT=0
DISPATCH_DTYPE_EXPLICIT=0
PROFILES=()
TEST_ARGS=()
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29570}
MPIRUN=${MPIRUN:-mpirun}
PYTHON=${PYTHON:-python3}
DRY_RUN=${DRY_RUN:-0}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --build-dir) shift; BUILD_DIR=${1:?missing --build-dir value} ;;
    --mori) shift; MORI_SOURCE_DIR=${1:?missing --mori value} ;;
    --profile) shift; PROFILES+=("${1:?missing --profile value}") ;;
    --world-size) shift; WORLD_SIZE=${1:?missing --world-size value}; WORLD_SIZE_EXPLICIT=1 ;;
    --kernel-type) shift; KERNEL_TYPE=${1:?missing --kernel-type value}; KERNEL_TYPE_EXPLICIT=1 ;;
    --api-mode) shift; API_MODE=${1:?missing --api-mode value}; API_MODE_EXPLICIT=1 ;;
    --dispatch-dtype) shift; DISPATCH_DTYPE=${1:?missing --dispatch-dtype value}; DISPATCH_DTYPE_EXPLICIT=1 ;;
    --num-tokens|--num-tokens-list|--max-num-inp-token-per-rank|--max-token-type-size|--hidden-dim|--topk|--num-experts-per-rank|--scale-dim|--scale-dtype)
      opt=$1; shift; TEST_ARGS+=("$opt" "${1:?missing $opt value}") ;;
    --no-combine-output-weights|--no-vllm-smoke|--downstream-sync-only|--require-multi-source-bf16|--stop-on-first-token-error)
      TEST_ARGS+=("$1") ;;
    --test-arg) shift; TEST_ARGS+=("${1:?missing --test-arg value}") ;;
    --dry-run) DRY_RUN=1 ;;
    *) ep_bench_die "unknown argument: $1" ;;
  esac
  shift
done

repo=$(ep_bench_repo_root)
BUILD_DIR=${BUILD_DIR:-$repo/build/ep-bench}
kiwi_ext_dir=${KIWI_EXT_DIR:-$BUILD_DIR/apps/ep/python}
test_py=$repo/apps/ep/tests/test_official_mori_compat.py

if [[ -z "${MORI_SOURCE_DIR:-}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "SKIP official MORI compatibility: set MORI_SOURCE_DIR or pass --mori" >&2
  exit 77
fi
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -d "$kiwi_ext_dir" ]] || ep_bench_die "missing kiwi_ep_ext directory: $kiwi_ext_dir"
  [[ -f "$test_py" ]] || ep_bench_die "missing test script: $test_py"
  [[ -d "${MORI_SOURCE_DIR:-}" ]] || { echo "SKIP official MORI compatibility: MORI_SOURCE_DIR not found: ${MORI_SOURCE_DIR:-}" >&2; exit 77; }
fi

mori_pythonpath=${MORI_SOURCE_DIR:+$MORI_SOURCE_DIR/python:$MORI_SOURCE_DIR}
pythonpath=$kiwi_ext_dir
if [[ -n "$mori_pythonpath" ]]; then
  pythonpath=$pythonpath:$mori_pythonpath
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  pythonpath=$pythonpath:$PYTHONPATH
fi

profile_expand() {
  local profile=$1
  PROFILE_ARGS=()
  PROFILE_WORLD_SIZE=$WORLD_SIZE
  PROFILE_KERNEL_TYPE=$KERNEL_TYPE
  PROFILE_API_MODE=$API_MODE
  PROFILE_DISPATCH_DTYPE=$DISPATCH_DTYPE

  case "$profile" in
    __default)
      ;;
    vllm-fp8-token-sync)
      [[ $WORLD_SIZE_EXPLICIT == 1 ]] || PROFILE_WORLD_SIZE=8
      [[ $API_MODE_EXPLICIT == 1 ]] || PROFILE_API_MODE=sync
      [[ $DISPATCH_DTYPE_EXPLICIT == 1 ]] || PROFILE_DISPATCH_DTYPE=fp8
      PROFILE_ARGS+=(
        --hidden-dim 7168 --topk 8 --num-experts-per-rank 32
        --max-token-type-size 2 --scale-dim 1 --scale-dtype float32
        --downstream-sync-only
      )
      ;;
    vllm-fp8-block-sync|sglang-fp8-block-sync)
      [[ $WORLD_SIZE_EXPLICIT == 1 ]] || PROFILE_WORLD_SIZE=8
      [[ $API_MODE_EXPLICIT == 1 ]] || PROFILE_API_MODE=sync
      [[ $DISPATCH_DTYPE_EXPLICIT == 1 ]] || PROFILE_DISPATCH_DTYPE=fp8
      PROFILE_ARGS+=(
        --hidden-dim 7168 --topk 8 --num-experts-per-rank 32
        --max-token-type-size 2 --scale-dim 56 --scale-dtype float32
        --downstream-sync-only
      )
      ;;
    *)
      ep_bench_die "unknown --profile: $profile"
      ;;
  esac
}

run_one_profile() {
  local profile=$1
  profile_expand "$profile"

  mori_pythonpath=${MORI_SOURCE_DIR:+$MORI_SOURCE_DIR/python:$MORI_SOURCE_DIR}
  pythonpath=$kiwi_ext_dir
  if [[ -n "$mori_pythonpath" ]]; then
    pythonpath=$pythonpath:$mori_pythonpath
  fi
  if [[ -n "${PYTHONPATH:-}" ]]; then
    pythonpath=$pythonpath:$PYTHONPATH
  fi

  if [[ "$profile" != "__default" ]]; then
    printf '# MORI compat profile %q expands to: world_size=%q kernel_type=%q api_mode=%q dispatch_dtype=%q test_args=' \
      "$profile" "$PROFILE_WORLD_SIZE" "$PROFILE_KERNEL_TYPE" "$PROFILE_API_MODE" "$PROFILE_DISPATCH_DTYPE"
    printf '%q ' "${PROFILE_ARGS[@]}" "${TEST_ARGS[@]}"
    printf '\n'
    printf '# MORI compat profile %q primary path: dispatch(tokens, weights, scales, topk_ids) -> combine(fused_expert_output, None, topk_ids)\n' "$profile"
  fi

  cmd=(
    env
    MASTER_ADDR="$MASTER_ADDR"
    MASTER_PORT="$MASTER_PORT"
    PYTHONPATH="$pythonpath"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    MORI_SHMEM_MODE="${MORI_SHMEM_MODE:-vmm_heap}"
    MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-512M}"
    MORI_SHMEM_VMM_CHUNK_SIZE="${MORI_SHMEM_VMM_CHUNK_SIZE:-64M}"
  )
  cmd+=(
    "$MPIRUN"
  )
  mapfile -d '' -t root_args < <(ep_bench_mpirun_root_args)
  cmd+=("${root_args[@]}" -np "$PROFILE_WORLD_SIZE")
  mapfile -d '' -t exports < <(ep_bench_mpirun_env_exports)
  cmd+=("${exports[@]}")
  cmd+=(
    "$PYTHON" "$test_py"
    --kernel-type "$PROFILE_KERNEL_TYPE"
    --api-mode "$PROFILE_API_MODE"
    --dispatch-dtype "$PROFILE_DISPATCH_DTYPE"
  )
  cmd+=("${PROFILE_ARGS[@]}" "${TEST_ARGS[@]}")
  if [[ -n "$GPU_PER_NODE" ]]; then
    cmd+=(--gpu-per-node "$GPU_PER_NODE")
  fi
  if [[ -n "${EXTRA_MORI_COMPAT_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_mori=($EXTRA_MORI_COMPAT_ARGS)
    cmd+=("${extra_mori[@]}")
  fi

  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    env PYTHONPATH="$pythonpath" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" "$PYTHON" - <<'PYPREFLIGHT' || { echo "SKIP official MORI compatibility: torch/mori import preflight failed" >&2; exit 77; }
import importlib.util
missing = [m for m in ("torch", "mori") if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PYPREFLIGHT
  fi

  ep_bench_run "${cmd[@]}"
}

if ((${#PROFILES[@]} == 0)); then
  PROFILES=(__default)
fi
for profile in "${PROFILES[@]}"; do
  run_one_profile "$profile"
done
exit 0
