#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Build KIWI EP and run focused smoke correctness tests.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/ep_bench/common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/ep_bench/run_kiwi_ep_smoke_correctness.sh [options]

This is a focused smoke gate, not the full EP correctness matrix.
Use scripts/ep_bench/run_kiwi_ep_full_correctness.sh for full EP validation.

Options:
  --build-dir DIR     Build directory (default: BUILD_DIR or build/ep-bench)
  --dry-run           Print commands without executing
  --skip-build        Reuse an existing build directory
  --no-python         Do not build/run Python graph correctness target
  -h, --help          Show this help

Environment knobs:
  KIWI_REPO, BUILD_DIR, CMAKE_BUILD_TYPE, HIP_ARCHITECTURES, CMAKE_GENERATOR,
  CMAKE_EXTRA_ARGS, BUILD_JOBS, RUN_TESTS, CTEST_EXTRA_ARGS,
  ALLOW_RUN_AS_ROOT.

Focused smoke tests:
  C++:    test_ep_block_count, test_dispatch, test_combine,
          test_combine_graph_replay, test_combine_graph_replay_auto_policy,
          test_ep_collective_primitives,
          test_dispatch_combine_harness,
          test_rcomp_readiness_layout
  Python: test_kiwi_ep_cudagraph_ws2 (requires torch, pybind11, and kiwi_ep_ext)

This script intentionally does not run test_official_mori_compat_* tests.
Do not report this focused smoke result as full correctness.
USAGE
}

BUILD_DIR=${BUILD_DIR:-}
SKIP_BUILD=${SKIP_BUILD:-0}
BUILD_PYTHON=${BUILD_PYTHON:-1}
RUN_TESTS=${RUN_TESTS:-1}
DRY_RUN=${DRY_RUN:-0}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --build-dir) shift; BUILD_DIR=${1:?missing --build-dir value} ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --no-python) BUILD_PYTHON=0 ;;
    *) ep_bench_die "unknown argument: $1" ;;
  esac
  shift
done

repo=$(ep_bench_repo_root)
BUILD_DIR=${BUILD_DIR:-$repo/build/ep-bench}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
BUILD_JOBS=${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}

cmake_args=(
  -S "$repo"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE"
  -DKIWI_BUILD_EP=ON
  -DKIWI_BUILD_TESTS=ON
  -DKIWI_BUILD_BENCHMARKS=ON
  -DKIWI_BUILD_PYTHON=$([[ "$BUILD_PYTHON" == "1" ]] && echo ON || echo OFF)
)
if [[ -n "${HIP_ARCHITECTURES:-}" ]]; then
  cmake_args+=(-DHIP_ARCHITECTURES="$HIP_ARCHITECTURES")
fi
if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
  cmake_args=(-G "$CMAKE_GENERATOR" "${cmake_args[@]}")
fi
if [[ -n "${CMAKE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_cmake=($CMAKE_EXTRA_ARGS)
  cmake_args+=("${extra_cmake[@]}")
fi

build_targets=(test_ep_block_count test_dispatch test_combine test_combine_graph_replay test_ep_collective_primitives test_dispatch_combine_harness test_rcomp_readiness_layout)
ctest_regex='^(test_ep_block_count|test_dispatch|test_combine|test_combine_graph_replay|test_combine_graph_replay_auto_policy|test_ep_collective_primitives|test_dispatch_combine_harness|test_rcomp_readiness_layout)$'
if [[ "$BUILD_PYTHON" == "1" ]]; then
  build_targets+=(kiwi_ep_ext)
  ctest_regex='^(test_ep_block_count|test_dispatch|test_combine|test_combine_graph_replay|test_combine_graph_replay_auto_policy|test_ep_collective_primitives|test_dispatch_combine_harness|test_rcomp_readiness_layout|test_kiwi_ep_cudagraph_ws2)$'
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  ep_bench_run cmake "${cmake_args[@]}"
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  ep_bench_run cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS" --target "${build_targets[@]}"
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  ctest_cmd=(ctest --test-dir "$BUILD_DIR" --output-on-failure -R "$ctest_regex")
  if [[ -n "${CTEST_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_ctest=($CTEST_EXTRA_ARGS)
    ctest_cmd+=("${extra_ctest[@]}")
  fi
  ep_bench_run "${ctest_cmd[@]}"
else
  ep_bench_note "RUN_TESTS=$RUN_TESTS; build completed without running tests"
fi
