#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Build KIWI EP and run the full EP correctness gate.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/ep_bench/common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/ep_bench/run_kiwi_ep_full_correctness.sh [options]

Options:
  --build-dir DIR     Build directory (default: BUILD_DIR or build/ep-bench)
  --dry-run           Print commands without executing
  --skip-build        Reuse an existing build directory
  -h, --help          Show this help

Environment knobs:
  KIWI_REPO, BUILD_DIR, CMAKE_BUILD_TYPE, HIP_ARCHITECTURES, CMAKE_GENERATOR,
  CMAKE_EXTRA_ARGS, BUILD_JOBS, RUN_TESTS, CTEST_EXTRA_ARGS,
  MORI_CTEST_EXTRA_ARGS, ALLOW_RUN_AS_ROOT.

This is the full EP correctness gate. It runs:
  1. the focused EP smoke gate from run_kiwi_ep_smoke_correctness.sh, and
  2. the official MORI compatibility CTest matrix: ^test_official_mori_compat_

Use this for EP dispatch/combine/barrier/counter/epoch/graph-replay changes
unless the user explicitly asks for a narrower check. If MORI is unavailable,
the official compatibility tests should skip with return code 77; semantic
mismatches remain hard failures.

USAGE
}

BUILD_DIR=${BUILD_DIR:-}
DRY_RUN=${DRY_RUN:-0}
SKIP_BUILD=${SKIP_BUILD:-0}

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --build-dir) shift; BUILD_DIR=${1:?missing --build-dir value} ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    *) ep_bench_die "unknown argument: $1" ;;
  esac
  shift
done

repo=$(ep_bench_repo_root)
BUILD_DIR=${BUILD_DIR:-$repo/build/ep-bench}

ep_bench_note "Running focused EP smoke correctness first"
focused_cmd=(
  "$SCRIPT_DIR/run_kiwi_ep_smoke_correctness.sh"
  --build-dir "$BUILD_DIR"
)
if [[ "$DRY_RUN" == "1" ]]; then
  focused_cmd+=(--dry-run)
fi
if [[ "$SKIP_BUILD" == "1" ]]; then
  focused_cmd+=(--skip-build)
fi
ep_bench_run "${focused_cmd[@]}"

if [[ "${RUN_TESTS:-1}" != "1" ]]; then
  ep_bench_note "RUN_TESTS=$RUN_TESTS; focused build completed without running tests"
  exit 0
fi

ep_bench_note "CTest inventory before official MORI compatibility matrix"
ep_bench_run ctest --test-dir "$BUILD_DIR" -N

ctest_cmd=(
  ctest --test-dir "$BUILD_DIR" --output-on-failure
  -R '^test_official_mori_compat_'
)
if [[ -n "${MORI_CTEST_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_mori_ctest=($MORI_CTEST_EXTRA_ARGS)
  ctest_cmd+=("${extra_mori_ctest[@]}")
elif [[ -n "${CTEST_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_ctest=($CTEST_EXTRA_ARGS)
  ctest_cmd+=("${extra_ctest[@]}")
fi

ep_bench_note "Running official MORI compatibility matrix"
ep_bench_run "${ctest_cmd[@]}"
