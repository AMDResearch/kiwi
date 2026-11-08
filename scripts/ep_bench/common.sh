#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Shared helpers for KIWI EP benchmark/profiling wrapper scripts.
# shellcheck shell=bash

if [[ -n "${KIWI_EP_BENCH_COMMON_SH:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
readonly KIWI_EP_BENCH_COMMON_SH=1

_ep_bench_common_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly EP_BENCH_SCRIPT_DIR="$_ep_bench_common_dir"
readonly EP_BENCH_REPO_ROOT_DEFAULT=$(cd -- "$EP_BENCH_SCRIPT_DIR/../.." && pwd)

_ep_bench_timestamp() {
  date -u +%Y%m%dT%H%M%SZ
}

ep_bench_die() {
  echo "ERROR: $*" >&2
  exit 2
}

ep_bench_note() {
  echo "Note: $*" >&2
}

ep_bench_repo_root() {
  local root=${KIWI_REPO:-$EP_BENCH_REPO_ROOT_DEFAULT}
  cd -- "$root" >/dev/null 2>&1 || ep_bench_die "KIWI_REPO does not exist: $root"
  pwd
}

ep_bench_output_dir() {
  local default_name=${1:-ep-bench-$(_ep_bench_timestamp)}
  local root
  root=$(ep_bench_repo_root)
  local out=${OUTPUT_DIR:-$root/ep-bench-logs/$default_name}
  mkdir -p -- "$out"
  cd -- "$out" >/dev/null 2>&1 || ep_bench_die "cannot enter OUTPUT_DIR: $out"
  pwd
}

ep_bench_print_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

ep_bench_run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ep_bench_print_cmd "$@"
  else
    ep_bench_print_cmd "$@"
    "$@"
  fi
}

ep_bench_bool() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    0|false|FALSE|no|NO|off|OFF|"") return 1 ;;
    *) ep_bench_die "invalid boolean value: $1" ;;
  esac
}

ep_bench_split_words() {
  # Split simple whitespace-separated env vars such as "1 8 64" into words.
  # Quoting inside the variable is intentionally not interpreted.
  local value=${1:-}
  local -n _out=$2
  # shellcheck disable=SC2206
  _out=($value)
}

ep_bench_make_pythonpath() {
  local first=$1
  if [[ -n "${PYTHONPATH:-}" ]]; then
    printf '%s:%s\n' "$first" "$PYTHONPATH"
  else
    printf '%s\n' "$first"
  fi
}

# Default launcher for the EP wrappers. Leave ranks unbound: KIWI binds each
# rank's proxy threads to its GPU's local CPUs within the rank's affinity mask,
# and Open MPI's default socket round-robin can bind a rank away from its GPU.
# Callers may override MPIRUN with site-specific mapping/binding flags.
EP_BENCH_DEFAULT_MPIRUN="mpirun --bind-to none"

ep_bench_mpirun_root_args() {
  if [[ "${ALLOW_RUN_AS_ROOT:-0}" == "1" ]]; then
    printf '%s\0' --allow-run-as-root
  fi
}

ep_bench_mpirun_env_exports() {
  # Open MPI style -x exports. Unknown MPI implementations usually ignore none
  # of these because the caller selects MPIRUN; override wrapper scripts if needed.
  # Export only variables that are actually set so wrappers do not create or warn
  # about optional ROCm/LCI tuning variables.
  local names=(
    MASTER_ADDR MASTER_PORT PYTHONPATH LD_LIBRARY_PATH PATH
    HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES LOCAL_WORLD_SIZE
    KIWI_LCI_DEVICE_NAME LCI_ATTR_DEVICE_NAME LCI_ATTR_USE_DMABUF
  )
  local name
  for name in "${names[@]}"; do
    if [[ -v $name ]]; then
      printf '%s\0' -x "$name"
    fi
  done
}

ep_bench_have_command() {
  command -v "$1" >/dev/null 2>&1
}
