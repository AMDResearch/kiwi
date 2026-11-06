#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Canonical Docker runtime wrapper for KIWI EP/MORI/IntelliKit runs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/ep_bench/common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/ep_bench/run_container.sh [options] [-- command ...]

Options:
  --image IMAGE        Docker image tag or digest (default: IMAGE env or kiwi-ep-bench:rocm7.2.4-pytorch2.10)
  --kiwi DIR          Host KIWI repository root to mount
  --mori DIR          Optional official MORI checkout to mount at /mori
  --intellikit DIR    Optional IntelliKit override to mount at /intellikit
  --output-dir DIR    Host output directory to mount at /outputs
  --dry-run           Print docker command without executing
  -h, --help          Show this help

Canonical runtime contract:
  * --network host, --ipc host, configurable --shm-size
  * /dev/kfd, /dev/dri, and /dev/infiniband device pass-through when present
  * /sys/class/infiniband, /sys/bus/pci, and /sys/class/net read-only for NIC selection
  * memlock unlimited, IPC_LOCK, SYS_PTRACE, seccomp=unconfined

Environment knobs:
  IMAGE, DOCKER, DOCKER_SHM_SIZE, PASS_INFINIBAND, EXTRA_DOCKER_ARGS,
  EP_BENCH_IMAGE_DIGEST,
  CONTAINER_KIWI_DIR, CONTAINER_OUTPUT_DIR.
  The canonical image already includes upstream IntelliKit under /opt/intellikit;
  --intellikit is only for testing a different local checkout/install.
USAGE
}

IMAGE=${IMAGE:-kiwi-ep-bench:rocm7.2.4-pytorch2.10}
DOCKER=${DOCKER:-docker}
DOCKER_SHM_SIZE=${DOCKER_SHM_SIZE:-64g}
PASS_INFINIBAND=${PASS_INFINIBAND:-1}
CONTAINER_KIWI_DIR=${CONTAINER_KIWI_DIR:-/workspace/kiwi}
CONTAINER_OUTPUT_DIR=${CONTAINER_OUTPUT_DIR:-/outputs}
CONTAINER_MORI_DIR=${CONTAINER_MORI_DIR:-/mori}
CONTAINER_INTELLIKIT_DIR=${CONTAINER_INTELLIKIT_DIR:-/intellikit}
cmd=(bash)

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --image) shift; IMAGE=${1:?missing --image value} ;;
    --kiwi) shift; KIWI_REPO=${1:?missing --kiwi value} ;;
    --mori) shift; MORI_SOURCE_DIR=${1:?missing --mori value} ;;
    --intellikit) shift; INTELLIKIT_ROOT=${1:?missing --intellikit value} ;;
    --output-dir) shift; OUTPUT_DIR=${1:?missing --output-dir value} ;;
    --dry-run) DRY_RUN=1 ;;
    --) shift; cmd=("$@"); break ;;
    *) ep_bench_die "unknown argument before --: $1" ;;
  esac
  shift
done

repo=$(ep_bench_repo_root)
out=$(ep_bench_output_dir container-$(_ep_bench_timestamp))

args=(
  "$DOCKER" run --rm
  --network host
  --ipc host
  --shm-size "$DOCKER_SHM_SIZE"
  --device /dev/kfd
  --device /dev/dri
  --group-add video
  --group-add render
  --cap-add SYS_PTRACE
  --cap-add IPC_LOCK
  --security-opt seccomp=unconfined
  --ulimit memlock=-1:-1
  --ulimit stack=67108864:67108864
  -v "$repo:$CONTAINER_KIWI_DIR:rw"
  -v "$out:$CONTAINER_OUTPUT_DIR:rw"
  -v /sys/class/infiniband:/sys/class/infiniband:ro
  -v /sys/bus/pci:/sys/bus/pci:ro
  -v /sys/class/net:/sys/class/net:ro
  -w "$CONTAINER_KIWI_DIR"
  -e "KIWI_REPO=$CONTAINER_KIWI_DIR"
  -e "EP_BENCH_IMAGE=$IMAGE"
  -e "EP_BENCH_IMAGE_DIGEST=${EP_BENCH_IMAGE_DIGEST:-}"
  -e "OUTPUT_DIR=$CONTAINER_OUTPUT_DIR"
  -e MPI_ALLOW_RUN_AS_ROOT=1
  -e MPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -e OMPI_ALLOW_RUN_AS_ROOT=1
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
)



if [[ -n "${MORI_SOURCE_DIR:-}" ]]; then
  [[ -d "$MORI_SOURCE_DIR" || "${DRY_RUN:-0}" == "1" ]] || ep_bench_die "MORI_SOURCE_DIR not found: $MORI_SOURCE_DIR"
  args+=(-v "$MORI_SOURCE_DIR:$CONTAINER_MORI_DIR:ro" -e "MORI_SOURCE_DIR=$CONTAINER_MORI_DIR")
fi
if [[ -n "${INTELLIKIT_ROOT:-}" ]]; then
  [[ -d "$INTELLIKIT_ROOT" || "${DRY_RUN:-0}" == "1" ]] || ep_bench_die "INTELLIKIT_ROOT not found: $INTELLIKIT_ROOT"
  args+=(-v "$INTELLIKIT_ROOT:$CONTAINER_INTELLIKIT_DIR:ro" -e "INTELLIKIT_ROOT=$CONTAINER_INTELLIKIT_DIR")
fi

if [[ "$PASS_INFINIBAND" == "1" ]]; then
  if [[ -d /dev/infiniband ]]; then
    shopt -s nullglob
    for dev in /dev/infiniband/*; do
      if [[ -c "$dev" || -b "$dev" ]]; then
        args+=(--device "$dev")
      else
        ep_bench_note "Skipping non-device InfiniBand path: $dev"
      fi
    done
    shopt -u nullglob
  else
    ep_bench_note "/dev/infiniband not found; running without RDMA devices"
  fi
  if [[ -d /usr/include/infiniband ]]; then
    args+=(-v /usr/include/infiniband:/usr/include/infiniband:ro)
  fi
fi

if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
  # Intentionally simple whitespace splitting for site-local additions.
  # shellcheck disable=SC2206
  extra=($EXTRA_DOCKER_ARGS)
  args+=("${extra[@]}")
fi

args+=("$IMAGE" "${cmd[@]}")
ep_bench_run "${args[@]}"
