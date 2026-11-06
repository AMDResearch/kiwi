#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Run kiwi-ep beside official MORI EP InterNode and InterNodeV1LL kernels.
# Python/MORI dependencies are expected in the EP benchmark Docker image.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_KIWI_REPO=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)

# EP_WORLD_SIZE is the number of GPU ranks in the benchmark.  WORLD_SIZE is
# accepted as a compatibility alias for older invocations, but MORI's official
# internode example uses WORLD_SIZE differently (node count), so this script
# passes that value explicitly only inside the MORI example command.
EP_WORLD_SIZE=${EP_WORLD_SIZE:-${WORLD_SIZE:-8}}
DISPATCH_DTYPE=${DISPATCH_DTYPE:-bf16}
COMBINE_DTYPE=${COMBINE_DTYPE:-bf16}
MAX_TOKENS=${MAX_TOKENS:-1}
HIDDEN_DIM=${HIDDEN_DIM:-7168}
ITERS=${ITERS:-10}
WARMUP=${WARMUP:-1}
GRAPH_REPLAY_ITERS=${GRAPH_REPLAY_ITERS:-10}
# MORI's official internode example fixes total experts at 256 and derives
# num_experts_per_rank as 256 / world_size.  Keep kiwi's one-shot default
# aligned with that behavior; override only when intentionally comparing a
# different expert layout.
if [[ -z "${NUM_EXPERTS_PER_RANK:-}" ]]; then
  if (( 256 % EP_WORLD_SIZE != 0 )); then
    echo "ERROR: default NUM_EXPERTS_PER_RANK requires EP_WORLD_SIZE to divide 256 (got $EP_WORLD_SIZE)" >&2
    exit 2
  fi
  NUM_EXPERTS_PER_RANK=$((256 / EP_WORLD_SIZE))
fi
NUM_EXPERTS_PER_TOKEN=${NUM_EXPERTS_PER_TOKEN:-8}

NUM_NODES=${NUM_NODES:-1}
MORI_NODE_RANK=${MORI_NODE_RANK:-0}
GPU_PER_NODE=${GPU_PER_NODE:-$EP_WORLD_SIZE}
NUM_QP=${NUM_QP:-1}
QUANT_TYPE=${QUANT_TYPE:-none}
MAX_RECV_TOTAL_TOKENS=${MAX_RECV_TOTAL_TOKENS:-0}
MORI_BLOCK_NUM=${MORI_BLOCK_NUM:-${DISPATCH_BLOCK_NUM:-}}
MORI_WARP_PER_BLOCK=${MORI_WARP_PER_BLOCK:-${DISPATCH_WARP_PER_BLOCK:-}}
MORI_RDMA_BLOCK_NUM=${MORI_RDMA_BLOCK_NUM:-}

KIWI_REPO=${KIWI_REPO:-$DEFAULT_KIWI_REPO}
MORI_SOURCE_DIR=${MORI_SOURCE_DIR:-}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29532}
MORI_MASTER_PORT=${MORI_MASTER_PORT:-29533}
LOG_DIR=${LOG_DIR:-$PWD/ep-mori-compare-logs}
DRY_RUN=${DRY_RUN:-0}
RUN_MODE=${RUN_MODE:-docker}  # docker (default) or bare (explicit custom env only)
RUN_KIWI=${RUN_KIWI:-1}
RUN_MORI_V1=${RUN_MORI_V1:-1}
RUN_MORI_V1LL=${RUN_MORI_V1LL:-1}

DOCKER=${DOCKER:-docker}
DOCKER_IMAGE=${DOCKER_IMAGE:-kiwi-ep-bench:rocm7.2.4-pytorch2.10}
EP_BENCH_IMAGE_DIGEST=${EP_BENCH_IMAGE_DIGEST:-}
ALLOW_UNQUALIFIED_LATEST=${ALLOW_UNQUALIFIED_LATEST:-0}
DOCKER_SHM_SIZE=${DOCKER_SHM_SIZE:-64g}
KIWI_MOUNT=${KIWI_MOUNT:-/workspace/kiwi}
MORI_MOUNT=${MORI_MOUNT:-/mori}
PYDEPS_DIR=${PYDEPS_DIR:-}
CONTAINER_PYDEPS_MOUNT=${CONTAINER_PYDEPS_MOUNT:-/pydeps}
HOST_INFINIBAND_INCLUDE_DIR=${HOST_INFINIBAND_INCLUDE_DIR:-/usr/include/infiniband}
# MORI's JIT path can need individual RDMA devices in the container, not just
# a /dev/infiniband bind.  Override if the site names differ.
DOCKER_INFINIBAND_DEVICES=${DOCKER_INFINIBAND_DEVICES:-"/dev/infiniband/rdma_cm /dev/infiniband/uverbs0 /dev/infiniband/uverbs1 /dev/infiniband/uverbs2 /dev/infiniband/uverbs3 /dev/infiniband/uverbs4 /dev/infiniband/uverbs5 /dev/infiniband/uverbs6 /dev/infiniband/uverbs7 /dev/infiniband/uverbs8"}
CONTAINER_KIWI_EXT_DIR=${CONTAINER_KIWI_EXT_DIR:-$KIWI_MOUNT/build/apps/ep/python}
CONTAINER_KIWI_PYTHONPATH=${CONTAINER_KIWI_PYTHONPATH:-$CONTAINER_KIWI_EXT_DIR}
CONTAINER_MORI_PYTHONPATH=${CONTAINER_MORI_PYTHONPATH:-$MORI_MOUNT/python:$MORI_MOUNT}
CONTAINER_LIB_LINK_DIR=${CONTAINER_LIB_LINK_DIR:-/tmp/mori-lib-links}
CONTAINER_LD_LIBRARY_PATH=${CONTAINER_LD_LIBRARY_PATH:-$CONTAINER_LIB_LINK_DIR:/opt/rocm/lib:/usr/lib/x86_64-linux-gnu:$MORI_MOUNT/python/mori}
MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-6G}
MORI_SHMEM_MODE=${MORI_SHMEM_MODE:-}
MORI_SHMEM_VMM_CHUNK_SIZE=${MORI_SHMEM_VMM_CHUNK_SIZE:-}
MORI_SKIP_VERIFY=${MORI_SKIP_VERIFY:-0}

# Dry runs only print commands, so show a placeholder when no checkout is set.
if [[ "$DRY_RUN" == "1" && -z "$MORI_SOURCE_DIR" ]]; then
  MORI_SOURCE_DIR=/path/to/mori
fi

if [[ -n "$PYDEPS_DIR" ]]; then
  CONTAINER_MORI_PYTHONPATH="$CONTAINER_PYDEPS_MOUNT:$CONTAINER_MORI_PYTHONPATH"
fi

HOST_KIWI_EXT_DIR=${HOST_KIWI_EXT_DIR:-$KIWI_REPO/build/apps/ep/python}
HOST_KIWI_PYTHONPATH=${HOST_KIWI_PYTHONPATH:-$HOST_KIWI_EXT_DIR${PYTHONPATH:+:$PYTHONPATH}}
HOST_MORI_PYTHONPATH=${HOST_MORI_PYTHONPATH:-$MORI_SOURCE_DIR/python:$MORI_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}}
HOST_MORI_LD_LIBRARY_PATH=${HOST_MORI_LD_LIBRARY_PATH:-/opt/rocm/lib:/usr/lib/x86_64-linux-gnu:$MORI_SOURCE_DIR/python/mori${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}

mkdir -p "$LOG_DIR"

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

shell_join() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
}

append_if_set() {
  local -n _arr=$1
  local flag=$2
  local value=$3
  if [[ -n "$value" ]]; then
    _arr+=("$flag" "$value")
  fi
}

if [[ "$RUN_MODE" != "docker" && "$RUN_MODE" != "bare" ]]; then
  echo "ERROR: RUN_MODE must be 'docker' or 'bare' (got '$RUN_MODE')" >&2
  exit 2
fi

if [[ "$RUN_MODE" == "docker" && "$DOCKER_IMAGE" == *:latest && "$ALLOW_UNQUALIFIED_LATEST" != "1" ]]; then
  echo "ERROR: DOCKER_IMAGE uses an unqualified :latest tag: $DOCKER_IMAGE" >&2
  echo "Use the canonical ep-bench image tag/digest or set ALLOW_UNQUALIFIED_LATEST=1 for historical reproduction only." >&2
  exit 2
fi

if [[ "$RUN_MODE" == "bare" ]]; then
  cat >&2 <<'MSG'
Warning: RUN_MODE=bare is for an explicitly activated custom environment only.
Bare hosts often lack the required torch/MORI/mpi4py Python environment, so
the default and documented path is RUN_MODE=docker.
MSG
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -d "$KIWI_REPO" ]]; then
    echo "ERROR: KIWI_REPO does not exist: $KIWI_REPO" >&2
    exit 2
  fi
  if [[ -z "$MORI_SOURCE_DIR" || ! -d "$MORI_SOURCE_DIR" ]]; then
    echo "ERROR: MORI_SOURCE_DIR is unset or does not exist: ${MORI_SOURCE_DIR:-<unset>}" >&2
    echo "Set MORI_SOURCE_DIR to a built official MORI checkout." >&2
    exit 2
  fi
fi

common_kiwi_args=(
  --cmd bench
  --timing graph
  --max-tokens "$MAX_TOKENS"
  --hidden-dim "$HIDDEN_DIM"
  --world-size "$EP_WORLD_SIZE"
  --dispatch-dtype "$DISPATCH_DTYPE"
  --num-experts-per-rank "$NUM_EXPERTS_PER_RANK"
  --num-experts-per-token "$NUM_EXPERTS_PER_TOKEN"
  --warmup "$WARMUP"
  --iters "$ITERS"
  --graph-replay-iters "$GRAPH_REPLAY_ITERS"
  --torch-dist auto
)
common_kiwi_args+=(--combine-dtype "$COMBINE_DTYPE")

common_mori_example_args=(
  --cmd bench
  # Official MORI's example still names the dispatch payload flag --dtype.
  --dtype "$DISPATCH_DTYPE"
  --max-tokens "$MAX_TOKENS"
  --num-qp "$NUM_QP"
  --quant-type "$QUANT_TYPE"
  --max-recv-total-tokens "$MAX_RECV_TOTAL_TOKENS"
)
common_mori_example_args+=(--combine-dtype "$COMBINE_DTYPE")
append_if_set common_mori_example_args --block-num "$MORI_BLOCK_NUM"
append_if_set common_mori_example_args --warp-per-block "$MORI_WARP_PER_BLOCK"
append_if_set common_mori_example_args --rdma-block-num "$MORI_RDMA_BLOCK_NUM"
if [[ "$MORI_SKIP_VERIFY" == "1" ]]; then
  common_mori_example_args+=(--skip-verify)
fi

mpirun_root_args=()
if [[ "$RUN_MODE" == "docker" ]]; then
  mpirun_root_args+=(--allow-run-as-root)
fi

if [[ "$RUN_MODE" == "docker" ]]; then
  docker_device_args=(--device /dev/kfd --device /dev/dri)
  for dev in $DOCKER_INFINIBAND_DEVICES; do
    if [[ "$DRY_RUN" == "1" || -e "$dev" ]]; then
      docker_device_args+=(--device "$dev")
    else
      echo "Warning: requested RDMA device does not exist and will not be passed: $dev" >&2
    fi
  done
  docker_mount_args=(
    -v "$KIWI_REPO:$KIWI_MOUNT:rw"
    -v "$MORI_SOURCE_DIR:$MORI_MOUNT:ro"
  )
  if [[ "$DRY_RUN" == "1" || -d "$HOST_INFINIBAND_INCLUDE_DIR" ]]; then
    docker_mount_args+=(-v "$HOST_INFINIBAND_INCLUDE_DIR:/usr/include/infiniband:ro")
  else
    echo "Warning: $HOST_INFINIBAND_INCLUDE_DIR not found; MORI JIT may fail to include infiniband/verbs.h" >&2
  fi
  if [[ -n "$PYDEPS_DIR" ]]; then
    docker_mount_args+=(-v "$PYDEPS_DIR:$CONTAINER_PYDEPS_MOUNT:ro")
  fi
  kiwi_script=$KIWI_MOUNT/apps/ep/benchmarks/python/bench_kiwi_ep.py
  mori_example=$MORI_MOUNT/examples/ops/dispatch_combine/test_dispatch_combine_internode.py
  bench_pythonpath=$CONTAINER_KIWI_PYTHONPATH
  bench_ld_library_path=$CONTAINER_LD_LIBRARY_PATH
  docker_common=(
    "$DOCKER" run --rm
    --network host
    --ipc host
    --shm-size "$DOCKER_SHM_SIZE"
    "${docker_device_args[@]}"
    --group-add video
    --group-add render
    --cap-add SYS_PTRACE
    --cap-add IPC_LOCK
    --security-opt seccomp=unconfined
    --ulimit memlock=-1:-1
    --ulimit stack=67108864:67108864
    "${docker_mount_args[@]}"
    -w "$KIWI_MOUNT"
    -e "PYTHONPATH=$CONTAINER_MORI_PYTHONPATH"
    -e "LD_LIBRARY_PATH=$CONTAINER_LD_LIBRARY_PATH"
    -e "MORI_SHMEM_HEAP_SIZE=$MORI_SHMEM_HEAP_SIZE"
    -e "EP_BENCH_IMAGE=$DOCKER_IMAGE"
    -e "EP_BENCH_IMAGE_DIGEST=$EP_BENCH_IMAGE_DIGEST"
    -e MPI_ALLOW_RUN_AS_ROOT=1
    -e MPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    -e OMPI_ALLOW_RUN_AS_ROOT=1
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    "$DOCKER_IMAGE"
  )
else
  kiwi_script=$KIWI_REPO/apps/ep/benchmarks/python/bench_kiwi_ep.py
  mori_example=$MORI_SOURCE_DIR/examples/ops/dispatch_combine/test_dispatch_combine_internode.py
  bench_pythonpath=$HOST_KIWI_PYTHONPATH
  bench_ld_library_path=$HOST_MORI_LD_LIBRARY_PATH
  docker_common=()
fi

kiwi_inner=(
  env
  MASTER_ADDR="$MASTER_ADDR"
  MASTER_PORT="$MASTER_PORT"
  PYTHONPATH="$bench_pythonpath"
  LD_LIBRARY_PATH="$bench_ld_library_path"
  mpirun "${mpirun_root_args[@]}" -np "$EP_WORLD_SIZE"
  -x MASTER_ADDR -x MASTER_PORT -x PYTHONPATH -x LD_LIBRARY_PATH
  python3 "$kiwi_script"
  "${common_kiwi_args[@]}"
)

make_mori_inner() {
  local official_kernel_type=$1
  if [[ "$RUN_MODE" == "docker" ]]; then
    printf '%s\0' \
      env \
      "GPU_PER_NODE=$GPU_PER_NODE" \
      "PYTHONPATH=$CONTAINER_MORI_PYTHONPATH" \
      "LD_LIBRARY_PATH=$CONTAINER_LD_LIBRARY_PATH" \
      "MORI_SHMEM_HEAP_SIZE=$MORI_SHMEM_HEAP_SIZE" \
      ${MORI_SHMEM_MODE:+"MORI_SHMEM_MODE=$MORI_SHMEM_MODE"} \
      ${MORI_SHMEM_VMM_CHUNK_SIZE:+"MORI_SHMEM_VMM_CHUNK_SIZE=$MORI_SHMEM_VMM_CHUNK_SIZE"} \
      torchrun --nnodes "$NUM_NODES" --nproc_per_node 1 \
      --node_rank "$MORI_NODE_RANK" \
      --master_addr "$MASTER_ADDR" \
      --master_port "$MORI_MASTER_PORT" \
      "$mori_example" \
      "${common_mori_example_args[@]}" \
      --kernel-type "$official_kernel_type"
  else
    printf '%s\0' \
      env \
      "RANK=$MORI_NODE_RANK" \
      "WORLD_SIZE=$NUM_NODES" \
      "GPU_PER_NODE=$GPU_PER_NODE" \
      "PYTHONPATH=$HOST_MORI_PYTHONPATH" \
      "LD_LIBRARY_PATH=$HOST_MORI_LD_LIBRARY_PATH" \
      "MORI_SHMEM_HEAP_SIZE=$MORI_SHMEM_HEAP_SIZE" \
      ${MORI_SHMEM_MODE:+"MORI_SHMEM_MODE=$MORI_SHMEM_MODE"} \
      ${MORI_SHMEM_VMM_CHUNK_SIZE:+"MORI_SHMEM_VMM_CHUNK_SIZE=$MORI_SHMEM_VMM_CHUNK_SIZE"} \
      python3 "$mori_example" \
      "${common_mori_example_args[@]}" \
      --kernel-type "$official_kernel_type"
  fi
}

run_step() {
  local label=$1
  local log_name=$2
  shift 2
  local -a inner=("$@")
  local -a cmd
  if [[ "$RUN_MODE" == "docker" ]]; then
    local prelude="set -e; mkdir -p '$CONTAINER_LIB_LINK_DIR';"
    prelude+=" [ -e /usr/lib/x86_64-linux-gnu/libmlx5.so.1 ] && ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 '$CONTAINER_LIB_LINK_DIR/libmlx5.so' || true;"
    prelude+=" [ -e /usr/lib/x86_64-linux-gnu/libibverbs.so.1 ] && ln -sf /usr/lib/x86_64-linux-gnu/libibverbs.so.1 '$CONTAINER_LIB_LINK_DIR/libibverbs.so' || true;"
    cmd=("${docker_common[@]}" bash -lc "$prelude $(shell_join "${inner[@]}")")
  else
    cmd=("${inner[@]}")
  fi

  echo
  echo "$label command:"
  print_cmd "${cmd[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  echo "$label log: $LOG_DIR/$log_name"
  "${cmd[@]}" 2>&1 | tee "$LOG_DIR/$log_name"
}

mapfile -d '' -t mori_v1_inner < <(make_mori_inner v1)
mapfile -d '' -t mori_v1ll_inner < <(make_mori_inner v1_ll)

cat <<EOF_SUMMARY
Mode: $RUN_MODE
Logs: $LOG_DIR
Docker image: $DOCKER_IMAGE
Resolved image digest: ${EP_BENCH_IMAGE_DIGEST:-not set; record docker image inspect output before publishing benchmark results}
KIWI repo host path: $KIWI_REPO
Official MORI host path: $MORI_SOURCE_DIR
Official MORI container mount: $MORI_MOUNT
MORI env in Docker: PYTHONPATH=$CONTAINER_MORI_PYTHONPATH, LD_LIBRARY_PATH=$CONTAINER_LD_LIBRARY_PATH
Kernel mapping: internode_v1 -> official --kernel-type v1 / InterNodeV1; internode_v1ll|internodeLL -> official --kernel-type v1_ll / InterNodeV1LL
Note: MORI's official internode example uses WORLD_SIZE as node count; this helper uses NUM_NODES=$NUM_NODES and GPU_PER_NODE=$GPU_PER_NODE.
MORI example launcher: torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=$MORI_NODE_RANK --master_port=$MORI_MASTER_PORT
Enabled steps: RUN_KIWI=$RUN_KIWI RUN_MORI_V1=$RUN_MORI_V1 RUN_MORI_V1LL=$RUN_MORI_V1LL
EOF_SUMMARY

if [[ "$RUN_KIWI" == "1" ]]; then
  run_step "KIWI graph benchmark" "kiwi_ep_graph.log" "${kiwi_inner[@]}"
else
  echo
  echo "Skipping KIWI graph benchmark because RUN_KIWI=$RUN_KIWI"
fi
if [[ "$RUN_MORI_V1" == "1" ]]; then
  run_step "MORI internode_v1 benchmark (v1 / InterNodeV1)" "mori_internode_v1.log" "${mori_v1_inner[@]}"
else
  echo
  echo "Skipping MORI v1 / InterNodeV1 because RUN_MORI_V1=$RUN_MORI_V1"
fi
if [[ "$RUN_MORI_V1LL" == "1" ]]; then
  run_step "MORI internode_v1ll benchmark (v1_ll / InterNodeV1LL)" "mori_internode_v1ll.log" "${mori_v1ll_inner[@]}"
else
  echo
  echo "Skipping MORI v1_ll / InterNodeV1LL because RUN_MORI_V1LL=$RUN_MORI_V1LL"
fi
