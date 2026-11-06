#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Collect reproducibility/provenance details for EP benchmark and profiling runs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/ep_bench/common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/ep_bench/collect_env.sh [--output-dir DIR] [--label NAME]

Collect host/container, ROCm, MPI, Python, Docker image, source, and dependency
provenance into OUTPUT_DIR (default: ep-bench-logs/env-<timestamp>).

Environment knobs:
  KIWI_REPO             KIWI checkout path (default: repository root)
  OUTPUT_DIR            Destination directory for env.txt and companion files
  EP_BENCH_IMAGE        Image tag/digest used for this run, if known
  EP_BENCH_IMAGE_DIGEST Resolved image digest, if known
  KIWI_EP_BENCH_BASE_TAG
                       Canonical ROCm/PyTorch base tag baked into the image
  LCI_SOURCE_DIR        Mounted/local LCI checkout to report
  MORI_SOURCE_DIR       Mounted/local official MORI checkout to report
  INTELLIKIT_ROOT       IntelliKit install root to report
  INTELLIKIT_SOURCE_DIR IntelliKit source checkout to report
  INTELLIKIT_BIN        IntelliKit profiling CLI (default image value: metrix)
USAGE
}

label=env
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --output-dir) shift; OUTPUT_DIR=${1:?missing --output-dir value} ;;
    --label) shift; label=${1:?missing --label value} ;;
    *) ep_bench_die "unknown argument: $1" ;;
  esac
  shift
done

repo=$(ep_bench_repo_root)
out=${OUTPUT_DIR:-$repo/ep-bench-logs/${label}-$(_ep_bench_timestamp)}
mkdir -p -- "$out"
env_txt=$out/env.txt

run_or_note() {
  local title=$1
  shift
  echo
  echo "## $title"
  if "$@"; then
    true
  else
    echo "(command failed or unavailable: $*)"
  fi
}

command_version() {
  local exe=$1
  shift || true
  if command -v "$exe" >/dev/null 2>&1; then
    local output
    output=$("$exe" "$@" 2>&1) || true
    printf '%s\n' "$output" | sed -n '1,40p'
  else
    echo "$exe: not found"
  fi
}

git_summary() {
  local name=$1
  local dir=$2
  echo
  echo "## git: $name"
  if [[ -d "$dir/.git" || -f "$dir/.git" ]]; then
    git -C "$dir" rev-parse --show-toplevel || true
    git -C "$dir" rev-parse HEAD || true
    git -C "$dir" status --short --branch || true
    git -C "$dir" remote -v || true
  elif [[ -n "$dir" ]]; then
    echo "not a git checkout or missing: $dir"
  else
    echo "not configured"
  fi
}

python_probe() {
  python3 - <<'PY' 2>&1 || true
import importlib.util
import os
import platform
import sys
print("python_executable=", sys.executable)
print("python_version=", sys.version.replace("\n", " "))
print("platform=", platform.platform())
for mod in ("torch", "mpi4py", "mori", "kiwi_ep_ext", "pybind11", "prettytable"):
    spec = importlib.util.find_spec(mod)
    print(f"module.{mod}.found={spec is not None}")
    if spec is not None and mod == "torch":
        import torch
        print("torch.__version__=", getattr(torch, "__version__", "unknown"))
        print("torch.version.hip=", getattr(torch.version, "hip", None))
        print("torch.cuda.is_available=", torch.cuda.is_available())
        try:
            print("torch.cuda.device_count=", torch.cuda.device_count())
            for idx in range(torch.cuda.device_count()):
                print(f"torch.cuda.device[{idx}]=", torch.cuda.get_device_name(idx))
        except Exception as exc:
            print("torch.cuda.device_probe_error=", repr(exc))
print("PYTHONPATH=", os.environ.get("PYTHONPATH", ""))
PY
}

docker_probe() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker: not found"
    return 0
  fi
  docker --version || true
  if [[ -n "${EP_BENCH_IMAGE:-}" ]]; then
    echo "EP_BENCH_IMAGE=$EP_BENCH_IMAGE"
    docker image inspect "$EP_BENCH_IMAGE" \
      --format 'id={{.Id}} repo_digests={{json .RepoDigests}} created={{.Created}}' 2>/dev/null || true
  fi
}

intellikit_probe() {
  echo "INTELLIKIT_ROOT=${INTELLIKIT_ROOT:-}"
  echo "INTELLIKIT_SOURCE_DIR=${INTELLIKIT_SOURCE_DIR:-}"
  echo "INTELLIKIT_VENV=${INTELLIKIT_VENV:-}"
  echo "INTELLIKIT_BIN=${INTELLIKIT_BIN:-}"
  if [[ -r /opt/ep-bench/intellikit-provenance.env ]]; then
    echo
    echo "## /opt/ep-bench/intellikit-provenance.env"
    cat /opt/ep-bench/intellikit-provenance.env
  fi

  echo
  echo "## IntelliKit command probes"
  local exe path
  local output
  for exe in intellikit metrix kerncap accordo metrix-mcp linex-mcp nexus-mcp \
             kerncap-mcp accordo-mcp hip-compiler-mcp hip-docs-mcp \
             amd-smi-mcp rocminfo-mcp uprof-profiler-mcp; do
    if path=$(command -v "$exe" 2>/dev/null); then
      echo "$exe.path=$path"
      case "$exe" in
        metrix)
          output=$("$exe" --version 2>&1) || true
          printf '%s\n' "$output" | sed -n '1,20p'
          ;;
        intellikit)
          output=$("$exe" --version 2>&1) || output=$("$exe" --help 2>&1) || true
          printf '%s\n' "$output" | sed -n '1,20p'
          ;;
        *)
          output=$("$exe" --help 2>&1) || true
          printf '%s\n' "$output" | sed -n '1,20p'
          ;;
      esac
    else
      echo "$exe.path=not found"
    fi
  done
}

{
  echo "# KIWI EP benchmark/profiling provenance"
  echo "collected_utc=$(_ep_bench_timestamp)"
  echo "label=$label"
  echo "repo=$repo"
  echo "output_dir=$out"
  echo "user=$(id -un 2>/dev/null || true) uid=$(id -u 2>/dev/null || true) groups=$(id -Gn 2>/dev/null || true)"
  echo "hostname=${HOSTNAME:-$(hostname 2>/dev/null || true)}"
  echo "uname=$(uname -a 2>/dev/null || true)"
  echo "pwd=$PWD"
  echo "EP_BENCH_IMAGE=${EP_BENCH_IMAGE:-}"
  echo "EP_BENCH_IMAGE_DIGEST=${EP_BENCH_IMAGE_DIGEST:-}"
  echo "KIWI_EP_BENCH_BASE_TAG=${KIWI_EP_BENCH_BASE_TAG:-}"

  if [[ -r /etc/os-release ]]; then
    echo
    echo "## /etc/os-release"
    cat /etc/os-release
  fi

  git_summary KIWI "$repo"
  git_summary LCI "${LCI_SOURCE_DIR:-}"
  git_summary MORI "${MORI_SOURCE_DIR:-}"
  git_summary IntelliKit "${INTELLIKIT_SOURCE_DIR:-${INTELLIKIT_ROOT:-}}"

  echo
  echo "## selected environment"
  env | sort | grep -E '^(KIWI|MORI|INTELLIKIT|KIWI_QUEUE|LCI|ROCM|HIP|HSA|ROCR|CUDA|MASTER|WORLD|RANK|LOCAL|OMPI|MPICH|MV2|SLURM|NCCL|RCCL|UCX|FI_|LD_LIBRARY_PATH|PYTHONPATH|PATH|EP_BENCH|DOCKER)=' || true

  echo
  echo "## tool versions"
  command_version git --version
  command_version cmake --version
  command_version ninja --version
  command_version make --version
  command_version hipcc --version
  command_version rocminfo --version
  command_version rocm-smi --version
  command_version mpirun --version
  command_version python3 --version

  run_or_note "ROCm agents" rocminfo
  run_or_note "rocm-smi" rocm-smi --showproductname --showdriverversion --showvbios --showtopo
  run_or_note "Python modules" python_probe
  run_or_note "Docker image" docker_probe
  run_or_note "IntelliKit" intellikit_probe
} >"$env_txt" 2>&1

if command -v python3 >/dev/null 2>&1; then
  python3 -m pip freeze >"$out/pip-freeze.txt" 2>/dev/null || true
fi

printf 'Wrote provenance to %s\n' "$env_txt"
