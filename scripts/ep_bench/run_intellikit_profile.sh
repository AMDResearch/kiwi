#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Stable profiling entrypoint for the in-image upstream IntelliKit install.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/ep_bench/common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/ep_bench/run_intellikit_profile.sh [options] -- workload ...

Options:
  --output-dir DIR   Profile output/provenance directory
  --dry-run          Print profiler command without executing
  --allow-missing    If IntelliKit is unavailable, print a documented placeholder and exit 0
  -h, --help         Show this help

Environment knobs:
  INTELLIKIT_ROOT       IntelliKit install root (default in image: /opt/intellikit)
  INTELLIKIT_VENV       IntelliKit venv (default: $INTELLIKIT_ROOT/venv)
  INTELLIKIT_BIN        Profiler executable (default: metrix from the in-image venv)
  INTELLIKIT_ARGS       Simple whitespace-separated profiler args. For metrix,
                        default is: profile --time-only --num-replays 1 --output OUT/metrix.json
  INTELLIKIT_WORKLOAD_MODE
                        auto|string|args|separator. Metrix expects the workload
                        as one shell-quoted command string, so auto selects string
                        for metrix and separator for other profilers.
  INTELLIKIT_SEPARATOR  Separator before workload in separator mode (default: "--")

Upstream IntelliKit does not currently provide a repo-root "intellikit" CLI.
The image installs the per-tool packages and defaults this profiling wrapper to
the upstream Metrix CLI (`metrix --version`, `metrix profile ...`). Override
INTELLIKIT_BIN/INTELLIKIT_ARGS/INTELLIKIT_WORKLOAD_MODE for another site-local
IntelliKit tool or profiler syntax.
USAGE
}

DRY_RUN=${DRY_RUN:-0}
ALLOW_MISSING=${ALLOW_MISSING:-0}
workload=()
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --output-dir) shift; OUTPUT_DIR=${1:?missing --output-dir value} ;;
    --dry-run) DRY_RUN=1 ;;
    --allow-missing) ALLOW_MISSING=1 ;;
    --) shift; workload=("$@"); break ;;
    *) ep_bench_die "unknown argument before --: $1" ;;
  esac
  shift
done

if ((${#workload[@]} == 0)); then
  ep_bench_die "missing workload after --"
fi

repo=$(ep_bench_repo_root)
out=$(ep_bench_output_dir intellikit-$(_ep_bench_timestamp))
"$SCRIPT_DIR/collect_env.sh" --output-dir "$out/provenance" --label intellikit >/dev/null || true

if [[ -z "${INTELLIKIT_BIN:-}" ]]; then
  if [[ -n "${INTELLIKIT_VENV:-}" && -x "$INTELLIKIT_VENV/bin/metrix" ]]; then
    INTELLIKIT_BIN=$INTELLIKIT_VENV/bin/metrix
  elif [[ -n "${INTELLIKIT_ROOT:-}" && -x "$INTELLIKIT_ROOT/venv/bin/metrix" ]]; then
    INTELLIKIT_BIN=$INTELLIKIT_ROOT/venv/bin/metrix
  elif [[ -n "${INTELLIKIT_ROOT:-}" && -x "$INTELLIKIT_ROOT/bin/metrix" ]]; then
    INTELLIKIT_BIN=$INTELLIKIT_ROOT/bin/metrix
  else
    INTELLIKIT_BIN=metrix
  fi
fi

if ! command -v "$INTELLIKIT_BIN" >/dev/null 2>&1 && [[ ! -x "$INTELLIKIT_BIN" ]]; then
  cat >&2 <<EOF_SKIP
IntelliKit Metrix profiler is not available.
The canonical image installs upstream IntelliKit at /opt/intellikit and sets
INTELLIKIT_BIN=/opt/intellikit/venv/bin/metrix. Set INTELLIKIT_ROOT,
INTELLIKIT_VENV, or INTELLIKIT_BIN/INTELLIKIT_ARGS for a different install.
Example:
  INTELLIKIT_BIN=/opt/intellikit/venv/bin/metrix \\
  INTELLIKIT_ARGS='profile --time-only --num-replays 1 --output $out/metrix.json' \\
  $0 -- ${workload[*]}
EOF_SKIP
  if [[ "$ALLOW_MISSING" == "1" ]]; then
    exit 0
  fi
  exit 77
fi

profiler_base=$(basename -- "$INTELLIKIT_BIN")
if [[ -n "${INTELLIKIT_ARGS:-}" ]]; then
  # Intentionally simple whitespace splitting for profiler args supplied through
  # the environment; use a small wrapper script for arguments requiring complex
  # shell quoting.
  # shellcheck disable=SC2206
  profiler_args=($INTELLIKIT_ARGS)
elif [[ "$profiler_base" == "metrix" ]]; then
  profiler_args=(profile --time-only --num-replays "${INTELLIKIT_NUM_REPLAYS:-1}" --output "$out/metrix.json")
else
  profiler_args=(profile --output "$out")
fi

workload_mode=${INTELLIKIT_WORKLOAD_MODE:-auto}
if [[ "$workload_mode" == "auto" ]]; then
  if [[ "$profiler_base" == "metrix" ]]; then
    workload_mode=string
  else
    workload_mode=separator
  fi
fi

cmd=(env)
cmd+=("$INTELLIKIT_BIN" "${profiler_args[@]}")
case "$workload_mode" in
  string)
    printf -v workload_cmd '%q ' "${workload[@]}"
    workload_cmd=${workload_cmd% }
    cmd+=("$workload_cmd")
    ;;
  args)
    cmd+=("${workload[@]}")
    ;;
  separator)
    if [[ "${INTELLIKIT_SEPARATOR---}" != "" ]]; then
      cmd+=("${INTELLIKIT_SEPARATOR---}")
    fi
    cmd+=("${workload[@]}")
    ;;
  *)
    ep_bench_die "invalid INTELLIKIT_WORKLOAD_MODE: $workload_mode"
    ;;
esac

ep_bench_run "${cmd[@]}"
printf 'Profile/provenance output directory: %s\n' "$out"
