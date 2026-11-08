#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Benchmark harness for the kiwi-ep Python operator.

The default path captures multiple dispatch+combine pairs in one CUDAGraph,
times one replay, and divides by the number of captured pairs to amortize the
fixed graph replay envelope. Eager timing remains available with
``--timing eager``.

Typical launch after building/importing ``kiwi_ep_ext``::

    MASTER_ADDR=127.0.0.1 MASTER_PORT=29532 \
    mpirun -np 8 -x MASTER_ADDR -x MASTER_PORT \
      python3 apps/ep/benchmarks/python/bench_kiwi_ep.py --cmd bench
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Any, Iterable, Optional

from kiwi_ep_bench_utils import (
    BenchInputs,
    DTypeSpec,
    LaunchConfig,
    ScaleSpec,
    default_launch_config,
    dispatch_kwargs,
    dtype_storage_hidden_elems,
    make_indices_for_rank,
    make_input_for_rank,
    make_scale_spec,
    make_scales_for_rank,
    make_synthetic_combine_input,
    quant_format_for_dtype,
    resolve_dtype,
    run_combine,
    run_dispatch,
    slot_payload_bytes,
    slot_stride_bytes,
    combine_slot_payload_bytes,
    combine_slot_stride_bytes,
)


_E2E_CAPTURE_WARNING_PRINTED = False

METRIC_DISPATCH_US = 0
METRIC_DISPATCH_BW_GBS = 1
METRIC_COMBINE_US = 2
METRIC_COMBINE_BW_GBS = 3
METRIC_E2E_US = 4
METRIC_E2E_BW_GBS = 5
METRIC_DISPATCH_BYTES_MIB = 6
METRIC_COMBINE_BYTES_MIB = 7
METRIC_E2E_BYTES_MIB = 8
METRIC_RECV_TOKENS = 9
METRIC_SPLIT_E2E_US = 10
METRIC_E2E_IS_FULL_GRAPH = 11


@dataclass(frozen=True)
class BenchConfig:
    rank: int
    local_rank: int
    world_size: int
    max_tokens: int
    model_hidden_dim: int
    dispatch_hidden_elems: int
    combine_hidden_elems: int
    num_experts_per_rank: int
    num_experts_per_token: int
    dispatch_dtype: DTypeSpec
    combine_dtype: DTypeSpec
    dispatch_scale: ScaleSpec
    combine_scale: ScaleSpec
    launch: LaunchConfig
    graph_replay_iters: int
    warmup: int
    iters: int
    seed: int
    debug_timing: bool


@dataclass(frozen=True)
class RoundMetrics:
    dispatch_us: float
    combine_us: float
    e2e_us: float
    split_e2e_us: float
    dispatch_bw_gbs: float
    combine_bw_gbs: float
    e2e_bw_gbs: float
    dispatch_bytes: float
    combine_bytes: float
    e2e_bytes: float
    recv_tokens: float
    e2e_is_full_graph: bool


def env_int(*names: str, default: Optional[int] = None) -> Optional[int]:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                continue
    return default


def launcher_rank_from_env() -> int:
    return env_int(
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
        "RANK",
        default=0,
    ) or 0


def _torch_dtype_name(dtype: Any) -> str:
    return str(dtype).split(".")[-1]


def merge_launch_args(args: argparse.Namespace, world_size: int) -> LaunchConfig:
    launch = default_launch_config(world_size, args.max_tokens)
    return LaunchConfig(
        dispatch_block_num=args.dispatch_block_num if args.dispatch_block_num is not None else launch.dispatch_block_num,
        dispatch_warp_per_block=(
            args.dispatch_warp_per_block if args.dispatch_warp_per_block is not None else launch.dispatch_warp_per_block
        ),
        combine_block_num=args.combine_block_num if args.combine_block_num is not None else launch.combine_block_num,
        combine_warp_per_block=(
            args.combine_warp_per_block if args.combine_warp_per_block is not None else launch.combine_warp_per_block
        ),
    )


def local_rank_from_env(global_rank: int, torch: Any) -> int:
    local = env_int(
        "LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        default=None,
    )
    if local is not None:
        return local
    device_count = torch.cuda.device_count()
    return global_rank % max(device_count, 1)


def local_world_size_from_env(torch: Any) -> int:
    value = env_int(
        "LOCAL_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
        "MV2_COMM_WORLD_LOCAL_SIZE",
        default=None,
    )
    if value is not None:
        return value
    slurm_tasks_per_node = os.environ.get("SLURM_TASKS_PER_NODE")
    if slurm_tasks_per_node:
        first = slurm_tasks_per_node.split(",", 1)[0]
        if "(" in first:
            first = first.split("(", 1)[0]
        try:
            return int(first)
        except ValueError:
            pass
    return max(torch.cuda.device_count(), 1)


def maybe_init_torch_dist(torch: Any, rank: int, world_size: int, args: argparse.Namespace) -> tuple[Optional[Any], bool, str]:
    """Initialize torch.distributed when safe, otherwise return a no-op status.

    The benchmark can be launched by MPI for LCI rather than torchrun.  For
    single-node MPI jobs we can safely bootstrap a gloo process group on
    localhost.  For multi-node jobs, require MASTER_ADDR/MASTER_PORT (or
    --master-addr/--master-port) to avoid accidentally hanging on 127.0.0.1.
    """

    try:
        import torch.distributed as dist  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on host install
        if args.torch_dist == "require":
            raise RuntimeError(f"torch.distributed import failed: {exc}") from exc
        return None, False, f"torch.distributed unavailable: {exc}"

    if not dist.is_available():
        if args.torch_dist == "require":
            raise RuntimeError("torch.distributed is not available in this PyTorch build")
        return None, False, "torch.distributed is not available"
    if dist.is_initialized():
        return dist, False, "using pre-initialized torch.distributed"
    if args.torch_dist == "never" or world_size <= 1:
        return None, False, "torch.distributed initialization disabled or single-rank run"

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    if args.master_addr:
        os.environ["MASTER_ADDR"] = args.master_addr
    if args.master_port:
        os.environ["MASTER_PORT"] = str(args.master_port)

    local_world = local_world_size_from_env(torch)
    if "MASTER_ADDR" not in os.environ:
        if local_world == world_size:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        else:
            msg = (
                "MASTER_ADDR is not set for a likely multi-node job; "
                "skipping torch.distributed gather. Set MASTER_ADDR/MASTER_PORT "
                "or pass --torch-dist require to make this fatal."
            )
            if args.torch_dist == "require":
                raise RuntimeError(msg)
            return None, False, msg
    os.environ.setdefault("MASTER_PORT", "29500")

    try:
        dist.init_process_group(
            backend=args.torch_dist_backend,
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:  # pragma: no cover - depends on distributed env
        if args.torch_dist == "require":
            raise
        return None, False, f"torch.distributed init failed; rank summaries will be local only: {exc}"
    return dist, True, f"initialized torch.distributed backend={dist.get_backend()}"


def dist_is_ready(dist: Optional[Any]) -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def synchronize_all(torch: Any, kiwi_ep_ext: Any, dist: Optional[Any]) -> None:
    torch.cuda.synchronize()
    if dist_is_ready(dist):
        dist.barrier()
    kiwi_ep_ext.barrier()


def collect_lci_device_summaries(
    torch: Any,
    dist: Optional[Any],
    rank: int,
    world_size: int,
    local_rank: int,
    lci_device_name: str,
    lci_device_source: str,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Gather the per-rank GPU/NIC selection used to create LCI contexts.

    The benchmark usually initializes a torch.distributed process group for
    rank-level summaries.  When that is unavailable, return only the local
    summary and let each rank print its own line so the log still exposes every
    process' requested HCA/NIC instead of misleadingly showing rank 0 only.
    """

    info = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "cuda_device": int(torch.cuda.current_device()),
        "cuda_device_name": torch.cuda.get_device_name(local_rank),
        "lci_device_name": lci_device_name or "<default>",
        "lci_device_source": lci_device_source,
    }

    if not dist_is_ready(dist):
        return [info], None

    try:
        gathered: list[Optional[dict[str, Any]]] = [None] * world_size
        dist.all_gather_object(gathered, info)
        return (
            sorted(
                (entry for entry in gathered if entry is not None),
                key=lambda entry: int(entry["rank"]),
            ),
            None,
        )
    except Exception as exc:  # pragma: no cover - depends on distributed env
        return [info], f"failed to gather per-rank LCI HCA/NIC selections: {exc}"


def print_lci_device_summaries(summaries: list[dict[str, Any]]) -> None:
    print("LCI requested HCA/NIC per rank:")
    for entry in summaries:
        print(
            "  "
            f"rank {entry['rank']} local_rank {entry['local_rank']} "
            f"cuda_device {entry['cuda_device']} ({entry['cuda_device_name']}) "
            f"-> {entry['lci_device_name']} [{entry['lci_device_source']}]"
        )


def generate_rank_inputs(torch: Any, cfg: BenchConfig) -> BenchInputs:
    """Generate MORI-style max-token benchmark inputs for this rank.

    MORI's test utility seeds one device generator and creates per-rank indices,
    weights, scales, then inputs.  Replaying the same generation order here gives
    each rank its MORI-equivalent slice without retaining every peer tensor.
    """

    device = torch.device("cuda", cfg.local_rank)
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    total_experts = cfg.num_experts_per_rank * cfg.world_size

    selected_indices = None
    selected_weights = None
    selected_scales = None
    selected_tokens = None

    for rank in range(cfg.world_size):
        indices = make_indices_for_rank(
            torch,
            cfg.max_tokens,
            cfg.num_experts_per_token,
            total_experts,
            device,
            rng,
        )
        if rank == cfg.rank:
            selected_indices = indices.to(torch.int32).contiguous()

    for rank in range(cfg.world_size):
        weights = torch.rand(
            (cfg.max_tokens, cfg.num_experts_per_token),
            dtype=torch.float32,
            device=device,
            generator=rng,
        )
        if rank == cfg.rank:
            selected_weights = weights.contiguous()

    for rank in range(cfg.world_size):
        scales = make_scales_for_rank(torch, cfg.max_tokens, cfg.dispatch_scale, device, rng)
        if rank == cfg.rank:
            selected_scales = scales.contiguous()

    for rank in range(cfg.world_size):
        tokens = make_input_for_rank(
            torch,
            cfg.max_tokens,
            cfg.dispatch_hidden_elems,
            cfg.dispatch_dtype,
            device,
            rng,
        )
        if rank == cfg.rank:
            selected_tokens = tokens.contiguous()

    assert selected_tokens is not None
    assert selected_weights is not None
    assert selected_scales is not None
    assert selected_indices is not None
    return BenchInputs(
        tokens=selected_tokens,
        weights=selected_weights,
        scales=selected_scales,
        indices=selected_indices,
    )


def validate_combine_input(combine_input: Any, cfg: BenchConfig) -> Any:
    if combine_input.size(1) != cfg.combine_hidden_elems:
        raise RuntimeError(
            "combine input hidden elements must match configured combine_hidden_elems "
            f"(got {combine_input.size(1)}, expected {cfg.combine_hidden_elems})"
        )
    return combine_input


def run_pair(torch: Any, op: Any, inputs: BenchInputs, cfg: BenchConfig) -> tuple[int, Any]:
    dispatch_output, _dispatch_weights, _dispatch_scales, dispatch_indices, recv_count = run_dispatch(op, inputs, cfg.launch)
    combine_input = make_synthetic_combine_input(torch, dispatch_output, cfg)
    validate_combine_input(combine_input, cfg)
    combine_output, _combine_weights = run_combine(op, combine_input, dispatch_indices, cfg.launch)
    total_recv = int(recv_count.cpu()[0].item())
    return total_recv, combine_output


def reset_and_sync(torch: Any, kiwi_ep_ext: Any, dist: Optional[Any], op: Any) -> None:
    synchronize_all(torch, kiwi_ep_ext, dist)
    op.reset()
    synchronize_all(torch, kiwi_ep_ext, dist)


def build_metrics(
    cfg: BenchConfig,
    total_recv_tokens: int,
    dispatch_us: float,
    combine_us: float,
    e2e_us: float,
    e2e_is_full_graph: bool,
) -> RoundMetrics:
    # Dispatch and combine transfer their full aligned slots.
    dispatch_slot = slot_stride_bytes(
        cfg.dispatch_hidden_elems,
        cfg.dispatch_dtype.elem_size,
        cfg.num_experts_per_token,
        cfg.dispatch_scale.dim,
        cfg.dispatch_scale.type_size,
    )
    combine_slot = combine_slot_stride_bytes(
        cfg.combine_hidden_elems,
        cfg.combine_dtype.elem_size,
        cfg.num_experts_per_token,
    )
    dispatch_bytes = float(total_recv_tokens * dispatch_slot)
    combine_bytes = float(total_recv_tokens * combine_slot)
    e2e_bytes = dispatch_bytes + combine_bytes

    def bw(bytes_count: float, latency_us: float) -> float:
        if latency_us <= 0:
            return 0.0
        return bytes_count / (latency_us * 1e-6) / 1e9

    split_e2e_us = dispatch_us + combine_us
    return RoundMetrics(
        dispatch_us=dispatch_us,
        combine_us=combine_us,
        e2e_us=e2e_us,
        split_e2e_us=split_e2e_us,
        dispatch_bw_gbs=bw(dispatch_bytes, dispatch_us),
        combine_bw_gbs=bw(combine_bytes, combine_us),
        e2e_bw_gbs=bw(e2e_bytes, e2e_us),
        dispatch_bytes=dispatch_bytes,
        combine_bytes=combine_bytes,
        e2e_bytes=e2e_bytes,
        recv_tokens=float(total_recv_tokens),
        e2e_is_full_graph=e2e_is_full_graph,
    )


def capture_split_graphs(torch: Any, op: Any, inputs: BenchInputs, cfg: BenchConfig) -> tuple[Any, Any, Any, Any, Any]:
    dispatch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(dispatch_graph):
        dispatch_output, _dispatch_weights, _dispatch_scales, dispatch_indices, _recv_count = run_dispatch(op, inputs, cfg.launch)

    combine_input = make_synthetic_combine_input(torch, dispatch_output, cfg)
    validate_combine_input(combine_input, cfg)
    combine_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(combine_graph):
        run_combine(op, combine_input, dispatch_indices, cfg.launch)

    return dispatch_graph, combine_graph, dispatch_output, dispatch_indices, combine_input


def capture_e2e_graph(
    torch: Any,
    op: Any,
    inputs: BenchInputs,
    combine_input: Any,
    cfg: BenchConfig,
    pairs: int,
) -> Any:
    e2e_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(e2e_graph):
        for _ in range(pairs):
            (
                _dispatch_output,
                _dispatch_weights,
                _dispatch_scales,
                dispatch_indices,
                _recv_count,
            ) = run_dispatch(op, inputs, cfg.launch)
            run_combine(op, combine_input, dispatch_indices, cfg.launch)
    return e2e_graph


def amortized_elapsed_ms(start: Any, end: Any, pairs: int) -> float:
    return start.elapsed_time(end) / pairs


def measure_graph_replay_us(torch: Any, graph: Any, iters: int) -> float:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) * 1000.0 / iters


def latest_debug_total_us(op: Any, phase_labels: list[Optional[str]]) -> Optional[float]:
    snapshot = op.get_debug_timing_snapshot()
    time_ticks = [int(v) for v in snapshot.get("time_ticks", [])]
    wall_clock_rate_khz = int(snapshot.get("wall_clock_rate_khz", 0))
    if not snapshot.get("enabled", False) or wall_clock_rate_khz <= 0:
        return None
    if not time_ticks or time_ticks[0] == 0:
        return None
    last_idx = 0
    for idx, phase_label in enumerate(phase_labels):
        if phase_label is not None and idx < len(time_ticks) and time_ticks[idx] != 0:
            last_idx = idx
    if last_idx <= 0:
        return None
    return (time_ticks[last_idx] - time_ticks[0]) * (1000.0 / float(wall_clock_rate_khz))


def print_graph_envelope_debug(
    torch: Any,
    kiwi_ep_ext: Any,
    dist: Optional[Any],
    op: Any,
    cfg: BenchConfig,
    iter_idx: int,
    dispatch_graph: Any,
    combine_ep_graph: Any,
    dispatch_event_us: float,
    combine_event_us: float,
) -> None:
    """Print where graph/event time sits outside the instrumented EP kernels."""

    # These graph replays include collective GPU/LCI communication.  All ranks
    # must replay them concurrently; only the printing is rank-ordered.
    dispatch_graph_event_us = measure_graph_replay_us(
        torch, dispatch_graph, cfg.graph_replay_iters)
    dispatch_kernel_us = latest_debug_total_us(
        op, dispatch_debug_labels())

    combine_ep_event_us = measure_graph_replay_us(
        torch, combine_ep_graph, cfg.graph_replay_iters)
    combine_kernel_us = latest_debug_total_us(
        op, combine_debug_labels())

    def fmt(value: Optional[float]) -> str:
        return "nan" if value is None else f"{value:.3f}"

    dispatch_graph_gap = (
        dispatch_graph_event_us - dispatch_kernel_us
        if dispatch_kernel_us is not None else None)
    dispatch_reported_gap = (
        dispatch_event_us - dispatch_kernel_us
        if dispatch_kernel_us is not None else None)
    combine_reported_gap = (
        combine_event_us - combine_kernel_us
        if combine_kernel_us is not None else None)
    combine_ep_gap = (
        combine_ep_event_us - combine_kernel_us
        if combine_kernel_us is not None else None)

    for ordered_rank in range(cfg.world_size):
        kiwi_ep_ext.barrier()
        if cfg.rank != ordered_rank:
            continue

        print(
            f"# py-envelope-debug rank={cfg.rank} iter={iter_idx} timing=graph "
            f"dispatch_reported_event_us={dispatch_event_us:.3f} "
            f"dispatch_graph_event_us={dispatch_graph_event_us:.3f} "
            f"dispatch_kernel_debug_us={fmt(dispatch_kernel_us)} "
            f"dispatch_reported_gap_us={fmt(dispatch_reported_gap)} "
            f"dispatch_graph_gap_us={fmt(dispatch_graph_gap)} "
            f"combine_reported_event_us={combine_event_us:.3f} "
            f"combine_ep_event_us={combine_ep_event_us:.3f} "
            f"combine_kernel_debug_us={fmt(combine_kernel_us)} "
            f"combine_reported_gap_us={fmt(combine_reported_gap)} "
            f"combine_ep_gap_us={fmt(combine_ep_gap)}"
        )
        sys.stdout.flush()

    kiwi_ep_ext.barrier()


def run_graph_bench_once(
    torch: Any,
    kiwi_ep_ext: Any,
    dist: Optional[Any],
    op: Any,
    inputs: BenchInputs,
    cfg: BenchConfig,
    iter_idx: int,
) -> RoundMetrics:
    global _E2E_CAPTURE_WARNING_PRINTED

    if cfg.graph_replay_iters <= 0:
        raise ValueError("--graph-replay-iters must be greater than zero for graph timing")

    # A real dispatch/combine pair discovers the dynamic receive count before
    # graph construction, just like MORI's benchmark.
    reset_and_sync(torch, kiwi_ep_ext, dist, op)
    total_recv_tokens, _ = run_pair(torch, op, inputs, cfg)
    torch.cuda.synchronize()

    synchronize_all(torch, kiwi_ep_ext, dist)
    dispatch_graph, combine_graph, dispatch_output, dispatch_indices, combine_input = (
        capture_split_graphs(torch, op, inputs, cfg))
    torch.cuda.synchronize()
    synchronize_all(torch, kiwi_ep_ext, dist)

    e2e_graph = None
    e2e_capture_error = None
    try:
        e2e_graph = capture_e2e_graph(
            torch, op, inputs, combine_input, cfg, cfg.graph_replay_iters)
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - requires graph-capable GPU runtime
        e2e_capture_error = exc
        torch.cuda.synchronize()
    synchronize_all(torch, kiwi_ep_ext, dist)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(cfg.graph_replay_iters)]
    mids = [torch.cuda.Event(enable_timing=True) for _ in range(cfg.graph_replay_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(cfg.graph_replay_iters)]
    e2e_start = torch.cuda.Event(enable_timing=True)
    e2e_end = torch.cuda.Event(enable_timing=True)

    for i in range(cfg.graph_replay_iters):
        starts[i].record()
        dispatch_graph.replay()
        mids[i].record()
        combine_graph.replay()
        ends[i].record()

    if e2e_graph is not None:
        e2e_start.record()
        e2e_graph.replay()
        e2e_end.record()

    torch.cuda.synchronize()

    dispatch_ms = sum(start.elapsed_time(mid) for start, mid in zip(starts, mids)) / cfg.graph_replay_iters
    combine_ms = sum(mid.elapsed_time(end) for mid, end in zip(mids, ends)) / cfg.graph_replay_iters
    if e2e_graph is not None:
        e2e_ms = amortized_elapsed_ms(
            e2e_start, e2e_end, cfg.graph_replay_iters)
        e2e_is_full_graph = True
    else:
        # Keep an e2e metric available for aggregation, but mark that it is the
        # split dispatch+combine sum rather than a full graph replay.
        e2e_ms = dispatch_ms + combine_ms
        e2e_is_full_graph = False
        if cfg.rank == 0 and e2e_capture_error is not None and not _E2E_CAPTURE_WARNING_PRINTED:
            print(f"Warning: e2e graph capture failed; reporting split sum for e2e: {e2e_capture_error}", file=sys.stderr)
            _E2E_CAPTURE_WARNING_PRINTED = True

    synchronize_all(torch, kiwi_ep_ext, dist)
    if cfg.debug_timing:
        # The combined timing loop replays dispatch and then combine, so the
        # shared debug buffer contains the final combine kernel.  Print combine
        # first.  Do not capture or replay a standalone combine-only graph
        # envelope: rcomp's combine-exit barrier guards the next dispatch, and
        # combine-only graph replay is not a valid EP protocol.  Then replay one
        # untimed dispatch graph to capture dispatch phase timestamps without
        # perturbing reported metrics.
        print_combine_debug_timing(kiwi_ep_ext, op, cfg, iter_idx, "graph")
        synchronize_all(torch, kiwi_ep_ext, dist)
        dispatch_graph.replay()
        torch.cuda.synchronize()
        print_dispatch_debug_timing(kiwi_ep_ext, op, cfg, iter_idx, "graph")
        synchronize_all(torch, kiwi_ep_ext, dist)
    return build_metrics(
        cfg,
        total_recv_tokens=total_recv_tokens,
        dispatch_us=dispatch_ms * 1000.0,
        combine_us=combine_ms * 1000.0,
        e2e_us=e2e_ms * 1000.0,
        e2e_is_full_graph=e2e_is_full_graph,
    )


def run_eager_bench_once(
    torch: Any,
    kiwi_ep_ext: Any,
    dist: Optional[Any],
    op: Any,
    inputs: BenchInputs,
    cfg: BenchConfig,
) -> RoundMetrics:
    if cfg.graph_replay_iters <= 0:
        raise ValueError("--timing-batch-iters must be greater than zero")

    # Discover the receive shape and create the synthetic expert output outside
    # the timed loop.
    reset_and_sync(torch, kiwi_ep_ext, dist, op)
    dispatch_output, _dispatch_weights, _dispatch_scales, dispatch_indices, recv_count = (
        run_dispatch(op, inputs, cfg.launch))
    combine_input = make_synthetic_combine_input(torch, dispatch_output, cfg)
    validate_combine_input(combine_input, cfg)
    run_combine(op, combine_input, dispatch_indices, cfg.launch)
    torch.cuda.synchronize()
    total_recv_tokens = int(recv_count.cpu()[0].item())

    reset_and_sync(torch, kiwi_ep_ext, dist, op)
    starts = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(cfg.graph_replay_iters)
    ]
    mids = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(cfg.graph_replay_iters)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(cfg.graph_replay_iters)
    ]

    for i in range(cfg.graph_replay_iters):
        starts[i].record()
        _dispatch_output, _dispatch_weights, _dispatch_scales, dispatch_indices, _recv_count = (
            run_dispatch(op, inputs, cfg.launch))
        mids[i].record()
        run_combine(op, combine_input, dispatch_indices, cfg.launch)
        ends[i].record()
    torch.cuda.synchronize()

    dispatch_ms = sum(
        start.elapsed_time(mid)
        for start, mid in zip(starts, mids)
    ) / cfg.graph_replay_iters
    combine_ms = sum(
        mid.elapsed_time(end)
        for mid, end in zip(mids, ends)
    ) / cfg.graph_replay_iters

    reset_and_sync(torch, kiwi_ep_ext, dist, op)
    batch_start = torch.cuda.Event(enable_timing=True)
    batch_end = torch.cuda.Event(enable_timing=True)
    batch_start.record()
    for _ in range(cfg.graph_replay_iters):
        _dispatch_output, _dispatch_weights, _dispatch_scales, batch_indices, _recv_count = (
            run_dispatch(op, inputs, cfg.launch))
        run_combine(op, combine_input, batch_indices, cfg.launch)
    batch_end.record()
    torch.cuda.synchronize()
    e2e_ms = (
        batch_start.elapsed_time(batch_end)
        / cfg.graph_replay_iters
    )

    synchronize_all(torch, kiwi_ep_ext, dist)
    return build_metrics(
        cfg,
        total_recv_tokens=total_recv_tokens,
        dispatch_us=dispatch_ms * 1000.0,
        combine_us=combine_ms * 1000.0,
        e2e_us=e2e_ms * 1000.0,
        e2e_is_full_graph=False,
    )


def warmup(torch: Any, kiwi_ep_ext: Any, dist: Optional[Any], op: Any, inputs: BenchInputs, cfg: BenchConfig) -> None:
    for _ in range(cfg.warmup):
        reset_and_sync(torch, kiwi_ep_ext, dist, op)
        run_pair(torch, op, inputs, cfg)
        torch.cuda.synchronize()
    synchronize_all(torch, kiwi_ep_ext, dist)


def metric_to_tensor_values(metrics: RoundMetrics) -> list[float]:
    values = [0.0] * (METRIC_E2E_IS_FULL_GRAPH + 1)
    values[METRIC_DISPATCH_US] = metrics.dispatch_us
    values[METRIC_DISPATCH_BW_GBS] = metrics.dispatch_bw_gbs
    values[METRIC_COMBINE_US] = metrics.combine_us
    values[METRIC_COMBINE_BW_GBS] = metrics.combine_bw_gbs
    values[METRIC_E2E_US] = metrics.e2e_us
    values[METRIC_E2E_BW_GBS] = metrics.e2e_bw_gbs
    values[METRIC_DISPATCH_BYTES_MIB] = metrics.dispatch_bytes / (1024.0**2)
    values[METRIC_COMBINE_BYTES_MIB] = metrics.combine_bytes / (1024.0**2)
    values[METRIC_E2E_BYTES_MIB] = metrics.e2e_bytes / (1024.0**2)
    values[METRIC_RECV_TOKENS] = metrics.recv_tokens
    values[METRIC_SPLIT_E2E_US] = metrics.split_e2e_us
    values[METRIC_E2E_IS_FULL_GRAPH] = 1.0 if metrics.e2e_is_full_graph else 0.0
    return values


def gather_metrics(torch: Any, dist: Optional[Any], metrics: RoundMetrics, local_rank: int) -> list[list[float]]:
    values = metric_to_tensor_values(metrics)
    if not dist_is_ready(dist):
        return [values]

    backend = str(dist.get_backend()).lower()
    device = torch.device("cuda", local_rank) if "nccl" in backend else torch.device("cpu")
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return [item.cpu().tolist() for item in gathered]


def _column(round_values: list[list[float]], idx: int) -> list[float]:
    return [row[idx] for row in round_values]


def fmt_int_list(values: Iterable[float]) -> str:
    return "[" + ", ".join(str(int(round(v))) for v in values) + "]"


def fmt_float_list(values: Iterable[float], digits: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{digits}f}" for v in values) + "]"


COMBINE_RCOMP_DEBUG_LABELS = [
    "entry",
    "rcomp_combine_work_start",
    "block0_pack_post_done",
    "local_pack_barrier_wait_start",
    "combine_exit_barrier_posted_after_pack",
    "all_blocks_local_pack_ready",
    "accumulation_start",
    "block0_accumulation_done",
    "all_blocks_accumulation_done",
    "wait_for_combine_exit_barrier",
    "combine_exit_barrier_complete",
]

DISPATCH_RCOMP_DEBUG_LABELS = [
    "entry",
    "count_prescan_done",
    "count_all2all_started",
    "rcomp_rdma_posted",
    "rcomp_ipc_copies_complete",
    "incoming_count_all2all_wait_start",
    "incoming_counts_observed",
    "rank_slot_maps_built",
    "unpack_barrier_released",
    "unpack_done",
]

def dispatch_debug_labels() -> list[Optional[str]]:
    return DISPATCH_RCOMP_DEBUG_LABELS


def combine_debug_labels() -> list[Optional[str]]:
    return COMBINE_RCOMP_DEBUG_LABELS


def print_debug_timing(
    kiwi_ep_ext: Any,
    op: Any,
    cfg: BenchConfig,
    iter_idx: int,
    timing: str,
    *,
    label: str,
    phase_labels: list[Optional[str]],
) -> None:
    """Print the latest per-rank device phase timing outside timed regions.

    The diagnostics are enabled only by explicit debug timing CLI flags and are
    disabled by default.
    """

    for ordered_rank in range(cfg.world_size):
        kiwi_ep_ext.barrier()
        if cfg.rank != ordered_rank:
            continue

        snapshot = op.get_debug_timing_snapshot()
        time_ticks = [int(v) for v in snapshot.get("time_ticks", [])]
        meta = [int(v) for v in snapshot.get("meta", [])]
        wall_clock_rate_khz = int(snapshot.get("wall_clock_rate_khz", 0))
        if not snapshot.get("enabled", False) or not time_ticks or time_ticks[0] == 0:
            print(
                f"# py-{label}-debug rank={cfg.rank} iter={iter_idx} timing={timing} "
                "unavailable=1"
            )
            sys.stdout.flush()
            continue

        def meta_at(idx: int) -> int:
            return meta[idx] if idx < len(meta) else 0

        tick_us = 1000.0 / float(wall_clock_rate_khz) if wall_clock_rate_khz > 0 else 0.0
        print(
            f"# py-{label}-debug rank={cfg.rank} iter={iter_idx} timing={timing} "
            f"wall_clock_rate_khz={wall_clock_rate_khz} "
            f"meta={{rank:{meta_at(0)},world:{meta_at(1)},tokens:{meta_at(2)},"
            f"field3:{meta_at(3)},field4:{meta_at(4)},blocks:{meta_at(5)},"
            f"remote_recv:{meta_at(6)},self_recv:{meta_at(7)}}}"
        )

        limit = min(len(phase_labels), len(time_ticks))
        previous_idx = 0
        for idx in range(1, limit):
            phase_label = phase_labels[idx]
            if phase_label is None or time_ticks[idx] == 0:
                continue
            delta_ticks = time_ticks[idx] - time_ticks[previous_idx]
            cumulative_ticks = time_ticks[idx] - time_ticks[0]
            line = (
                f"# py-{label}-debug rank={cfg.rank} iter={iter_idx} "
                f"phase={phase_label} "
                f"delta_ticks={delta_ticks} cumulative_ticks={cumulative_ticks}"
            )
            if tick_us > 0.0:
                line += f" delta_us={delta_ticks * tick_us:.3f} cumulative_us={cumulative_ticks * tick_us:.3f}"
            print(line)
            previous_idx = idx
        sys.stdout.flush()

    kiwi_ep_ext.barrier()


def print_dispatch_debug_timing(kiwi_ep_ext: Any, op: Any, cfg: BenchConfig, iter_idx: int, timing: str) -> None:
    print_debug_timing(
        kiwi_ep_ext,
        op,
        cfg,
        iter_idx,
        timing,
        label="dispatch",
        phase_labels=dispatch_debug_labels(),
    )


def print_combine_debug_timing(kiwi_ep_ext: Any, op: Any, cfg: BenchConfig, iter_idx: int, timing: str) -> None:
    print_debug_timing(
        kiwi_ep_ext,
        op,
        cfg,
        iter_idx,
        timing,
        label="combine",
        phase_labels=combine_debug_labels(),
    )


def print_round_summaries(rounds: list[list[list[float]]], cfg: BenchConfig, timing: str) -> None:
    if not rounds:
        print("No benchmark rounds were run")
        return

    gathered_world = len(rounds[0])
    summary_is_global = gathered_world == cfg.world_size
    if not summary_is_global:
        print(
            "Warning: global per-rank metric gather is unavailable or incomplete "
            f"(gathered {gathered_world}/{cfg.world_size} rank(s)). "
            "The benchmark summary below is rank-local/incomplete; max-rank "
            "latency and avg-rank bandwidth are not valid multi-rank summaries. "
            "Set up torch.distributed or pass --torch-dist require to fail instead."
        )
        print()

    print("Dispatch result:")
    best_dispatch_algo_bw = 0.0
    min_dispatch_latency_us = float("inf")
    for i, values in enumerate(rounds):
        durations = _column(values, METRIC_DISPATCH_US)
        bws = _column(values, METRIC_DISPATCH_BW_GBS)
        bytes_mib = _column(values, METRIC_DISPATCH_BYTES_MIB)
        recv_tokens = _column(values, METRIC_RECV_TOKENS)
        lat = max(durations)
        algo_bw = sum(bws) / gathered_world
        avg_recv = sum(recv_tokens) / gathered_world
        ll_scale = (cfg.max_tokens * cfg.num_experts_per_token) / (avg_recv + 0.01)
        best_dispatch_algo_bw = max(best_dispatch_algo_bw, algo_bw)
        min_dispatch_latency_us = min(min_dispatch_latency_us, lat)
        print(
            f"Round {i} duration(us) {fmt_int_list(durations)} "
            f"bandwidth(GB/s) {fmt_float_list(bws)} "
            f"bytes(MiB) {fmt_float_list(bytes_mib)} "
            f"recv_tokens {fmt_int_list(recv_tokens)} "
            f"lat {lat:.0f} bw {algo_bw:.2f} / {algo_bw * ll_scale:.2f}"
        )

    print()
    print("Combine result:")
    best_combine_algo_bw = 0.0
    min_combine_latency_us = float("inf")
    for i, values in enumerate(rounds):
        durations = _column(values, METRIC_COMBINE_US)
        bws = _column(values, METRIC_COMBINE_BW_GBS)
        bytes_mib = _column(values, METRIC_COMBINE_BYTES_MIB)
        recv_tokens = _column(values, METRIC_RECV_TOKENS)
        lat = max(durations)
        algo_bw = sum(bws) / gathered_world
        avg_recv = sum(recv_tokens) / gathered_world
        ll_scale = (cfg.max_tokens * cfg.num_experts_per_token) / (avg_recv + 0.01)
        best_combine_algo_bw = max(best_combine_algo_bw, algo_bw)
        min_combine_latency_us = min(min_combine_latency_us, lat)
        print(
            f"Round {i} duration(us) {fmt_int_list(durations)} "
            f"bandwidth(GB/s) {fmt_float_list(bws)} "
            f"bytes(MiB) {fmt_float_list(bytes_mib)} "
            f"recv_tokens {fmt_int_list(recv_tokens)} "
            f"lat {lat:.0f} bw {algo_bw:.2f} / {algo_bw * ll_scale:.2f}"
        )

    print()
    print("End-to-end result:")
    if timing == "graph":
        if all(min(_column(values, METRIC_E2E_IS_FULL_GRAPH)) >= 1.0 for values in rounds):
            print(
                "Note: e2e is one replay of a graph containing "
                f"{cfg.graph_replay_iters} dispatch+combine pairs, divided by "
                f"{cfg.graph_replay_iters}; split results use separate graph replays."
            )
        else:
            print("Note: full e2e graph was unavailable on at least one round/rank; e2e falls back to split dispatch+combine sum.")
    else:
        print(
            "Note: eager e2e uses one event pair around "
            f"{cfg.graph_replay_iters} dispatch+combine iterations. "
            "Split dispatch/combine values use per-iteration events and include "
            "Python event-record pacing."
        )

    best_e2e_algo_bw = 0.0
    min_e2e_latency_us = float("inf")
    for i, values in enumerate(rounds):
        durations = _column(values, METRIC_E2E_US)
        split_durations = _column(values, METRIC_SPLIT_E2E_US)
        bws = _column(values, METRIC_E2E_BW_GBS)
        bytes_mib = _column(values, METRIC_E2E_BYTES_MIB)
        lat = max(durations)
        algo_bw = sum(bws) / gathered_world
        best_e2e_algo_bw = max(best_e2e_algo_bw, algo_bw)
        min_e2e_latency_us = min(min_e2e_latency_us, lat)
        print(
            f"Round {i} e2e(us) {fmt_int_list(durations)} "
            f"split_sum(us) {fmt_int_list(split_durations)} "
            f"bandwidth(GB/s) {fmt_float_list(bws)} "
            f"bytes(MiB) {fmt_float_list(bytes_mib)} "
            f"lat {lat:.0f} bw {algo_bw:.2f}"
        )

    print()
    print("Performance summary:")
    if not summary_is_global:
        print("  scope: rank-local/incomplete; not a valid multi-rank aggregate")
    print(
        f"  best dispatch avg-rank bandwidth: {best_dispatch_algo_bw:.2f} GB/s; "
        f"best max-rank latency: {min_dispatch_latency_us:.0f} us"
    )
    print(
        f"  best combine  avg-rank bandwidth: {best_combine_algo_bw:.2f} GB/s; "
        f"best max-rank latency: {min_combine_latency_us:.0f} us"
    )
    print(
        f"  best e2e      avg-rank bandwidth: {best_e2e_algo_bw:.2f} GB/s; "
        f"best max-rank latency: {min_e2e_latency_us:.0f} us"
    )


def run_bench(
    torch: Any,
    kiwi_ep_ext: Any,
    dist: Optional[Any],
    op: Any,
    inputs: BenchInputs,
    cfg: BenchConfig,
    timing: str,
) -> None:
    warmup(torch, kiwi_ep_ext, dist, op, inputs, cfg)
    rounds: list[list[list[float]]] = []
    for iter_idx in range(cfg.iters):
        if timing == "graph":
            metrics = run_graph_bench_once(torch, kiwi_ep_ext, dist, op, inputs, cfg, iter_idx)
        else:
            metrics = run_eager_bench_once(torch, kiwi_ep_ext, dist, op, inputs, cfg)
        rounds.append(gather_metrics(torch, dist, metrics, cfg.local_rank))
        if cfg.debug_timing and timing == "eager":
            # Eager timing leaves the combine snapshot in the shared debug
            # buffer.  Print it, then issue one untimed dispatch to capture the
            # dispatch snapshot without perturbing reported metrics.
            print_combine_debug_timing(kiwi_ep_ext, op, cfg, iter_idx, timing)
            synchronize_all(torch, kiwi_ep_ext, dist)
            reset_and_sync(torch, kiwi_ep_ext, dist, op)
            run_dispatch(op, inputs, cfg.launch)
            torch.cuda.synchronize()
            print_dispatch_debug_timing(kiwi_ep_ext, op, cfg, iter_idx, timing)
        synchronize_all(torch, kiwi_ep_ext, dist)

    if cfg.rank == 0:
        print_round_summaries(rounds, cfg, timing)


def run_stress(
    torch: Any,
    kiwi_ep_ext: Any,
    dist: Optional[Any],
    op: Any,
    inputs: BenchInputs,
    cfg: BenchConfig,
    timing: str,
) -> None:
    warmup(torch, kiwi_ep_ext, dist, op, inputs, cfg)
    if timing == "eager":
        for i in range(cfg.iters):
            if cfg.rank == 0:
                print(f"Stress round {i} begin (eager)")
            run_eager_bench_once(torch, kiwi_ep_ext, dist, op, inputs, cfg)
            if cfg.debug_timing:
                print_combine_debug_timing(kiwi_ep_ext, op, cfg, i, timing)
                synchronize_all(torch, kiwi_ep_ext, dist)
                reset_and_sync(torch, kiwi_ep_ext, dist, op)
                run_dispatch(op, inputs, cfg.launch)
                torch.cuda.synchronize()
                print_dispatch_debug_timing(kiwi_ep_ext, op, cfg, i, timing)
        return

    reset_and_sync(torch, kiwi_ep_ext, dist, op)
    dispatch_output, _dispatch_weights, _dispatch_scales, dispatch_indices, _recv_count = run_dispatch(op, inputs, cfg.launch)
    combine_input = make_synthetic_combine_input(torch, dispatch_output, cfg)
    validate_combine_input(combine_input, cfg)
    run_combine(op, combine_input, dispatch_indices, cfg.launch)
    torch.cuda.synchronize()
    synchronize_all(torch, kiwi_ep_ext, dist)
    graph = capture_e2e_graph(torch, op, inputs, combine_input, cfg, pairs=1)
    torch.cuda.synchronize()
    synchronize_all(torch, kiwi_ep_ext, dist)

    for i in range(cfg.iters):
        if cfg.rank == 0:
            print(f"Stress round {i} begin (graph e2e replay)")
        graph.replay()
        torch.cuda.synchronize()
        synchronize_all(torch, kiwi_ep_ext, dist)
        if cfg.debug_timing:
            print_combine_debug_timing(kiwi_ep_ext, op, cfg, i, timing)
            synchronize_all(torch, kiwi_ep_ext, dist)
            reset_and_sync(torch, kiwi_ep_ext, dist, op)
            run_dispatch(op, inputs, cfg.launch)
            torch.cuda.synchronize()
            print_dispatch_debug_timing(kiwi_ep_ext, op, cfg, i, timing)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark kiwi-ep dispatch/combine timing and bandwidth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cmd", choices=["bench", "stress"], default="bench", help="Run a benchmark or repeated stress replay")
    parser.add_argument(
        "--timing",
        choices=["graph", "eager"],
        default="graph",
        help=(
            "Timing mode; graph captures multiple dispatch+combine pairs and "
            "eager uses event-bracketed kernel launches"
        ),
    )
    dtype_choices = ["bf16", "bfloat16", "fp32", "float32", "fp8", "fp8_e4m3_fnuz", "fp8_e4m3", "fp4"]
    parser.add_argument(
        "--dispatch-dtype",
        dest="dispatch_dtype",
        default="bf16",
        choices=dtype_choices,
        help="Dispatch payload dtype",
    )
    parser.add_argument(
        "--combine-dtype",
        default="bf16",
        choices=dtype_choices,
        help="Combine payload dtype",
    )
    parser.add_argument("--max-tokens", type=int, default=8, help="Maximum input tokens per rank")
    parser.add_argument("--hidden-dim", type=int, default=7168, help="Logical hidden dimension")
    parser.add_argument(
        "--dispatch-scale-dim",
        type=int,
        default=0,
        help="Explicit dispatch scale vector dimension per token; 0 disables dispatch scales",
    )
    parser.add_argument(
        "--dispatch-scale-dtype",
        choices=["none", "float32", "e8m0"],
        default="none",
        help="Explicit dispatch scale dtype used to derive dispatch_scale_type_size",
    )
    parser.add_argument(
        "--combine-scale-dim",
        type=int,
        default=0,
        help="Explicit combine scale vector dimension per token; 0 disables combine scales",
    )
    parser.add_argument(
        "--combine-scale-dtype",
        choices=["none", "float32", "e8m0"],
        default="none",
        help="Explicit combine scale dtype used to derive combine_scale_type_size",
    )
    parser.add_argument("--world-size", type=int, default=None, help="EP world size; defaults to LCI rank count")
    parser.add_argument("--num-experts-per-rank", type=int, default=None, help="Experts per rank; defaults to 256 // world_size")
    parser.add_argument(
        "--num-experts-per-token",
        type=int,
        default=8,
        help="Top-k experts per token",
    )
    parser.add_argument(
        "--timing-batch-iters",
        "--graph-replay-iters",
        dest="graph_replay_iters",
        type=int,
        default=10,
        help=(
            "Dispatch+combine pairs captured in the amortized E2E graph; for "
            "eager timing, iterations per event-bracketed batch"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup dispatch+combine rounds")
    parser.add_argument("--iters", type=int, default=10, help="Measured rounds for bench; replay rounds for stress")
    parser.add_argument("--dispatch-block-num", type=int, default=None, help="MORI dispatch block_num compatibility knob")
    parser.add_argument("--dispatch-warp-per-block", type=int, default=None, help="MORI dispatch warp_per_block compatibility knob")
    parser.add_argument("--combine-block-num", type=int, default=None, help="MORI combine block_num compatibility knob")
    parser.add_argument("--combine-warp-per-block", type=int, default=None, help="MORI combine warp_per_block compatibility knob")
    parser.add_argument("--max-recv-total-tokens", type=int, default=0, help="Accepted for MORI CLI compatibility; ignored by kiwi-ep")
    parser.add_argument("--zero-copy", type=int, choices=[0, 1], default=1, help="Accepted for MORI CLI compatibility; kiwi-ep always uses its registered buffers")
    parser.add_argument("--quant-type", choices=["none", "fp8_direct_cast"], default="none", help="Accepted for MORI CLI compatibility; ignored by kiwi-ep")
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=0,
        help="Kernel grid block count for dispatch/combine; 0 selects the "
        "library's measured per-GPU/NIC recommendation, while a positive "
        "value overrides it",
    )
    parser.add_argument("--num-proxy-threads", type=int, default=4, help="Number of CPU proxy threads; one LCI device is allocated per proxy thread (num_devices is derived, not tunable)")
    parser.add_argument("--queue-capacity", type=int, default=2048, help="Per-queue ring size in 64-byte lines (power of two). Messages are line-aligned and a line carries up to 32 B of payload (a put descriptor uses 2 lines, a multicast put 6); each queue costs queue_capacity*64B pinned host memory")
    parser.add_argument(
        "--rcomp-backend",
        choices=["auto", "rdma-only", "hybrid"],
        default="auto",
        help=(
            "RCOMP payload backend; auto uses the library's measured "
            "per-GPU/NIC launch policy (apps/ep/ep_tuning.hpp)"
        ),
    )
    parser.add_argument(
        "--lci-device-name",
        default=None,
        help=(
            "LCI HCA/NIC device name. Defaults to KIWI_LCI_DEVICE_NAME or "
            "LCI_ATTR_DEVICE_NAME when set; otherwise auto-selects a GPU-local active HCA."
        ),
    )
    parser.add_argument("--seed", type=int, default=123, help="Device RNG seed for MORI-style input generation")
    parser.add_argument(
        "--debug-timeline",
        dest="debug_timing",
        action="store_true",
        help=(
            "Enable low-level, transport-specific dispatch and combine "
            "phase timeline diagnostics and "
            "print the latest per-rank phase deltas after each measured Python round."
        ),
    )
    parser.add_argument(
        "--torch-dist",
        choices=["auto", "never", "require"],
        default="auto",
        help="Use torch.distributed all_gather for per-rank summaries",
    )
    parser.add_argument("--torch-dist-backend", default="gloo", help="torch.distributed backend for auto initialization")
    parser.add_argument("--master-addr", default=None, help="MASTER_ADDR override for torch.distributed auto initialization")
    parser.add_argument("--master-port", default=None, help="MASTER_PORT override for torch.distributed auto initialization")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")
    if args.num_experts_per_token <= 0:
        raise ValueError("--num-experts-per-token must be positive")
    if args.graph_replay_iters <= 0:
        raise ValueError("--timing-batch-iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.num_blocks < 0:
        raise ValueError("--num-blocks must be non-negative")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    # Keep --help and py_compile independent of built GPU extensions.
    import torch
    import kiwi_ep_ext

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda/HIP is not available; build/run this benchmark on a ROCm GPU host")
    if args.timing == "graph" and not hasattr(torch.cuda, "CUDAGraph"):
        raise SystemExit("torch.cuda.CUDAGraph is unavailable; rerun with --timing eager for diagnostics")

    try:
        dispatch_scale = make_scale_spec(
            torch, args.dispatch_scale_dim, args.dispatch_scale_dtype,
            flag_prefix="--dispatch")
        combine_scale = make_scale_spec(
            torch, args.combine_scale_dim, args.combine_scale_dtype,
            flag_prefix="--combine")
    except ValueError as exc:
        parser.error(str(exc))

    dispatch_dtype = resolve_dtype(torch, args.dispatch_dtype)
    combine_dtype = resolve_dtype(torch, args.combine_dtype)

    launcher_rank = launcher_rank_from_env()
    pre_lci_local_rank = local_rank_from_env(launcher_rank, torch)
    if pre_lci_local_rank >= torch.cuda.device_count():
        raise SystemExit(
            f"launcher local rank {pre_lci_local_rank} has no visible GPU "
            f"(torch.cuda.device_count()={torch.cuda.device_count()})"
        )
    torch.cuda.set_device(pre_lci_local_rank)

    if args.lci_device_name:
        lci_device_name = args.lci_device_name
        lci_device_source = "--lci-device-name"
    elif os.environ.get("KIWI_LCI_DEVICE_NAME"):
        lci_device_name = os.environ["KIWI_LCI_DEVICE_NAME"]
        lci_device_source = "KIWI_LCI_DEVICE_NAME"
    elif os.environ.get("LCI_ATTR_DEVICE_NAME"):
        lci_device_name = os.environ["LCI_ATTR_DEVICE_NAME"]
        lci_device_source = "LCI_ATTR_DEVICE_NAME"
    else:
        lci_device_name = kiwi_ep_ext.get_hca_name_for_current_gpu()
        lci_device_source = "auto GPU-local get_hca_name_for_current_gpu()"
    lci_ctx = kiwi_ep_ext.LciContext(lci_device_name or "")
    rank = int(kiwi_ep_ext.get_rank())
    nranks = int(kiwi_ep_ext.get_nranks())
    world_size = args.world_size if args.world_size is not None else nranks
    if world_size != nranks:
        raise SystemExit(
            f"--world-size ({world_size}) must match the LCI rank count from the launcher ({nranks}) for kiwi-ep"
        )
    if rank >= world_size:
        raise SystemExit(f"LCI rank {rank} is outside --world-size {world_size}")

    local_rank = local_rank_from_env(rank, torch)
    if local_rank != pre_lci_local_rank:
        raise SystemExit(
            f"launcher rank/local-rank environment changed after LCI init: "
            f"pre-LCI local_rank={pre_lci_local_rank}, LCI rank={rank} maps to local_rank={local_rank}. "
            "Cannot safely auto-select a GPU-local NIC after LCI is initialized."
        )
    if local_rank >= torch.cuda.device_count():
        raise SystemExit(
            f"local rank {local_rank} has no visible GPU (torch.cuda.device_count()={torch.cuda.device_count()})"
        )
    torch.cuda.set_device(local_rank)

    if args.num_experts_per_rank is None:
        args.num_experts_per_rank = 256 // world_size
    total_experts = args.num_experts_per_rank * world_size
    if args.num_experts_per_token > total_experts:
        raise SystemExit(
            f"--num-experts-per-token ({args.num_experts_per_token}) exceeds total experts ({total_experts})"
        )

    dist, dist_owned, dist_status = maybe_init_torch_dist(torch, rank, world_size, args)
    lci_device_summaries, lci_device_summary_warning = collect_lci_device_summaries(
        torch, dist, rank, world_size, local_rank, lci_device_name,
        lci_device_source)
    launch = merge_launch_args(args, world_size)
    cfg = BenchConfig(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        max_tokens=args.max_tokens,
        model_hidden_dim=args.hidden_dim,
        dispatch_hidden_elems=dtype_storage_hidden_elems(dispatch_dtype, args.hidden_dim),
        combine_hidden_elems=dtype_storage_hidden_elems(combine_dtype, args.hidden_dim),
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=args.num_experts_per_token,
        dispatch_dtype=dispatch_dtype,
        combine_dtype=combine_dtype,
        dispatch_scale=dispatch_scale,
        combine_scale=combine_scale,
        launch=launch,
        graph_replay_iters=args.graph_replay_iters,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        debug_timing=args.debug_timing,
    )

    if rank == 0:
        print(
            f"Running kiwi-ep {args.cmd}: timing={args.timing}, max_tokens={cfg.max_tokens}, "
            f"model_hidden_dim={cfg.model_hidden_dim}, dispatch_hidden_elems={cfg.dispatch_hidden_elems}, "
            f"combine_hidden_elems={cfg.combine_hidden_elems}, "
            f"dispatch_dtype={dispatch_dtype.cli_name}({_torch_dtype_name(dispatch_dtype.torch_dtype)}, "
            f"dispatch_scale_dim={dispatch_scale.dim}, dispatch_scale_type_size={dispatch_scale.type_size}, "
            f"dispatch_scale_dtype={dispatch_scale.cli_name}), "
            f"combine_dtype={combine_dtype.cli_name}({_torch_dtype_name(combine_dtype.torch_dtype)}, "
            f"combine_scale_dim={combine_scale.dim}, combine_scale_type_size={combine_scale.type_size}, "
            f"combine_scale_dtype={combine_scale.cli_name}), "
            f"world_size={cfg.world_size}, experts_per_rank={cfg.num_experts_per_rank}, "
            f"topk={cfg.num_experts_per_token}, timing_batch_iters={cfg.graph_replay_iters}, "
            f"warmup={cfg.warmup}, iters={cfg.iters}, "
            f"num_blocks={args.num_blocks or 'auto'}, "
            f"num_proxy_threads={args.num_proxy_threads}, "
            f"rcomp_backend={args.rcomp_backend}, "
            "protocol=rcomp"
        )
        dispatch_payload = slot_payload_bytes(
            cfg.dispatch_hidden_elems, dispatch_dtype.elem_size, cfg.num_experts_per_token,
            dispatch_scale.dim, dispatch_scale.type_size)
        dispatch_stride = slot_stride_bytes(
            cfg.dispatch_hidden_elems, dispatch_dtype.elem_size, cfg.num_experts_per_token,
            dispatch_scale.dim, dispatch_scale.type_size)
        combine_payload = combine_slot_payload_bytes(
            cfg.combine_hidden_elems, combine_dtype.elem_size, cfg.num_experts_per_token,
        )
        combine_stride = combine_slot_stride_bytes(
            cfg.combine_hidden_elems, combine_dtype.elem_size, cfg.num_experts_per_token,
        )
        print(
            f"Launch config: dispatch_block_num={launch.dispatch_block_num}, "
            f"dispatch_warp_per_block={launch.dispatch_warp_per_block}, "
            f"combine_block_num={launch.combine_block_num}, "
            f"combine_warp_per_block={launch.combine_warp_per_block}"
        )
        print(
            "Byte accounting: "
            f"dispatch_slot_payload={dispatch_payload}B dispatch_slot_stride={dispatch_stride}B; "
            f"combine_slot_payload={combine_payload}B combine_slot_stride={combine_stride}B "
            "(slot strides are 64B aligned)"
        )
        print(f"torch.distributed: {dist_status}")
        print_lci_device_summaries(lci_device_summaries)
        if lci_device_summary_warning:
            print(f"Warning: {lci_device_summary_warning}")
        if args.debug_timing:
            print("Debug timing: enabled for dispatch and combine (per-rank phase lines print after each measured round)")
        if args.max_recv_total_tokens:
            print("Warning: --max-recv-total-tokens is accepted for MORI CLI compatibility but ignored by kiwi-ep")
        if args.zero_copy != 1:
            print("Warning: --zero-copy is accepted for MORI CLI compatibility but ignored by kiwi-ep")
        if args.quant_type != "none":
            print("Warning: --quant-type is accepted for MORI CLI compatibility but ignored by kiwi-ep")
        print(
            "Combine input uses a separate synthetic expert-output buffer; "
            "dispatch output is used only for routing/count metadata."
        )
        print("-" * 80)
    elif not dist_is_ready(dist):
        print_lci_device_summaries(lci_device_summaries)

    op = None
    try:
        op = kiwi_ep_ext.KiwiEpOp(
            device_id=local_rank,
            world_size=world_size,
            hidden_dim=args.hidden_dim,
            topk=args.num_experts_per_token,
            num_experts_per_rank=args.num_experts_per_rank,
            max_num_inp_token_per_rank=args.max_tokens,
            dispatch_dtype=dispatch_dtype.torch_dtype,
            combine_dtype=combine_dtype.torch_dtype,
            num_blocks=args.num_blocks,
            num_proxy_threads=args.num_proxy_threads,
            queue_capacity=args.queue_capacity,
            dispatch_elem_size=dispatch_dtype.elem_size,
            dispatch_scale_dim=dispatch_scale.dim,
            dispatch_scale_type_size=dispatch_scale.type_size,
            combine_scale_dim=combine_scale.dim,
            combine_scale_type_size=combine_scale.type_size,
            dispatch_scale_dtype=dispatch_scale.torch_dtype or torch.float32,
            device_name=lci_device_name or "",
            dispatch_quant_format=quant_format_for_dtype(dispatch_dtype),
            combine_quant_format=quant_format_for_dtype(combine_dtype),
            rcomp_backend=args.rcomp_backend,
        )
        if rank == 0:
            policy = op.launch_policy(args.max_tokens)
            print(
                f"Launch policy: num_blocks={policy['num_blocks']}, "
                f"rcomp_backend={policy['rcomp_backend']}, "
                f"gpu_arch={policy['gpu_arch'] or 'unknown'}, "
                f"nic_family={policy['nic_family'] or 'unknown'}"
            )
        if args.debug_timing:
            op.set_debug_timing_enabled(True)
        inputs = generate_rank_inputs(torch, cfg)
        synchronize_all(torch, kiwi_ep_ext, dist)
        if args.cmd == "bench":
            run_bench(torch, kiwi_ep_ext, dist, op, inputs, cfg, args.timing)
        else:
            run_stress(torch, kiwi_ep_ext, dist, op, inputs, cfg, args.timing)
        synchronize_all(torch, kiwi_ep_ext, dist)
    finally:
        if op is not None:
            del op
        if dist_owned and dist_is_ready(dist):
            dist.destroy_process_group()
        del lci_ctx

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
