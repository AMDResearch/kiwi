#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Correctness smoke test for kiwi-ep under torch.cuda.CUDAGraph.

The graph replay still relies on the normal always-running host proxy threads to
execute LCI puts.  The graph-compatible contract tested here is that dispatch
and combine themselves enqueue only stream work during capture/replay, and GPU
kernels observe proxy/LCI completion through GPU-visible state.
"""

import gc
import os
import sys
import time
import weakref


def skip(msg: str) -> None:
    print(f"SKIP: {msg}", file=sys.stderr)
    sys.exit(77)


try:
    import torch
except Exception as exc:  # pragma: no cover - depends on test host
    skip(f"PyTorch unavailable: {exc}")

try:
    import kiwi_ep_ext
except Exception as exc:  # pragma: no cover - depends on build config
    skip(f"kiwi_ep_ext unavailable: {exc}")


if not torch.cuda.is_available():
    skip("HIP device unavailable")
if not hasattr(torch.cuda, "CUDAGraph"):
    skip("torch.cuda.CUDAGraph unavailable")


def env_int(*names: str, default: int = 0) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def make_inputs(device, rank: int, world_size: int, num_tokens: int,
                hidden_dim: int, topk: int, experts_per_rank: int):
    base = torch.arange(num_tokens * hidden_dim, dtype=torch.float32,
                        device=device).reshape(num_tokens, hidden_dim)
    tokens = base + rank * 100000.0

    total_experts = world_size * experts_per_rank
    indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    for t in range(num_tokens):
        for k in range(topk):
            dest_rank = (rank + k) % world_size
            local_expert = (t + k) % experts_per_rank
            indices[t, k] = (dest_rank * experts_per_rank + local_expert) % total_experts

    weights = torch.ones((num_tokens, topk), dtype=torch.float32, device=device)
    scales = torch.empty((0,), dtype=torch.float32, device=device)
    return tokens.contiguous(), weights.contiguous(), scales, indices.contiguous()


def run_round(op, tokens, weights, scales, indices, combine_out=None):
    out, _out_w, _out_s, out_idx, recv_count = op.dispatch(
        tokens, weights, scales, indices, block_num=64, warp_per_block=4)
    if combine_out is None:
        combined, _ = op.combine(
            out, None, out_idx, block_num=64, warp_per_block=4)
    else:
        combined, _ = op.combine(
            out, None, out_idx, out=combine_out,
            block_num=64, warp_per_block=4)
    return recv_count, combined


def assert_combine_rejected(op, expert_output, indices, output, expected):
    try:
        op.combine(expert_output, None, indices, out=output)
    except RuntimeError as exc:
        if expected not in str(exc):
            raise AssertionError(
                f"expected {expected!r} in combine error, got {exc!s}") from exc
    else:
        raise AssertionError(f"combine unexpectedly accepted invalid out: {expected}")


def main() -> int:
    global_rank = env_int("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", default=0)
    local_rank = env_int("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
                         "MPI_LOCALRANKID", "SLURM_LOCALID", default=0)
    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        skip(f"global rank {global_rank}: resolved local rank {local_rank} is "
             f"outside {device_count} visible HIP/CUDA devices")

    # Keep all local GPUs visible: EP resolves same-node peers by PCI BDF.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    current_device = torch.cuda.current_device()
    pci_bdf = getattr(
        torch.cuda.get_device_properties(current_device), "pci_bus_id", "<unknown>")
    print(
        f"[Rank {global_rank}] device mapping: global_rank={global_rank} "
        f"local_rank={local_rank} visible_devices={device_count} "
        f"current_hip_cuda_device={current_device} pci_bdf={pci_bdf}",
        file=sys.stderr,
    )

    lci_ctx = kiwi_ep_ext.LciContext()
    rank = kiwi_ep_ext.get_rank()
    world_size = kiwi_ep_ext.get_nranks()
    if world_size < 2:
        skip("CUDAGraph EP test requires at least 2 MPI ranks")
    num_tokens = 8
    hidden_dim = 64
    topk = min(2, world_size)
    experts_per_rank = 2
    max_num_inp_token_per_rank = 16

    op = kiwi_ep_ext.KiwiEpOp(
        device_id=local_rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        topk=topk,
        num_experts_per_rank=experts_per_rank,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        dispatch_dtype=torch.float32,
        combine_dtype=torch.float32,
        num_blocks=4,
        num_proxy_threads=4,
        dispatch_elem_size=4,
    )

    tokens, weights, scales, indices = make_inputs(
        device, rank, world_size, num_tokens, hidden_dim, topk, experts_per_rank)

    # Validate caller-owned outputs before any distributed work is enqueued.
    # The capacity check below needs a real dispatch count and is covered after
    # the eager direct-output round.
    for invalid_output, expected in (
        (
            torch.empty((num_tokens, hidden_dim), dtype=torch.float32),
            "must be a CUDA tensor",
        ),
        (
            torch.empty(
                (num_tokens, hidden_dim), dtype=torch.float16, device=device),
            "dtype must match",
        ),
        (
            torch.empty(
                (num_tokens, hidden_dim + 1),
                dtype=torch.float32,
                device=device,
            ),
            "must have shape",
        ),
        (
            torch.empty(
                (hidden_dim, num_tokens), dtype=torch.float32, device=device
            ).transpose(0, 1),
            "must be contiguous",
        ),
    ):
        assert_combine_rejected(op, tokens, indices, invalid_output, expected)

    if torch.cuda.device_count() > 1:
        wrong_device = torch.device(
            "cuda", (local_rank + 1) % torch.cuda.device_count())
        assert_combine_rejected(
            op,
            tokens,
            indices,
            torch.empty(
                (num_tokens, hidden_dim), dtype=torch.float32, device=wrong_device
            ),
            "must be on the KiwiEpOp device",
        )

    # Eager reference on the same operator. reset() is outside capture; captured
    # dispatch/combine must do their own graph-safe per-round local setup.
    kiwi_ep_ext.barrier()
    op.reset()
    kiwi_ep_ext.barrier()
    ref_count, ref_combined = run_round(op, tokens, weights, scales, indices)
    torch.cuda.synchronize()
    # MORI returns recv_count as an alias of operator-owned storage.  kiwi-ep
    # follows that performance-oriented lifetime contract in eager and graph
    # mode, so tests snapshot the scalar before any later dispatch/reset can
    # legitimately reuse the internal count buffer.
    ref_count_cpu = int(ref_count.cpu()[0].item())
    ref_combined_cpu = ref_combined[:num_tokens].detach().cpu()

    # Caller-provided eager output must be used directly and remain available
    # through the split combine_recv API.
    kiwi_ep_ext.barrier()
    op.reset()
    kiwi_ep_ext.barrier()
    direct_output = torch.empty_like(tokens)
    (
        direct_expert_output,
        _direct_weights,
        _direct_scales,
        direct_indices,
        direct_count,
    ) = op.dispatch(tokens, weights, scales, indices)
    direct_combined, _ = op.combine(
        direct_expert_output, None, direct_indices, out=direct_output)
    torch.cuda.synchronize()
    if direct_combined.data_ptr() != direct_output.data_ptr():
        raise AssertionError("combine did not return the caller-provided output")
    combine_recv_output, _ = op.combine_recv()
    if combine_recv_output.data_ptr() != direct_output.data_ptr():
        raise AssertionError("combine_recv did not return the caller-provided output")
    direct_count_cpu = int(direct_count.cpu()[0].item())
    if direct_count_cpu != ref_count_cpu:
        raise AssertionError(
            f"direct-output recv_count mismatch "
            f"got={direct_count_cpu} expected={ref_count_cpu}")
    if not torch.equal(direct_output[:num_tokens].detach().cpu(), ref_combined_cpu):
        raise AssertionError("direct-output eager result differs from reference")

    assert_combine_rejected(
        op,
        direct_expert_output,
        direct_indices,
        torch.empty(
            (num_tokens - 1, hidden_dim), dtype=torch.float32, device=device
        ),
        "token capacity",
    )

    # Start graph capture from a clean globally synchronized point.  Do not put
    # reset/barrier inside the graph; the replay path must be self-contained.
    kiwi_ep_ext.barrier()
    op.reset()
    kiwi_ep_ext.barrier()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _warmup_count, _warmup_combined = run_round(op, tokens, weights, scales, indices)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    if _warmup_combined.data_ptr() != ref_combined.data_ptr():
        raise AssertionError("combine must reuse its operator-owned output buffer")

    kiwi_ep_ext.barrier()
    op.reset()
    kiwi_ep_ext.barrier()
    torch.cuda.synchronize()

    def capture_direct_output_graph():
        graph_output = torch.empty_like(tokens)
        graph_output_ptr = graph_output.data_ptr()
        graph_output_ref = weakref.ref(graph_output)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                graph_count, graph_combined = run_round(
                    op, tokens, weights, scales, indices, graph_output)
        except Exception as exc:
            print(f"[Rank {rank}] graph capture failed: {exc}", file=sys.stderr)
            return None
        if graph_combined.data_ptr() != graph_output_ptr:
            raise AssertionError(
                "captured combine did not return the caller-provided output")
        return graph, graph_count, graph_output_ref

    captured = capture_direct_output_graph()
    if captured is None:
        return 1
    graph, graph_count, graph_output_ref = captured
    gc.collect()
    if graph_output_ref() is None:
        raise AssertionError(
            "combine did not retain the caller-provided graph output")

    # Replays intentionally have no host-side inter-rank barrier between them.
    # Skew alternating ranks so rcomp's combine-exit guard must hold the fast
    # rank's prior round until the slow peer reaches its matching safe point.
    for replay in range(6):
        if rank == (replay % min(world_size, 2)):
            time.sleep(0.05)
        try:
            graph.replay()
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"[Rank {rank}] graph replay {replay} failed: {exc}", file=sys.stderr)
            return 1

        got_count = int(graph_count.cpu()[0].item())
        if got_count != ref_count_cpu:
            print(f"[Rank {rank}] replay {replay}: recv_count mismatch "
                  f"got={got_count} expected={ref_count_cpu}", file=sys.stderr)
            return 1
        graph_output = graph_output_ref()
        if graph_output is None:
            raise AssertionError(
                "caller-provided graph output was released before replay")
        got_combined = graph_output[:num_tokens].detach().cpu()
        if not torch.equal(got_combined, ref_combined_cpu):
            max_diff = (got_combined - ref_combined_cpu).abs().max().item()
            print(f"[Rank {rank}] replay {replay}: combined output mismatch "
                  f"max_diff={max_diff}", file=sys.stderr)
            return 1

    kiwi_ep_ext.barrier()
    if rank == 0:
        print("kiwi-ep CUDAGraph dispatch/combine correctness: PASSED")
    del op
    del lci_ctx
    return 0


if __name__ == "__main__":
    sys.exit(main())
