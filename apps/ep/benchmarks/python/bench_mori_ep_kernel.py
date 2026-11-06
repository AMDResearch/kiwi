#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Single-node wrapper around official MORI EP benchmark classes.

This helper intentionally depends on an official MORI checkout/install.  It
reuses ``tests.python.ops.bench_dispatch_combine.EpDispatchCombineBenchmark``
inside one host while exposing a concise kernel-type switch for side-by-side
kiwi-ep experiments.

For internode/v1_ll comparison, prefer MORI's official example
``examples/ops/dispatch_combine/test_dispatch_combine_internode.py``.  That
example accepts ``--kernel-type v0|v1|v1_ll|async_ll`` whereas the stock
``tests/python/ops/bench_dispatch_combine.py`` script does not expose such a
CLI switch in the installed tree.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from typing import Any


KERNEL_TYPE_ALIASES = {
    "internode": "internode",
    "v0": "internode",
    "internode_v1": "internode_v1",
    "v1": "internode_v1",
    "internode_v1ll": "internode_v1ll",
    "internodell": "internode_v1ll",
    "internode_ll": "internode_v1ll",
    "v1_ll": "internode_v1ll",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MORI's single-node EP dispatch/combine benchmark classes with an explicit kernel type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mori-source-dir", default=os.environ.get("MORI_SOURCE_DIR"), help="Official MORI source checkout; adds <dir> and <dir>/python to sys.path")
    parser.add_argument(
        "--kernel-type",
        type=lambda value: value.strip().lower(),
        choices=sorted(KERNEL_TYPE_ALIASES),
        required=True,
        help=(
            "MORI EP kernel type alias. Mapping: internode/v0 -> InterNode; "
            "internode_v1/v1 -> InterNodeV1; internode_v1ll/internodeLL/v1_ll -> InterNodeV1LL."
        ),
    )
    parser.add_argument("--cmd", choices=["bench", "stress"], default="bench")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--dispatch-dtype", choices=["bf16", "fp8_e4m3_fnuz", "fp8_e4m3", "fp4"], default="bf16")
    parser.add_argument("--combine-dtype", choices=["bf16", "fp8_e4m3_fnuz", "fp8_e4m3", "fp4"], default="bf16")
    parser.add_argument("--hidden-dim", type=int, default=7168, help="Logical hidden dimension")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--num-experts-per-rank", type=int, default=None)
    parser.add_argument("--num-experts-per-token", type=int, default=8)
    parser.add_argument("--graph-replay-iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--zero-copy", type=int, choices=[0, 1], default=1)
    parser.add_argument("--quant-type", choices=["none", "fp8_direct_cast"], default="none")
    parser.add_argument("--gpu-per-node", type=int, default=None, help="MORI gpu_per_node; defaults to world_size")
    parser.add_argument("--rdma-block-num", type=int, default=None, help="Override MORI rdma_block_num for v1 kernels")
    parser.add_argument("--dispatch-block-num", type=int, default=None)
    parser.add_argument("--dispatch-warp-per-block", type=int, default=None)
    parser.add_argument("--combine-block-num", type=int, default=None)
    parser.add_argument("--combine-warp-per-block", type=int, default=None)
    parser.add_argument("--max-recv-total-tokens", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=0, help="Torch distributed port; 0 chooses a free local port before spawning")
    parser.add_argument("--heap-size", default=os.environ.get("MORI_SHMEM_HEAP_SIZE", "6G"), help="MORI_SHMEM_HEAP_SIZE")
    return parser


def add_mori_to_path(mori_source_dir: str | None) -> None:
    if not mori_source_dir:
        return
    source = os.path.abspath(os.path.expanduser(mori_source_dir))
    python_dir = os.path.join(source, "python")
    for path in (python_dir, source):
        if path not in sys.path:
            sys.path.insert(0, path)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def kernel_defaults(kernel_type: str) -> tuple[int, int, int]:
    """Return (block_num, warp_num_per_block, rdma_block_num)."""

    if kernel_type == "internode":
        return 16, 16, 0
    if kernel_type == "internode_v1":
        return 96, 8, 64
    if kernel_type == "internode_v1ll":
        return 256, 8, 128
    raise ValueError(f"unknown kernel_type {kernel_type}")


def worker(rank: int, world_size: int, raw_args: dict[str, Any]) -> None:
    add_mori_to_path(raw_args.get("mori_source_dir"))
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(raw_args["heap_size"]))

    import torch
    import mori
    from tests.python.ops.bench_dispatch_combine import (
        EpDispatchCombineBenchmark,
        _DATA_TYPE_MAP,
        _get_default_launch_config,
    )
    from tests.python.ops.dispatch_combine_test_utils import _is_fp4x2_dtype
    from tests.python.utils import TorchDistContext

    kernel_map = {
        "internode": mori.ops.EpDispatchCombineKernelType.InterNode,
        "internode_v1": mori.ops.EpDispatchCombineKernelType.InterNodeV1,
        "internode_v1ll": mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
    }
    dispatch_data_type = _DATA_TYPE_MAP[raw_args["dispatch_dtype"]]
    combine_data_type = _DATA_TYPE_MAP[raw_args["combine_dtype"]]
    dispatch_is_fp4 = _is_fp4x2_dtype(dispatch_data_type)
    combine_is_fp4 = _is_fp4x2_dtype(combine_data_type)
    if dispatch_is_fp4 != combine_is_fp4:
        raise ValueError(
            "Unsupported official MORI benchmark dtype combination: "
            "FP4/non-FP4 cross-type combine would require packed/unpacked "
            f"shape conversion (dispatch_dtype={raw_args['dispatch_dtype']}, "
            f"combine_dtype={raw_args['combine_dtype']})"
        )

    model_hidden_dim = raw_args["hidden_dim"]
    dispatch_hidden_elems = model_hidden_dim // 2 if dispatch_is_fp4 else model_hidden_dim
    combine_hidden_elems = model_hidden_dim // 2 if combine_is_fp4 else model_hidden_dim

    block_num, warp_num_per_block, rdma_block_num = kernel_defaults(raw_args["kernel_type"])
    if raw_args.get("rdma_block_num") is not None:
        rdma_block_num = int(raw_args["rdma_block_num"])

    # MORI's public config keeps generic names: `data_type` is the dispatch
    # input payload dtype, while `hidden_dim` is the logical model hidden size.
    # The benchmark keeps phase-specific dispatch_hidden_elems/combine_hidden_elems separately.
    config = mori.ops.EpDispatchCombineConfig(
        data_type=dispatch_data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=model_hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=raw_args["max_tokens"],
        num_experts_per_rank=raw_args["num_experts_per_rank"],
        num_experts_per_token=raw_args["num_experts_per_token"],
        max_total_recv_tokens=raw_args["max_recv_total_tokens"],
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        use_external_inp_buf=not bool(raw_args["zero_copy"]),
        gpu_per_node=raw_args["gpu_per_node"],
        kernel_type=kernel_map[raw_args["kernel_type"]],
        quant_type=raw_args["quant_type"],
    )

    with TorchDistContext(rank=rank, world_size=world_size, master_port=raw_args["master_port"]):
        mori.shmem.shmem_torch_process_group_init("default")
        op = mori.ops.EpDispatchCombineOp(config)
        benchmark = EpDispatchCombineBenchmark(
            config,
            combine_data_type=combine_data_type,
            combine_hidden_dim=combine_hidden_elems,
            dispatch_hidden_dim=dispatch_hidden_elems,
        )
        launch = _get_default_launch_config(
            world_size=world_size,
            max_num_inp_token_per_rank=raw_args["max_tokens"],
            use_external_inp_buf=config.use_external_inp_buf,
        )
        dispatch_block_num = raw_args["dispatch_block_num"] or launch.dispatch_block_num
        dispatch_warp_per_block = raw_args["dispatch_warp_per_block"] or launch.dispatch_warp_per_block
        combine_block_num = raw_args["combine_block_num"] or launch.combine_block_num
        combine_warp_per_block = raw_args["combine_warp_per_block"] or launch.combine_warp_per_block

        if rank == 0:
            print(
                f"Running official MORI {raw_args['cmd']}: kernel_type={raw_args['kernel_type']}, "
                f"max_tokens={raw_args['max_tokens']}, "
                f"model_hidden_dim={model_hidden_dim}, "
                f"dispatch_hidden_elems={dispatch_hidden_elems}, combine_hidden_elems={combine_hidden_elems}, "
                f"dispatch_dtype={raw_args['dispatch_dtype']}, "
                f"combine_dtype={raw_args['combine_dtype']}, "
                f"world_size={world_size}, "
                f"experts_per_rank={raw_args['num_experts_per_rank']}, topk={raw_args['num_experts_per_token']}, "
                f"gpu_per_node={raw_args['gpu_per_node']}, rdma_block_num={rdma_block_num}"
            )
            print(
                f"Launch config: dispatch_block_num={dispatch_block_num}, "
                f"dispatch_warp_per_block={dispatch_warp_per_block}, "
                f"combine_block_num={combine_block_num}, combine_warp_per_block={combine_warp_per_block}"
            )
            print("-" * 80)

        if raw_args["cmd"] == "bench":
            benchmark.run(
                op,
                dispatch_block_num=dispatch_block_num,
                dispatch_warp_per_block=dispatch_warp_per_block,
                combine_block_num=combine_block_num,
                combine_warp_per_block=combine_warp_per_block,
                warmup=raw_args["warmup"],
                iters=raw_args["iters"],
                graph_replay_iters=raw_args["graph_replay_iters"],
            )
        else:
            benchmark.stress(
                op,
                dispatch_block_num=dispatch_block_num,
                dispatch_warp_per_block=dispatch_warp_per_block,
                combine_block_num=combine_block_num,
                combine_warp_per_block=combine_warp_per_block,
            )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.kernel_type = KERNEL_TYPE_ALIASES[args.kernel_type.lower()]
    if args.num_experts_per_rank is None:
        args.num_experts_per_rank = 256 // args.world_size
    if args.gpu_per_node is None:
        args.gpu_per_node = args.world_size
    if args.master_port == 0:
        args.master_port = free_port()
    add_mori_to_path(args.mori_source_dir)
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(args.heap_size))

    import torch

    torch.multiprocessing.spawn(
        worker,
        args=(args.world_size, vars(args)),
        nprocs=args.world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
