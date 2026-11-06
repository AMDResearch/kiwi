# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Reusable helpers for KIWI-EP Python benchmark adapters.

This module intentionally has no dependency on ``kiwi_ep_ext`` and imports no
``torch`` module at import time.  Callers pass their torch module/object into
helpers that need dtype or tensor construction, which keeps simple import checks
usable on hosts without the built extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DTypeSpec:
    """Runtime dtype metadata used by KIWI-EP benchmark helpers."""

    cli_name: str
    torch_dtype: Any
    elem_size: int

    @property
    def is_fp4(self) -> bool:
        return self.cli_name == "fp4"


@dataclass(frozen=True)
class ScaleSpec:
    """Explicit numeric scale layout for one EP phase."""

    dim: int
    type_size: int
    torch_dtype: Any
    cli_name: str


@dataclass(frozen=True)
class LaunchConfig:
    dispatch_block_num: int
    dispatch_warp_per_block: int
    combine_block_num: int
    combine_warp_per_block: int


@dataclass
class BenchInputs:
    tokens: Any
    weights: Any
    scales: Any
    indices: Any


def resolve_dtype(torch: Any, name: str) -> DTypeSpec:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "f32": "fp32",
        "fp8": "fp8_e4m3_fnuz",
        "fp8_fnuz": "fp8_e4m3_fnuz",
        "fp8_e4m3fnuz": "fp8_e4m3_fnuz",
        "fp8_e4m3_fnuz": "fp8_e4m3_fnuz",
        "fp8_e4m3fn": "fp8_e4m3",
        "fp8_e4m3": "fp8_e4m3",
        "fp4": "fp4",
    }
    key = aliases.get(key, key)

    if key == "bf16":
        return DTypeSpec("bf16", torch.bfloat16, 2)
    if key == "fp32":
        return DTypeSpec("fp32", torch.float32, 4)
    if key == "fp8_e4m3_fnuz":
        dtype = getattr(torch, "float8_e4m3fnuz", None)
        if dtype is None:
            raise ValueError("this PyTorch build does not expose torch.float8_e4m3fnuz")
        return DTypeSpec("fp8_e4m3_fnuz", dtype, 1)
    if key == "fp8_e4m3":
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise ValueError("this PyTorch build does not expose torch.float8_e4m3fn")
        return DTypeSpec("fp8_e4m3", dtype, 1)
    if key == "fp4":
        dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if dtype is None:
            raise ValueError("this PyTorch build does not expose torch.float4_e2m1fn_x2")
        return DTypeSpec("fp4", dtype, 1)

    raise ValueError(f"unsupported dtype {name!r}")


def dtype_storage_hidden_elems(dtype: DTypeSpec, logical_hidden_dim: int) -> int:
    if dtype.is_fp4:
        if logical_hidden_dim % 2:
            raise ValueError("FP4 logical --hidden-dim must be even because storage is packed two values per byte")
        return logical_hidden_dim // 2
    return logical_hidden_dim


def quant_format_for_dtype(dtype: DTypeSpec) -> str:
    if dtype.is_fp4:
        return "fp4"
    if dtype.cli_name.startswith("fp8"):
        return "fp8"
    return "none"


def resolve_scale_dtype(torch: Any, name: str) -> tuple[str, Any, int]:
    key = name.strip().lower().replace("-", "_")
    if key in ("none", "no", "null"):
        return "none", None, 0
    if key in ("float32", "fp32", "f32"):
        return "float32", torch.float32, 4
    if key == "e8m0":
        for attr in ("float8_e8m0fnu", "float8_e8m0fnuz"):
            dtype = getattr(torch, attr, None)
            if dtype is not None:
                return "e8m0", dtype, 1
        # Older PyTorch builds may not expose the shell dtype, but E8M0 scales
        # are a one-byte explicit layout. Use uint8 storage rather than
        # deriving anything from the payload dtype.
        return "e8m0", torch.uint8, 1
    raise ValueError(f"unsupported scale dtype {name!r}")


def make_scale_spec(torch: Any, dim: int, dtype_name: str, *, flag_prefix: str) -> ScaleSpec:
    if dim < 0:
        raise ValueError(f"{flag_prefix}-scale-dim must be non-negative")
    cli_name, torch_dtype, type_size = resolve_scale_dtype(torch, dtype_name)
    if dim == 0:
        return ScaleSpec(dim=0, type_size=0, torch_dtype=None, cli_name="none")
    if cli_name == "none":
        raise ValueError(f"{flag_prefix}-scale-dtype must not be 'none' when scale dim is nonzero")
    return ScaleSpec(dim=dim, type_size=type_size, torch_dtype=torch_dtype, cli_name=cli_name)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def slot_payload_bytes(physical_hidden_elems: int, elem_size: int, topk: int, scale_dim: int, scale_type_size: int) -> int:
    return (
        physical_hidden_elems * elem_size
        + topk * 4      # int32 expert indices
        + topk * 4      # float32 routing weights
        + scale_dim * scale_type_size
        + 4             # int32 source token id
    )


def slot_stride_bytes(physical_hidden_elems: int, elem_size: int, topk: int, scale_dim: int, scale_type_size: int) -> int:
    return align_up(slot_payload_bytes(physical_hidden_elems, elem_size, topk, scale_dim, scale_type_size), 64)


def combine_slot_payload_bytes(physical_hidden_elems: int, elem_size: int, topk: int) -> int:
    return physical_hidden_elems * elem_size + topk * 4


def combine_slot_stride_bytes(physical_hidden_elems: int, elem_size: int, topk: int) -> int:
    return align_up(combine_slot_payload_bytes(physical_hidden_elems, elem_size, topk), 64)


def default_launch_config(world_size: int, max_tokens: int) -> LaunchConfig:
    """Use MORI's benchmark defaults as compatibility launch metadata.

    kiwi-ep currently ignores these launch kwargs in the binding, but accepting
    and printing MORI-like values makes command lines comparable.
    """

    if world_size <= 4:
        if max_tokens > 128:
            return LaunchConfig(768, 8, 72, 4)
        if max_tokens > 64:
            return LaunchConfig(216, 6, 72, 4)
        return LaunchConfig(223, 6, 72, 4)
    if max_tokens > 1024:
        return LaunchConfig(80, 16, 80, 4)
    return LaunchConfig(64, 16, 64, 4)


def make_indices_for_rank(torch: Any, num_tokens: int, topk: int, total_experts: int, device: Any, rng: Any) -> Any:
    indices = torch.empty((num_tokens, topk), dtype=torch.int64, device=device)
    for token in range(num_tokens):
        perm = torch.randperm(total_experts, generator=rng, device=device)
        indices[token] = perm[:topk]
    return indices.to(torch.int32)


def make_input_for_rank(torch: Any, num_tokens: int, physical_hidden_elems: int, dtype: DTypeSpec, device: Any, rng: Any) -> Any:
    if dtype.is_fp4:
        raw = torch.randint(0, 256, (num_tokens, physical_hidden_elems), dtype=torch.uint8, device=device, generator=rng)
        return raw.view(dtype.torch_dtype)
    base = torch.randn((num_tokens, physical_hidden_elems), dtype=torch.float32, device=device, generator=rng)
    if dtype.torch_dtype is torch.float32:
        return base
    return base.to(dtype.torch_dtype)


def make_scales_for_rank(torch: Any, num_tokens: int, scale: ScaleSpec, device: Any, rng: Any) -> Any:
    if scale.dim <= 0:
        return torch.empty((num_tokens, 0), dtype=torch.float32, device=device)
    scale_dtype = scale.torch_dtype or torch.float32
    if scale_dtype is torch.uint8:
        return torch.randint(0, 256, (num_tokens, scale.dim), dtype=torch.uint8, device=device, generator=rng)
    return torch.rand((num_tokens, scale.dim), dtype=torch.float32, device=device, generator=rng).to(scale_dtype)


def dispatch_kwargs(launch: LaunchConfig) -> dict[str, int]:
    return {
        "block_num": launch.dispatch_block_num,
        "warp_per_block": launch.dispatch_warp_per_block,
    }


def combine_kwargs(launch: LaunchConfig) -> dict[str, int]:
    return {
        "block_num": launch.combine_block_num,
        "warp_per_block": launch.combine_warp_per_block,
    }


def run_dispatch(op: Any, inputs: BenchInputs, launch: LaunchConfig) -> tuple[Any, Any, Any, Any, Any]:
    return op.dispatch(
        inputs.tokens,
        inputs.weights,
        inputs.scales,
        inputs.indices,
        **dispatch_kwargs(launch),
    )


def make_synthetic_combine_input(
    torch: Any,
    dispatch_output: Any,
    cfg: Optional[Any] = None,
    *,
    combine_hidden_elems: Optional[int] = None,
    combine_dtype: Optional[DTypeSpec] = None,
    seed: Optional[int] = None,
    rank: int = 0,
) -> Any:
    """Create independent expert-output rows after dispatch fixed routing metadata.

    ``cfg`` may be any object exposing ``combine_hidden_elems``,
    ``combine_dtype``, ``seed``, and optionally ``rank``.  The explicit keyword
    form is provided for external adapters that do not use the benchmark's full
    config object.
    """

    if cfg is not None:
        combine_hidden_elems = getattr(cfg, "combine_hidden_elems")
        combine_dtype = getattr(cfg, "combine_dtype")
        seed = getattr(cfg, "seed")
        rank = getattr(cfg, "rank", rank)
    if combine_hidden_elems is None or combine_dtype is None or seed is None:
        raise ValueError("combine_hidden_elems, combine_dtype, and seed are required when cfg is not provided")

    device = dispatch_output.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed + 1_000_003 + rank)
    return make_input_for_rank(
        torch,
        dispatch_output.size(0),
        combine_hidden_elems,
        combine_dtype,
        device,
        rng,
    ).contiguous()


def run_combine(op: Any, combine_input: Any, dispatch_indices: Any, launch: LaunchConfig) -> tuple[Any, Any]:
    # Correct MORI benchmark semantics: combine weights are not the original
    # [num_tokens, topk] routing weights.  Passing None asks kiwi-ep to sum the
    # already-produced expert rows, matching MORI's default benchmark path.
    return op.combine(
        combine_input,
        None,
        dispatch_indices,
        **combine_kwargs(launch),
    )


__all__ = [
    "BenchInputs",
    "DTypeSpec",
    "LaunchConfig",
    "ScaleSpec",
    "align_up",
    "combine_slot_payload_bytes",
    "combine_slot_stride_bytes",
    "combine_kwargs",
    "default_launch_config",
    "dispatch_kwargs",
    "dtype_storage_hidden_elems",
    "make_indices_for_rank",
    "make_input_for_rank",
    "make_scale_spec",
    "make_scales_for_rank",
    "make_synthetic_combine_input",
    "quant_format_for_dtype",
    "resolve_dtype",
    "resolve_scale_dtype",
    "run_combine",
    "run_dispatch",
    "slot_payload_bytes",
    "slot_stride_bytes",
]
