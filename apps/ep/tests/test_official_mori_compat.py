#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Drop-in correctness gate against the official ROCm/MORI EP operator.

The test uses the public ``mori.ops.EpDispatchCombineOp`` API as the reference
for the KIWI Python binding.  It is intentionally a correctness gate for using
``kiwi-ep`` as a MORI-EP replacement in SGLang/vLLM, not a compatibility shim:
KIWI output is compared as exposed by the production binding, without rewriting
source metadata, synthesizing local rows, or repairing combine output.

Missing PyTorch, official MORI, or insufficient visible GPUs exit 77 so CTest
can mark the case skipped.  A built ``kiwi_ep_ext`` import failure, an API shape
mismatch, a missing split API, or a semantic mismatch is a test failure.
"""

import argparse
import os
import subprocess
import sys
from typing import Dict, Optional, Tuple

SKIP_RETURN_CODE = 77


def skip(reason: str) -> None:
    print(f"SKIP official MORI compatibility test: {reason}", file=sys.stderr)
    raise SystemExit(SKIP_RETURN_CODE)


def import_or_skip():
    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover - depends on host env
        skip(f"PyTorch is not importable ({exc})")

    try:
        import kiwi_ep_ext
    except Exception as exc:  # pragma: no cover - depends on build dir
        raise RuntimeError(
            f"kiwi_ep_ext is not importable from the built test tree ({exc})"
        ) from exc

    mori_source_dir = os.environ.get("MORI_SOURCE_DIR")
    if mori_source_dir:
        sys.path.insert(0, os.path.join(mori_source_dir, "python"))

    try:
        import mori
        import mori.ops
        import mori.shmem
    except Exception as exc:  # pragma: no cover - depends on host env
        skip(f"official MORI Python package is not importable ({exc})")

    return torch, dist, kiwi_ep_ext, mori


def env_int(*names: str, default: int = 0) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def parse_token_counts(spec: Optional[str], default: int) -> list[int]:
    if not spec:
        return [default]

    counts: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            counts.extend(range(start, end + step, step))
        else:
            counts.append(int(part))

    if not counts:
        raise ValueError(f"empty --num-tokens-list specification: {spec!r}")
    if any(count < 0 for count in counts):
        raise ValueError(f"negative token count in --num-tokens-list: {spec!r}")
    return counts


def dtype_from_name(torch, name: str):
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "fp8":
        for attr in ("float8_e4m3fnuz", "float8_e4m3fn"):
            if hasattr(torch, attr):
                return getattr(torch, attr)
        skip("PyTorch has no supported FP8 dtype")
    if name == "fp4":
        dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if dtype is None:
            skip("PyTorch has no float4_e2m1fn_x2 dtype")
        return dtype
    raise ValueError(f"unsupported dtype {name}")


def dtype_element_size(torch, dtype) -> int:
    return torch.empty((1,), dtype=dtype).element_size()


def dtype_storage_hidden_elems(dtype_name: str, logical_hidden_dim: int) -> int:
    if dtype_name == "fp4":
        if logical_hidden_dim % 2 != 0:
            raise AssertionError("FP4 logical hidden dim must be even")
        return logical_hidden_dim // 2
    return logical_hidden_dim


def config_hidden_dim(dispatch_dtype_name: str, logical_hidden_dim: int) -> int:
    # MORI's public config uses the logical hidden size. FP4 tensors are packed
    # in storage, but SGLang still passes the original hidden_size here.
    return logical_hidden_dim


def scale_dtype_from_name(torch, dispatch_dtype_name: str, scale_dtype_name: str):
    if scale_dtype_name == "float32":
        return torch.float32
    if scale_dtype_name == "e8m0":
        for attr in ("float8_e8m0fnu", "float8_e8m0fnuz"):
            if hasattr(torch, attr):
                return getattr(torch, attr)
        skip("PyTorch has no E8M0 scale dtype")
    if scale_dtype_name != "auto":
        raise ValueError(f"unsupported scale dtype {scale_dtype_name}")
    if dispatch_dtype_name == "fp4":
        return scale_dtype_from_name(torch, dispatch_dtype_name, "e8m0")
    return torch.float32


def combine_tolerance(combine_dtype_name: str) -> Tuple[float, float]:
    if combine_dtype_name == "bfloat16":
        return 1.0e-2, 1.0e-2
    if combine_dtype_name in ("fp8", "fp4"):
        return 2.5e-1, 0.0
    return 1.0e-5, 1.0e-6


def expert_id_for_rank(dest_rank: int, token: int, k: int,
                       num_experts_per_rank: int) -> int:
    return dest_rank * num_experts_per_rank + (
        (token * 3 + k * 5 + dest_rank) % num_experts_per_rank
    )


def remote_rank_for_ordinal(src_rank: int, ordinal: int, world_size: int) -> int:
    return (src_rank + 1 + (ordinal % (world_size - 1))) % world_size


def dest_rank_for_pattern(src_rank: int, token: int, k: int,
                          topk: int, world_size: int) -> int:
    if world_size <= 1:
        return src_rank

    r1 = remote_rank_for_ordinal(src_rank, 0, world_size)
    r2 = remote_rank_for_ordinal(src_rank, 1, world_size) if world_size > 2 else r1

    mod = token % 6
    if mod == 0:
        return src_rank
    if mod == 1:
        return r1 if k == 1 else src_rank
    if mod == 2:
        return r1 if k < min(topk, 2) else src_rank
    if mod == 3:
        return r1 if (k % 2) == 0 else src_rank
    if mod == 4:
        if world_size > 2:
            if k == 0:
                return r1
            if k == 1:
                return r2
            return src_rank if k == 2 else r1
        return r1 if k < min(topk, 3) else src_rank
    return remote_rank_for_ordinal(src_rank, k, world_size)


def make_fp4_tensor(torch, rank: int, shape: Tuple[int, int], device, seed: int):
    rows, hidden_elems = shape
    base = torch.arange(rows * hidden_elems, dtype=torch.int32, device=device).reshape(rows, hidden_elems)
    low = (base + rank * 7 + seed) & 0x0F
    high = ((base * 3) + rank * 11 + seed) & 0x0F
    raw = (low | (high << 4)).to(torch.uint8)
    return raw.view(torch.float4_e2m1fn_x2)


def make_inputs(torch, rank: int, world_size: int, num_tokens: int,
                logical_hidden_dim: int, topk: int, num_experts_per_rank: int,
                dispatch_dtype_name: str, dispatch_dtype, scale_dim: int,
                scale_dtype, device):
    hidden_dim = dtype_storage_hidden_elems(dispatch_dtype_name, logical_hidden_dim)
    index_rows = []
    weight_rows = []
    for token in range(num_tokens):
        idx_row = []
        weight_row = []
        for k in range(topk):
            dest_rank = dest_rank_for_pattern(rank, token, k, topk, world_size)
            idx_row.append(expert_id_for_rank(dest_rank, token, k,
                                              num_experts_per_rank))
            weight_row.append(0.125 * float(1 + ((rank + token + 2 * k) % 5)))
        index_rows.append(idx_row)
        weight_rows.append(weight_row)

    if dispatch_dtype_name == "fp4":
        tokens = make_fp4_tensor(torch, rank, (num_tokens, hidden_dim), device, seed=3)
    elif num_tokens == 0:
        tokens = torch.empty((0, hidden_dim), dtype=dispatch_dtype, device=device)
    else:
        token_rows = []
        for token in range(num_tokens):
            if dispatch_dtype_name == "fp8":
                # Keep values well inside FP8 finite range.  The generic
                # integer-like pattern is intentionally large for BF16/FP32
                # bitwise source tracking, but overflows FP8 and can create
                # NaNs before dispatch/combine semantics are exercised.
                token_rows.append([
                    1.0
                    + float(rank) * 0.5
                    + float(token) * 0.03125
                    + float(dim % 64) * 0.001953125
                    for dim in range(hidden_dim)
                ])
            else:
                token_rows.append([
                    float(rank * 100000 + token * 1000 + dim) + 0.25
                    for dim in range(hidden_dim)
                ])
        tokens = torch.tensor(
            token_rows, dtype=torch.float32, device=device).to(dispatch_dtype)

    if num_tokens == 0:
        indices = torch.empty((0, topk), dtype=torch.int32, device=device)
        weights = torch.empty((0, topk), dtype=torch.float32, device=device)
    else:
        indices = torch.tensor(index_rows, dtype=torch.int32, device=device)
        weights = torch.tensor(weight_rows, dtype=torch.float32, device=device)
    if scale_dim > 0:
        if num_tokens == 0:
            scales = torch.empty((0, scale_dim), dtype=scale_dtype, device=device)
        else:
            scale_rows = []
            for token in range(num_tokens):
                scale_rows.append([
                    1.0 + float(rank) * 0.125 + float(token) * 0.015625
                    + float(s) * 0.001953125
                    for s in range(scale_dim)
                ])
            scales = torch.tensor(scale_rows, dtype=torch.float32, device=device).to(scale_dtype)
    else:
        scales = torch.empty((num_tokens, 0), dtype=torch.float32, device=device)
    return tokens, indices, weights, scales


def decode_mori_src_pos(raw_pos: int, world_size: int,
                        max_num_inp_token_per_rank: int) -> Tuple[int, int]:
    stride = world_size * max_num_inp_token_per_rank
    return int(raw_pos) // stride, int(raw_pos) % stride


def decode_legacy_kiwi_src_pos(raw_pos: int,
                               max_num_inp_token_per_rank: int) -> Tuple[int, int]:
    return int(raw_pos) // max_num_inp_token_per_rank, int(raw_pos) % max_num_inp_token_per_rank


def decode_src_pos_for_expert_output(raw_pos: int, world_size: int,
                                     max_num_inp_token_per_rank: int) -> Tuple[int, int]:
    src_rank, src_token = decode_mori_src_pos(
        raw_pos, world_size, max_num_inp_token_per_rank)
    if 0 <= src_rank < world_size and 0 <= src_token < max_num_inp_token_per_rank:
        return src_rank, src_token

    src_rank, src_token = decode_legacy_kiwi_src_pos(
        raw_pos, max_num_inp_token_per_rank)
    if 0 <= src_rank < world_size and 0 <= src_token < max_num_inp_token_per_rank:
        return src_rank, src_token

    raise AssertionError(
        f"source position is neither MORI nor legacy KIWI encoded: raw={raw_pos}"
    )


def expected_local_mori_src_positions(rank: int, world_size: int,
                                      num_tokens: int, topk: int,
                                      num_experts_per_rank: int,
                                      max_num_inp_token_per_rank: int) -> set[int]:
    stride = world_size * max_num_inp_token_per_rank
    local_rows = set()
    for token in range(num_tokens):
        for k in range(topk):
            dest_rank = dest_rank_for_pattern(rank, token, k, topk, world_size)
            expert = expert_id_for_rank(dest_rank, token, k,
                                        num_experts_per_rank)
            if expert // num_experts_per_rank == rank:
                local_rows.add(rank * stride + token)
                break
    return local_rows


def expected_tokens_with_min_unique_dests(rank: int, world_size: int,
                                          num_tokens: int, topk: int,
                                          min_unique: int) -> list[int]:
    tokens = []
    for token in range(num_tokens):
        unique = {
            dest_rank_for_pattern(rank, token, k, topk, world_size)
            for k in range(topk)
        }
        if len(unique) >= min_unique:
            tokens.append(token)
    return tokens


def make_expert_output(torch, rank: int, src_order: list[int], recv_count: int,
                       max_recv: int, combine_hidden_elems: int, world_size: int,
                       max_num_inp_token_per_rank: int, combine_dtype_name: str,
                       combine_dtype, device,
                       bf16_rounding_probe: bool = False):
    if combine_dtype_name == "fp4":
        raw = torch.zeros((max_recv, combine_hidden_elems), dtype=torch.uint8, device=device)
        hidden_elem_idx = torch.arange(combine_hidden_elems, dtype=torch.int32, device=device)
        for row in range(recv_count):
            src_rank, src_token = decode_src_pos_for_expert_output(
                src_order[row], world_size, max_num_inp_token_per_rank)
            low = (hidden_elem_idx + rank + src_rank + src_token) & 0x0F
            high = ((hidden_elem_idx * 3) + rank + src_token) & 0x0F
            raw[row].copy_((low | (high << 4)).to(torch.uint8))
        return raw.view(combine_dtype)

    rows = torch.zeros((max_recv, combine_hidden_elems), dtype=torch.float32, device=device)
    dims = torch.arange(combine_hidden_elems, dtype=torch.float32, device=device) * 0.0009765625
    for row in range(recv_count):
        src_rank, src_token = decode_src_pos_for_expert_output(
            src_order[row], world_size, max_num_inp_token_per_rank)
        if bf16_rounding_probe:
            first_remote = (remote_rank_for_ordinal(src_rank, 0, world_size)
                            if world_size > 1 else src_rank)
            base = 1.0 if rank == first_remote else 0.00390625
            rows[row].fill_(base)
        else:
            base = (
                1.0
                + float(rank) * 0.25
                + float(src_rank) * 0.03125
                + float(src_token) * 0.00390625
            )
            rows[row].copy_(base + dims)
    return rows.to(combine_dtype)


def make_combine_weight_override(torch, src_order: list[int], recv_count: int,
                                 max_recv: int, topk: int, world_size: int,
                                 max_num_inp_token_per_rank: int, device):
    rows = torch.zeros((max_recv, topk), dtype=torch.float32, device=device)
    for row in range(recv_count):
        src_rank, src_token = decode_src_pos_for_expert_output(
            src_order[row], world_size, max_num_inp_token_per_rank)
        for k in range(topk):
            rows[row, k] = (
                10.0
                + float(src_rank) * 0.5
                + float(src_token) * 0.03125
                + float(k) * 0.001953125
            )
    return rows


DispatchRow = Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"], "torch.Tensor"]


def tensor_equal(torch, a, b) -> bool:
    try:
        if tuple(a.shape) != tuple(b.shape):
            return False
        if a.dtype != b.dtype:
            return False
        if a.device != b.device:
            return False
        return bool(torch.equal(a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8)))
    except Exception:
        return False


def tensor_max_diff(torch, a, b):
    try:
        return torch.max(torch.abs(a.float() - b.float())).item()
    except Exception:
        return "unavailable"


def tensors_allclose(torch, a, b, atol: float, rtol: float) -> bool:
    try:
        return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))
    except Exception:
        return tensor_equal(torch, a, b)


def dispatch_table(torch, name: str, op, output, weights, scales, indices, recv_count,
                   ) -> Tuple[Dict[int, DispatchRow], int, list[int]]:
    torch.cuda.synchronize()
    count = int(recv_count[:1].detach().cpu()[0].item())
    src_pos = op.get_dispatch_src_token_pos()[:count].detach().cpu().tolist()

    out_cpu = output[:count].detach().cpu()
    weights_cpu = weights[:count].detach().cpu()
    scales_cpu = scales[:count].detach().cpu() if scales is not None else None
    indices_cpu = indices[:count].detach().cpu()
    table: Dict[int, DispatchRow] = {}
    for i, raw_pos in enumerate(src_pos):
        key = int(raw_pos)
        if key in table:
            raise AssertionError(
                f"{name}: duplicate raw dispatch source position {key}; "
                f"raw positions {src_pos}"
            )
        table[key] = (
            out_cpu[i].clone(),
            weights_cpu[i].clone(),
            scales_cpu[i].clone() if scales_cpu is not None else None,
            indices_cpu[i].clone(),
        )
    return table, count, [int(x) for x in src_pos]


def compare_dispatch_tables(torch, rank: int,
                            kiwi_rows: Dict[int, DispatchRow],
                            mori_rows: Dict[int, DispatchRow]) -> int:
    errors = 0
    kiwi_keys = set(kiwi_rows)
    mori_keys = set(mori_rows)
    if kiwi_keys != mori_keys:
        missing = sorted(mori_keys - kiwi_keys)
        extra = sorted(kiwi_keys - mori_keys)
        print(f"[Rank {rank}] dispatch src_pos set mismatch: "
              f"missing_from_kiwi={missing} extra_in_kiwi={extra}",
              file=sys.stderr)
        errors += 1

    for key in sorted(kiwi_keys & mori_keys):
        kiwi_out, kiwi_w, kiwi_s, kiwi_idx = kiwi_rows[key]
        mori_out, mori_w, mori_s, mori_idx = mori_rows[key]
        if not tensor_equal(torch, kiwi_idx, mori_idx):
            print(f"[Rank {rank}] index mismatch raw_src_pos={key}: "
                  f"kiwi={kiwi_idx.tolist()} mori={mori_idx.tolist()}",
                  file=sys.stderr)
            errors += 1
        if not tensor_equal(torch, kiwi_w, mori_w):
            print(f"[Rank {rank}] weight bit mismatch raw_src_pos={key}: "
                  f"kiwi={kiwi_w.tolist()} mori={mori_w.tolist()}",
                  file=sys.stderr)
            errors += 1
        if (kiwi_s is None) != (mori_s is None):
            print(f"[Rank {rank}] scale presence mismatch raw_src_pos={key}: "
                  f"kiwi={kiwi_s is not None} mori={mori_s is not None}",
                  file=sys.stderr)
            errors += 1
        elif kiwi_s is not None and mori_s is not None and not tensor_equal(torch, kiwi_s, mori_s):
            print(f"[Rank {rank}] scale bit mismatch raw_src_pos={key}",
                  file=sys.stderr)
            errors += 1
        if kiwi_out.dtype != mori_out.dtype:
            print(f"[Rank {rank}] hidden dtype mismatch raw_src_pos={key}: "
                  f"kiwi={kiwi_out.dtype} mori={mori_out.dtype}",
                  file=sys.stderr)
            errors += 1
        if not tensor_equal(torch, kiwi_out, mori_out):
            print(f"[Rank {rank}] hidden bit mismatch raw_src_pos={key}: "
                  f"max_diff={tensor_max_diff(torch, kiwi_out, mori_out)}",
                  file=sys.stderr)
            errors += 1
    return errors


def compare_required_dispatch_keys(rank: int, label: str,
                                   kiwi_rows: Dict[int, DispatchRow],
                                   required_keys: set[int]) -> int:
    missing = sorted(required_keys - set(kiwi_rows))
    if missing:
        print(f"[Rank {rank}] kiwi dispatch is missing required {label} "
              f"MORI source-position rows: {missing}", file=sys.stderr)
        return 1
    return 0


def compare_tensor_api(rank: int, label: str,
                       kiwi_tensor: Optional["torch.Tensor"],
                       mori_tensor: Optional["torch.Tensor"]) -> int:
    errors = 0
    if mori_tensor is None:
        if kiwi_tensor is not None:
            print(f"[Rank {rank}] {label}: MORI returned None but kiwi "
                  "returned a tensor", file=sys.stderr)
            return 1
        return 0
    if kiwi_tensor is None:
        print(f"[Rank {rank}] {label}: MORI returned a tensor but kiwi "
              "returned None", file=sys.stderr)
        return 1

    kiwi_shape = tuple(int(x) for x in kiwi_tensor.shape)
    mori_shape = tuple(int(x) for x in mori_tensor.shape)
    if kiwi_shape != mori_shape:
        print(f"[Rank {rank}] {label}: shape mismatch: "
              f"kiwi={kiwi_shape} mori={mori_shape}", file=sys.stderr)
        errors += 1
    if kiwi_tensor.dtype != mori_tensor.dtype:
        print(f"[Rank {rank}] {label}: dtype mismatch: "
              f"kiwi={kiwi_tensor.dtype} mori={mori_tensor.dtype}",
              file=sys.stderr)
        errors += 1
    if kiwi_tensor.device != mori_tensor.device:
        print(f"[Rank {rank}] {label}: device mismatch: "
              f"kiwi={kiwi_tensor.device} mori={mori_tensor.device}",
              file=sys.stderr)
        errors += 1
    if kiwi_tensor.is_contiguous() != mori_tensor.is_contiguous():
        print(f"[Rank {rank}] {label}: contiguity mismatch: "
              f"kiwi={kiwi_tensor.is_contiguous()} "
              f"mori={mori_tensor.is_contiguous()}",
              file=sys.stderr)
        errors += 1
    kiwi_stride = tuple(int(x) for x in kiwi_tensor.stride())
    mori_stride = tuple(int(x) for x in mori_tensor.stride())
    if kiwi_stride != mori_stride:
        print(f"[Rank {rank}] {label}: stride mismatch: "
              f"kiwi={kiwi_stride} mori={mori_stride}",
              file=sys.stderr)
        errors += 1
    return errors


def compare_dispatch_api_tensors(torch, rank: int, kiwi_op, mori_op,
                                 combine_dtype, combine_hidden_elems: int,
                                 kiwi_out, kiwi_w, kiwi_s, kiwi_idx,
                                 kiwi_count,
                                 mori_out, mori_w, mori_s, mori_idx,
                                 mori_count) -> int:
    errors = 0
    errors += compare_tensor_api(rank, "dispatch hidden buffer",
                                 kiwi_out, mori_out)
    errors += compare_tensor_api(rank, "dispatch weight buffer",
                                 kiwi_w, mori_w)
    errors += compare_tensor_api(rank, "dispatch scale buffer",
                                 kiwi_s, mori_s)
    errors += compare_tensor_api(rank, "dispatch index buffer",
                                 kiwi_idx, mori_idx)
    errors += compare_tensor_api(rank, "dispatch recv-count tensor",
                                 kiwi_count, mori_count)

    try:
        kiwi_registered = kiwi_op.get_registered_combine_input_buffer(
            combine_dtype, combine_hidden_elems)
    except Exception as exc:
        print(f"[Rank {rank}] registered combine input buffer: "
              f"kiwi call failed: {exc}", file=sys.stderr)
        errors += 1
        kiwi_registered = None
    mori_registered = mori_op.get_registered_combine_input_buffer(
        combine_dtype, combine_hidden_elems)
    errors += compare_tensor_api(rank, "registered combine input buffer",
                                 kiwi_registered, mori_registered)

    try:
        kiwi_src_pos = kiwi_op.get_dispatch_src_token_pos()
    except Exception as exc:
        print(f"[Rank {rank}] dispatch source-position tensor: "
              f"kiwi call failed: {exc}", file=sys.stderr)
        errors += 1
        kiwi_src_pos = None
    mori_src_pos = mori_op.get_dispatch_src_token_pos()
    errors += compare_tensor_api(rank, "dispatch source-position tensor",
                                 kiwi_src_pos, mori_src_pos)
    if kiwi_src_pos is not None:
        errors += compare_decoded_src_positions(
            rank, kiwi_op, mori_op, kiwi_src_pos, kiwi_count)
    return errors


def compare_decoded_src_positions(rank: int, kiwi_op, mori_op,
                                  kiwi_src_pos, kiwi_count) -> int:
    """Decode kiwi's dispatch source positions with decode_send_flat_idx.

    Every received row must decode to a valid (src_rank, src_token_id) and
    match MORI's decoding of the same raw value.
    """
    count = int(kiwi_count[:1].detach().cpu()[0].item())
    raw_positions = kiwi_src_pos[:count].detach().cpu().tolist()
    per_rank = kiwi_op.max_num_tokens_to_send_per_rank()
    world_size = kiwi_op.max_num_tokens_to_send() // per_rank
    errors = 0
    for raw_pos in raw_positions:
        try:
            kiwi_decoded = tuple(kiwi_op.decode_send_flat_idx(int(raw_pos)))
        except Exception as exc:
            print(f"[Rank {rank}] decode_send_flat_idx({raw_pos}): "
                  f"kiwi call failed: {exc}", file=sys.stderr)
            return errors + 1
        mori_decoded = tuple(mori_op.decode_send_flat_idx(int(raw_pos)))
        src_rank, src_token = kiwi_decoded
        if (kiwi_decoded != mori_decoded
                or not 0 <= src_rank < world_size
                or not 0 <= src_token < per_rank):
            print(f"[Rank {rank}] decode_send_flat_idx({raw_pos}) mismatch: "
                  f"kiwi={kiwi_decoded} mori={mori_decoded} "
                  f"world_size={world_size} per_rank={per_rank}",
                  file=sys.stderr)
            errors += 1
    return errors



def compare_direct_dispatch_outputs(
        torch, rank: int, label: str,
        kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count,
        mori_out, mori_w, mori_s, mori_idx, mori_count) -> Tuple[int, int, int]:
    """Compare only tensors returned by dispatch(), with no auxiliary MORI APIs."""
    errors = 0
    errors += compare_tensor_api(rank, f"{label}: dispatch hidden buffer",
                                 kiwi_out, mori_out)
    errors += compare_tensor_api(rank, f"{label}: dispatch weight buffer",
                                 kiwi_w, mori_w)
    errors += compare_tensor_api(rank, f"{label}: dispatch scale buffer",
                                 kiwi_s, mori_s)
    errors += compare_tensor_api(rank, f"{label}: dispatch index buffer",
                                 kiwi_idx, mori_idx)
    errors += compare_tensor_api(rank, f"{label}: dispatch recv-count tensor",
                                 kiwi_count, mori_count)

    torch.cuda.synchronize()
    kiwi_total = int(kiwi_count[:1].detach().cpu()[0].item())
    mori_total = int(mori_count[:1].detach().cpu()[0].item())
    if kiwi_total != mori_total:
        print(f"[Rank {rank}] {label}: dispatch recv count mismatch: "
              f"kiwi={kiwi_total} mori={mori_total}", file=sys.stderr)
        errors += 1

    prefix = min(kiwi_total, mori_total)
    for name, kiwi_tensor, mori_tensor in (
            ("hidden", kiwi_out, mori_out),
            ("weights", kiwi_w, mori_w),
            ("indices", kiwi_idx, mori_idx),
    ):
        if kiwi_tensor is not None and mori_tensor is not None:
            kiwi = kiwi_tensor[:prefix].detach().cpu()
            mori = mori_tensor[:prefix].detach().cpu()
            if not tensor_equal(torch, kiwi, mori):
                print(f"[Rank {rank}] {label}: dispatch {name} returned by "
                      f"dispatch() differ in first {prefix} rows: "
                      f"max_diff={tensor_max_diff(torch, kiwi, mori)}",
                      file=sys.stderr)
                errors += 1
    if (kiwi_s is None) != (mori_s is None):
        print(f"[Rank {rank}] {label}: dispatch scale presence mismatch: "
              f"kiwi={kiwi_s is not None} mori={mori_s is not None}",
              file=sys.stderr)
        errors += 1
    elif kiwi_s is not None and mori_s is not None:
        kiwi = kiwi_s[:prefix].detach().cpu()
        mori = mori_s[:prefix].detach().cpu()
        if not tensor_equal(torch, kiwi, mori):
            print(f"[Rank {rank}] {label}: dispatch scales returned by "
                  f"dispatch() differ in first {prefix} rows", file=sys.stderr)
            errors += 1

    return errors, kiwi_total, mori_total


def _tensor_row_bytes(torch, tensor, row: int) -> bytes:
    if tensor is None:
        return b"<none>"
    return bytes(tensor[row].detach().cpu().contiguous().view(torch.uint8).tolist())


def compare_dispatch_multiset_outputs(
        torch, rank: int, label: str,
        kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count,
        mori_out, mori_w, mori_s, mori_idx, mori_count) -> Tuple[int, int, int]:
    """Compare dispatch() rows as an unordered public-output multiset.

    MORI and KIWI do not have to expose received rows in the same order for the
    downstream sync-only path: downstream experts consume the returned rows and
    feed results back to the same operator instance for combine().  This still
    verifies that the public dispatch tensors contain the same row payloads
    without using layout/source-position helper APIs.
    """
    errors = 0
    errors += compare_tensor_api(rank, f"{label}: dispatch hidden buffer",
                                 kiwi_out, mori_out)
    errors += compare_tensor_api(rank, f"{label}: dispatch weight buffer",
                                 kiwi_w, mori_w)
    errors += compare_tensor_api(rank, f"{label}: dispatch scale buffer",
                                 kiwi_s, mori_s)
    errors += compare_tensor_api(rank, f"{label}: dispatch index buffer",
                                 kiwi_idx, mori_idx)
    errors += compare_tensor_api(rank, f"{label}: dispatch recv-count tensor",
                                 kiwi_count, mori_count)

    torch.cuda.synchronize()
    kiwi_total = int(kiwi_count[:1].detach().cpu()[0].item())
    mori_total = int(mori_count[:1].detach().cpu()[0].item())
    if kiwi_total != mori_total:
        print(f"[Rank {rank}] {label}: dispatch recv count mismatch: "
              f"kiwi={kiwi_total} mori={mori_total}", file=sys.stderr)
        errors += 1

    if (kiwi_s is None) != (mori_s is None):
        print(f"[Rank {rank}] {label}: dispatch scale presence mismatch: "
              f"kiwi={kiwi_s is not None} mori={mori_s is not None}",
              file=sys.stderr)
        errors += 1

    prefix = min(kiwi_total, mori_total)
    if prefix == 0 or errors:
        return errors, kiwi_total, mori_total

    from collections import Counter

    def signature(out, weights, scales, indices, row: int) -> tuple[bytes, bytes, bytes, bytes]:
        return (
            _tensor_row_bytes(torch, out, row),
            _tensor_row_bytes(torch, weights, row),
            _tensor_row_bytes(torch, scales, row) if scales is not None else b"<none>",
            _tensor_row_bytes(torch, indices, row),
        )

    kiwi_rows = Counter(signature(kiwi_out, kiwi_w, kiwi_s, kiwi_idx, i)
                        for i in range(prefix))
    mori_rows = Counter(signature(mori_out, mori_w, mori_s, mori_idx, i)
                        for i in range(prefix))
    if kiwi_rows != mori_rows:
        missing = sum((mori_rows - kiwi_rows).values())
        extra = sum((kiwi_rows - mori_rows).values())
        print(f"[Rank {rank}] {label}: dispatch row multiset mismatch: "
              f"missing_from_kiwi={missing} extra_in_kiwi={extra}",
              file=sys.stderr)
        errors += 1

    return errors, kiwi_total, mori_total


def make_downstream_expert_output(torch, dispatch_out, recv_count,
                                  combine_dtype_name: str, combine_dtype,
                                  logical_hidden_dim: int):
    """Create vLLM/SGLang-like fused expert output from public dispatch rows.

    Downstream fused expert outputs use the effective combine payload dtype
    derived from MORI's max_token_type_size rather than the dispatch dtype.
    """
    count = int(recv_count[:1].detach().cpu()[0].item())
    combine_hidden_elems = dtype_storage_hidden_elems(combine_dtype_name, logical_hidden_dim)
    expert = torch.zeros((dispatch_out.size(0), combine_hidden_elems),
                         dtype=combine_dtype, device=dispatch_out.device)
    if count == 0:
        return expert
    if dispatch_out.size(1) != combine_hidden_elems:
        raise AssertionError(
            "downstream sync expert output requires dispatch rows that match "
            f"effective combine hidden elements; got dispatch hidden dim "
            f"{dispatch_out.size(1)} vs combine_hidden_elems {combine_hidden_elems}")
    expert[:count].copy_(dispatch_out[:count].float().to(combine_dtype))
    return expert



def compare_optional_weights(torch, rank: int, label: str,
                             kiwi_weight: Optional["torch.Tensor"],
                             mori_weight: Optional["torch.Tensor"],
                             num_tokens: int) -> int:
    if mori_weight is None:
        if kiwi_weight is not None:
            print(f"[Rank {rank}] {label}: MORI returned no output weights, "
                  "but kiwi did", file=sys.stderr)
            return 1
        return 0

    if kiwi_weight is None:
        print(f"[Rank {rank}] {label}: MORI returned output weights, "
              "but kiwi returned None", file=sys.stderr)
        return 1

    errors = compare_tensor_api(rank, f"{label}: combine output-weight buffer",
                                kiwi_weight, mori_weight)
    kiwi = kiwi_weight[:num_tokens].detach().cpu()
    mori = mori_weight[:num_tokens].detach().cpu()
    if not tensors_allclose(torch, kiwi, mori, atol=1.0e-6, rtol=0.0):
        print(f"[Rank {rank}] {label}: combine output-weight mismatch: "
              f"max_diff={tensor_max_diff(torch, kiwi, mori)}",
              file=sys.stderr)
        errors += 1
    return errors


def compare_combine_outputs(torch, rank: int, label: str,
                            kiwi_output, kiwi_weight,
                            mori_output, mori_weight,
                            num_tokens: int, combine_dtype_name: str,
                            strict_tolerance: bool = False) -> int:
    errors = 0
    if strict_tolerance:
        atol, rtol = 1.0e-4, 0.0
    else:
        atol, rtol = combine_tolerance(combine_dtype_name)
    errors += compare_tensor_api(rank, f"{label}: combine output buffer",
                                 kiwi_output, mori_output)
    kiwi = kiwi_output[:num_tokens].detach().cpu()
    mori = mori_output[:num_tokens].detach().cpu()
    if kiwi.dtype != mori.dtype:
        print(f"[Rank {rank}] {label}: combine dtype mismatch: "
              f"kiwi={kiwi.dtype} mori={mori.dtype}", file=sys.stderr)
        errors += 1
    if not tensors_allclose(torch, kiwi, mori, atol=atol, rtol=rtol):
        print(f"[Rank {rank}] {label}: combine output mismatch: "
              f"max_diff={tensor_max_diff(torch, kiwi, mori)} "
              f"atol={atol} rtol={rtol}", file=sys.stderr)
        errors += 1
    errors += compare_optional_weights(
        torch, rank, label, kiwi_weight, mori_weight, num_tokens)
    return errors


def compare_layout_api(rank: int, kiwi_op, mori_op) -> int:
    errors = 0
    for name in (
            "max_num_tokens_to_recv",
            "max_num_tokens_to_recv_per_rank",
            "max_num_tokens_to_send",
            "max_num_tokens_to_send_per_rank",
    ):
        try:
            kiwi_value = getattr(kiwi_op, name)()
        except Exception as exc:
            print(f"[Rank {rank}] {name}: kiwi call failed: {exc}",
                  file=sys.stderr)
            errors += 1
            continue
        mori_value = getattr(mori_op, name)()
        if kiwi_value != mori_value:
            print(f"[Rank {rank}] {name} mismatch: "
                  f"kiwi={kiwi_value} mori={mori_value}", file=sys.stderr)
            errors += 1
    return errors


def call_kiwi(rank: int, label: str, fn, *args):
    try:
        return fn(*args), 0
    except Exception as exc:
        print(f"[Rank {rank}] {label}: kiwi call failed: {exc}",
              file=sys.stderr)
        return None, 1


def mori_correctness_block_num(args) -> int:
    """Return the MORI launch geometry used by this correctness harness.

    Official MORI's IntraNode combine kernel can over-decompose small
    correctness-reference shapes when launched with the broad default block
    count.  The symptom is a MORI-only partial/zero combine buffer in the
    reference path (first observed for empty-scale hidden_dim=32 and reproduced
    for quantized scale-bearing hidden_dim=128/max_token_type_size=2).  That is
    a MORI reference-launch artifact, not behavior KIWI should mimic.  Keep the
    assertions exact and still exercise all combine forms, but launch the
    IntraNode MORI reference path with one block so every combine warp owns real
    hidden elements.  KIWI's production launch path is unchanged.
    """
    if args.kernel_type == "IntraNode":
        return 1
    return args.block_num


def barrier_all(torch, dist, kiwi_ep_ext):
    torch.cuda.synchronize()
    dist.barrier()
    kiwi_ep_ext.barrier()


def reset_ops(torch, dist, kiwi_ep_ext, mori_op, kiwi_op):
    mori_op.reset()
    kiwi_op.reset()
    barrier_all(torch, dist, kiwi_ep_ext)


def precondition_intranode_empty_scale_path(
        torch, dist, kiwi_ep_ext, rank: int, world_size: int,
        mori_op, kiwi_op, tokens, indices, weights, scales,
        args) -> None:
    """Reset-isolated symmetric warmup for the MORI empty-scale IntraNode path.

    The warmup intentionally mirrors both combine forms that the real test
    still compares below: override weights and combine(None weights).  It is
    reset-isolated, uses both MORI and KIWI operators, and does not replace,
    relax, or skip any of the exact assertions in the official comparison.
    """
    if args.kernel_type != "IntraNode" or args.scale_dim != 0 or args.num_tokens == 0:
        return

    combine_hidden_elems = dtype_storage_hidden_elems(
        args.combine_dtype_name_effective, args.hidden_dim)
    mori_block_num = mori_correctness_block_num(args)

    for use_override_weights in (True, False):
        reset_ops(torch, dist, kiwi_ep_ext, mori_op, kiwi_op)

        mori_out, mori_w, mori_s, mori_idx, mori_count = mori_op.dispatch(
            tokens, weights, scales, indices,
            block_num=mori_block_num,
            rdma_block_num=args.rdma_block_num,
            warp_per_block=args.warp_num_per_block,
        )
        barrier_all(torch, dist, kiwi_ep_ext)

        kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count = kiwi_op.dispatch(
            tokens, weights, scales, indices)
        barrier_all(torch, dist, kiwi_ep_ext)

        _mori_rows, mori_total, mori_order = dispatch_table(
            torch, "mori-empty-scale-precondition", mori_op, mori_out,
            mori_w, mori_s, mori_idx, mori_count)
        _kiwi_rows, kiwi_total, kiwi_order = dispatch_table(
            torch, "kiwi-empty-scale-precondition", kiwi_op, kiwi_out,
            kiwi_w, kiwi_s, kiwi_idx, kiwi_count)

        mori_expert = make_expert_output(
            torch, rank, mori_order, mori_total,
            mori_out.size(0), combine_hidden_elems, world_size,
            args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
            args.combine_dtype_effective, tokens.device,
            bf16_rounding_probe=args.require_multi_source_bf16)
        kiwi_expert = make_expert_output(
            torch, rank, kiwi_order, kiwi_total,
            kiwi_out.size(0), combine_hidden_elems, world_size,
            args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
            args.combine_dtype_effective, tokens.device,
            bf16_rounding_probe=args.require_multi_source_bf16)

        mori_combine_w_arg = None
        kiwi_combine_w_arg = None
        if use_override_weights:
            mori_combine_w_arg = make_combine_weight_override(
                torch, mori_order, mori_total, mori_w.size(0), args.topk,
                world_size, args.max_num_inp_token_per_rank, tokens.device)
            kiwi_combine_w_arg = make_combine_weight_override(
                torch, kiwi_order, kiwi_total, kiwi_w.size(0), args.topk,
                world_size, args.max_num_inp_token_per_rank, tokens.device)

        mori_op.combine(
            mori_expert, mori_combine_w_arg, indices,
            block_num=mori_block_num,
            rdma_block_num=args.rdma_block_num,
            warp_per_block=args.warp_num_per_block,
        )
        barrier_all(torch, dist, kiwi_ep_ext)
        kiwi_op.combine(kiwi_expert, kiwi_combine_w_arg, indices)
        barrier_all(torch, dist, kiwi_ep_ext)

    reset_ops(torch, dist, kiwi_ep_ext, mori_op, kiwi_op)


def run_sync_round(torch, dist, kiwi_ep_ext, rank: int, world_size: int,
                   mori_op, kiwi_op, tokens, indices, weights, scales,
                   args, label: str) -> int:
    combine_hidden_elems = dtype_storage_hidden_elems(
        args.combine_dtype_name_effective, args.hidden_dim)
    mori_block_num = mori_correctness_block_num(args)
    errors = 0
    reset_ops(torch, dist, kiwi_ep_ext, mori_op, kiwi_op)

    mori_out, mori_w, mori_s, mori_idx, mori_count = mori_op.dispatch(
        tokens, weights, scales, indices,
        block_num=mori_block_num,
        rdma_block_num=args.rdma_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_result, call_errors = call_kiwi(
        rank, f"{label} dispatch", kiwi_op.dispatch,
        tokens, weights, scales, indices)
    errors += call_errors
    if kiwi_result is None:
        return errors
    kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count = kiwi_result
    barrier_all(torch, dist, kiwi_ep_ext)

    mori_rows, mori_total, mori_order = dispatch_table(
        torch, f"mori-{label}", mori_op, mori_out, mori_w, mori_s,
        mori_idx, mori_count)
    kiwi_rows, kiwi_total, kiwi_order = dispatch_table(
        torch, f"kiwi-{label}", kiwi_op, kiwi_out, kiwi_w, kiwi_s,
        kiwi_idx, kiwi_count)

    errors += compare_layout_api(rank, kiwi_op, mori_op)
    errors += compare_dispatch_api_tensors(
        torch, rank, kiwi_op, mori_op,
        args.combine_dtype_effective, combine_hidden_elems,
        kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count,
        mori_out, mori_w, mori_s, mori_idx, mori_count)

    if args.num_tokens > 0:
        local_keys = expected_local_mori_src_positions(
            rank, world_size, args.num_tokens, args.topk,
            args.num_experts_per_rank, args.max_num_inp_token_per_rank)
        if not local_keys:
            raise AssertionError("test routing pattern did not exercise local experts")
        errors += compare_required_dispatch_keys(rank, "local", kiwi_rows, local_keys)
    if kiwi_total != mori_total:
        print(f"[Rank {rank}] {label} recv count mismatch: "
              f"kiwi={kiwi_total} mori={mori_total}", file=sys.stderr)
        errors += 1
    errors += compare_dispatch_tables(torch, rank, kiwi_rows, mori_rows)

    if args.require_multi_source_bf16:
        multi_source_tokens = expected_tokens_with_min_unique_dests(
            rank, world_size, args.num_tokens, args.topk, 3)
        if not multi_source_tokens:
            raise AssertionError(
                "routing pattern did not exercise any token with at "
                "least three unique destination ranks")

    mori_expert = make_expert_output(
        torch, rank, mori_order, mori_total,
        mori_out.size(0), combine_hidden_elems, world_size,
        args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
        args.combine_dtype_effective, tokens.device,
        bf16_rounding_probe=args.require_multi_source_bf16)
    kiwi_expert = make_expert_output(
        torch, rank, kiwi_order, kiwi_total,
        kiwi_out.size(0), combine_hidden_elems, world_size,
        args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
        args.combine_dtype_effective, tokens.device,
        bf16_rounding_probe=args.require_multi_source_bf16)

    mori_combine_w_arg = None
    kiwi_combine_w_arg = None
    if args.combine_output_weights:
        mori_combine_w_arg = make_combine_weight_override(
            torch, mori_order, mori_total, mori_w.size(0), args.topk,
            world_size, args.max_num_inp_token_per_rank, tokens.device)
        kiwi_combine_w_arg = make_combine_weight_override(
            torch, kiwi_order, kiwi_total, kiwi_w.size(0), args.topk,
            world_size, args.max_num_inp_token_per_rank, tokens.device)

    mori_combine, mori_combine_w = mori_op.combine(
        mori_expert, mori_combine_w_arg, indices,
        block_num=mori_block_num,
        rdma_block_num=args.rdma_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_combine_result, call_errors = call_kiwi(
        rank, f"{label} combine", kiwi_op.combine,
        kiwi_expert, kiwi_combine_w_arg, indices)
    errors += call_errors
    if kiwi_combine_result is not None:
        kiwi_combine, kiwi_combine_w = kiwi_combine_result
        barrier_all(torch, dist, kiwi_ep_ext)
        errors += compare_combine_outputs(
            torch, rank, f"{label} combine",
            kiwi_combine, kiwi_combine_w,
            mori_combine, mori_combine_w,
            args.num_tokens, args.combine_dtype_name_effective,
            strict_tolerance=args.require_multi_source_bf16)

    if args.vllm_smoke:
        errors += run_vllm_smoke(
            torch, dist, kiwi_ep_ext, rank, world_size, mori_op, kiwi_op,
            tokens, indices, weights, scales, args, f"{label} vLLM-like")

    return errors



def run_downstream_sync_round(torch, dist, kiwi_ep_ext, rank: int, world_size: int,
                              mori_op, kiwi_op, tokens, indices, weights, scales,
                              args, label: str) -> int:
    """Focused downstream path: dispatch() then combine(..., None, topk_ids).

    This intentionally avoids MORI layout helpers, registered combine buffers,
    split APIs, and source-position helpers so vLLM/SGLang-named profiles cover
    only the public synchronous default-external-input-buffer path.
    """
    mori_block_num = mori_correctness_block_num(args)
    errors = 0

    mori_out, mori_w, mori_s, mori_idx, mori_count = mori_op.dispatch(
        tokens, weights, scales, indices,
        block_num=mori_block_num,
        rdma_block_num=args.rdma_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_result, call_errors = call_kiwi(
        rank, f"{label} dispatch", kiwi_op.dispatch,
        tokens, weights, scales, indices)
    errors += call_errors
    if kiwi_result is None:
        return errors
    kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count = kiwi_result
    barrier_all(torch, dist, kiwi_ep_ext)

    dispatch_errors, kiwi_total, mori_total = compare_dispatch_multiset_outputs(
        torch, rank, label,
        kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count,
        mori_out, mori_w, mori_s, mori_idx, mori_count)
    errors += dispatch_errors

    # Stand in for downstream fused expert outputs using only public dispatch
    # rows.  Each implementation may return rows in a different order, so build
    # expert outputs from that implementation's returned rows and let combine()
    # consume them with the operator's own recorded dispatch layout.
    mori_expert = make_downstream_expert_output(
        torch, mori_out, mori_count,
        args.combine_dtype_name_effective, args.combine_dtype_effective,
        args.hidden_dim)
    kiwi_expert = make_downstream_expert_output(
        torch, kiwi_out, kiwi_count,
        args.combine_dtype_name_effective, args.combine_dtype_effective,
        args.hidden_dim)

    mori_combine, mori_w_out = mori_op.combine(
        mori_expert, None, indices,
        block_num=mori_block_num,
        rdma_block_num=args.rdma_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_combine_result, call_errors = call_kiwi(
        rank, f"{label} combine(None weights)", kiwi_op.combine,
        kiwi_expert, None, indices)
    errors += call_errors
    if kiwi_combine_result is not None:
        kiwi_combine, kiwi_w_out = kiwi_combine_result
        barrier_all(torch, dist, kiwi_ep_ext)
        errors += compare_combine_outputs(
            torch, rank, f"{label} combine(None weights)",
            kiwi_combine, kiwi_w_out, mori_combine, mori_w_out,
            args.num_tokens, args.combine_dtype_name_effective,
            strict_tolerance=args.require_multi_source_bf16)

    print(f"[Rank {rank}] {label}: downstream sync public API path complete: "
          f"dispatch() recv_counts=(kiwi={kiwi_total}, mori={mori_total}) "
          "combine(..., None, topk_ids)")
    return errors

def run_vllm_smoke(torch, dist, kiwi_ep_ext, rank: int, world_size: int,
                   mori_op, kiwi_op, tokens, indices, weights, scales,
                   args, label: str) -> int:
    combine_hidden_elems = dtype_storage_hidden_elems(
        args.combine_dtype_name_effective, args.hidden_dim)
    mori_block_num = mori_correctness_block_num(args)
    errors = 0
    reset_ops(torch, dist, kiwi_ep_ext, mori_op, kiwi_op)

    mori_out, mori_w, mori_s, mori_idx, mori_count = mori_op.dispatch(
        tokens, weights, scales, indices,
        block_num=mori_block_num,
        rdma_block_num=args.rdma_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_result, call_errors = call_kiwi(
        rank, f"{label} dispatch", kiwi_op.dispatch,
        tokens, weights, scales, indices)
    errors += call_errors
    if kiwi_result is None:
        return errors
    kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count = kiwi_result
    barrier_all(torch, dist, kiwi_ep_ext)

    _mori_rows, mori_total, mori_order = dispatch_table(
        torch, f"mori-{label}", mori_op, mori_out, mori_w, mori_s,
        mori_idx, mori_count)
    _kiwi_rows, kiwi_total, kiwi_order = dispatch_table(
        torch, f"kiwi-{label}", kiwi_op, kiwi_out, kiwi_w, kiwi_s,
        kiwi_idx, kiwi_count)

    mori_expert = make_expert_output(
        torch, rank, mori_order, mori_total,
        mori_out.size(0), combine_hidden_elems, world_size,
        args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
        args.combine_dtype_effective, tokens.device,
        bf16_rounding_probe=args.require_multi_source_bf16)
    kiwi_expert = make_expert_output(
        torch, rank, kiwi_order, kiwi_total,
        kiwi_out.size(0), combine_hidden_elems, world_size,
        args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
        args.combine_dtype_effective, tokens.device,
        bf16_rounding_probe=args.require_multi_source_bf16)

    mori_combine, mori_w_out = mori_op.combine(
        mori_expert, None, indices,
        block_num=mori_block_num,
        rdma_block_num=args.rdma_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_combine_result, call_errors = call_kiwi(
        rank, f"{label} combine(None weights)", kiwi_op.combine,
        kiwi_expert, None, indices)
    errors += call_errors
    if kiwi_combine_result is not None:
        kiwi_combine, kiwi_w_out = kiwi_combine_result
        barrier_all(torch, dist, kiwi_ep_ext)
        errors += compare_combine_outputs(
            torch, rank, f"{label} combine(None weights)",
            kiwi_combine, kiwi_w_out, mori_combine, mori_w_out,
            args.num_tokens, args.combine_dtype_name_effective,
            strict_tolerance=args.require_multi_source_bf16)
    return errors


def run_split_round(torch, dist, kiwi_ep_ext, rank: int, world_size: int,
                    mori_op, kiwi_op, tokens, indices, weights, scales,
                    args, label: str) -> int:
    combine_hidden_elems = dtype_storage_hidden_elems(
        args.combine_dtype_name_effective, args.hidden_dim)
    mori_block_num = mori_correctness_block_num(args)
    errors = 0
    reset_ops(torch, dist, kiwi_ep_ext, mori_op, kiwi_op)

    mori_out, mori_w, mori_s, mori_idx, mori_count = mori_op.dispatch_send(
        tokens, weights, scales, indices,
        block_num=mori_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    mori_op.dispatch_recv(warp_per_block=args.warp_num_per_block)
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_result, call_errors = call_kiwi(
        rank, f"{label} dispatch_send", kiwi_op.dispatch_send,
        tokens, weights, scales, indices)
    errors += call_errors
    if not hasattr(kiwi_op, "dispatch_recv"):
        print(f"[Rank {rank}] {label}: kiwi missing dispatch_recv()",
              file=sys.stderr)
        errors += 1
        return errors
    _, recv_errors = call_kiwi(rank, f"{label} dispatch_recv", kiwi_op.dispatch_recv)
    errors += recv_errors
    if kiwi_result is None:
        return errors
    kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count = kiwi_result
    barrier_all(torch, dist, kiwi_ep_ext)

    mori_rows, mori_total, mori_order = dispatch_table(
        torch, f"mori-{label}", mori_op, mori_out, mori_w, mori_s,
        mori_idx, mori_count)
    kiwi_rows, kiwi_total, kiwi_order = dispatch_table(
        torch, f"kiwi-{label}", kiwi_op, kiwi_out, kiwi_w, kiwi_s,
        kiwi_idx, kiwi_count)

    errors += compare_layout_api(rank, kiwi_op, mori_op)
    errors += compare_dispatch_api_tensors(
        torch, rank, kiwi_op, mori_op,
        args.combine_dtype_effective, combine_hidden_elems,
        kiwi_out, kiwi_w, kiwi_s, kiwi_idx, kiwi_count,
        mori_out, mori_w, mori_s, mori_idx, mori_count)
    if args.num_tokens > 0:
        errors += compare_required_dispatch_keys(
            rank, "local", kiwi_rows,
            expected_local_mori_src_positions(
                rank, world_size, args.num_tokens, args.topk,
                args.num_experts_per_rank, args.max_num_inp_token_per_rank))
    if kiwi_total != mori_total:
        print(f"[Rank {rank}] {label} recv count mismatch: "
              f"kiwi={kiwi_total} mori={mori_total}", file=sys.stderr)
        errors += 1
    errors += compare_dispatch_tables(torch, rank, kiwi_rows, mori_rows)

    mori_expert = make_expert_output(
        torch, rank, mori_order, mori_total,
        mori_out.size(0), combine_hidden_elems, world_size,
        args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
        args.combine_dtype_effective, tokens.device,
        bf16_rounding_probe=args.require_multi_source_bf16)
    kiwi_expert = make_expert_output(
        torch, rank, kiwi_order, kiwi_total,
        kiwi_out.size(0), combine_hidden_elems, world_size,
        args.max_num_inp_token_per_rank, args.combine_dtype_name_effective,
        args.combine_dtype_effective, tokens.device,
        bf16_rounding_probe=args.require_multi_source_bf16)

    mori_combine, mori_w_out = mori_op.combine_send(
        mori_expert, None, indices,
        block_num=mori_block_num,
        warp_per_block=args.warp_num_per_block,
    )
    mori_op.combine_recv(warp_per_block=args.warp_num_per_block)
    barrier_all(torch, dist, kiwi_ep_ext)

    kiwi_combine_result, call_errors = call_kiwi(
        rank, f"{label} combine_send(None weights)", kiwi_op.combine_send,
        kiwi_expert, None, indices)
    errors += call_errors
    if hasattr(kiwi_op, "combine_recv"):
        _, recv_errors = call_kiwi(rank, f"{label} combine_recv", kiwi_op.combine_recv)
        errors += recv_errors
    else:
        print(f"[Rank {rank}] {label}: kiwi missing combine_recv()",
              file=sys.stderr)
        errors += 1
    if kiwi_combine_result is not None:
        kiwi_combine, kiwi_w_out = kiwi_combine_result
        barrier_all(torch, dist, kiwi_ep_ext)
        errors += compare_combine_outputs(
            torch, rank, f"{label} split combine",
            kiwi_combine, kiwi_w_out, mori_combine, mori_w_out,
            args.num_tokens, args.combine_dtype_name_effective,
            strict_tolerance=args.require_multi_source_bf16)
    return errors


def init_torch_dist(torch, dist, mori, rank: int, world_size: int, local_rank: int):
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
    except TypeError:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world_size,
        )
    try:
        torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    except RuntimeError:
        pass
    mori.shmem.shmem_torch_process_group_init("default")
    return device


def mori_kernel_type(mori, name: str):
    try:
        return getattr(mori.ops.EpDispatchCombineKernelType, name)
    except AttributeError as exc:
        skip(f"official MORI does not expose kernel type {name} ({exc})")


def first_slurm_hostname() -> Optional[str]:
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return None
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    for line in output.splitlines():
        host = line.strip()
        if host:
            return host
    return None


def default_master_port(args, world_size: int) -> str:
    # All ranks see the same argv, so this yields a stable per-case port when
    # the test is launched outside CTest.  CTest still injects explicit unique
    # ports so parallel external runners do not collide on a fixed default.
    seed = (
        f"{args.kernel_type}:{args.api_mode}:{args.dispatch_dtype}:"
        f"{args.scale_dim}:{args.scale_dtype}:{args.hidden_dim}:"
        f"{args.topk}:{world_size}"
    )
    return str(29537 + (sum(ord(ch) for ch in seed) % 1000))


def configure_torch_rendezvous(args, world_size: int, local_size: int) -> None:
    if "MASTER_ADDR" not in os.environ:
        if local_size < world_size:
            master_addr = first_slurm_hostname()
            if master_addr is None:
                skip("multi-node torch.distributed rendezvous requires "
                     "MASTER_ADDR or a SLURM_NODELIST resolvable by scontrol")
            os.environ["MASTER_ADDR"] = master_addr
        else:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", default_master_port(args, world_size))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="logical hidden dim; FP4 uses half as many storage hidden elements")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument(
        "--max-num-inp-token-per-rank",
        dest="max_num_inp_token_per_rank",
        type=int,
        default=32,
        help="MORI-compatible max input tokens per rank",
    )
    parser.add_argument("--num-tokens-list", default=None,
                        help=("comma-separated token counts/ranges to sweep, "
                              "e.g. '0,1-256'; each count must be <= "
                              "--max-num-inp-token-per-rank"))
    parser.add_argument("--num-experts-per-rank", type=int, default=4)
    parser.add_argument("--dispatch-dtype", choices=("float32", "bfloat16", "fp8", "fp4"),
                        default="float32")
    parser.add_argument("--scale-dim", type=int, default=0)
    parser.add_argument("--scale-dtype", choices=("auto", "float32", "e8m0"),
                        default="auto")
    parser.add_argument("--kernel-type",
                        choices=("IntraNode", "InterNodeV1", "InterNodeV1LL", "AsyncLL"),
                        default="InterNodeV1")
    parser.add_argument("--api-mode", choices=("sync", "split", "both"),
                        default="sync")
    parser.add_argument("--gpu-per-node", type=int, default=0,
                        help="0 means use MPI local size")
    parser.add_argument("--block-num", type=int, default=96)
    parser.add_argument("--rdma-block-num", type=int, default=64)
    parser.add_argument("--warp-num-per-block", type=int, default=8)
    parser.add_argument("--max-token-type-size", type=int, default=0,
                        help=("override MORI max_token_type_size; quantized "
                              "SGLang/vLLM paths use the original activation "
                              "dtype size rather than the packed dispatch dtype"))
    parser.add_argument("--stop-on-first-token-error", action="store_true",
                        help=("in a token-count sweep, stop after the first "
                              "failing token count"))
    parser.add_argument("--no-combine-output-weights",
                        dest="combine_output_weights", action="store_false",
                        help=("do not pass dispatch weights into combine; "
                              "MORI then does not expose combine output weights"))
    parser.add_argument("--no-vllm-smoke",
                        dest="vllm_smoke", action="store_false",
                        help=("skip the vLLM/SGLang-like "
                              "dispatch->expert->combine(..., None, topk_ids) "
                              "synchronous smoke round"))
    parser.add_argument("--downstream-sync-only", action="store_true",
                        help=("exercise only downstream public sync APIs: "
                              "dispatch(tokens, weights, scales, topk_ids) and "
                              "combine(fused_expert_output, None, topk_ids); "
                              "skip layout, registered-buffer, split, and "
                              "source-position helper assertions"))
    parser.add_argument("--require-multi-source-bf16", action="store_true",
                        help=("require a bf16 route with at least three unique "
                              "destination ranks and use rounding-sensitive "
                              "expert outputs"))
    parser.set_defaults(combine_output_weights=True, vllm_smoke=True)
    args = parser.parse_args()

    if args.downstream_sync_only and args.api_mode != "sync":
        raise AssertionError("--downstream-sync-only requires --api-mode sync")
    if args.api_mode == "split" and args.kernel_type != "AsyncLL":
        raise AssertionError("split API coverage must use --kernel-type AsyncLL")
    if args.require_multi_source_bf16:
        if args.dispatch_dtype != "bfloat16":
            raise AssertionError("--require-multi-source-bf16 requires --dispatch-dtype bfloat16")
    token_counts = parse_token_counts(args.num_tokens_list, args.num_tokens)
    if max(token_counts) > args.max_num_inp_token_per_rank:
        raise AssertionError(
            f"requested token count {max(token_counts)} exceeds "
            f"--max-num-inp-token-per-rank={args.max_num_inp_token_per_rank}")

    torch, dist, kiwi_ep_ext, mori = import_or_skip()
    dispatch_dtype = dtype_from_name(torch, args.dispatch_dtype)
    dispatch_elem_size = dtype_element_size(torch, dispatch_dtype)
    model_hidden_dim = config_hidden_dim(args.dispatch_dtype, args.hidden_dim)
    scale_dtype = scale_dtype_from_name(torch, args.dispatch_dtype, args.scale_dtype)
    dispatch_scale_type_size = dtype_element_size(torch, scale_dtype) if args.scale_dim > 0 else 0
    max_token_type_size = (
        args.max_token_type_size if args.max_token_type_size > 0 else dispatch_elem_size)
    args.max_token_type_size_effective = max_token_type_size
    if args.dispatch_dtype in ("fp8", "fp4") and max_token_type_size == dispatch_elem_size:
        combine_dtype = dispatch_dtype
        combine_dtype_name = args.dispatch_dtype
    elif max_token_type_size == dtype_element_size(torch, torch.bfloat16):
        combine_dtype = torch.bfloat16
        combine_dtype_name = "bfloat16"
    elif max_token_type_size == dtype_element_size(torch, torch.float32):
        combine_dtype = torch.float32
        combine_dtype_name = "float32"
    else:
        raise AssertionError(
            f"cannot derive explicit combine dtype from max_token_type_size={max_token_type_size}")
    args.combine_dtype_effective = combine_dtype
    args.combine_dtype_name_effective = combine_dtype_name
    combine_quant_format = combine_dtype_name if combine_dtype_name in ("fp8", "fp4") else "none"

    rank = env_int("OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK")
    world_size = env_int("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE", default=1)
    local_rank = env_int("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK", default=rank)
    local_size = env_int("OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS",
                         "MV2_COMM_WORLD_LOCAL_SIZE", default=world_size)
    if args.require_multi_source_bf16 and world_size < 4:
        skip("multi-source bf16 probe requires at least 4 ranks")
    configure_torch_rendezvous(args, world_size, local_size)

    device_count = torch.cuda.device_count()
    if device_count < local_size or local_rank >= device_count:
        skip("official MORI compatibility test requires one visible GPU per "
             f"local MPI rank (visible_gpus={device_count}, "
             f"local_size={local_size}, local_rank={local_rank})")

    device = init_torch_dist(torch, dist, mori, rank, world_size, local_rank)
    lci_ctx = None
    mori_op = None
    kiwi_op = None
    try:
        lci_ctx = kiwi_ep_ext.LciContext()
        if kiwi_ep_ext.get_rank() != rank or kiwi_ep_ext.get_nranks() != world_size:
            raise AssertionError(
                f"LCI rank/world mismatch: lci=({kiwi_ep_ext.get_rank()},"
                f" {kiwi_ep_ext.get_nranks()}) env=({rank}, {world_size})"
            )

        kernel_type = mori_kernel_type(mori, args.kernel_type)
        gpu_per_node = args.gpu_per_node if args.gpu_per_node > 0 else local_size
        # MORI names this public payload dtype `data_type` and exposes one
        # generic scale layout. In these compatibility cases that layout maps
        # to KIWI dispatch/input scales; combine/expert-output scales remain 0.
        mori_cfg = mori.ops.EpDispatchCombineConfig(
            data_type=dispatch_dtype,
            rank=rank,
            world_size=world_size,
            hidden_dim=model_hidden_dim,
            scale_dim=args.scale_dim,
            scale_type_size=dispatch_scale_type_size,
            max_token_type_size=max_token_type_size,
            max_num_inp_token_per_rank=args.max_num_inp_token_per_rank,
            num_experts_per_rank=args.num_experts_per_rank,
            num_experts_per_token=args.topk,
            warp_num_per_block=args.warp_num_per_block,
            block_num=mori_correctness_block_num(args),
            use_external_inp_buf=True,
            kernel_type=kernel_type,
            gpu_per_node=gpu_per_node,
            rdma_block_num=args.rdma_block_num,
            num_qp_per_pe=1,
        )
        mori_op = mori.ops.EpDispatchCombineOp(mori_cfg)
        try:
            kiwi_op = kiwi_ep_ext.KiwiEpOp(
                device_id=local_rank,
                world_size=world_size,
                hidden_dim=model_hidden_dim,
                topk=args.topk,
                num_experts_per_rank=args.num_experts_per_rank,
                max_num_inp_token_per_rank=args.max_num_inp_token_per_rank,
                dispatch_dtype=dispatch_dtype,
                combine_dtype=combine_dtype,
                num_blocks=4,
                num_proxy_threads=4,
                dispatch_elem_size=dispatch_elem_size,
                dispatch_scale_dim=args.scale_dim,
                dispatch_scale_type_size=dispatch_scale_type_size,
                combine_scale_dim=0,
                combine_scale_type_size=0,
                dispatch_scale_dtype=scale_dtype,
                dispatch_quant_format=args.dispatch_dtype if args.dispatch_dtype in ("fp8", "fp4") else "none",
                combine_quant_format=combine_quant_format,
            )
        except Exception as exc:
            print(f"[Rank {rank}] KiwiEpOp construction failed: {exc}",
                  file=sys.stderr)
            return 1

        dist.barrier()
        kiwi_ep_ext.barrier()

        errors = 0
        original_num_tokens = args.num_tokens
        for num_tokens in token_counts:
            args.num_tokens = num_tokens
            tokens, indices, weights, scales = make_inputs(
                torch, rank, world_size, args.num_tokens, args.hidden_dim,
                args.topk, args.num_experts_per_rank, args.dispatch_dtype, dispatch_dtype,
                args.scale_dim, scale_dtype, device)

            precondition_intranode_empty_scale_path(
                torch, dist, kiwi_ep_ext, rank, world_size, mori_op, kiwi_op,
                tokens, indices, weights, scales, args)

            scale_variants = [
                (scales, "scale-tensor" if args.scale_dim > 0 else "empty-scale-tensor")
            ]
            if args.scale_dim == 0:
                scale_variants.append((None, "scales-none"))

            token_errors = 0
            for dispatch_scales, scale_label in scale_variants:
                if args.downstream_sync_only:
                    token_errors += run_downstream_sync_round(
                        torch, dist, kiwi_ep_ext, rank, world_size, mori_op, kiwi_op,
                        tokens, indices, weights, dispatch_scales, args,
                        f"{args.kernel_type}/downstream-sync/tokens{num_tokens}/{scale_label}")
                    continue
                if args.api_mode in ("sync", "both"):
                    token_errors += run_sync_round(
                        torch, dist, kiwi_ep_ext, rank, world_size, mori_op, kiwi_op,
                        tokens, indices, weights, dispatch_scales, args,
                        f"{args.kernel_type}/sync/tokens{num_tokens}/{scale_label}")
                if args.api_mode in ("split", "both"):
                    token_errors += run_split_round(
                        torch, dist, kiwi_ep_ext, rank, world_size, mori_op, kiwi_op,
                        tokens, indices, weights, dispatch_scales, args,
                        f"{args.kernel_type}/split/tokens{num_tokens}/{scale_label}")
            errors += token_errors
            print(f"[Rank {rank}] token-count case complete: "
                  f"tokens={num_tokens} errors={token_errors}")
            if token_errors and args.stop_on_first_token_error:
                stop = torch.tensor([1], dtype=torch.int32, device=device)
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                break
            if args.stop_on_first_token_error:
                stop = torch.tensor([0], dtype=torch.int32, device=device)
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if int(stop.cpu()[0].item()) != 0:
                    break
        args.num_tokens = original_num_tokens

        print(f"[Rank {rank}] official MORI drop-in comparison: "
              f"kernel={args.kernel_type} api={args.api_mode} "
              f"dispatch_dtype={args.dispatch_dtype} "
              f"effective_combine_dtype={args.combine_dtype_name_effective} "
              f"max_token_type_size={args.max_token_type_size_effective} "
              f"scale_dim={args.scale_dim} "
              f"world_size={world_size} gpu_per_node={gpu_per_node} "
              f"token_counts={token_counts} "
              f"errors={errors}")
        return 0 if errors == 0 else 1
    finally:
        del kiwi_op
        del mori_op
        del lci_ctx
        try:
            mori.shmem.shmem_finalize()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
