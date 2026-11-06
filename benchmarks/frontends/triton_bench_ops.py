# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Device and launch helpers the Triton harness kernels share.

The device functions mirror lines of the C++ harness kernels
(benchmarks/invoke/frontends, benchmarks/queue_baselines/frontends,
apps/p2p/benchmarks/frontends). Copied into the build tree's python/.
"""

import re

import torch
import triton
import triton.language as tl

import kiwi_triton as kiwi


@triton.jit
def timestamp(WALL: tl.constexpr):
    """wall_clock64() (s_memrealtime) or clock64() (s_memtime), resolved
    immediately (s_waitcnt lgkmcnt(0)), as the C++ timestamp<kWall>()."""
    if WALL:
        t = tl.inline_asm_elementwise("s_memrealtime $0\n s_waitcnt lgkmcnt(0)", "=s", [],
                                      dtype=tl.uint64, is_pure=False, pack=1)
    else:
        t = tl.inline_asm_elementwise("s_memtime $0\n s_waitcnt lgkmcnt(0)", "=s", [],
                                      dtype=tl.uint64, is_pure=False, pack=1)
    return t


@triton.jit
def wall_clock64():
    """wall_clock64(): s_memrealtime."""
    return timestamp(True)


@triton.jit
def wait_vmcnt0():
    """asm volatile("s_waitcnt vmcnt(0)" ::: "memory")."""
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=s,~{memory}", [], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def drain_memory_ops():
    """asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory")."""
    tl.inline_asm_elementwise("s_waitcnt vmcnt(0) lgkmcnt(0)", "=s,~{memory}", [],
                              dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def u64_ptr(addr):
    """An integer address as a uint64 pointer."""
    return tl.cast(addr, tl.uint64).to(tl.pointer_type(tl.uint64))


def require_no_scratch(compiled):
    """The C++ benchmarks' require_no_scratch() for a compiled Triton kernel: a
    payload in scratch would make each push wait for the previous one."""
    size = re.search(r"\.private_segment_fixed_size:\s+(\d+)", compiled.asm["amdgcn"])
    if size is None or int(size.group(1)) != 0:
        raise RuntimeError(f"Triton kernel {compiled.name} uses scratch (private segment "
                           f"{size.group(1) if size else 'unknown'} bytes)")
    return compiled


def launch_empty(kernel, grid, *args, **kwargs):
    """Compile ``kernel`` for these arguments and check it before launching it;
    then launch it, with arguments that make it do no work, and wait, so the
    module is loaded outside any timed region."""
    require_no_scratch(kiwi.warmup(kernel, grid, *args, **kwargs))
    kiwi.launch(kernel, grid, *args, **kwargs)
    torch.cuda.synchronize()
