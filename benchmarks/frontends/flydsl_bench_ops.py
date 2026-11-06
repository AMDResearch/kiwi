# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Low-level device operations the FlyDSL harness kernels share.

Each mirrors a line of the C++ harness kernels (benchmarks/invoke/frontends,
apps/p2p/benchmarks/frontends). Copied into the build tree's python/.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm
import flydsl.expr as fx


def i64():
    return ir.IntegerType.get_signless(64)


def const_i64(value):
    return arith.ConstantOp(i64(), value).result


def _ir(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def as_i64(value):
    """Zero-extend an integer or index DSL value to an i64 ir.Value."""
    value = _ir(value)
    if isinstance(value.type, ir.IndexType):
        return arith.IndexCastUIOp(i64(), value).result
    width = ir.IntegerType(value.type).width
    if width < 64:
        return arith.ExtUIOp(i64(), value).result
    return value


def _global_ptr(addr):
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_i64(addr)).result


def wall_clock64():
    """wall_clock64(): s_memrealtime."""
    return fx.Uint64(llvm.call_intrinsic(i64(), "llvm.amdgcn.s.memrealtime", [], [], []))


def wait_vmcnt0():
    """asm volatile("s_waitcnt vmcnt(0)" ::: "memory")."""
    llvm.inline_asm(None, [], "s_waitcnt vmcnt(0)", "~{memory}", has_side_effects=True)


def load_system(addr):
    """*(volatile uint64_t*)addr as an i64 ir.Value: global_load_dwordx2 ... sc0 sc1
    (readiness_bench::load_system)."""
    return llvm.LoadOp(i64(), _global_ptr(addr), alignment=8, volatile_=True).result


def load_volatile_u64(addr):
    """*(volatile uint64_t*)addr."""
    return fx.Uint64(load_system(addr))


def store_u64(addr, value):
    """*(uint64_t*)addr = value."""
    llvm.StoreOp(as_i64(value), _global_ptr(addr), alignment=8)

