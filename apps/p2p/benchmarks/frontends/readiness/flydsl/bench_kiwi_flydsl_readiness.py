# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""FlyDSL frontend of the KIWI single-PUT readiness harness.

Same options, host code and result line (backend=kiwi_flydsl) as the
harness's C++ frontend (kiwi_readiness_bench); sender_kernel and
receiver_kernel are the two rank branches of its run_protocol kernel. Run
with exactly two ranks.
"""

import torch  # noqa: F401  (load the HIP runtime FlyDSL uses before the extension)
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, scf
import kiwi_flydsl as kiwi
import kiwi_readiness_bench

from flydsl_bench_ops import as_i64 as _as_i64, const_i64 as _c64, load_system

# readiness_put_t {u64 source; u64 remote_offset; i32 source_heap;
# i32 target_heap; i32 rank;} plus 4 bytes of padding.
DESC_TYPES = (fx.Uint64, fx.Uint64, fx.Int32, fx.Int32, fx.Int32, fx.Int32)


def _add(a, b):
    return arith.AddIOp(a, b).result


def _mul(a, b):
    return arith.MulIOp(a, b).result


def _spin_until_at_least(addr, epoch):
    loop = scf.WhileOp([], [])
    before = loop.before.blocks.append()
    with ir.InsertionPoint(before):
        pending = arith.CmpIOp(arith.CmpIPredicate.ult, load_system(addr), epoch).result
        scf.ConditionOp(pending, [])
    after = loop.after.blocks.append()
    with ir.InsertionPoint(after):
        scf.YieldOp([])


def _thread0_waits(target, epoch):
    """if (threadIdx.x == 0) while (load_system(target) < epoch) {}"""
    tid = _as_i64(fx.thread_idx.x)
    first = arith.CmpIOp(arith.CmpIPredicate.eq, tid, _c64(0)).result
    wait = scf.IfOp(first, [], has_else=False)
    with ir.InsertionPoint(wait.regions[0].blocks[0]):
        _spin_until_at_least(_as_i64(target), epoch)
        scf.YieldOp([])


def _threads_wait_all_slots(target, depth, epoch):
    """for (slot = threadIdx.x; slot < depth; slot += blockDim.x) wait(slot)"""
    index = ir.IndexType.get()
    lb = arith.IndexCastUIOp(index, _as_i64(fx.thread_idx.x)).result
    ub = arith.IndexCastUIOp(index, _as_i64(depth)).result
    step = arith.IndexCastUIOp(index, _as_i64(fx.block_dim.x)).result
    loop = scf.ForOp(lb, ub, step)
    with ir.InsertionPoint(loop.body):
        slot = _as_i64(loop.induction_variable)
        _spin_until_at_least(_add(_as_i64(target), _mul(slot, _c64(8))), epoch)
        scf.YieldOp([])


@flyc.kernel
def sender_kernel(ctx: kiwi.DeviceContext, peer: fx.Int32, source: fx.Int64,
                  target: fx.Int64, source_heap: fx.Int32, target_heap: fx.Int32,
                  cb: fx.Int32, iters: fx.Int32, depth: fx.Int32):
    """Rank 0: post `depth` epoch writes, then wait for the ACK."""
    h = ctx.get_device_handle(0)
    for e in range(iters):
        epoch = _add(_as_i64(e), _c64(1))
        for slot in range(depth):
            slot64 = _as_i64(slot)
            value = _add(_mul(_as_i64(e), _as_i64(depth)), slot64)
            src = _add(_as_i64(source), _mul(value, _c64(8)))
            h = h.invoke(cb, fx.Uint64(src), fx.Uint64(_mul(slot64, _c64(8))),
                         source_heap, target_heap, peer, fx.Int32(0),
                         arg_types=DESC_TYPES)
        _thread0_waits(target, epoch)
        fx.barrier()
    ctx.save_device_handle(0, h)


@flyc.kernel
def receiver_kernel(ctx: kiwi.DeviceContext, peer: fx.Int32, source: fx.Int64,
                    target: fx.Int64, source_heap: fx.Int32, target_heap: fx.Int32,
                    cb: fx.Int32, iters: fx.Int32, depth: fx.Int32):
    """Rank 1: wait for all `depth` slots, then post the epoch ACK."""
    h = ctx.get_device_handle(0)
    for e in range(iters):
        epoch = _add(_as_i64(e), _c64(1))
        _threads_wait_all_slots(target, depth, epoch)
        fx.barrier()
        src = _add(_as_i64(source), _mul(_mul(_as_i64(e), _as_i64(depth)), _c64(8)))
        h = h.invoke(cb, fx.Uint64(src), fx.Uint64(_c64(0)),
                     source_heap, target_heap, peer, fx.Int32(0),
                     arg_types=DESC_TYPES)
        fx.barrier()
    ctx.save_device_handle(0, h)


@flyc.jit
def launch_protocol(ctx: kiwi.DeviceContext, rank: fx.Constexpr[int], peer: fx.Int32,
           source: fx.Int64, target: fx.Int64, source_heap: fx.Int32,
           target_heap: fx.Int32, cb: fx.Int32, iters: fx.Int32, depth: fx.Int32,
           stream: fx.Stream = fx.Stream(None)):
    kernel = sender_kernel if rank == 0 else receiver_kernel
    kernel(ctx, peer, source, target, source_heap, target_heap, cb, iters, depth).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def launch(ctx, rank, peer, source, target, source_heap, target_heap, cb_id, iters,
           depth):
    launch_protocol(kiwi.Endpoints(ctx), rank, peer, source, target, source_heap,
                    target_heap, cb_id, iters, depth)


if __name__ == "__main__":
    kiwi_readiness_bench.main(
        launch, "kiwi_flydsl",
        "KIWI single-PUT GPU-polled readiness benchmark (FlyDSL kernel)")
