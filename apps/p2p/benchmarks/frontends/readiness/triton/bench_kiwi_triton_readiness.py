# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Triton frontend of the KIWI single-PUT readiness harness.

Same options, host code and result line (backend=kiwi_triton) as the
harness's C++ frontend (kiwi_readiness_bench); run_protocol mirrors the C++
run_protocol kernel, with the rank as a compile-time constant. Run with
exactly two ranks.
"""

import torch  # noqa: F401  (load the HIP runtime Triton uses before the extension)
import triton
import triton.language as tl

import kiwi_triton as kiwi
import kiwi_readiness_bench
from triton_bench_ops import require_no_scratch

# sizeof(readiness_put_t): {u64 source; u64 remote_offset; i32 source_heap;
# i32 target_heap; i32 rank;} plus 4 bytes of padding.
DESC_BYTES = tl.constexpr(32)


@triton.jit
def load_system(ptr):
    """readiness_bench::load_system: global_load_dwordx2 ... sc0 sc1."""
    return tl.load(ptr, cache_modifier=".cv", volatile=True)


@triton.jit
def readiness_put(source, remote_offset, source_heap, target_heap, rank):
    """ArgumentSerializer<readiness_put_t>: the descriptor's four payload words."""
    lane = tl.arange(0, 64)
    w0 = tl.cast(source, tl.uint64)
    w1 = tl.cast(remote_offset, tl.uint64)
    w2 = (tl.cast(tl.cast(source_heap, tl.uint32), tl.uint64)
          | (tl.cast(tl.cast(target_heap, tl.uint32), tl.uint64) << 32))
    w3 = tl.cast(tl.cast(rank, tl.uint32), tl.uint64)
    return tl.where(lane == 0, w0, tl.where(lane == 1, w1, tl.where(lane == 2, w2,
                    tl.where(lane == 3, w3, 0)))).to(tl.uint64)


@triton.jit(do_not_specialize=["peer", "source", "target", "source_heap", "target_heap",
                                "cb_id", "iters", "depth"])
def run_protocol(device_context, peer, source, target, source_heap, target_heap, cb_id, iters,
                 depth, RANK: tl.constexpr):
    # gctx.get_handle(): one block, one endpoint.
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(
        device_context, 0)
    src = tl.cast(source, tl.uint64)
    tgt = tl.cast(target, tl.uint64).to(tl.pointer_type(tl.uint64))
    for e in range(iters):
        epoch = tl.cast(e, tl.uint64) + 1
        if RANK == 0:
            for slot in range(depth):
                desc = readiness_put(src + ((epoch - 1) * depth + slot) * 8, slot * 8,
                                     source_heap, target_heap, peer)
                producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached,
                                               log2_capacity, cb_id, desc, DESC_BYTES)
            # if (threadIdx.x == 0) while (load_system(target) < epoch) {}
            while load_system(tgt) < epoch:
                pass
        else:
            # for (slot = threadIdx.x; slot < depth; slot += blockDim.x)
            #     while (load_system(target + slot) < epoch) {}
            lane = tl.arange(0, 64)
            for base in range(0, depth, 64):
                slots = base + lane
                pending = True
                valid = slots < depth
                while pending:
                    values = tl.load(tgt + slots, mask=valid, other=0, cache_modifier=".cv",
                                     volatile=True)
                    pending = tl.max(tl.where(valid & (values < epoch), 1, 0), 0) != 0
            tl.debug_barrier()
            desc = readiness_put(src + (epoch - 1) * depth * 8, 0, source_heap, target_heap,
                                 peer)
            producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached,
                                           log2_capacity, cb_id, desc, DESC_BYTES)
        tl.debug_barrier()
    # ~ghandle_t(): save_device_handle().
    kiwi.save_device_handle(device_context, 0, producer, cached)


_checked = set()


def launch(ctx, rank, peer, source, target, source_heap, target_heap, cb_id, iters, depth):
    args = (ctx, peer, source, target, source_heap, target_heap, cb_id, iters, depth)
    if rank not in _checked:
        require_no_scratch(kiwi.warmup(run_protocol, (1,), *args, RANK=rank))
        _checked.add(rank)
    kiwi.launch(run_protocol, (1,), *args, RANK=rank)


if __name__ == "__main__":
    kiwi_readiness_bench.main(
        launch, "kiwi_triton",
        "KIWI single-PUT GPU-polled readiness benchmark (Triton kernel)")
