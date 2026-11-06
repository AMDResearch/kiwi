# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""KIWI invoke from Triton kernels.

Triton device functions that mirror the C++ device side of KIWI-Invoke over
the SPSC device-to-host queue:

- ``load_device_handle`` is ``DeviceContext::get_device_handle(endpoint)``;
  ``load_handle`` loads a ``DeviceHandle`` at a given address.
- ``try_push`` is ``SPSCD2HQueue::DeviceHandle::push_lanes`` with
  ``detail::write_message``: a 32-bit credit check, a rare credit refresh
  over PCIe, and one 16-byte chunk store per active lane. ``push`` retries it
  until it succeeds, as C++ ``while (!handle.push_lanes(...)) {}``.
- ``try_invoke`` and ``invoke`` are ``DeviceHandle::invoke(callback_id,
  args...)`` (once, and retried) for arguments already packed into per-lane
  payload words as ``ArgumentSerializer`` packs them; ``invoke_u32`` packs a
  ``uint32_t`` followed by zero padding (the invoke benchmarks'
  ``FixedPayload<N>``).
- ``save_device_handle`` is ``DeviceContext::save_device_handle``.

Targets: AMD GPUs that run 64-lane wavefronts (CDNA, gfx9), like the C++
device-to-host queues. One 64-lane wavefront owns an endpoint and a ``[64]``
tensor holds one element per lane, so kernels must run with ``num_warps=1``:
launch them with :func:`launch` (or compile them with :func:`warmup`). The
handle is a tuple ``(chunks, consumer_index, producer_index,
consumer_index_cached, log2_capacity)``; use the queue state each call
returns, as with the by-value C++ handle.
"""

import triton
import triton.language as tl

# kiwi::invoke::DeviceContext and SPSCD2HQueue::DeviceHandle layout. The tests
# check these against the C++ headers.
CONTEXT_HANDLES_OFFSET = 0
HANDLE_SIZE = 40
CHUNKS_OFFSET = 0
CONSUMER_INDEX_OFFSET = 8
PRODUCER_INDEX_OFFSET = 16
CONSUMER_INDEX_CACHED_OFFSET = 24
LOG2_CAPACITY_OFFSET = 32  # uint32_t
CHUNK_BYTES = 16
CHUNK_PAYLOAD_BYTES = 8
CHUNKS_PER_LINE = 4
MAX_MESSAGE_BYTES = 512  # kMaxMessageBytes
MAX_CAPACITY_LINES = 1 << 31  # kMaxCapacityLines: the push checks credit in 32 bits

# Device-code copies: Triton JIT functions only read tl.constexpr globals.
_CONTEXT_HANDLES_OFFSET = tl.constexpr(CONTEXT_HANDLES_OFFSET)
_HANDLE_SIZE = tl.constexpr(HANDLE_SIZE)
_CHUNKS_OFFSET = tl.constexpr(CHUNKS_OFFSET)
_CONSUMER_INDEX_OFFSET = tl.constexpr(CONSUMER_INDEX_OFFSET)
_PRODUCER_INDEX_OFFSET = tl.constexpr(PRODUCER_INDEX_OFFSET)
_CONSUMER_INDEX_CACHED_OFFSET = tl.constexpr(CONSUMER_INDEX_CACHED_OFFSET)
_LOG2_CAPACITY_OFFSET = tl.constexpr(LOG2_CAPACITY_OFFSET)
_CHUNK_BYTES = tl.constexpr(CHUNK_BYTES)
_CHUNK_PAYLOAD_BYTES = tl.constexpr(CHUNK_PAYLOAD_BYTES)
_CHUNKS_PER_LINE = tl.constexpr(CHUNKS_PER_LINE)
_LINE_PAYLOAD_BYTES = tl.constexpr(CHUNK_PAYLOAD_BYTES * CHUNKS_PER_LINE)
_MAX_MESSAGE_BYTES = tl.constexpr(MAX_MESSAGE_BYTES)


def _one_wave(kwargs):
    if kwargs.get("num_warps", 1) != 1:
        raise ValueError("KIWI Triton kernels run one 64-lane wavefront per program: "
                         "num_warps must be 1")
    return dict(kwargs, num_warps=1)


def launch(kernel, grid, *args, **kwargs):
    """``kernel[grid](*args, **kwargs)`` with ``num_warps=1``; returns the compiled kernel."""
    return kernel[grid](*args, **_one_wave(kwargs))


def warmup(kernel, grid, *args, **kwargs):
    """Compile ``kernel`` for these arguments, with ``num_warps=1``, without
    launching it; returns the compiled kernel (``.asm["amdgcn"]`` is its ISA)."""
    return kernel.warmup(*args, grid=grid, **_one_wave(kwargs))


@triton.jit
def _trap():
    """__builtin_trap(): s_trap 2 aborts the queue (the kernel faults)."""
    tl.inline_asm_elementwise("s_trap 2", "=s", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _load_u64(addr):
    return tl.load(addr.to(tl.pointer_type(tl.uint64)))


@triton.jit
def load_handle(handle):
    """Load an SPSCD2HQueue::DeviceHandle at address ``handle``.

    Returns ``(chunks, consumer_index, producer_index, consumer_index_cached,
    log2_capacity)`` as uint64 scalars.
    """
    base = tl.cast(handle, tl.uint64)
    chunks = _load_u64(base + _CHUNKS_OFFSET)
    consumer_index = _load_u64(base + _CONSUMER_INDEX_OFFSET)
    producer = _load_u64(base + _PRODUCER_INDEX_OFFSET)
    cached = _load_u64(base + _CONSUMER_INDEX_CACHED_OFFSET)
    log2_capacity = tl.load((base + _LOG2_CAPACITY_OFFSET).to(tl.pointer_type(tl.uint32)))
    return chunks, consumer_index, producer, cached, log2_capacity.to(tl.uint64)


@triton.jit
def load_device_handle(device_context, endpoint):
    """DeviceContext::get_device_handle(endpoint); see load_handle."""
    handles = _load_u64(tl.cast(device_context, tl.uint64) + _CONTEXT_HANDLES_OFFSET)
    return load_handle(handles + tl.cast(endpoint, tl.uint64) * _HANDLE_SIZE)


@triton.jit
def write_message(chunks, log2_capacity, base, word, metadata, BYTES: tl.constexpr):
    """detail::write_message: lane c stores chunk c = {data lo, sig 0, data hi,
    sig 1} (LL8) for payload word c of a BYTES-byte message at absolute line
    ``base``, as two 8-byte halves ``data lo | sig 0 << 32`` and
    ``data hi | sig 1 << 32``.

    Each 8-byte half must be single-copy atomic; the C++ Chunk store is one
    global_store_dwordx4 per active lane. tl.join keeps a lane's two halves in
    one thread, and the alignment hint lets Triton store them with one 16-byte
    vector store: the ring and every chunk are 16-byte aligned, but a pointer
    made from an integer is otherwise assumed 8-byte aligned and the store
    would split in two.
    """
    LINES: tl.constexpr = 1 if BYTES == 0 else (BYTES + _LINE_PAYLOAD_BYTES - 1) // _LINE_PAYLOAD_BYTES
    lane = tl.arange(0, 64)
    line = base + (lane // _CHUNKS_PER_LINE).to(tl.uint64)
    epoch = (line >> log2_capacity) & 0xFF
    data = tl.where(lane * _CHUNK_PAYLOAD_BYTES < BYTES, word, 0).to(tl.uint64)
    # Signal pair sig 0 | sig 1 << 32 (C++ make_signal): the epoch in both
    # signals; lane 0's header has metadata bits 0-7 at 8-15, the length at
    # 16-31 and metadata bits 8-31 at 40-63. The header carries no epoch, so
    # it is loop-invariant in a push loop; every lane ORs in its line's epoch.
    meta = metadata.to(tl.uint64)
    header = tl.where(lane == 0, ((meta & 0xFF) << 8) | (BYTES << 16) | ((meta >> 8) << 40),
                      0).to(tl.uint64)
    signal = header | epoch | (epoch << 32)
    low32 = tl.full((), 0xFFFFFFFF, tl.uint64)
    half0 = (data & low32) | (signal << 32)
    half1 = (data >> 32) | (signal & (low32 << 32))
    mask = (tl.full((), 1, tl.uint64) << log2_capacity) - 1
    chunk = (line & mask) * _CHUNKS_PER_LINE + (lane % _CHUNKS_PER_LINE).to(tl.uint64)
    ring = tl.multiple_of(chunks.to(tl.pointer_type(tl.uint64)), _CHUNK_BYTES)
    ptrs = (ring + chunk * 2)[:, None] + tl.arange(0, 2)[None, :]
    active = lane < LINES * _CHUNKS_PER_LINE
    tl.store(ptrs, tl.join(half0, half1), mask=active[:, None])


@triton.jit
def _load_published(consumer_index):
    """Lane 0 reads the published consumer index over PCIe and the wave takes
    its value (C++ load_published + broadcast_lane0), so every lane agrees on
    the refreshed credit."""
    lane = tl.arange(0, 64)
    ptrs = consumer_index.to(tl.pointer_type(tl.uint64)) + lane * 0
    fresh = tl.load(ptrs, mask=lane == 0, other=0, cache_modifier=".cv", volatile=True)
    return tl.max(fresh, 0)


@triton.jit
def try_push(chunks, consumer_index, producer, cached, log2_capacity, word, metadata,
             BYTES: tl.constexpr):
    """SPSCD2HQueue::DeviceHandle::push_lanes(word, BYTES, metadata).

    Lane c's ``word`` is payload word c. Returns ``(producer_index,
    consumer_index_cached, ok)``; nothing is written unless ``ok``.
    """
    tl.static_assert(BYTES >= 0, "negative message length")
    tl.static_assert(BYTES <= _MAX_MESSAGE_BYTES, "message longer than kMaxMessageBytes")
    LINES: tl.constexpr = 1 if BYTES == 0 else (BYTES + _LINE_PAYLOAD_BYTES - 1) // _LINE_PAYLOAD_BYTES
    # log2_capacity is a uint32_t, and the credit check runs in 32 bits, as
    # C++: lines in flight never exceed the capacity, which the host keeps at
    # most kMaxCapacityLines, so used + lines cannot wrap.
    cap = tl.full((), 1, tl.uint32) << log2_capacity.to(tl.uint32)
    ok = True
    if (producer - cached).to(tl.uint32) + LINES > cap:
        # Rare path: a message longer than the ring could never fit.
        if LINES > cap:
            _trap()
        cached = _load_published(consumer_index)
        ok = (producer - cached).to(tl.uint32) + LINES <= cap
    # Write under ok instead of returning early, as C++.
    if ok:
        write_message(chunks, log2_capacity, producer, word, metadata, BYTES)
        producer = producer + LINES
    return producer, cached, ok


@triton.jit
def push(chunks, consumer_index, producer, cached, log2_capacity, word, metadata,
         BYTES: tl.constexpr):
    """``while (!handle.push_lanes(word, BYTES, metadata)) {}``.

    Returns ``(producer_index, consumer_index_cached)``.
    """
    ok = False
    while not ok:
        producer, cached, ok = try_push(chunks, consumer_index, producer, cached,
                                        log2_capacity, word, metadata, BYTES)
    return producer, cached


@triton.jit
def try_invoke(chunks, consumer_index, producer, cached, log2_capacity, callback_id, word,
               BYTES: tl.constexpr):
    """handle.invoke(callback_id, args...) for serialized arguments: one attempt.

    ``word`` is a ``[64]`` uint64 tensor whose element c holds argument bytes
    ``[8c, 8c + 8)`` as packed by ``ArgumentSerializer`` (little-endian, no
    padding between arguments); BYTES is their total size. The callback id is
    a ``uint8_t``. Returns ``(producer_index, consumer_index_cached, ok)``.
    """
    return try_push(chunks, consumer_index, producer, cached, log2_capacity, word,
                    tl.cast(callback_id, tl.uint32) & 0xFF, BYTES)


@triton.jit
def invoke(chunks, consumer_index, producer, cached, log2_capacity, callback_id, word,
           BYTES: tl.constexpr):
    """``while (!handle.invoke(callback_id, args...)) {}``; see try_invoke.

    Returns ``(producer_index, consumer_index_cached)``.
    """
    return push(chunks, consumer_index, producer, cached, log2_capacity, word,
                tl.cast(callback_id, tl.uint32) & 0xFF, BYTES)


@triton.jit
def invoke_u32(chunks, consumer_index, producer, cached, log2_capacity, callback_id, value,
               BYTES: tl.constexpr):
    """invoke() for a payload that is one uint32 followed by zero padding."""
    lane = tl.arange(0, 64)
    word = tl.where(lane == 0, tl.cast(tl.cast(value, tl.uint32), tl.uint64), 0).to(tl.uint64)
    return invoke(chunks, consumer_index, producer, cached, log2_capacity, callback_id, word,
                  BYTES)


@triton.jit
def save_device_handle(device_context, endpoint, producer, cached):
    """DeviceContext::save_device_handle(endpoint, handle): persists queue progress.

    Lane 0 stores in C++; here every lane stores the same values to the same
    addresses.
    """
    handles = _load_u64(tl.cast(device_context, tl.uint64) + _CONTEXT_HANDLES_OFFSET)
    base = handles + tl.cast(endpoint, tl.uint64) * _HANDLE_SIZE
    tl.store((base + _PRODUCER_INDEX_OFFSET).to(tl.pointer_type(tl.uint64)), producer)
    tl.store((base + _CONSUMER_INDEX_CACHED_OFFSET).to(tl.pointer_type(tl.uint64)), cached)
