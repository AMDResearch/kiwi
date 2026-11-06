# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""GPU tests for invoking C++ KIWI callbacks from Triton kernels.

Typed callbacks are registered on a C++ ``kiwi::invoke::Context`` owned by
the test-only ``kiwi_invoke_test_ext`` module; they record what they receive.
kiwi_triton takes arguments already packed into payload words, so the tests
pack them on the host (``ArgumentSerializer`` order) and the kernels load
them.

Exits 77 (CTest skip) when Triton, PyTorch, or a ROCm GPU is unavailable.
"""

import re
import struct
import sys
import time
import unittest

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"SKIP: {exc}")
    sys.exit(77)

import kiwi_triton as kiwi  # noqa: E402
import kiwi_invoke_test_ext as ext  # noqa: E402

TIMEOUT = 60.0
# Message sizes at the chunk (8 B) and line (32 B) boundaries, up to the
# 512-byte limit, one callback each.
BOUNDARY_BYTES = (0, 8, 9, 32, 33, 512)


@triton.jit
def _payload(words, row):
    """Payload words of message ``row`` of a host-packed [rows, 64] tensor."""
    return tl.load(words + row * 64 + tl.arange(0, 64)).to(tl.uint64, bitcast=True)


@triton.jit
def invoke_rows_kernel(ctx, cb, words, n, BYTES: tl.constexpr):
    """Invoke ``cb`` with message i = row i of ``words`` on the endpoint
    ``program_id``, then save the handle."""
    endpoint = tl.program_id(0)
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(
        ctx, endpoint)
    for i in range(n):
        producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity,
                                       cb, _payload(words, endpoint * n + i), BYTES)
    kiwi.save_device_handle(ctx, endpoint, producer, cached)


@triton.jit
def _boundary_round(chunks, consumer_index, producer, cached, log2_capacity, words, i):
    """One invoke per size in BOUNDARY_BYTES; callback k gets row 6i + k."""
    producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity, 0,
                                   _payload(words, i * 6), 0)
    producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity, 1,
                                   _payload(words, i * 6 + 1), 8)
    producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity, 2,
                                   _payload(words, i * 6 + 2), 9)
    producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity, 3,
                                   _payload(words, i * 6 + 3), 32)
    producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity, 4,
                                   _payload(words, i * 6 + 4), 33)
    producer, cached = kiwi.invoke(chunks, consumer_index, producer, cached, log2_capacity, 5,
                                   _payload(words, i * 6 + 5), 512)
    return producer, cached


@triton.jit
def boundary_kernel(ctx, words, n):
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(ctx, 0)
    for i in range(n):
        producer, cached = _boundary_round(chunks, consumer_index, producer, cached,
                                           log2_capacity, words, i)


@triton.jit
def boundary_once_kernel(ctx, words):
    """One round with no loop around it, for the ISA check."""
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(ctx, 0)
    producer, cached = _boundary_round(chunks, consumer_index, producer, cached, log2_capacity,
                                       words, 0)


@triton.jit
def try_kernel(ctx, attempts):
    # Attempts go to endpoint 0 and the report to endpoint 1, so the host can
    # wait for the report without draining endpoint 0.
    lane = tl.arange(0, 64)
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(ctx, 0)
    accepted = 0
    for i in range(attempts):
        word = tl.where(lane == 0, i, 0).to(tl.uint64)
        producer, cached, ok = kiwi.try_invoke(chunks, consumer_index, producer, cached,
                                               log2_capacity, 0, word, 4)
        if ok:
            accepted += 1
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(ctx, 1)
    producer, cached = kiwi.invoke_u32(chunks, consumer_index, producer, cached, log2_capacity,
                                       1, accepted, 4)


@triton.jit
def oversized_kernel(ctx):
    chunks, consumer_index, producer, cached, log2_capacity = kiwi.load_device_handle(ctx, 0)
    producer, cached = kiwi.invoke_u32(chunks, consumer_index, producer, cached, log2_capacity,
                                       0, 0, 513)


def pack_rows(messages):
    """[len(messages), 64] int64 device tensor; row r holds message r's payload words."""
    rows = torch.zeros((len(messages), 64), dtype=torch.int64)
    for r, message in enumerate(messages):
        message = message + bytes(-len(message) % 8)
        words = struct.unpack(f"<{len(message) // 8}q", message)
        rows[r, :len(words)] = torch.tensor(words, dtype=torch.int64)
    return rows.cuda()


def typed_message(i):
    """ArgumentSerializer<int32_t, float, uint64_t>(i, i * 0.5, i * 3 + 2**32)."""
    return struct.pack("<ifQ", i, i * 0.5, i * 3 + 2**32)


def typed_expected(n):
    return [(i, i * 0.5, i * 3 + 2**32) for i in range(n)]


def boundary_messages(i):
    base = i * 1000
    words = [base + k for k in range(4)]
    return [b"", struct.pack("<Q", base), struct.pack("<QB", base, i),
            struct.pack("<4Q", *words), struct.pack("<4QB", *words, i),
            struct.pack("<64Q", *(base + k for k in range(64)))]


def unpack_all(fmt, records):
    return [struct.unpack(fmt, r) for r in records]


class TritonInvokeTest(unittest.TestCase):
    def finish(self, rec, count, endpoint_id=0):
        executed = 0
        deadline = time.monotonic() + TIMEOUT
        while executed < count:
            self.assertLess(time.monotonic(), deadline, f"ran {executed} of {count} callbacks")
            executed += rec.progress(endpoint_id, count - executed)
        torch.cuda.synchronize()
        self.assertEqual(rec.progress(endpoint_id), 0)

    def invoke_rows(self, rec, cb, messages, nbytes, endpoints=1):
        rows = pack_rows(messages)
        kiwi.launch(invoke_rows_kernel, (endpoints,), rec.device_context, cb, rows,
                    len(messages) // endpoints, BYTES=nbytes)
        return rows

    def test_layout_matches_cpp_headers(self):
        for name in ("CONTEXT_HANDLES_OFFSET", "HANDLE_SIZE", "CHUNKS_OFFSET",
                     "CONSUMER_INDEX_OFFSET", "PRODUCER_INDEX_OFFSET",
                     "CONSUMER_INDEX_CACHED_OFFSET", "LOG2_CAPACITY_OFFSET", "CHUNK_BYTES",
                     "CHUNK_PAYLOAD_BYTES", "CHUNKS_PER_LINE", "MAX_CAPACITY_LINES"):
            self.assertEqual(getattr(kiwi, name), getattr(ext, name), name)
        self.assertEqual(kiwi.MAX_MESSAGE_BYTES, ext.MAX_ARG_BYTES)

    def test_typed_cpp_callback(self):
        rec = ext.RecordingContext(1, 64)
        rec.add_bytes_recorder(0)
        cb = rec.add_typed_recorder()
        # The callback id is a uint8_t: cb + 256 is cb.
        rows = self.invoke_rows(rec, cb + 256, [typed_message(i) for i in range(300)], 16)
        self.finish(rec, 300)
        del rows
        self.assertEqual(rec.records(0), [])
        self.assertEqual(rec.typed_records(), typed_expected(300))

    def test_chunk_and_line_boundaries(self):
        rec = ext.RecordingContext(1, 64)
        for nbytes in BOUNDARY_BYTES:
            rec.add_bytes_recorder(nbytes)
        # 22 lines per round, so 40 rounds wrap the 64-line ring many times.
        n = 40
        expected = [boundary_messages(i) for i in range(n)]
        rows = pack_rows([m for round_ in expected for m in round_])
        kiwi.launch(boundary_kernel, (1,), rec.device_context, rows, n)
        self.finish(rec, len(BOUNDARY_BYTES) * n)
        for k in range(len(BOUNDARY_BYTES)):
            self.assertEqual(rec.records(k), [round_[k] for round_ in expected], k)

    def test_push_isa(self):
        rec = ext.RecordingContext(1, 64)
        rows = pack_rows(boundary_messages(0))
        isa = kiwi.warmup(boundary_once_kernel, (1,), rec.device_context, rows).asm["amdgcn"]
        instructions = [line.split()[0] for line in isa.splitlines()
                        if re.match(r"\s+[a-z]", line)]
        stores = [i for i in instructions if i.startswith("global_store")]
        # One 16-byte chunk store per invoke (single-copy atomic); no scratch.
        self.assertEqual(stores, ["global_store_dwordx4"] * len(BOUNDARY_BYTES))
        self.assertFalse([i for i in instructions if i.startswith("scratch_")])
        self.assertIn(".private_segment_fixed_size: 0", isa)

    def test_queue_wraps_under_backpressure(self):
        # 8-line queue: the kernel stalls on a full queue many times and the
        # epochs wrap, so the host must drain concurrently.
        rec = ext.RecordingContext(1, 8)
        cb = rec.add_typed_recorder()
        rows = self.invoke_rows(rec, cb, [typed_message(i) for i in range(1000)], 16)
        self.finish(rec, 1000)
        del rows
        self.assertEqual(rec.typed_records(), typed_expected(1000))

    def test_try_invoke_reports_full_queue(self):
        rec = ext.RecordingContext(2, 4)
        tried = rec.add_bytes_recorder(4)
        accepted = rec.add_bytes_recorder(4)
        # Endpoint 0 is not drained until the report arrives on endpoint 1,
        # so exactly its capacity of attempts succeed.
        kiwi.launch(try_kernel, (1,), rec.device_context, 10)
        self.finish(rec, 1, endpoint_id=1)
        self.finish(rec, 4, endpoint_id=0)
        self.assertEqual(unpack_all("<i", rec.records(tried)), [(i,) for i in range(4)])
        self.assertEqual(unpack_all("<i", rec.records(accepted)), [(4,)])

    def test_saved_progress_persists_across_launches(self):
        rec = ext.RecordingContext(1, 16)
        cb = rec.add_bytes_recorder(4)
        for launch in range(3):
            rows = self.invoke_rows(
                rec, cb, [struct.pack("<i", launch * 50 + i) for i in range(50)], 4)
            self.finish(rec, 50)
            del rows
        self.assertEqual(unpack_all("<i", rec.records(cb)), [(i,) for i in range(150)])

    def test_program_endpoints(self):
        endpoints, n = 4, 100
        rec = ext.RecordingContext(endpoints, 16)
        cb = rec.add_bytes_recorder(8)
        rows = self.invoke_rows(
            rec, cb, [struct.pack("<ii", ep, i) for ep in range(endpoints) for i in range(n)],
            8, endpoints)
        executed = 0
        seen = 0
        deadline = time.monotonic() + TIMEOUT
        while executed < endpoints * n:
            self.assertLess(time.monotonic(), deadline)
            for ep in range(endpoints):
                executed += rec.progress(ep)
                got = unpack_all("<ii", rec.records(cb))
                self.assertTrue(all(e == ep for e, _ in got[seen:]))
                seen = len(got)
        torch.cuda.synchronize()
        del rows
        got = unpack_all("<ii", rec.records(cb))
        for ep in range(endpoints):
            self.assertEqual([i for e, i in got if e == ep], list(range(n)))

    def test_invalid_arguments(self):
        rec = ext.RecordingContext(1, 64)
        with self.assertRaises(triton.CompilationError) as raised:
            kiwi.warmup(oversized_kernel, (1,), rec.device_context)
        # The static assertion is the root cause of the call-site errors.
        error = raised.exception
        while error.__cause__ is not None:
            error = error.__cause__
        self.assertIn("kMaxMessageBytes", str(error))
        with self.assertRaisesRegex(ValueError, "num_warps must be 1"):
            kiwi.launch(try_kernel, (1,), rec.device_context, 0, num_warps=4)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no ROCm GPU visible")
        sys.exit(77)
    unittest.main()
