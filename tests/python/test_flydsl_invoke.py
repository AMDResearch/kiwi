# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""GPU tests for invoking C++ KIWI callbacks from FlyDSL kernels.

Typed callbacks are registered on a C++ ``kiwi::invoke::Context`` owned by
the test-only ``kiwi_invoke_test_ext`` module; they record what they receive.

Exits 77 (CTest skip) when FlyDSL, PyTorch, or a ROCm GPU is unavailable.
"""

import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import time
import unittest

try:
    import torch
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import range_constexpr
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"SKIP: {exc}")
    sys.exit(77)

import kiwi_flydsl as kiwi  # noqa: E402
from kiwi_flydsl import _layout  # noqa: E402
import kiwi_invoke_test_ext as ext  # noqa: E402

TIMEOUT = 60.0


@flyc.kernel
def scalar_kernel(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32):
    h = ctx.get_device_handle(0)
    for i in range(n):
        h = h.invoke(cb, i, fx.Float32(fx.Int32(i)) * 0.5, fx.Uint64(i) * 3 + 0x100000000)


@flyc.jit
def launch_scalar(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32,
                  stream: fx.Stream = fx.Stream(None)):
    scalar_kernel(ctx, cb, n).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


MIXED_TYPES = (fx.Uint8, fx.Int64, fx.Int16, fx.Float64, fx.Boolean, fx.Uint32)
MIXED_FORMAT = struct.Struct("<Bqhd?I")


@flyc.kernel
def mixed_kernel(ctx: kiwi.DeviceContext, n: fx.Int32):
    h = ctx.get_device_handle(0)
    for i in range(n):
        # Int16 - 300 widens to Int32; arg_types casts it back to the
        # registered 2-byte field.
        h = h.invoke(
            1,
            i,
            fx.Int64(i) * -7,
            fx.Int16(i) - 300,
            fx.Float64(fx.Int32(i)) / 4.0,
            i % 2 == 0,
            fx.Uint32(i) | 0x80000000,
            arg_types=MIXED_TYPES,
        )


@flyc.jit
def launch_mixed(ctx: kiwi.DeviceContext, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    mixed_kernel(ctx, n).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


# Message sizes at the chunk (8 B) and line (32 B) boundaries, up to the
# 512-byte limit: one 1-line message per callback except 33 B (2 lines) and
# 512 B (16 lines).
BOUNDARY_BYTES = (0, 8, 9, 32, 33, 512)


@flyc.kernel
def boundary_kernel(ctx: kiwi.DeviceContext, n: fx.Int32):
    h = ctx.get_device_handle(0)
    for i in range(n):
        base = fx.Uint64(i) * 1000
        h = h.invoke(0)
        h = h.invoke(1, base)
        h = h.invoke(2, base, fx.Uint8(i))
        h = h.invoke(3, *[base + k for k in range_constexpr(4)])
        h = h.invoke(4, *[base + k for k in range_constexpr(4)], fx.Uint8(i))
        h = h.invoke(5, *[base + k for k in range_constexpr(64)])


@flyc.jit
def launch_boundary(ctx: kiwi.DeviceContext, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    boundary_kernel(ctx, n).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def endpoint_kernel(ctx: kiwi.DeviceContext, n: fx.Int32):
    h = ctx.get_device_handle()
    for i in range(n):
        h = h.invoke(0, fx.block_idx.x, i)


@flyc.jit
def launch_endpoints(ctx: kiwi.DeviceContext, blocks: fx.Constexpr[int], n: fx.Int32,
                     stream: fx.Stream = fx.Stream(None)):
    endpoint_kernel(ctx, n).launch(grid=(blocks, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def try_kernel(ctx: kiwi.DeviceContext, attempts: fx.Int32):
    # Attempts go to endpoint 0 and the report to endpoint 1, so the host can
    # wait for the report without draining endpoint 0.
    h = ctx.get_device_handle(0)
    report = ctx.get_device_handle(1)
    accepted = fx.Int32(0)
    for i in range(attempts):
        h, ok = h.try_invoke(0, i)
        if ok:
            accepted = accepted + 1
    report = report.invoke(1, accepted)


@flyc.jit
def launch_try(ctx: kiwi.DeviceContext, attempts: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    try_kernel(ctx, attempts).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def persistent_kernel(ctx: kiwi.DeviceContext, start: fx.Int32, n: fx.Int32):
    h = ctx.get_device_handle(0)
    for i in range(n):
        h = h.invoke(0, start + i)
    ctx.save_device_handle(0, h)


@flyc.jit
def launch_persistent(ctx: kiwi.DeviceContext, start: fx.Int32, n: fx.Int32,
                      stream: fx.Stream = fx.Stream(None)):
    persistent_kernel(ctx, start, n).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def python_int_arg_kernel(ctx: kiwi.DeviceContext):
    h = ctx.get_device_handle(0)
    h = h.invoke(0, 5)


@flyc.jit
def launch_python_int_arg(ctx: kiwi.DeviceContext, stream: fx.Stream = fx.Stream(None)):
    python_int_arg_kernel(ctx).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def oversized_kernel(ctx: kiwi.DeviceContext):
    h = ctx.get_device_handle(0)
    h = h.invoke(0, *[fx.Uint64(k) for k in range_constexpr(65)])


@flyc.jit
def launch_oversized(ctx: kiwi.DeviceContext, stream: fx.Stream = fx.Stream(None)):
    oversized_kernel(ctx).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.kernel
def bad_callback_id_kernel(ctx: kiwi.DeviceContext):
    h = ctx.get_device_handle(0)
    h = h.invoke(256)


@flyc.jit
def launch_bad_callback_id(ctx: kiwi.DeviceContext, stream: fx.Stream = fx.Stream(None)):
    bad_callback_id_kernel(ctx).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def endpoints(rec):
    return kiwi.Endpoints(rec.device_context)


def unpack_all(fmt, records):
    return [struct.unpack(fmt, r) for r in records]


class FlyDSLInvokeTest(unittest.TestCase):
    def finish(self, rec, count, endpoint_id=0):
        executed = 0
        deadline = time.monotonic() + TIMEOUT
        while executed < count:
            self.assertLess(time.monotonic(), deadline, f"ran {executed} of {count} callbacks")
            executed += rec.progress(endpoint_id, count - executed)
        torch.cuda.synchronize()
        self.assertEqual(rec.progress(endpoint_id), 0)

    def test_layout_matches_cpp_headers(self):
        for name in ("CONTEXT_HANDLES_OFFSET", "CONTEXT_NUM_ENDPOINTS_OFFSET",
                     "HANDLE_SIZE", "CHUNKS_OFFSET", "CONSUMER_INDEX_OFFSET",
                     "PRODUCER_INDEX_OFFSET", "CONSUMER_INDEX_CACHED_OFFSET",
                     "LOG2_CAPACITY_OFFSET", "CHUNK_BYTES", "CHUNK_PAYLOAD_BYTES",
                     "CHUNKS_PER_LINE", "MAX_ARG_BYTES", "MAX_CAPACITY_LINES"):
            self.assertEqual(getattr(_layout, name), getattr(ext, name), name)

    def test_typed_cpp_callback_and_dynamic_callback_id(self):
        rec = ext.RecordingContext(1, 64)
        rec.add_bytes_recorder(0)
        cb = rec.add_typed_recorder()
        self.assertEqual(cb, 1)

        launch_scalar(endpoints(rec), cb, 300)
        self.finish(rec, 300)
        self.assertEqual(rec.records(0), [])
        self.assertEqual(rec.typed_records(), [(i, i * 0.5, i * 3 + 2**32) for i in range(300)])

    def test_unaligned_mixed_argument_types(self):
        rec = ext.RecordingContext(1, 128)
        rec.add_bytes_recorder(0)
        cb = rec.add_bytes_recorder(MIXED_FORMAT.size)

        launch_mixed(endpoints(rec), 200)
        self.finish(rec, 200)
        self.assertEqual(rec.records(0), [])
        expected = [
            (i & 0xFF, -7 * i, i - 300, i / 4.0, i % 2 == 0, i | 0x80000000)
            for i in range(200)
        ]
        self.assertEqual([MIXED_FORMAT.unpack(r) for r in rec.records(cb)], expected)

    def test_chunk_and_line_boundaries(self):
        rec = ext.RecordingContext(1, 64)
        for nbytes in BOUNDARY_BYTES:
            rec.add_bytes_recorder(nbytes)

        # 22 lines per round, so 40 rounds wrap the 64-line ring many times.
        n = 40
        launch_boundary(endpoints(rec), n)
        self.finish(rec, len(BOUNDARY_BYTES) * n)
        self.assertEqual(rec.records(0), [b""] * n)
        for i in range(n):
            base = i * 1000
            words = [base + k for k in range(4)]
            self.assertEqual(rec.records(1)[i], struct.pack("<Q", base))
            self.assertEqual(rec.records(2)[i], struct.pack("<QB", base, i))
            self.assertEqual(rec.records(3)[i], struct.pack("<4Q", *words))
            self.assertEqual(rec.records(4)[i], struct.pack("<4QB", *words, i))
            self.assertEqual(rec.records(5)[i],
                             struct.pack("<64Q", *(base + k for k in range(64))))

    def test_push_isa(self):
        # Recompile boundary_kernel in a child process with an empty FlyDSL
        # cache (under HOME) and dump its final ISA.
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, HOME=tmp, FLYDSL_DUMP_IR="1",
                       FLYDSL_DUMP_DIR=os.path.join(tmp, "dump"))
            subprocess.run(
                [sys.executable, __file__, "FlyDSLInvokeTest.test_chunk_and_line_boundaries"],
                env=env, check=True, capture_output=True, timeout=TIMEOUT)
            dumps = list(pathlib.Path(tmp, "dump").glob("boundary_kernel*/*final_isa.s"))
            self.assertEqual(len(dumps), 1, "boundary_kernel ISA was not dumped")
            isa = dumps[0].read_text()

        instructions = [line.split()[0] for line in isa.splitlines()
                        if re.match(r"\s+[a-z]", line)]
        stores = [i for i in instructions if i.startswith("global_store")]
        # One 16-byte chunk store per invoke; no flat, scratch or buffer
        # accesses, and wave-uniform values through readfirstlane.
        self.assertEqual(stores, ["global_store_dwordx4"] * len(BOUNDARY_BYTES))
        for prefix in ("flat_", "scratch_", "buffer_", "ds_bpermute"):
            self.assertFalse([i for i in instructions if i.startswith(prefix)], prefix)
        self.assertIn(".private_segment_fixed_size: 0", isa)
        self.assertIn("v_readfirstlane_b32", instructions)
        self.assertRegex(isa, r"global_load_dwordx2 .* sc0 sc1")

    def test_queue_wraps_under_backpressure(self):
        # 8-line queue: the kernel stalls on a full queue many times and the
        # epochs wrap, so the host must drain concurrently.
        rec = ext.RecordingContext(1, 8)
        rec.add_bytes_recorder(0)
        cb = rec.add_typed_recorder()
        launch_scalar(endpoints(rec), cb, 1000)
        self.finish(rec, 1000)
        self.assertEqual(rec.typed_records(), [(i, i * 0.5, i * 3 + 2**32) for i in range(1000)])

    def test_stress_16k_invokes_into_16k_queue(self):
        capacity = n = 16 * 1024
        expected = [(i, i * 0.5, i * 3 + 2**32) for i in range(n)]
        # Each 16-byte invoke is one line, so 16K invokes fill the 16K-line
        # ring. after_kernel: the kernel finishes with the queue exactly full before
        # the host drains it. concurrent: the host drains while it produces.
        for drain in ("after_kernel", "concurrent"):
            with self.subTest(drain=drain):
                rec = ext.RecordingContext(1, capacity)
                cb = rec.add_typed_recorder()
                launch_scalar(endpoints(rec), cb, 0)
                torch.cuda.synchronize()

                start = time.perf_counter()
                launch_scalar(endpoints(rec), cb, n)
                if drain == "after_kernel":
                    torch.cuda.synchronize()
                self.finish(rec, n)
                elapsed = time.perf_counter() - start

                self.assertEqual(rec.typed_records(), expected)
                print(f"\n  stress {drain}: {n} invokes in {elapsed * 1e3:.1f} ms "
                      f"({n / elapsed / 1e6:.2f} M invokes/s)", file=sys.stderr)

    def test_default_endpoint_is_block_index(self):
        blocks = 4
        n = 100
        rec = ext.RecordingContext(blocks, 16)
        rec.add_bytes_recorder(8)
        launch_endpoints(endpoints(rec), blocks, n)

        executed = 0
        seen = 0
        deadline = time.monotonic() + TIMEOUT
        while executed < blocks * n:
            self.assertLess(time.monotonic(), deadline)
            for ep in range(blocks):
                executed += rec.progress(ep)
                got = unpack_all("<ii", rec.records(0))
                self.assertTrue(all(b == ep for b, _ in got[seen:]))
                seen = len(got)
        torch.cuda.synchronize()
        got = unpack_all("<ii", rec.records(0))
        for ep in range(blocks):
            self.assertEqual([i for b, i in got if b == ep], list(range(n)))

    def test_try_invoke_reports_full_queue(self):
        rec = ext.RecordingContext(2, 4)
        tried = rec.add_bytes_recorder(4)
        accepted = rec.add_bytes_recorder(4)

        # Endpoint 0 is not drained until the report arrives on endpoint 1,
        # so exactly its capacity of attempts succeed.
        launch_try(endpoints(rec), 10)
        self.finish(rec, 1, endpoint_id=1)
        self.finish(rec, 4, endpoint_id=0)
        self.assertEqual(unpack_all("<i", rec.records(tried)), [(i,) for i in range(4)])
        self.assertEqual(unpack_all("<i", rec.records(accepted)), [(4,)])

    def test_saved_progress_persists_across_launches(self):
        rec = ext.RecordingContext(1, 16)
        cb = rec.add_bytes_recorder(4)
        for launch in range(3):
            launch_persistent(endpoints(rec), launch * 50, 50)
            self.finish(rec, 50)
        self.assertEqual(unpack_all("<i", rec.records(cb)), [(i,) for i in range(150)])

    def test_invalid_arguments(self):
        rec = ext.RecordingContext(1, 64)
        with self.assertRaises(ValueError):
            kiwi.Endpoints(0)
        ctx = endpoints(rec)
        with self.assertRaisesRegex(TypeError, "FlyDSL numeric"):
            launch_python_int_arg(ctx)
        with self.assertRaisesRegex(ValueError, "limit is 512 bytes"):
            launch_oversized(ctx)
        with self.assertRaisesRegex(ValueError, "outside 0..255"):
            launch_bad_callback_id(ctx)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no ROCm GPU visible")
        sys.exit(77)
    unittest.main()
