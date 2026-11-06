# KIWI Python Frontends

This directory holds the optional Python kernel frontends for KIWI-Invoke.
Both emit KIWI's wire format from the kernel, so callbacks registered on a
C++ `kiwi::invoke::Context` run unchanged:

- `kiwi_flydsl/` (`KIWI_BUILD_FLYDSL=ON`): FlyDSL; see the
  [FlyDSL invoke guide](../docs/flydsl.md).
- `kiwi_triton.py` (`KIWI_BUILD_TRITON=ON`): Triton, described here.

Neither option enables the P2P or EP applications. The build places every
frontend module and benchmark extension in the build tree's `python/`; put
that directory on `PYTHONPATH`.

## Requirements

- An AMD GPU with 64-lane wavefronts (CDNA, gfx9) and a HIP runtime
- Python 3 with PyTorch (ROCm), Triton and pybind11
- CMake and a C++/HIP build toolchain

Build against the HIP runtime that PyTorch and Triton load, and pass
`-DPython3_EXECUTABLE=...` if that Python is not the first on `PATH`.

## Configure, Build And Test

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DKIWI_BUILD_TRITON=ON -DKIWI_BUILD_TESTS=ON -DHIP_ARCHITECTURES=gfx942
cmake --build build -j
ctest --test-dir build -R triton --output-on-failure
```

## KIWI-Invoke From Triton

`kiwi_triton.py` mirrors the C++ device side of KIWI-Invoke over the SPSC
device-to-host queue:

- `load_device_handle(device_context, endpoint)` is
  `DeviceContext::get_device_handle()`; it returns the handle as
  `(chunks, consumer_index, producer_index, consumer_index_cached,
  log2_capacity)`. `load_handle()` loads a handle at a given address.
- `try_push()` is `SPSCD2HQueue::DeviceHandle::push_lanes()`: a 32-bit credit
  check, a rare credit refresh (lane 0 reads the published consumer index and
  the wave takes its value), a trap for a message longer than the ring, and
  one 16-byte store per active lane. `push()` retries it, as C++
  `while (!handle.push_lanes(...)) {}`.
- `try_invoke()` and `invoke()` are the same for an invoke: the callback id
  (a `uint8_t`) is the metadata and the payload is the arguments packed as
  `ArgumentSerializer` packs them (lane c holds bytes `[8c, 8c + 8)`).
  `invoke_u32()` packs one `uint32_t` followed by zero padding.
- `save_device_handle()` persists queue progress for a later launch.

Use the queue state each call returns, as with the by-value C++ handle. One
64-lane wavefront owns an endpoint, so kernels run with `num_warps=1`:
launch them with `kiwi_triton.launch(kernel, grid, *args)` and compile them
ahead of a launch with `kiwi_triton.warmup(kernel, grid, *args)`. Inline asm
is used only for the trap.

[`examples/triton`](../examples/triton/README.md) is a minimal kernel that
invokes a C++ callback; build it with `KIWI_BUILD_EXAMPLES=ON`.

`test_triton_invoke` (`tests/python/`) invokes C++ callbacks through the
test-only `kiwi_invoke_test_ext` context it shares with the FlyDSL tests: the
handle layout against the C++ headers, a typed callback, message sizes at the
chunk and line boundaries up to 512 bytes, queue wrap under backpressure,
`try_invoke` on a full queue, saved progress across launches, per-program
endpoints, and invalid arguments. It also checks the ISA: one
`global_store_dwordx4` per invoke and no scratch.

## Benchmarks

With `KIWI_BUILD_BENCHMARKS=ON`, each cross-frontend benchmark harness gets a
Triton frontend (see
[`benchmarks/README.md`](../benchmarks/README.md#cross-frontend-harnesses)):
the harness's `common/` host code, a copy of the standalone C++ benchmark's,
runs through the harness's driver (`kiwi_invoke_latency_bench`,
`kiwi_invoke_throughput_bench`, `kiwi_readiness_bench`, `kiwi_queue_bench`),
so options, timed regions and output are those of the harness's C++ frontend
(`cpp/`); only the kernel launch comes from Python. The FlyDSL frontends use
the same drivers. Every Triton kernel is compiled and checked for scratch
before its first launch, and loaded before the timed region.

| Triton frontend | C++ frontend (standalone original) | Also needs |
|---|---|---|
| `benchmarks/invoke/frontends/latency/triton/bench_invoke_triton_latency.py` | `harness_invoke_latency_cpp` (`bench_invoke_latency`) | |
| `benchmarks/invoke/frontends/throughput/triton/bench_invoke_triton_throughput.py` | `harness_invoke_throughput_cpp` (`bench_invoke_throughput`) | |
| `apps/p2p/benchmarks/frontends/readiness/triton/bench_kiwi_triton_readiness.py` | `harness_kiwi_readiness_cpp` (`bench_kiwi_readiness`) | `KIWI_BUILD_P2P=ON` |
| `benchmarks/queue_baselines/frontends/triton/bench_triton_queue_baselines.py` | `harness_queue_baselines_cpp` (`bench_queue_baselines -q kiwi-spsc`) | |

From the repository root, with the build in `build/` (scripts and
executables are at the mirrored build paths):

```bash
export PYTHONPATH="$PWD/build/python"
B=build/benchmarks
taskset -c 8 python3 $B/invoke/frontends/latency/triton/bench_invoke_triton_latency.py \
    -c 1024 -n 1000000 -s
taskset -c 8-23 python3 $B/invoke/frontends/throughput/triton/bench_invoke_triton_throughput.py \
    -m 10000000 -e 1 -t 1 --poll-mode direct
mpirun -np 2 -x PYTHONPATH python3 \
    build/apps/p2p/benchmarks/frontends/readiness/triton/bench_kiwi_triton_readiness.py --depth 1
python3 $B/queue_baselines/frontends/triton/bench_triton_queue_baselines.py \
    -m latency -n 100000 --warmup 1000
```

The throughput kernel supports at most 256 endpoints (a Triton program is one
wavefront). `bench_triton_queue_baselines.py` is the `kiwi-spsc-triton` row of
the GPU-to-CPU queue baselines; see
[`benchmarks/queue_baselines/README.md`](../benchmarks/queue_baselines/README.md#cross-frontend-harness).

## Skipped Tests

`test_triton_invoke` and `test_kiwi_triton_spsc_baseline` exit with code 77
when PyTorch, Triton, or a usable HIP device is unavailable, which CTest
reports as skipped (check for "Skipped"; run the script directly to see the
reason). The other benchmark runs (`test_triton_*_bench`) do not skip.
