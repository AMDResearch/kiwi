# KIWI application agent instructions

These instructions apply to `apps/` and supplement the repository-root `AGENTS.md`.

## Project Overview

`kiwi` is a HIP/C++ prototype for GPU-initiated communication on top of LCI. It has two library targets:

- `kiwi_p2p`: GPU kernels enqueue communication callbacks through `invoke`; CPU proxy threads execute LCI operations.
- `kiwi_ep`: work-in-progress MoE dispatch/combine built directly on `invoke`.

P2P and EP are independent applications. They share LCI and
`apps/common/topology/`; neither is part of the KIWI core.
Run all commands from the KIWI repository root.

The project is ROCm/HIP-oriented and defaults to `gfx942` if `rocminfo` cannot auto-detect `HIP_ARCHITECTURES`.

## Build Commands

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKIWI_BUILD_P2P=ON -DKIWI_BUILD_EP=ON -DKIWI_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

Useful options:

```bash
-DKIWI_BUILD_TESTS=ON
-DKIWI_BUILD_BENCHMARKS=ON
-DKIWI_BUILD_PYTHON=OFF
-DHIP_ARCHITECTURES=gfx942
```

The queue/invocation dependency is the KIWI core in this checkout.
CMake fetches `uiuc-hpc/lci` and CLI11 for benchmarks / CLI-driven test harnesses.

## Key APIs

### P2P

- `kiwi::p2p::context_t`: host-side owner of LCI devices, `kiwi::invoke::Context`, symmetric heaps, and proxy threads.
- `kiwi::p2p::gcontext_t`: GPU-side context passed into kernels.
- `kiwi::p2p::ghandle_t`: move-only wave-level handle used inside kernels.
- `ghandle_t::post_p()`: fire-and-forget 8-byte value send.
- `ghandle_t::post_put()`: RDMA put of a local buffer.
- `ghandle_t::try_invoke()`: invoke a custom callback registered with `context_t::register_callback()`.

Lifecycle reminders:

1. Allocate/register symmetric heaps before spawning proxy threads.
2. Spawn proxy threads before kernels that issue callbacks.
3. Join proxy threads before freeing symmetric heaps.
4. Full-wavefront participation is expected for GPU-side communication calls.

### EP (work in progress)

- `kiwi::ep::config_t`: world size, expert count, hidden dimension, top-k, queue/proxy configuration, element sizes, and heap layout.
- `kiwi::ep::ep_context_t`: owns EP communication/all-to-all state, the RDMA-registered symmetric heap, device workspace, receive buffers, and RCOMP route metadata.
- `kiwi::ep::dispatch_enqueue()` is the asynchronous, graph-capturable dispatch
  path; `kiwi::ep::dispatch()` calls it and then synchronizes the stream.
- `kiwi::ep::combine()` sends expert outputs back to source ranks and accumulates weighted outputs.

## Tests

Registered CTest targets include:

- P2P CTest targets: `test_hello`, `test_context`, `test_comp_counter`, `test_memory`, and `test_pingpong`. The `test_lci_put_atomicity` executable is a manual two-node hardware test and is not registered with CTest.
- EP C++ executables: `test_dispatch`, `test_combine`, `test_combine_graph_replay`, `test_ep_block_count`, `test_ep_collective_primitives`, `test_dispatch_combine_harness`, `test_rcomp_readiness_layout`, and `test_rcomp_hybrid_ipc`. CTest also registers configuration-specific cases using these executables.
- EP Python/CUDA-graph smoke test: `test_kiwi_ep_cudagraph_ws2` when Python support is enabled.
- Official MORI compatibility tests: `test_official_mori_compat_*` when Python support is enabled and an official MORI source is configured. These cover the broad EP configuration matrix.

Run `ctest --output-on-failure` from the build directory. MPI-backed tests are registered with multiple ranks.

For EP work, be precise about validation scope:

- `scripts/ep_bench/run_kiwi_ep_smoke_correctness.sh` is a focused smoke gate. It does not run the official MORI compatibility matrix.
- `scripts/ep_bench/run_kiwi_ep_full_correctness.sh` is the full EP correctness gate. It runs the focused smoke gate and then the official MORI compatibility CTest matrix.
- For changes to EP dispatch, combine, barriers, counters, epochs, graph replay, routing, or public MORI-compatible behavior, run the full EP correctness gate in the Docker workflow unless the user explicitly asks for a narrower check. If the full gate is not run, say that clearly in the final report.
- **Full MORI correctness is not satisfied by skipped tests.** Provision and mount the official MORI checkout before declaring full EP correctness:
  ```bash
  MORI_SOURCE_DIR=/path/to/built/mori \
  scripts/ep_bench/run_container.sh -- \
    scripts/ep_bench/run_kiwi_ep_full_correctness.sh --build-dir /workspace/kiwi/build/full
  ```
  The mounted checkout must contain an importable built official package:
  `PYTHONPATH="$MORI_SOURCE_DIR/python" python3 -c 'import mori, mori.ops, mori.shmem'`
  must succeed. A source-only checkout missing `python/mori/libmori_pybinds.so`
  must be built using the official MORI installation procedure before the gate
  runs. Follow `docs/ep-benchmark-image.md` for the complete image, DMA-BUF,
  and dependency-override command. A MORI `SKIP_RETURN_CODE=77` result means
  the full MORI gate was **not run**; do not report it as acceptable full
  correctness or as a passing validation.
- Validation reports must include the exact command or CTest regex, the number of tests discovered, the number run, and pass/fail/skip counts. Do not report focused smoke results as full correctness.

## Benchmarks

P2P benchmark executables:

- `bench_latency`
- `bench_bandwidth`
- `bench_loadstore_latency`
- `bench_lci_loopback`
- `bench_lci_put_latency`
- `bench_lci_put_bandwidth`
- `bench_kiwi_readiness`, `bench_lci_readiness`, `bench_ibverbs_readiness`: matched single-write ping-pong/throughput controls (see `apps/README.md`)
- `frontends/readiness/`: cross-frontend harness for `bench_kiwi_readiness` (`common/` host code, `cpp/harness_kiwi_readiness_cpp`, `flydsl/` with `KIWI_BUILD_FLYDSL=ON`, `triton/` with `KIWI_BUILD_TRITON=ON`); see `apps/README.md`

EP performance benchmarking is provided by
`apps/ep/benchmarks/python/bench_kiwi_ep.py` and the wrapper
`scripts/ep_bench/run_kiwi_ep_benchmark.sh`. Native microbenchmarks are
`bench_ep_collectives`, `bench_ep_copy`, and `bench_ep_hidden_copy`. The native
dispatch/combine CLI harness is a test target: `test_dispatch_combine_harness`.

`bench_ep_collectives` is a standalone native microbenchmark for EP collective-control overhead. It compares direct CPU-side LCI `barrier_x` and fixed-size uint64 `alltoall_x` with GPU-initiated EP barrier and count all-to-all. The RCOMP hybrid count transport publishes packed epoch/count rows into GPU symmetric-heap ingress, with HIP IPC gathers for same-node peers and one proxy-posted RDMA PUT per remote node. The all-to-all cases exchange one uint64 per source/destination pair, matching the EP dispatch count-vector control path; RCOMP uses receiver-owned symmetric-heap readiness flags. Build it with:

```bash
cmake --build build --target bench_ep_collectives -j$(nproc)
```

Typical 8-rank MI300X run:

```bash
mpirun -np 8 build/apps/ep/benchmarks/bench_ep_collectives \
  --warmup 100 --iters 1000 \
  --num-blocks 1 --num-proxy-threads 1
```

The benchmark maps each MPI rank to a HIP device using `OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `SLURM_LOCALID`, `LOCAL_RANK`, or rank modulo visible GPUs; `--hip-device` is available for single-process/debug overrides. Useful knobs include `--warmup`, `--iters`, `--num-blocks`, `--num-proxy-threads`, `--lci-device-name`, `--skip-lci`, and `--skip-kiwi`. Output contains parseable per-rank `RESULT` lines and rank-aggregated `SUMMARY` lines.

For EP correctness and performance on GPU nodes, `docs/ep-benchmark-image.md` is the canonical Docker/image workflow. Use `scripts/ep_bench/run_container.sh` with the canonical image (`kiwi-ep-bench:rocm7.2.4-pytorch2.10` when available). Do not treat missing host Python/Torch dependencies as a blocker until the Docker path has been tried.

When reporting EP performance numbers, include dispatch latency, combine latency, total latency, total bandwidth, and whether end-to-end timing is a captured graph or a split dispatch+combine sum. Include image identity/provenance for Docker runs.

Use each benchmark/test harness `--help` output for current options.

## Code Organization

- `apps/p2p/context.hip.hpp` / `.cpp`: P2P context and GPU handles.
- `apps/p2p/cmd_desc.hip.hpp`: callback command descriptors.
- `apps/p2p/comp_counter.hip.hpp`: completion counter helpers.
- `apps/ep/ep_config.hpp`: EP configuration and heap/slot layout.
- `apps/ep/ep_context.hip.hpp` / `.cpp`: EP resource management and public entry points.
- `apps/ep/dispatch.hip.cpp`: fused dispatch implementation.
- `apps/ep/combine.hip.cpp`: combine phase.
- `apps/p2p/tests/` and `apps/ep/tests/`: application correctness tests.
- `apps/p2p/benchmarks/` and `apps/ep/benchmarks/`: application benchmarks.

## Change Checklist

- Update `README.md` and public comments when APIs, build flags, tests, or benchmark names change.
- Prefer CTest for validation; run the narrow test first for focused changes, then `ctest --output-on-failure` before broad changes.
- Keep GPU-side communication calls wave-uniform unless intentionally changing the protocol.
- Do not commit generated build trees, scratch output, or planning/task directories.
