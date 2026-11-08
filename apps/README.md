# KIWI applications: P2P and EP

Run build commands from the [KIWI repository root](../README.md).
P2P and EP are independent prototype applications built on the invocation
layer, sharing LCI and topology support. Neither application is KIWI core.

KIWI is a HIP/C++ prototype for GPU-initiated communication on top of [LCI](https://github.com/uiuc-hpc/lci). The main focus is the **P2P layer** (`kiwi_p2p`), where GPU kernels enqueue communication requests through `kiwi::invoke` and CPU proxy threads execute the corresponding LCI operations.

The repository also contains an experimental **EP layer** (`kiwi_ep`) for MoE-style dispatch/combine work. Treat `kiwi_ep` as work in progress; its APIs and performance characteristics are still evolving.

The code currently targets AMD ROCm/HIP systems and defaults to the MI300X `gfx942` architecture when CMake cannot auto-detect a GPU architecture.

## Repository Layout

```text
apps/p2p/            kiwi_p2p context, GPU handles, descriptors, and counters
apps/ep/             Work-in-progress MoE dispatch/combine application
apps/p2p/tests/      P2P correctness tests
apps/ep/tests/       Work-in-progress EP tests
apps/p2p/benchmarks/ P2P, LCI, and load/store microbenchmarks
apps/ep/benchmarks/  EP Python graph benchmark and native microbenchmarks
apps/common/topology/ Topology support shared only by the applications
cmake/          Shared compile option helpers
```

## Requirements

- ROCm/HIP toolchain.
- CMake 3.21 or newer.
- C++17 compiler support.
- MPI runtime for multi-rank tests and benchmarks.
- Access to the CMake FetchContent dependencies:
  - `kiwi_queue`/`kiwi_invoke` comes directly from the KIWI core.
  - `uiuc-hpc/lci` for LCI.
  - CLI11 when benchmarks or CLI-driven tests are enabled.

## Build

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKIWI_BUILD_P2P=ON -DKIWI_BUILD_EP=ON
make -j$(nproc)
```

Useful CMake options:

```bash
cmake .. \
  -DKIWI_BUILD_P2P=ON \
  -DKIWI_BUILD_EP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DKIWI_BUILD_TESTS=ON \
  -DKIWI_BUILD_BENCHMARKS=ON \
  -DKIWI_BUILD_PYTHON=OFF \
  -DHIP_ARCHITECTURES=gfx942
```

`HIP_ARCHITECTURES` is auto-detected with `rocminfo` when possible. If auto-detection fails, CMake falls back to `gfx942`.

`kiwi_ep` uses the RCOMP payload protocol. In automatic mode, context
construction provisions endpoints and optional HIP IPC resources from the
maximum token capacity, while each dispatch and combine selects its grid and
payload backend from that call's active token count. The lookup lives in
`apps/ep/ep_tuning.hpp`, with one measured table per GPU architecture and NIC
family under `apps/ep/tuning/` (currently `gfx942_cx7.cpp` and
`gfx950_pollara.cpp`). Context construction detects the GPU architecture from
the HIP device and the NIC family from `config_t::device_name` or the HCA
nearest the GPU, unless `config_t::gpu_arch` and `config_t::nic_family` are
set. `ionic_*` HCAs map to `pollara`; ConnectX-7 is identified from its
Mellanox PCI device ID rather than the generic `mlx5_*` name. A matching table
applies to the world sizes and active token counts it measured, and a token
count between two rows uses the larger row. Everything else uses the generic
schedule: 12 blocks at one token, 20 at two, 32 from 3 through 32, 64 from 33
through 64, and 256 above 64; ws8 selects `rdma-only` below 8 active tokens and
`hybrid` at 8 or more, while other world sizes select `rdma-only` below 4 and
`hybrid` at 4 or more.
Explicit `num_blocks` and `rcomp_backend` settings remain fixed, and native
`config_t` defaults remain explicit 4-block, `rdma-only` settings. Raw proxy
devices explicitly disable LCI SHM and have no packet pool; high-level setup
collectives use the runtime default LCI device. Unless overridden through LCI's
environment variables, EP sizes that control-only pool to 8,192 packets and the
default device's receive queue to 4,096 entries. Its 512-byte packets accommodate
EP's largest 256-byte setup exchange while reducing the pool to 4 MiB. Each
remote RDMA payload PUT is
followed on the same QP by a one-byte write to a receiver-owned readiness flag
in the symmetric heap. Set `config_t::rcomp_backend =
rcomp_backend_t::HYBRID`, pass `rcomp_backend="hybrid"` through the Python
bindings, or use `--rcomp-backend hybrid` in the EP benchmark harness to direct
exact-hostname-local, non-self payloads through HIP IPC while remote peers
retain the RDMA-plus-readiness path. Hybrid setup exchanges PCI Bus IDs,
resolves each exact-hostname peer to an ordinal visible in the local process,
and strictly requires peer access, native peer atomics, and a successful
symmetric-heap import; otherwise context construction fails on that rank. Self
remains a direct GPU copy. The native dispatch/combine harness supports
`--routing all-ranks --verify-results`; a 2+2 mixed-host run can additionally
assert `--assert-ipc-peer-count 1 --assert-rdma-peer-count 2` to prove local
IPC, remote RDMA, and dispatch/combine data correctness together.

## Main Components

### `kiwi_p2p`

The P2P layer exposes a host-side `kiwi::p2p::context_t` and GPU-side `kiwi::p2p::gcontext_t`/`kiwi::p2p::ghandle_t`:

- `context_t` owns LCI devices, a `kiwi::invoke::Context`, symmetric heap registrations, and CPU proxy threads.
- `gcontext_t` is passed by value into kernels.
- `ghandle_t` is a wave-level GPU handle. Its device methods assume a full 64-thread wavefront participates.
- `post_p()` sends an 8-byte immediate-style value.
- `post_put()` sends a buffer via LCI put.
- `try_invoke()` can target custom callbacks registered through `context_t::register_callback()`.

Typical lifecycle:

```cpp
kiwi::p2p::context_t ctx(/*num_queues=*/4, /*num_devices=*/4);
int heap_id = ctx.register_symmetric(buffer, bytes);  // registers on all context devices
ctx.spawn_proxy_threads();

auto gctx = ctx.get_device_context();
kernel<<<blocks, 64>>>(gctx, heap_id, ...);
hipDeviceSynchronize();

ctx.join_proxy_threads();
ctx.deregister_symmetric(heap_id);
```

Symmetric heap registration is collective across ranks and should be done before proxy threads are spawned. `register_symmetric(buffer, bytes)` registers the application-owned buffer on all LCI devices owned by the context so proxy threads can use any configured device. Deregister symmetric heaps after proxy threads have joined; this removes the LCI registrations but does not free application-owned buffers.

For latency experiments that need a dedicated proxy core, call
`ctx.spawn_proxy_threads(/*num_threads=*/1, /*first_core=*/49)` after pinning
the main/control thread separately. When `first_core >= 0`, proxy thread `i`
pins itself to `first_core + i`. Otherwise, as in EP, proxy threads are bound
to the GPU's PCI-local CPUs within the process affinity mask (keeping the
inherited mask when they do not intersect) and the scheduler places them
within that set. `KIWI_PROXY_AFFINITY_DEBUG=1` logs the chosen and actual CPUs.

## Tests

Build with `KIWI_BUILD_TESTS=ON` and run from the build directory:

```bash
ctest --output-on-failure
```

P2P CTest executables:

- `apps/p2p/tests/test_hello`
- `apps/p2p/tests/test_context`
- `apps/p2p/tests/test_comp_counter`
- `apps/p2p/tests/test_memory`
- `apps/p2p/tests/test_pingpong`

`apps/p2p/tests/test_lci_put_atomicity` is a manual two-node hardware test and
is intentionally not registered with CTest.

Several tests are registered through MPI with two ranks.

Important EP test executables:

- `apps/ep/tests/test_dispatch`
- `apps/ep/tests/test_combine`
- `apps/ep/tests/test_combine_graph_replay`
- `apps/ep/tests/test_ep_block_count`
- `apps/ep/tests/test_ep_collective_primitives` — repeated EP-owned barrier and count all-to-all coverage.
- `apps/ep/tests/test_dispatch_combine_harness` — native C++ EP dispatch/combine smoke/debug harness, not the canonical performance benchmark.
- `apps/ep/tests/test_rcomp_readiness_layout`
- `apps/ep/tests/test_rcomp_hybrid_ipc`

CTest also registers configuration-specific cases using these executables.
With `KIWI_BUILD_PYTHON=ON`, it adds the Python graph-replay smoke test and,
when official MORI is configured, the `test_official_mori_compat_*` matrix.

## Benchmarks

Build with `KIWI_BUILD_BENCHMARKS=ON`.

P2P benchmarks:

```bash
./apps/p2p/benchmarks/bench_latency --queues 1 --check
./apps/p2p/benchmarks/bench_bandwidth --wgs 8 --depth 1024
./apps/p2p/benchmarks/bench_loadstore_latency --gpu0 0 --gpu1 1 --check
./apps/p2p/benchmarks/bench_lci_loopback --size 8
./apps/p2p/benchmarks/bench_lci_put_latency --size 8 --cpu-mem
./apps/p2p/benchmarks/bench_lci_put_bandwidth --threads 8 --devices 8 --depth 1024 --size 8192
```

Benchmark defaults and options are defined with CLI11 in each benchmark source file.
For positive message sizes, `bench_latency` and `bench_bandwidth` model the
KIWI-EP remote payload protocol: one callback posts the payload PUT and then a
one-byte receiver-owned readiness PUT on the same proxy/device. Receivers poll
and reset the readiness byte instead of requesting transport remote
completion. This is the default `kiwi::p2p::ghandle_t::post_put` protocol;
reported bandwidth counts payload bytes only.

`bench_lci_put_bandwidth` is a direct LCI CPU-threaded `post_put` bandwidth baseline: rank 0 CPU threads post bounded-window data PUTs with remote completion, rank 1 receiver/progress threads ACK each iteration with small PUTs, and rank 0 waits for ACK plus local completions before slot reuse. It is not a KIWI benchmark and not a sender-local injection-only/no-remote-completion test.
`bench_latency` also accepts `--main-core` and `--proxy-core` for experiments that must isolate the benchmark control thread from the proxy/progress thread. `bench_lci_put_bandwidth` accepts `--main-core` and `--thread-core` for explicit control/worker-thread pinning.

The three `*_readiness` benchmarks compare one 8-byte network write per
logical operation across KIWI, direct LCI, and raw ibverbs. Each runs with
exactly two ranks: in every round, rank 0 writes `--depth` epoch values into
rank 1's target slots, and rank 1 waits for all of them before writing one
epoch ACK back. `--depth 1` measures ping-pong latency; a larger depth (for
example 128) measures acknowledged single-endpoint throughput. Each run checks
that every target slot holds the final epoch.

| Benchmark | Write | Receiver |
|---|---|---|
| `bench_kiwi_readiness` | one KIWI invoke descriptor; the proxy callback issues one `lci::net_post_put_x` | the GPU kernel polls its target and posts the reply descriptor |
| `bench_lci_readiness` | one `lci::net_post_put_x`, the network-layer put that KIWI's proxy callback also issues; no remote completion | `--receiver gpu` (default) or `--receiver cpu` |
| `bench_ibverbs_readiness` | one `IBV_WR_RDMA_WRITE`, no immediate data | `--receiver gpu` (default) or `--receiver cpu` |

- `--receiver gpu`: a persistent GPU kernel polls an ordinary GPU target and
  notifies the posting CPU thread through a pinned host word. This matches
  the KIWI completion path.
- `--receiver cpu`: the posting CPU thread polls a fine-grained GPU target
  directly. Add `--cpu-memory` to use registered CPU source and target
  buffers instead.
- Sources are GPU buffers unless `--cpu-memory` is given.
- `bench_ibverbs_readiness` requires `--ib-device`. `--gid-index` defaults to
  3 (RoCEv2); use 0 on InfiniBand. `--dmabuf` registers GPU memory through
  DMA-BUF, which some NICs require. Each rank signals only the last WRITE of a
  batch and waits for that send completion after posting.
- For LCI and KIWI, choose the NIC with `LCI_ATTR_DEVICE_NAME` and enable
  DMA-BUF registration with `LCI_ATTR_USE_DMABUF=1`.
- DMA-BUF keeps GPU buffers in VRAM only if the kernel supports PCIe
  peer-to-peer DMA (`CONFIG_PCI_P2PDMA`). Without it, amdgpu moves the
  exported buffer into host memory and the benchmarks silently measure host
  memory. On such systems (for example some 5.15 kernels), use
  `LCI_ATTR_USE_DMABUF=0` and omit `--dmabuf`.
- `bench_kiwi_readiness --main-core C --proxy-core P` pins the control and
  proxy threads to distinct CPUs.

Rank 0 prints one `result` line. `latency_us` is half the measured round-trip
time and appears only with `--depth 1`. `mops` counts forward writes per
second and excludes the ACK. Pin each rank to a NIC-local CPU and GPU (for
example with a per-rank `taskset` wrapper). Basic invocations:

```bash
mpirun -np 2 ./apps/p2p/benchmarks/bench_kiwi_readiness --depth 1
mpirun -np 2 ./apps/p2p/benchmarks/bench_lci_readiness --depth 1 --receiver cpu
mpirun -np 2 ./apps/p2p/benchmarks/bench_ibverbs_readiness --depth 128 --receiver gpu \
    --ib-device mlx5_0 --gid-index 3 --dmabuf
```

`bench_kiwi_readiness` also has a cross-frontend harness,
`apps/p2p/benchmarks/frontends/readiness/`, with the layout of the invoke
harnesses (see
[`benchmarks/README.md`](../benchmarks/README.md#cross-frontend-harnesses)):
`common/` holds a copy of its host code templated on a kernel launcher, with
the `kiwi_readiness_bench_ext` bridge and `kiwi_readiness_bench` driver for
Python kernels, and each frontend supplies only the protocol kernel: `cpp/`
(`harness_kiwi_readiness_cpp`, the standalone kernel), `flydsl/`
(`KIWI_BUILD_FLYDSL=ON`) and `triton/` (`KIWI_BUILD_TRITON=ON`). The Python
kernels invoke the callback through `gcontext_t::invoke_device_context()`.
Every frontend takes the standalone options and prints its `result` line with
`backend=kiwi_cpp`, `kiwi_flydsl` or `kiwi_triton`; a Python frontend first
compiles its kernel in an untimed one-round run.
From the repository root, with the build in `build/`:

```bash
H=build/apps/p2p/benchmarks/frontends/readiness
mpirun -np 2 $H/cpp/harness_kiwi_readiness_cpp --depth 1
mpirun -np 2 -x PYTHONPATH="$PWD/build/python" python3 \
    $H/flydsl/bench_kiwi_flydsl_readiness.py --depth 1
mpirun -np 2 -x PYTHONPATH="$PWD/build/python" python3 \
    $H/triton/bench_kiwi_triton_readiness.py --depth 1
```

`-x` exports the variable to every rank, including ranks on other nodes;
export per-rank settings such as `LCI_ATTR_DEVICE_NAME` and
`HIP_VISIBLE_DEVICES` the same way or through a per-rank wrapper.


EP native microbenchmarks:

```bash
cmake --build build --target bench_ep_collectives -j$(nproc)
mpirun -np 8 build/apps/ep/benchmarks/bench_ep_collectives \
  --warmup 10000 --iters 100000 \
  --num-blocks 1 --num-proxy-threads 1
```

The other native EP targets are `bench_ep_copy` and `bench_ep_hidden_copy`.
The canonical MORI-style graph benchmark is
`apps/ep/benchmarks/python/bench_kiwi_ep.py` and is normally launched through
`scripts/ep_bench/run_kiwi_ep_benchmark.sh`.

`bench_ep_collectives` compares direct CPU-side LCI `barrier_x` and fixed-size
uint64 `alltoall_x` with GPU-initiated EP barrier and count all-to-all. The
GPU path uses the RCOMP hybrid count transport: it publishes packed epoch/count
rows into GPU symmetric-heap ingress, with same-node receivers gathering
through HIP IPC and proxies posting one RDMA PUT per remote node. The all-to-all
cases exchange one uint64 per source/destination pair, matching the EP dispatch
count-vector control path; payload readiness uses receiver-owned
symmetric-heap flags. It maps MPI ranks
to HIP devices using
`OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `SLURM_LOCALID`, `LOCAL_RANK`,
or rank modulo visible GPUs; use `--hip-device` only for single-process/debug
overrides. Useful knobs include `--warmup`, `--iters`, `--num-blocks`,
`--num-proxy-threads`, `--lci-device-name`, `--skip-lci`, and
`--skip-kiwi`. Output contains parseable per-rank `RESULT` lines and
rank-aggregated `SUMMARY` lines.

For the reproducible KIWI EP / MORI EP Docker workflow, provenance collection,
and profiling entrypoints, see `docs/ep-benchmark-image.md`.

## Development Notes

- GPU-side P2P methods are wave-level operations; launch kernels with 64 threads per participating block/wavefront unless the code path explicitly states otherwise.
- `kiwi::p2p::ghandle_t` is move-only and persists `invoke` queue state back to device memory in its destructor.
- `context_t::requested_device_name` is only honored for the first context that initializes the process-wide LCI runtime.
- `kiwi_ep` is work in progress; avoid presenting its APIs as stable in public docs.
- The Python `kiwi_ep_ext.KiwiEpOp` dispatch/combine path is intended to be
  allocation-free and asynchronous after operator construction, matching the
  MORI hot-path contract. Callers must provide correctly typed contiguous input
  tensors; dispatch returns views of operator-owned receive storage, and combine
  returns an operator-owned output buffer that is reused by later calls. Both
  operations enqueue on the current Torch stream without synchronizing it. The
  path remains `torch.cuda.CUDAGraph` compatible while offloading RDMA puts
  through the always-running host proxy threads. Build with
  `KIWI_BUILD_PYTHON=ON` and run `test_kiwi_ep_cudagraph_ws2` for the focused
  graph replay correctness check.
- Avoid adding long-lived planning artifacts to the repository root. Keep durable task state in issues/PRs.
