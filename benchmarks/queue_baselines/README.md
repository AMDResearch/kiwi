# GPU->CPU queue baselines

Standalone microbenchmarks that compare the KIWI-Queue device-to-host handoff
with the reverse-offload queues of other systems. A single harness
(`bench_queue_baselines`) drives every queue through one adapter interface
(`qb_common.hpp`). All queues therefore share the same kernels, pinned proxy
threads, acknowledgement path, timers, and per-message validation. Each queue
keeps its own descriptor format, memory placement, and publish protocol.

## Queues

| `--queue` | Source | Descriptor | Device publish (steady state) |
|---|---|---|---|
| `kiwi-spsc` | this repo | 16B message in a 64B line of 16B chunks | one 16B store per lane; indices in registers; no atomics |
| `kiwi-mpsc` | this repo | 16B message in a 64B line of 16B chunks | lane-0 agent-scope fetch-add line claim; cached credit; one 16B store per lane |
| `kiwi-invoke` | this repo | `invoke(cb, word0, tag)`: 16B argument message over `kiwi-spsc`'s queue | as `kiwi-spsc`; arguments serialized in registers |
| `mscclpp-fifo` | MSCCL++ `07ce0f3e` (after v0.9.0), unmodified `FifoDeviceHandle::push` | 16B trigger | lane-0 agent-scope fetch-add; `fst` store; L2 writeback + system release store of `snd` |
| `llvm-rpc` | LLVM `libc/shared/rpc.h` `cca9d7b4`, unmodified | 64B lane buffer | lane-0 port lock (`fetch_or`); 64B buffer; system release fence; outbox flip |
| `uccl-ep-fifo` | UCCL `3ec055f3` `ep/include/fifo_*.hpp`, unmodified (hipify-equivalent include shim) | 16B trigger (32B slot) | lane-0 system-scope fetch-add; `snd` store; L2 writeback + release store of `fst` |
| `nvshmem-proxy` | NVSHMEM `cffd2a5` proxy channel, protocol re-implementation | 32B put request | device fence; lane-0 device-scope byte reservation; four 8B entries, each with a lap-parity flag |
| `rccl-gin-proxy` | RCCL `rocm-systems` `5909ad99` (NCCL 2.32.3 base), unmodified `ncclGinApi_Put` proxy backend (RCCL's own hipify build step) | 128B GFD | context loads; lane-0 system-scope fetch-add; `ci` load every push; eight 16B system-scope stores (`global_store_dwordx4 sc0 sc1`), per-qword valid bits |
| `rocshmem-ro` | rocSHMEM 3.8.0 (`rocm-systems` `5909ad99`) RO queue, protocol re-implementation | 192B element (PUT_NBI writes 36B + valid) | lane-0 spinlock claim; field stores; fence, valid, fence |
| `raw-store` | lower bound, not a queue | 64B slot | one wave-wide 64B store, seq in lane 7; no capacity check (latency and never-wrapping push-cost only) |
| `rocm-hostcall` | ROCm device-libs `__ockl_call_host_function` + CLR listener, unmodified | 64B lane packet | synchronous: CAS free-list pop, CAS ready-list push, doorbell + interrupt; waits for the host |

Third-party headers are fetched at configure time at pinned revisions with
SHA-256 verification. Nothing is vendored. Every pin is the upstream default
branch as of 2026-10-16 (MSCCL++, UCCL and LLVM `main`, `rocm-systems`
`develop`, NVSHMEM `devel`).

Adapter notes:

- **MSCCL++**: the host `Fifo` lives in `libmscclpp`, so its allocation
  (`hipHostMalloc(Mapped)` ring and tail, device head and tail cache),
  `poll()`, and `pop()` are reproduced line for line from `src/core/fifo.cc`
  at the same commit.
- **LLVM RPC**: a push is `Client::open` + `Port::send` + close, and a pop is
  `Server::try_open` + `Port::recv` + close. This is send-only RPC. `rpc.h`
  declares its mailbox, header and packet pointers global (`RPC_GLOBAL`), so
  the device accesses them with `global_*` operations. Two choices
  favor the baseline:
  - The doorbell is left null, so there are no interrupts and the server
    busy-polls.
  - Only lane 0 opens the port, so a message writes one 64B lane buffer rather
    than 4 KiB for a full wave.

  LLVM builds the server as plain C++, so it lives in `llvm_rpc_server.cpp`.
  Per-producer order is not guaranteed; the harness checks set semantics only.
- **RCCL GIN**: the device side is RCCL's GIN proxy backend
  (`ncclGinApi_Put<NCCL_NET_DEVICE_GIN_PROXY>::call`: `buildGfd` then
  `postGfd`), compiled from the pinned headers after the steps RCCL's own
  build applies (`hipify-perl -experimental`, `hip_compat.h` as is, `nccl.h`
  from `nccl.h.in`). The GIN proxy host side lives in `librccl`, so its
  allocation, progress pass, GFD poll (nontemporal slot reset + `sfence`), and
  in-order `ci` release are reproduced line for line from
  `src/gin/gin_host_proxy.cc` and `src/gin/gin_host.cc`. A progress pass
  takes the progress threads' reader lock, releases completed `ci`s, then
  polls up to `NCCL_GIN_PROXY_POLL_BATCH` (default 32) GFDs, peeking at the
  next slot after each one as upstream does to decide request aggregation. `ci` is in pinned host memory, because RCCL
  leaves GDRCopy off by default; `QB_RCCL_GIN_CIS=device` selects RCCL's
  GDRCopy path (uncached VRAM).
- **Re-implementations** (NVSHMEM, rocSHMEM): each cites the upstream
  file and line for every step, and its header documents the deviations.
  The main ones are:
  - NVSHMEM: CUDA `volatile` maps to relaxed system-scope atomics, matching PTX.
  - rocSHMEM: the consumer skips upstream's second thread and idle sleep. This
    favors the baseline.
- **rocm-hostcall** is synchronous. `latency` uses the call's own completion,
  `throughput` counts calls per second on the runtime's listener thread, and
  `push-cost`/`idle-poll` do not apply.
- **kiwi-spsc / kiwi-mpsc**: lane i passes payload word i to `push_lanes()`,
  so the message goes from registers to the ring with no staging. Both hosts
  use the `HostPollState` fast path; the queues differ only in how producers
  claim lines (registers vs. a shared device-memory fetch-add).
- **kiwi-invoke**: KIWI-Invoke over its default SPSC queue, one endpoint per
  wave. The device calls `DeviceContext::DeviceHandle::invoke()` with the
  message's two words as the callback arguments; the harness loads the handle
  per wave and makes it wave-uniform, as `get_device_handle()` does. The host
  checks `HostPollState::ready()`, then calls `Context::progress(endpoint, 1)`,
  which pops the message, looks up the callback, deserializes the arguments,
  and runs the registered `std::function`; the callback hands the message to
  the harness's validator.

## Build

```bash
cmake -S . -B build/qb -DCMAKE_BUILD_TYPE=Release -DHIP_ARCHITECTURES=gfx942 \
  -DKIWI_BUILD_BENCHMARKS=ON -DKIWI_BUILD_QUEUE_BASELINES=ON
cmake --build build/qb -j --target bench_queue_baselines
```

## Modes

Run the binary pinned to CPUs on the GPU's NUMA node, for example
`taskset -c 8-23`. Worker threads use the first CPUs in the affinity mask.

| `--mode` | What it measures |
|---|---|
| `latency` | One wave: push, host pop, host ack, wave sees ack. RTT percentiles. |
| `push-cost` | GPU cost per `push()`, in shader cycles (`clock64()`, default) or, with `--timer wall`, in ns from the constant 100 MHz `wall_clock64()` (10 ns resolution), with `--waves` concurrent producers and active proxy threads. Two patterns. **Isolated** (default): stores are drained before each timed push and the timer bracket is subtracted. **`--burst`**: back-to-back pushes; per-wave loop time / pushes. Size `-c` so rings never wrap (the `push` sweep profile does), so proxies cannot throttle producers. |
| `loaded-latency` | Probe RTT (wave 0) while `--waves` background waves push through the same queues and proxy threads. `--bg-rate` fixes the total offered load (Mops, throttled on the GPU) so every queue sees the same load; unthrottled, the probe mostly measures proxy backlog. |
| `floor` | Cost of the push-cost timer bracket around an empty op, an LDS stage round trip, and one wave-wide 64B store to device, coherent host, and KIWI uncached host memory. |
| `throughput` | `--waves` producer waves x `--iters` messages; `--threads` pinned proxy threads; `--queues` queues (0 = one per wave). Wall-clock Mops/s. |
| `idle-poll` | Host ns per poll of an empty queue, across `--queues` queues. |

Examples:

Full sweeps: `run_sweep.sh <binary> quick|latency|push|throughput|paper [reps]`.
On a SLURM cluster, `slurm_sweep.sh` builds in node-local scratch, records
the environment, and runs a profile on one exclusive node.

```bash
B=build/qb/benchmarks/queue_baselines/bench_queue_baselines
taskset -c 8-23 $B -q kiwi-spsc -m latency -n 100000
taskset -c 8-23 $B -q mscclpp-fifo -m push-cost -n 100000 -c 4096
# library-default sharing (many waves per queue) vs one queue per wave
taskset -c 8-23 $B -q mscclpp-fifo -m throughput -w 64 --queues 4 -t 4 -n 20000
taskset -c 8-23 $B -q mscclpp-fifo -m throughput -w 64 -t 4 -n 20000
```

Each run prints a machine-readable `RESULT key=value ...` line. `valid=1`
means every message passed these checks: correct producer-to-queue mapping,
matching tag, no duplicates, exact count, and FIFO order for queues that
guarantee it.

## Cross-frontend harness

`frontends/` runs the `kiwi-spsc` row with producers written in different
languages on identical host code (the pattern of
[`../README.md`](../README.md#cross-frontend-harnesses)). It needs only the
KIWI queue, so it is built with `KIWI_BUILD_BENCHMARKS=ON`, without the
third-party queues.

- `common/`: `qb_harness.hpp`, a copy of this benchmark's host side for the
  latency, push-cost, throughput and idle-poll modes (proxy threads, ack
  path, timers, validation, `RESULT` lines) templated on a kernel launcher,
  and `kiwi_queue_bench_ext` and the `kiwi_queue_bench` driver, which run it
  for Python kernels.
- `cpp/`: `harness_queue_baselines_cpp`, this benchmark's `kiwi-spsc` kernels
  and launcher (`queue=kiwi-spsc-cpp`).
- `triton/`: `bench_triton_queue_baselines.py` (`KIWI_BUILD_TRITON=ON`), the
  same kernels in Triton, pushing with `kiwi_triton.push`
  (`queue=kiwi-spsc-triton`). A Triton program is one wave, so
  `--waves-per-block` must be 1, and `--check-overlap` is C++ only.

Every frontend takes this benchmark's options for those modes, without `-q`.
From the repository root, with the build in `build/`:

```bash
export PYTHONPATH="$PWD/build/python"
H=build/benchmarks/queue_baselines/frontends
taskset -c 8-23 $H/cpp/harness_queue_baselines_cpp -m latency -n 100000
taskset -c 8-23 python3 $H/triton/bench_triton_queue_baselines.py -m latency -n 100000
taskset -c 8-23 python3 $H/triton/bench_triton_queue_baselines.py -m push-cost -w 1024 --timer wall
taskset -c 8-23 python3 $H/triton/bench_triton_queue_baselines.py -m throughput -w 256 -t 8
```

## Harness fairness notes

These were found while calibrating against `benchmarks/queue` on MI300X.
Each applies to every queue.

- **Every producer wave of a multi-wave run is resident at once.** The
  kernels carry `__launch_bounds__(256)`, the largest workgroup the harness
  launches (`--waves-per-block` <= 4). Without a bound the compiler assumes
  1024-thread workgroups. A stack array it promotes to LDS is then reserved
  for 1024 threads per workgroup: 32 KiB for the RCCL GIN push descriptor.
  That allowed only 2 RCCL waves per CU (608 on MI300X), so a "W=1024" RCCL
  push-cost run ran about 608 waves at a time, and its W=1024 push cost was
  about half the real value. Each launch also checks with
  `hipOccupancyMaxActiveBlocksPerMultiprocessor` that its whole grid can be
  resident, and fails otherwise.
  `--check-overlap` (push-cost, throughput) also records each wave's start
  and end and reports `overlap_waves`, the waves still running when the last
  one started, and `start_spread_us`; a run is then valid only if
  `overlap_waves` equals `--waves`. It is a diagnostic, off by default: it
  changes the code around the timed push and moves the measured cost (MSCCL++
  at W=1024 by about -20%). Measurements run without it; the bound leaves the
  KIWI, MSCCL++/UCCL-EP push and throughput, NVSHMEM and raw-store kernels
  instruction for instruction unchanged.

- **The latency kernel waits for its stores (`s_waitcnt vmcnt(0)`) after each
  push**, before polling the ack, as `bench_queue_latency` does. Without the
  wait, ack polling races ahead of the posted write and every queue's RTT
  grows by about 0.85 us.
- **Single-producer modes pass the handle by value.** The throughput kernel
  loads per-wave handles from memory and calls each adapter's `globalize()`.
  Without this, queue pointers compile to `flat_*` operations. A flat store
  holds `lgkmcnt`, so later LDS or scalar waits stall until the host-memory
  store is acknowledged. KIWI's handle
  also carries the producer and credit indices, so its `globalize()` moves the
  whole handle to SGPRs (`wave_uniform()`). Left in VGPRs, the indices put an
  `s_waitcnt vmcnt(0)` at the top of the push loop, which waits for the
  previous host-memory store, and W=1 throughput drops to a third.
- **Where the ack lives matters for latency.** With `--ack-mem host`, every GPU
  poll of the ack is a PCIe read (about 1.3 us), so the RTT comes in whole
  poll round trips. Which side of a poll the host's ack lands on depends on
  each queue's timing. On MI300X, kiwi needed an extra poll in 35% of
  iterations and MSCCL++ in 1%, and background load shifts that mix.
  `--ack-mem device` (the default) puts the ack in fine-grained VRAM. The host
  writes it with an `sfence`, since the write otherwise sits in the
  write-combining buffer for about 1 ms, and the GPU polls it locally. That
  removes the artifact at a cost of about +0.4 us for every queue.
- **Each process launch settles into its own timing mode.** Latency moves by
  about +/-0.4 us between launches, so latency points use many short launches
  and report the median and range.
- **Proxy polling interferes with GPU writes.** A consumer spinning on the
  cache line the GPU is writing slows the GPU's posted writes. Queues that pack
  several descriptors per line, or publish with several stores, suffer the
  most. `--idle-pause N` backs off after an empty polling round and must be
  swept per queue: MSCCL++ on 64 waves over 4 queues with 4 threads goes from
  12 to 56 Mops at `--idle-pause 1000`.
- **Validation state is cache-line aligned per producer** to avoid false
  sharing between proxy threads.
- **Message construction is nearly free.** The tag is a bitwise NOT of word
  0 and filler words use XOR/shift, because building the message runs inside
  the timed push. A 64-bit hash here added 190-290 cycles, mostly to queues
  with wide descriptors.
- **Push payloads are built without scratch.** A per-lane array indexed by
  lane spills to scratch and adds about 800 cycles per push.
