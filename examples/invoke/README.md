# invoke ROCm-XIO SDMA ping-pong

`invoke_sdma_ping_pong` compares three 2-rank, rank-to-GPU ping-pong paths:

- `device`: device-initiated ROCm-XIO SDMA baseline. By default this is
  reported with host-side amortized timing around kernel launch/synchronize;
  `--device-timing` adds GPU `wall_clock64()` diagnostics.
- `invoke`: GPU kernel invokes a invoke callback; the CPU callback/proxy
  immediately issues ROCm-XIO host SDMA `put_signal()` to the peer GPU, then the
  peer GPU waits/signals in the same ping-pong pattern.
- `host`: pure CPU-thread initiated ROCm-XIO host SDMA baseline. Both MPI
  ranks call `createHostQueue()`/`getHostHandle()` and issue `put_signal()`
  directly from the CPU; no invoke GPU-to-CPU callback is used. Rank 0 times
  this path with CPU `steady_clock`, and the CPU wait side directly polls
  host-pinned, CPU-visible completion signal memory updated by SDMA.
- `both` (or `all`): runs invoke, device, and host modes and prints a compact
  latency summary. Default summary rows are host-side timers; optional GPU
  diagnostic rows are printed only when `--device-timing` (or `--gpu-timing`)
  is passed.

ROCm-XIO host API revisions differ in the formal argument names for
`SdmaQueueHostHandle::put_signal()`: some declare
`put_signal(src, dst, size, flag_ptr, flag_value)`, while others declare
`put_signal(dst, src, size, flag_ptr, flag_value)` and internally submit
`CopyLinearPacket(src, dst, size)`. The benchmark's CMake configure step
inspects the fetched ROCm-XIO source and defines the correct call order. The
benchmark code wraps host call sites to make the intended semantic direction
explicit: each rank copies its local source buffer into the peer rank's
destination buffer and then signals the peer.

Timing semantics:

Before each size/mode run, source buffers are filled with rank-specific payload
patterns and destination buffers are cleared. After each mode/size, both ranks
copy back and validate their destination buffer against the peer's source
pattern. This payload check is outside the timed region and is intended to catch
reversed `put_signal()` copy direction or no-copy bugs that signal-only tests
would otherwise miss.

Host-side timing is the default and primary result:

- `invoke` always prints `invoke-host-sdma-host-amortized`. Rank-side CPU
  `steady_clock` brackets the measured kernel launch, kernel completion, proxy
  stop/join/drain, and host SDMA `quiet()`; the maximum rank duration is divided
  by the final calibrated iteration count.
- `device` always prints `device-initiated-host-amortized`. CPU `steady_clock`
  brackets the measured device kernel launch and synchronize; the maximum rank
  duration is divided by the final calibrated iteration count. This makes the
  device baseline comparable to the default invoke host-side amortized result.
- `host` prints `host-sdma-host-pinned-signal-cpu-timed`, a CPU-timed rank-0
  round-trip distribution while the CPU directly polls host-pinned, CPU-visible
  completion signal memory. This avoids repeated `hipMemcpyDeviceToHost` polling
  of GPU signal memory in the pure host baseline.

GPU timing is optional and off by default. Passing `--device-timing` (alias
`--gpu-timing`) additionally prints rank-0 `wall_clock64()` per-iteration
diagnostic rows:

- `invoke-host-sdma-gpu-diagnostic`: from invoke callback invocation until
  the peer response signal is observed by the GPU.
- `device-initiated-gpu-diagnostic`: from device SDMA `putSignal()` until the
  peer response signal is observed by the GPU.

For long enough calibrated runs, host-amortized and GPU diagnostic averages
should be in the same range. For short smoke tests, the host-amortized rows can
include visible fixed launch/proxy overhead.

## Build

Configure queue with invoke examples enabled. This example uses CMake
`FetchContent` to fetch ROCm-XIO from `https://github.com/ROCm/rocm-xio.git`.

```bash
cmake -S . -B build \
  -DKIWI_BUILD_EXAMPLES=ON \
  -DKIWI_BUILD_INVOKE=ON \
  -DKIWI_BUILD_INVOKE_SDMA_PING_PONG=ON \
  -DHIP_ARCHITECTURES=gfx942
cmake --build build --target invoke_sdma_ping_pong -j
```

If ROCm-XIO is already available as a source checkout, or the build host has no
network access, point `FetchContent` at a local source checkout instead. For
example:

```bash
git clone https://github.com/ROCm/rocm-xio.git /tmp/rocm-xio
cmake -S . -B build \
  -DKIWI_BUILD_EXAMPLES=ON \
  -DKIWI_BUILD_INVOKE=ON \
  -DKIWI_BUILD_INVOKE_SDMA_PING_PONG=ON \
  -DFETCHCONTENT_SOURCE_DIR_ROCM_XIO=/tmp/rocm-xio
```

The `FetchContent_Declare` still names the SSH repository; the
`FETCHCONTENT_SOURCE_DIR_ROCM_XIO` override is only a configure-time workaround
for hosts without GitHub SSH access.

## Run

Run exactly two MPI ranks on a node with two mutually peer-accessible AMD GPUs:

```bash
mpirun -n 2 build/examples/invoke/invoke_sdma_ping_pong --mode both
```

Useful options:

```bash
--mode <invoke|device|host|both|all>
--iterations <N>        # initial measured iterations
--warmup <N>            # untimed warmup iterations
--min-duration-ms <N>   # kiwi/invoke/device adapt iterations until host amortized timing reaches N ms (default 1000; 0 disables)
--device-timing       # also print GPU wall_clock64 diagnostic rows (default off)
--gpu-timing          # alias for --device-timing
--sizes 64,256,1024,4096
--queue-capacity <N>
--pin-core <N>          # pin the main thread; invoke proxy defaults to the same core
--progress-core <N>     # pin the dedicated invoke proxy/progress thread
--no-pin                # leave CPU affinity unchanged (default)
```

## CPU affinity / pinning

The invoke mode uses a separate host proxy/progress thread. The proxy thread
progresses invoke callbacks and issues the host SDMA `put_signal()` calls;
the main thread launches the kernel, synchronizes, then signals the proxy to
stop and joins/drains it. The progress loop is intentionally stopped by the main
thread after kernel completion, not by counting completed callbacks.

For reproducible latency, pin each MPI rank/proxy to CPU cores that are
topologically close to its GPU, for example:

```bash
mpirun \
  -n 1 build/examples/invoke/invoke_sdma_ping_pong \
    --mode both --sizes 64,256 --iterations 100 --warmup 10 --min-duration-ms 1000 --pin-core 8 --progress-core 9 \
  : -n 1 build/examples/invoke/invoke_sdma_ping_pong \
    --mode both --sizes 64,256 --iterations 100 --warmup 10 --min-duration-ms 1000 --pin-core 16 --progress-core 17
```

Passing the same `--pin-core` value to `mpirun -n 2` pins both ranks to that
core; use MPI app contexts, rank-specific launch wrappers, or your scheduler's
CPU binding support when each rank should use a different core.

In `host` mode, the completion signals are one shared host-pinned 64-bit word
per MPI rank. Each rank registers both its local signal and the peer shared
signal with HIP and passes the mapped host-memory device pointer to ROCm-XIO
`put_signal()` so the SDMA atomic completion update is visible to a direct CPU
polling loop. If a platform cannot register/map the shared host signal for SDMA
atomics, the benchmark reports that HIP registration/mapping failure instead of
falling back to GPU-signal `hipMemcpy` polling.

`--pin-core` pins the main thread and, unless `--progress-core` is also passed,
the invoke proxy thread. `--progress-core` can be used to put the proxy on a
separate core; if it is passed alone it also pins the main thread for backwards
compatibility with the previous alias behavior. The benchmark prints the
requested/current CPU affinity for every rank and reports the proxy thread
affinity in invoke mode. Topology-aware core selection is platform-specific;
use tools such as `rocminfo`, `rocm-smi topo`, `lstopo`, or cluster
documentation to pick GPU-local cores. If no core is specified, the benchmark
reports the inherited affinity mask and does not pin.

## Duration control

`--min-duration-ms` defaults to 1000 for invoke and device modes so the
measured batch runs long enough to amortize fixed launch/proxy overhead in the
default host-side metric. The first short measured run is used as calibration;
if its host whole-run duration is shorter than the requested minimum, all ranks
increase the timed iteration count and rerun. Use a small value such as
`--min-duration-ms 10`, or `--min-duration-ms 0`, for quick smoke tests.
