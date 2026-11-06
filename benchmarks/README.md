# Benchmarks

This directory contains performance benchmarks for **queue**, **invoke**, and low-level hardware behavior used by the queue implementations.

## Structure

```text
benchmarks/
├── queue/          # Queue latency, throughput, and overhead benchmarks
│   ├── bench_queue_latency
│   ├── bench_queue_throughput
│   ├── bench_queue_overhead
│   ├── bench_host_to_device_latency
│   ├── bench_mpmc_h2d_latency
│   └── bench_mpmc_h2d_throughput
├── invoke/         # Callback bridge benchmarks
│   ├── bench_invoke_latency
│   ├── bench_invoke_throughput
│   └── bench_invoke_overhead
└── hardware/       # Hardware characterization benchmarks
    ├── bench_pcie_latency
    └── bench_host_mapped_grid_poll
```

## Building

Benchmarks are built when `KIWI_BUILD_BENCHMARKS=ON` (off by default). CLI11 is fetched automatically by CMake FetchContent.

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKIWI_BUILD_BENCHMARKS=ON
make -j$(nproc)
```

Executables are written under:

- `build/benchmarks/queue/`
- `build/benchmarks/invoke/`
- `build/benchmarks/hardware/`

## queue Benchmarks

### `bench_queue_latency`

Measures round-trip GPU→CPU→GPU latency using the SPSC (default) or MPSC
(`--mpsc`) device-to-host queue.

```bash
cd build/benchmarks/queue
./bench_queue_latency [OPTIONS]
```

Common options:

- `-c,--capacity INT` — queue capacity in 64-byte lines, default `1024`.
- `-n,--iterations INT` — measured iterations, default `1000000`.
- `--mpsc` — use `MPSCD2HQueue`.
- `-s,--stats` — print latency percentiles.
- `--bytes INT` — message bytes pushed with `push_reg`, default `56` (one of
  8, 24, 32, 48, 56, 64, 112, 168, 240, 448, 512).
- `-v,--verbose` — verbose output.

Every push and invoke benchmark kernel is checked with `require_no_scratch()`
before launch and the benchmark exits if a kernel uses scratch.

### `bench_queue_throughput`

Measures device-to-host message throughput with one or more CPU consumer
threads. SPSC (default) uses one queue per GPU producer wavefront; with
`--mpsc`, wavefront `w` pushes to MPSC queue `w % Q`.

```bash
./bench_queue_throughput -m 1000000 -g 1 -t 1
```

Common options:

- `-c,--capacity INT` — queue capacity in 64-byte lines, default `1024`.
- `-m,--messages INT` — total messages, default `1000000`.
- `-g,--gpu-blocks INT` — GPU producer wavefronts, default `1`.
- `-t,--cpu-threads INT` — CPU consumer threads, default `1`.
- `--mpsc` — use `MPSCD2HQueue`.
- `-q,--queues INT` — MPSC queues, default one per consumer thread.
- `--poll-mode direct|host-poll-state` — consumer polling, default
  `host-poll-state`.
- `--bytes` — as for `bench_queue_latency`.
- `-v,--verbose` — verbose output.

### `bench_queue_overhead`

Measures GPU push overhead for the SPSC (default) or MPSC (`--mpsc`)
device-to-host queue.

```bash
./bench_queue_overhead -m 100000
```

Common options:

- `-c,--capacity INT` — queue capacity in 64-byte lines, default `1024`.
- `-m,--messages INT` — number of messages, default `100000`.
- `-w,--warmup INT` — warmup iterations, default auto.
- `--mpsc` — use `MPSCD2HQueue`.
- `--bytes` — as for `bench_queue_latency`.
- `-v,--verbose` — verbose output.

### `bench_host_to_device_latency`

Measures round-trip CPU→GPU→CPU latency using the SPSC host-to-device queue.

```bash
./bench_host_to_device_latency --stats
./bench_host_to_device_latency --device-slots --stats
```

Common options:

- `-c,--capacity INT` — queue capacity, default `1024`.
- `-n,--iterations INT` — measured iterations, default `1000000`.
- `-w,--warmup INT` — warmup iterations, default `2000000`.
- `-s,--stats` — print latency percentiles.
- `--device-slots` — use fine-grained device VRAM for queue slots instead of host coherent slots.
- `-v,--verbose` — verbose output.

### `bench_mpmc_h2d_latency`

Measures round-trip CPU→GPU→CPU latency using the MPMC host-to-device queue.

```bash
./bench_mpmc_h2d_latency --stats
```

Common options:

- `-c,--capacity INT` — queue capacity, default `1024`.
- `-n,--iterations INT` — measured iterations, default `1000000`.
- `-w,--warmup INT` — warmup iterations, default `2000000`.
- `-s,--stats` — print latency percentiles.
- `-v,--verbose` — verbose output.

### `bench_mpmc_h2d_throughput`

Measures MPMC host-to-device throughput with multiple CPU producers and GPU consumer wavefronts.

```bash
./bench_mpmc_h2d_throughput -m 1000000 -t 4 -g 4 -p
```

Common options:

- `-c,--capacity INT` — queue capacity, default `1024`.
- `-m,--messages INT` — total messages, default `1000000`.
- `-t,--cpu-threads INT` — CPU producer threads, default `1`.
- `-g,--gpu-blocks INT` — GPU consumer blocks/wavefronts, default `1`.
- `-p,--pin-cpu` — pin CPU producer threads to cores.
- `-v,--verbose` — per-thread/block breakdown.

## invoke Benchmarks

### `bench_invoke_latency`

Measures round-trip GPU→CPU→GPU latency using invoke callbacks.

```bash
cd build/benchmarks/invoke
./bench_invoke_latency --stats
```

Common options:

- `-c,--capacity INT` — queue capacity in 64-byte lines, default `1024`.
- `-n,--iterations INT` — measured iterations, default `1000000`.
- `-w,--warmup INT` — warmup iterations, default `2000000`.
- `-p,--payload-size INT` — argument bytes (4, 8, 16, 32, 52, 112, 224,
  512), default `4`.
- `-s,--stats` — print latency percentiles.
- `-v,--verbose` — verbose output.

### `bench_invoke_throughput`

Measures callback throughput with invoke.

```bash
./bench_invoke_throughput -m 1000000 -e 1 -t 1
```

Common options:

- `-c,--capacity INT` — queue capacity in 64-byte lines, default `1024`.
- `-m,--messages INT` — total callbacks, default `1000000`.
- `-e,--endpoints INT` — parallel endpoints/queues, default `1`.
- `-t,--cpu-threads INT` — CPU consumer threads, default `1`.
- `--poll-mode direct|host-poll-state` — consumer polling, default
  `host-poll-state`.
- `-v,--verbose` — verbose output.

### `bench_invoke_overhead`

Measures GPU invoke overhead with invoke.

```bash
./bench_invoke_overhead -m 100000
```

Common options:

- `-c,--capacity INT` — queue capacity in 64-byte lines, default `1024`.
- `-m,--messages INT` — number of invokes, default `100000`.
- `-w,--warmup INT` — warmup iterations, default auto.
- `-v,--verbose` — verbose output.

## Cross-frontend harnesses

The benchmarks above are standalone C++ programs. To compare kernels written
in different languages on identical host code, `bench_invoke_latency` and
`bench_invoke_throughput` also have a cross-frontend harness, built with the
benchmarks:

```text
invoke/frontends/
├── latency/
│   ├── common/   # host code + Python bridge (kiwi_invoke_latency_bench_ext, driver)
│   ├── cpp/      # harness_invoke_latency_cpp
│   ├── flydsl/   # bench_invoke_flydsl_latency.py (KIWI_BUILD_FLYDSL)
│   └── triton/   # bench_invoke_triton_latency.py (KIWI_BUILD_TRITON)
└── throughput/   # the same for bench_invoke_throughput
frontends/        # Python helpers shared by the frontends of every harness
```

- `common/` holds a copy of the standalone benchmark's host code (setup,
  callbacks, warmup, timed region, validation, output) as a header templated
  on a kernel launcher, a pybind11 extension that runs it with a Python
  launcher, and the Python driver with the standalone options.
- `cpp/` is the standalone benchmark's HIP kernel and launch, built with the
  same options, plus a CLI. It runs through `common/` like every other
  frontend, so it is the reference for comparing frontends and is itself
  checked against the standalone benchmark.
- Each Python frontend directory (`flydsl/`, `triton/`) holds only its
  kernels and a launcher. A Python kernel is compiled and loaded by an
  untimed launch before the timed region.

Every frontend takes the standalone options and prints the standalone output
(the title names the frontend). Python extensions and drivers are built into
`build/python`; executables and scripts land at the mirrored build paths. The
P2P readiness benchmark (`apps/p2p/benchmarks/frontends/readiness/`; see
[`apps/README.md`](../apps/README.md)) and the `kiwi-spsc` queue baseline
(`queue_baselines/frontends/`; see
[`queue_baselines/README.md`](queue_baselines/README.md#cross-frontend-harness))
have the same harness. CTest runs each frontend briefly. From the repository
root, with the build in `build/`; run frontends in the same environment and
pin them the same way when comparing:

```bash
H=build/benchmarks/invoke/frontends
export PYTHONPATH="$PWD/build/python"
taskset -c 8 $H/latency/cpp/harness_invoke_latency_cpp -c 1024 -n 1000000 -s
taskset -c 8 python3 $H/latency/flydsl/bench_invoke_flydsl_latency.py -c 1024 -n 1000000 -s
taskset -c 8 python3 $H/latency/triton/bench_invoke_triton_latency.py -c 1024 -n 1000000 -s
taskset -c 8-23 $H/throughput/cpp/harness_invoke_throughput_cpp \
    -m 10000000 -e 1 -t 1 --poll-mode direct
taskset -c 8-23 python3 $H/throughput/flydsl/bench_invoke_flydsl_throughput.py \
    -m 10000000 -e 1 -t 1 --poll-mode direct
taskset -c 8-23 python3 $H/throughput/triton/bench_invoke_triton_throughput.py \
    -m 10000000 -e 1 -t 1 --poll-mode direct
```

FlyDSL frontends need `KIWI_BUILD_FLYDSL=ON` (see the
[FlyDSL invoke guide](../docs/flydsl.md)) and Triton frontends
`KIWI_BUILD_TRITON=ON` (see [`python/README.md`](../python/README.md)). The
Triton throughput kernel supports at most 256 endpoints (a Triton program is
one wavefront).

## Hardware Benchmark

### `bench_pcie_latency`

Characterizes CPU/GPU signaling latency without the queue API.

```bash
cd build/benchmarks/hardware
./bench_pcie_latency [iterations] [warmup]
```

Defaults are `100000` measured iterations and `10000` warmup iterations.

### `bench_host_mapped_grid_poll`

Characterizes persistent-grid polling of host-mapped memory.

## Adding New Benchmarks

For queue:

1. Create a new `.hip.cpp` file in `benchmarks/queue/`.
2. Include `<kiwi/queue/queue.hpp>` and `<CLI/CLI.hpp>`.
3. Define a local parameter struct for benchmark-specific options.
4. Add the benchmark to `benchmarks/queue/CMakeLists.txt` using `add_queue_benchmark()`.

For invoke:

1. Create a new `.hip.cpp` file in `benchmarks/invoke/`.
2. Include `<kiwi/invoke/invoke.hpp>` and `<CLI/CLI.hpp>`.
3. Define a local parameter struct for benchmark-specific options.
4. Add the benchmark to `benchmarks/invoke/CMakeLists.txt` using `add_invoke_benchmark()`.
