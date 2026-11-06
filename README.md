# KIWI

**Kernel-Initiated Workload-defined Invocations**

KIWI's core provides GPU/host queues and workload-defined callback invocation
for AMD ROCm/HIP systems. P2P communication and expert-parallel (EP)
dispatch/combine are **prototype applications built on the invocation layer**,
not parts of the core.

## Layout

```text
include/kiwi/queue/          Core queue headers
include/kiwi/invoke/         Header-only invocation layer
src/queue/               Compiled queue implementation
tests/{queue,invoke}/     Core correctness tests
tests/python/            Optional FlyDSL and Triton invoke tests
tests/layout/            Host-only build/wrapper layout checks
python/                  Optional FlyDSL and Triton invoke frontends
benchmarks/              Core queue/invocation and hardware benchmarks
examples/                Core usage examples
apps/p2p/                P2P sources, headers, tests, benchmarks
apps/ep/                 EP sources, headers, tests, benchmarks, python/
apps/common/topology/    Existing support shared only by the applications
cmake/                   Shared build helpers
scripts/                 Application benchmark/correctness launchers
docker/                  EP benchmark image
docs/                    Core API, FlyDSL, and EP benchmark-image guides
```

Public APIs use `kiwi::queue`, `kiwi::invoke`, `kiwi::p2p`, and `kiwi::ep`.
Application headers remain beside their sources.

## Build the core

Requirements: CMake 3.21+, a C++17/ROCm HIP toolchain, and ROCr extended-scope
fine-grained host memory support for queue allocation.

```bash
cmake -S . -B build/core -DHIP_ARCHITECTURES=gfx942
cmake --build build/core -j
```

This default builds only `kiwi_queue` and `kiwi_invoke`, without fetching LCI or
requiring MPI, PyTorch, GoogleTest, or CLI11. Set the architecture for your GPU;
`gfx942` targets MI300X. Disable `KIWI_BUILD_INVOKE` for queues alone.

Core tests, benchmarks, and examples are opt-in:

```bash
cmake -S . -B build/core-tests -DHIP_ARCHITECTURES=gfx942 \
  -DKIWI_BUILD_TESTS=ON -DKIWI_BUILD_BENCHMARKS=ON
cmake --build build/core-tests -j
ctest --test-dir build/core-tests --output-on-failure
```

Tests fetch GoogleTest; benchmarks fetch CLI11. `KIWI_BUILD_EXAMPLES=ON`
enables examples. The optional SDMA example additionally needs ROCm-XIO and MPI;
set `KIWI_BUILD_INVOKE_SDMA_PING_PONG=OFF` for only the small core examples.
See the [core API guide](docs/core.md).

`KIWI_BUILD_FLYDSL=ON` builds the optional pure-Python `kiwi_flydsl` package
so FlyDSL kernels can invoke callbacks registered on a C++ invoke context, and
adds `test_flydsl_invoke` to CTest when tests are enabled. It requires
invocation, Python 3, and FlyDSL; its tests also need pybind11. See the
[FlyDSL invoke guide](docs/flydsl.md).

## Build applications

Both applications are off by default and can be enabled independently:

```bash
# P2P only
cmake -S . -B build/p2p -DKIWI_BUILD_P2P=ON -DHIP_ARCHITECTURES=gfx942
# EP only (does not build or depend on kiwi_p2p)
cmake -S . -B build/ep -DKIWI_BUILD_EP=ON -DHIP_ARCHITECTURES=gfx942
# Both apps, with their tests and benchmarks
cmake -S . -B build/apps -DKIWI_BUILD_P2P=ON -DKIWI_BUILD_EP=ON \
  -DKIWI_BUILD_TESTS=ON -DKIWI_BUILD_BENCHMARKS=ON -DHIP_ARCHITECTURES=gfx942
cmake --build build/apps -j
ctest --test-dir build/apps --output-on-failure
```

Application builds require invocation and fetch LCI. Their tests/benchmarks
require MPI and CLI11. `KIWI_BUILD_PYTHON=ON` adds EP's `kiwi_ep_ext` binding
and requires `KIWI_BUILD_EP=ON`, PyTorch, and pybind11.
`KIWI_BUILD_TRITON=ON` adds the optional Triton frontend (`kiwi_triton`); it
requires Python, PyTorch, Triton, pybind11, and a HIP runtime, but does not
enable either application.
The test and benchmark options apply to the core and every enabled application.

All configurations use the root CMake entry point. External dependency
overrides such as `FETCHCONTENT_SOURCE_DIR_LCI`, `FETCHCONTENT_SOURCE_DIR_CLI11`,
and `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` are available.

See the [application guide](apps/README.md) for APIs and benchmark details.
See the [Python frontend guide](python/README.md) for the Triton frontend,
its tests and benchmarks.

Agent instructions follow the same ownership boundary: repository-wide and
core guidance is in `AGENTS.md`, while application-specific guidance is in
`apps/AGENTS.md`.

## Validation and containers

Run wrappers from the KIWI root. The container wrapper mounts the **whole KIWI
checkout** at `/workspace/kiwi`; `KIWI_REPO` and `--kiwi` refer to the KIWI
root:

```bash
scripts/ep_bench/run_container.sh -- bash
python3 -m unittest discover -s tests/layout -v
```

EP build artifacts are under `build/<configuration>/apps/ep/`, with Python
bindings under `apps/ep/python/` inside that build directory. P2P artifacts are
under `apps/p2p/`; core test/benchmark artifacts are under `tests/` and
`benchmarks/`.

Follow the [EP benchmark image guide](docs/ep-benchmark-image.md) for GPU
validation. Full MORI compatibility requires an importable built official MORI package and
execution of the full matrix; unavailable or skipped tests are not passes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

KIWI is released under the [MIT license](LICENSE). A few queue-baseline
adapters under `benchmarks/queue_baselines/adapters/` contain portions adapted
from MSCCL++, UCCL, RCCL, NVSHMEM and PyTorch; those portions keep their
original licenses and are listed in [NOTICES.txt](NOTICES.txt).
