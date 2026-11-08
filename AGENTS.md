# KIWI agent instructions

KIWI core is the queue and invocation mechanism. P2P and EP are prototype
applications built on the invocation layer, not core components.
Dependencies must point from applications to core, never the reverse.
Preserve existing public APIs unless the task explicitly calls for migration.

- For application work, also read the more specific `apps/AGENTS.md`.
- For cross-component build changes, apply both this file and the application
  instructions.

## Repository hygiene

This tree is prepared for open-source release. Every source file starts with
the AMD copyright line and `SPDX-License-Identifier: MIT`; keep that header on
new files. Code copied or adapted from another project keeps the original
copyright line, adds `Modifications Copyright © Advanced Micro Devices, Inc.`,
uses the upstream license in its SPDX line, and is listed in `NOTICES.txt`. Do not commit cluster or host names, internal IP addresses, personal
paths, internal links, or results from unreleased hardware. Cluster-specific
runbooks and dated investigation or benchmark reports do not belong in this
repository.

## Core development

The core is a HIP/C++ library for low-latency control-message queues between
AMD GPUs and the host CPU. `invoke` is the callback-dispatch layer built on
those queues. Applications are opt-in and must not introduce dependencies into
the default core build.

Use `#include <kiwi/queue/queue.hpp>` for the queue API. Canonical aliases are:

- `kiwi::queue::SPSCD2HQueue`: one GPU wavefront pushes, one CPU thread pops.
- `kiwi::queue::MPSCD2HQueue`: multiple GPU wavefronts push, one CPU thread pops.
- `kiwi::queue::SPSCH2DQueue`: one CPU thread pushes, one GPU wavefront pops.
- `kiwi::queue::MPMCH2DQueue`: multiple CPU threads push, multiple GPU wavefronts pop.

Core invariants:

- Capacity is a non-zero power of two.
- Device queue operations are wavefront-cooperative and use a 64-thread
  wavefront.
- Both D2H queues use a 16-byte chunk wire format (LL8: per lane two halves
  of 4 B payload + 4 B signal, messages up to 512 B, capacity in 64-byte
  lines) and one push
  API: `push_reg`, `push_shared`, `push_global`, `push_lanes`, and
  `push_lane_masked` (each active lane pushes its own message). invoke
  mirrors it: `invoke`, `invoke_shared`, `invoke_lane_masked`.
  No push path may use scratch (`test_queue_push_isa`), except that
  `push_lane_masked` spills large per-lane messages: from 512 B, or from
  192 B inside an SPSC retry loop, with the default launch bounds
  (`docs/core.md`). SPSC pushes use no atomics.
- `push_lane_masked` is wave-collective too: all 64 lanes call it from
  converged code with a per-lane `active` flag. A call is all-or-nothing
  like `push_lanes` (SPSC returns `false` when short; MPSC waits), and a
  call longer than the ring (`capacity` lines) traps. Lanes on the two
  sides of a divergent branch are concurrent threads to the compiler, so
  they must never share per-lane producer state through plain loads and
  stores, and the SPSC producer index is a register every lane must keep
  equal.
- H2D queues use 64-byte wave-slots with 56 payload bytes; lane 7 is an epoch
  or sequence.
- H2D device consumers call `flush_consumer_index()` before returning.
- Do not introduce device-side virtual calls. Device-callable methods stay inline
  in HIP headers.

Build and test the core from the repository root:

```bash
cmake -S . -B build/core-tests -DCMAKE_BUILD_TYPE=Release \
  -DKIWI_BUILD_TESTS=ON -DHIP_ARCHITECTURES=gfx942
cmake --build build/core-tests -j
ctest --test-dir build/core-tests --output-on-failure
```

Useful options are `KIWI_BUILD_BENCHMARKS`, `KIWI_BUILD_EXAMPLES`, and
`KIWI_BUILD_INVOKE`. The corresponding source trees are `include/kiwi/queue/`,
`include/kiwi/invoke/`, `src/queue/`, `tests/{queue,invoke}/`,
`benchmarks/{queue,invoke,hardware}/`, and `examples/{queue,invoke}/`.

When changing queue internals, run the narrow direction/concurrency test first.
Changes involving memory ordering or visibility also require the relevant
`test_queue_hardware` or `test_queue_wavefront_atomicity` coverage. Run the
full configured CTest suite before sharing a broad change.
