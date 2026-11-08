# KIWI core: queues and invocations

Run the commands below from the [KIWI repository root](../README.md).
The public namespaces are `kiwi::queue` and `kiwi::invoke`.

The KIWI queue layer provides low-latency control-message transport between AMD GPUs and the host CPU. It provides SPSC queues for both directions (`GPU -> CPU` and `CPU -> GPU`), an MPSC `GPU -> CPU` queue, and an MPMC `CPU -> GPU` queue. The invocation layer provides callback dispatch on top of those queues.

Device-to-host queues carry messages of up to **512 bytes** in **16-byte chunks**, each two 8-byte halves of 4 B payload + 4 B signal (LL8). Host-to-device queues use **64-byte wave-slots** with 56 bytes of payload. Device-side queue operations are intended to be called by a full 64-lane wavefront.

## Features

- GPU-to-CPU and CPU-to-GPU queues.
- SPSC queues with no atomics on the single-producer/single-consumer state.
- Device-to-host push API (`SPSCD2HQueue`, `MPSCD2HQueue`): `push_reg`,
  `push_shared`, `push_global` and `push_lanes`. Messages are 0 to 512 bytes with 32-bit
  metadata; none of these pushes uses scratch.
- Lane-masked device-to-host pushes (`push_lane_masked`, and
  `invoke_lane_masked` in invoke): each active lane of the wavefront pushes
  its own message, all-or-nothing per call. Large per-lane messages may spill
  to scratch.
- An MPSC device-to-host queue: producer wavefronts claim lines with one
  device-memory fetch-add per push; the single host consumer is the SPSC one.
- MPMC host-to-device queues using atomic slot claiming and Vyukov-style
  sequence numbers.
- Epoch-based readiness, so slots and lines do not need separate clearing.
- Cached producer/consumer progress to reduce PCIe traffic on hot paths.
- Benchmark and hardware-assumption test suites for MI300X/ROCm systems.

## Requirements

- AMD GPU with ROCm/HIP support. The default CMake fallback architecture is `gfx942` for MI300X.
- ROCm/HIP installation, usually under `/opt/rocm` or pointed to by `ROCM_PATH`/`HIP_PATH`.
- ROCr with an allocatable extended-scope fine-grained host memory pool.
- CMake 3.21 or newer.
- C++17 compiler support.
- Network access during CMake configure if tests or benchmarks are enabled, because CMake FetchContent downloads GoogleTest and CLI11.

## Build and Test

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKIWI_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

Useful CMake options:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DKIWI_BUILD_TESTS=ON \
  -DKIWI_BUILD_BENCHMARKS=ON \
  -DKIWI_BUILD_EXAMPLES=ON \
  -DKIWI_BUILD_INVOKE=ON
```

CMake auto-detects `HIP_ARCHITECTURES` with `rocminfo` when possible. Override it explicitly if needed:

```bash
cmake .. -DHIP_ARCHITECTURES=gfx942
```

## Basic Device-to-Host Usage

```cpp
#include <kiwi/queue/queue.hpp>
#include <hip/hip_runtime.h>
#include <cstring>
#include <memory>

__global__ void producer(kiwi::queue::SPSCD2HQueue::DeviceHandle q, int num_msgs) {
    for (int i = 0; i < num_msgs; ++i) {
        uint32_t value = static_cast<uint32_t>(i);

        // A full wavefront cooperates on the push. Retry if the queue is full.
        while (!q.push_reg(value)) {}
    }
}

int main() {
    auto queue = std::make_unique<kiwi::queue::SPSCD2HQueue>(1024);

    producer<<<1, 64>>>(queue->device_handle(), 1000);

    uint8_t buf[kiwi::queue::kMaxMessageBytes];
    int received = 0;
    while (received < 1000) {
        size_t bytes = 0;
        if (queue->pop(buf, /*metadata=*/nullptr, &bytes)) {
            uint32_t value = 0;
            std::memcpy(&value, buf, sizeof(value));  // bytes == 4
            // process(value);
            ++received;
        }
    }

    hipDeviceSynchronize();
}
```

Important notes:

- Queue capacity must be a non-zero power of two. Device-to-host capacity is
  the ring size in 64-byte lines, at most `kiwi::queue::kMaxCapacityLines`
  (2^31) so the producer's credit check stays a 32-bit scalar compare; larger
  capacities throw `queue_error`. Messages are line-aligned and a line
  carries up to 32 B of payload, so a message of `B` bytes uses
  `kiwi::queue::message_lines(B)` lines (a 56-byte KIWI-EP put descriptor 2,
  a 512-byte message 16). The ring must hold the largest message: a longer
  push traps, and `invoke::Context::register_callback()` throws
  `std::invalid_argument` for arguments the rings cannot hold.
- Device-side operations need a full 64-lane wavefront (wave64 GFX9 target)
  with every lane passing the same arguments, except the per-lane arguments
  of `push_lane_masked` and `invoke_lane_masked`.
- `pop(buf, &metadata, &bytes)` returns one whole message, or `false` without
  consuming anything if any part of it is not visible yet.

### Device-to-host push API

`SPSCD2HQueue::DeviceHandle` and `MPSCD2HQueue::DeviceHandle` provide four pushes of one message of up to
`kMaxMessageBytes` (512) bytes with 32-bit metadata:

| Push | Payload source |
| --- | --- |
| `push_reg(const T& msg, uint32_t metadata = 0)` | registers; `sizeof(T)` bytes |
| `push_shared(const void* lds, size_t bytes, uint32_t metadata = 0)` | 8-byte aligned shared memory |
| `push_global(const void* ptr, size_t bytes, uint32_t metadata = 0)` | 8-byte aligned global memory |
| `push_lanes(uint64_t word, uint32_t bytes, uint32_t metadata)` | lane `i` passes payload word `i` |

`push_shared` and `push_global` read `round_up(bytes, 8)` bytes. On a full
ring an SPSC push returns `false` without writing; callers loop on
`!push(...)`. An MPSC push has already claimed its lines, so it waits for the
consumer instead and always returns `true`.

`SPSCD2HQueue` and `MPSCD2HQueue` have the same ring format, push API, and
host consumer (`pop`, `host_poll_state`), so `invoke::Context<Q>` and the
applications work with either. They differ in how a producer finds where to
write:

| | `SPSCD2HQueue` | `MPSCD2HQueue` |
| --- | --- | --- |
| Producers per queue | one wavefront | any number of wavefronts |
| Claim | producer index in registers | lane-0 agent-scope fetch-add on a device-memory index |
| Full ring | push returns `false` | push waits (`s_sleep`) for credit |
| `device_handle()` | queue must be drained; publishes the exact consumer position | any time; handles may be copied and used concurrently |

Messages of one wavefront are popped in push order in both queues.

### Lane-masked pushes

The pushes above send one message per call, one payload word per lane.
`push_lane_masked` sends one message per active lane, each with its own
payload, length and metadata:

```cpp
template<typename T>
bool push_lane_masked(const T& message, uint32_t bytes, uint32_t metadata, bool active);
```

- It is wavefront-collective on both queues, like every push: all 64 lanes
  call it from converged code (debug builds trap otherwise). Each lane passes
  its own arguments; a lane with `active` false pushes nothing, and the
  values of its other arguments are ignored: its `message` contents, `bytes`
  and `metadata` need not be meaningful. Its `message` must still be a valid
  reference to a `T` object, as any C++ reference must, and may be read: for
  lanes past the end of an array, clamp the index or pass a dummy object
  instead of forming `data[i]` out of bounds.
- An active lane sends the first `bytes` bytes of its own `message`.
  `bytes > sizeof(T)` traps in every build.
- The active lanes' messages take consecutive lines in lane order, so they
  are popped in lane order, after the wavefront's earlier pushes. Each lane
  writes every chunk of its own message.
- A call is all-or-nothing and returns `bool`, like `push_lanes`:
  - `SPSCD2HQueue`: one credit check for the whole call, which re-reads the
    consumer index over PCIe only when the cached credit looks short. If the
    active lanes' lines do not fit, nothing is written and the call returns
    `false`; retry it with `while (!h.push_lane_masked(...)) {}`. The
    producer index advances in registers, with no atomics. It mixes freely
    with the collective pushes and persists across launches like them.
  - `MPSCD2HQueue`: lane 0 claims the active lanes' lines with one fetch-add
    per call and waits until the consumer has freed all of them, as the
    collective pushes do; the call returns `true`. The handle's credit cache
    is updated as in the collective pushes.
- A call may take the whole ring: the consumer publishes its position on
  every pop, so once it has popped every earlier message, a waiting call has
  credit for up to `capacity` lines. A call whose active lanes need more
  lines than the ring holds traps in every build, like a message longer than
  the ring. The largest call, 64 messages of 512 B, takes 1024 lines.
- Each lane holds its whole message in registers and writes it in
  straight-line code, building each chunk just before its store: about
  `sizeof(T) / 4 + 18` VGPRs. With the default launch bounds (1024 threads,
  128 VGPRs), messages up to 384 B use no scratch; at 512 B the message
  alone fills the budget and 60-90 B per lane spill.
- An SPSC retry loop (`while (!h.push_lane_masked(...)) {}`) costs more: the
  chunks do not change between retries, so the compiler builds all of them
  before the loop and keeps them live, about 2.5 times the message. 128 B
  takes about 90-100 VGPRs without scratch; from 192 B the push spills
  (12-108 B per lane at 192 B, about 1 KB at 512 B). With
  `__launch_bounds__(256)` (512 VGPRs) a 512-byte message takes 330-390
  VGPRs and no scratch. MPSC calls need no retry loop.
- Examples above: gfx942, ROCm 6.3 and 7.14, each lane's message loaded
  from global memory; exact counts depend on the kernel.

There is no per-lane push from divergent code (`if (c) push(a); else
push(b);`) or from part of the wavefront; pass `active` instead. The SPSC
producer index is a register that every lane must keep equal, and lanes on
the two sides of a divergent branch are concurrent threads to the compiler:
it may sink one side's plain store past the other side's load, so they cannot
share producer state through memory either. On MPSC, a call from part of the
wavefront could not update the credit cache the collective pushes share, so
every such call re-read the consumer index over PCIe. A divergent per-lane
push, if needed later, would be a separate API.

```cpp
__global__ void producer(SPSCD2HQueue::DeviceHandle h) {
    Desc d = make_desc(threadIdx.x);
    const bool has_work = d.bytes != 0;
    while (!h.push_lane_masked(d, sizeof(Desc), kDesc, has_work)) {}  // whole wavefront
    while (!h.push_reg(summary())) {}                                 // collective
}
```

`invoke::DeviceContext<Q>::DeviceHandle::invoke_lane_masked(callback_id,
active, args...)` is the invoke counterpart: each active lane serializes its
own arguments (as `invoke` does) and pushes them as one message with its own
`callback_id`, through `push_lane_masked`, whose result it returns, as
`invoke` returns the push's. The host dispatches the messages like any other
invoke. As with `push_lane_masked`, an inactive lane's `callback_id` and
argument values are ignored, but its arguments must be valid references to
readable objects; it may read them. For example,
`invoke_lane_masked(id, i < n, data[i < n ? i : 0])` rather than `data[i]`.

### Cached host polling for device-to-host queues

`host_poll_state()` returns an opaque host-side state that caches the next
message's chunk-0 signal and expected epoch. Its `ready()` check avoids
reading queue consumer state. Refresh it only after a real progress attempt
that might consume messages—not on the common empty polling path:

```cpp
auto poll_state = queue->host_poll_state();
if (poll_state.ready()) {
    context.progress(endpoint_id);
    poll_state.refresh();  // progress may have advanced the queue
}
```

Use the state only while the queue remains alive. `ready()` is a hint: it can
be true while `pop()` still returns false, when chunk 0 of a message is visible
before its other chunks.

## Public Queue Types

The main header is:

```cpp
#include <kiwi/queue/queue.hpp>
```

Canonical aliases:

| Alias | Direction | Concurrency | Producer | Consumer |
| --- | --- | --- | --- | --- |
| `kiwi::queue::SPSCD2HQueue` | Device to host | SPSC | one GPU wavefront | one CPU thread |
| `kiwi::queue::MPSCD2HQueue` | Device to host | MPSC | multiple GPU wavefronts | one CPU thread |
| `kiwi::queue::SPSCH2DQueue` | Host to device | SPSC | one CPU thread | one GPU wavefront |
| `kiwi::queue::MPMCH2DQueue` | Host to device | MPMC | multiple CPU threads | multiple GPU wavefronts |

Device-to-host queues use the chunk format below; host-to-device queues use 56-byte payload wave-slots. Host-to-device queues require each consuming GPU block/wavefront to call `flush_consumer_index()` before the kernel returns so host-side producers can see final consumer progress.

## Payload Layout

### Device to host

The ring is `capacity` 64-byte lines of four 16-byte chunks. Lane `c` of the
pushing wavefront writes chunk `c` of a message with one 16-byte store:

```text
line = 64 B   +-------------+-------------+-------------+-------------+
              |   chunk 0   |   chunk 1   |   chunk 2   |   chunk 3   |
              +-------------+-------------+-------------+-------------+
chunk = 16 B  | data lo (4) | sig 0 (4)   | data hi (4) | sig 1 (4)   |
              +-------------+-------------+-------------+-------------+
```

- `data lo` / `data hi` are payload bytes `[8c, 8c + 4)` / `[8c + 4, 8c + 8)`
  of the message.
- Both signals of every chunk hold the lap epoch of the chunk's line (bits
  0-7). Chunk 0's signals also hold the header: sig 0 has metadata bits 0-7
  (bits 8-15) and the length (bits 16-31); sig 1 has metadata bits 8-31
  (bits 8-31).
- A message of `B` bytes starts on a line and occupies `ceil(max(1, ceil(B/8)) / 4)`
  lines (one line up to 32 bytes, 16 lines for 512 bytes).
- The host consumes a message only when every signal of its lines, padding
  included, carries its line's expected epoch, so it never reads a partly
  written message. Each signal guards the 4 data bytes beside it, so this
  relies only on aligned 8-byte halves of GPU stores and host loads being
  single-copy atomic (as NCCL's LL protocol), not on 16-byte atomicity. The
  host consumer is x86-64 only.
- `invoke` sends each callback as one message: the callback id is the
  metadata and the serialized arguments are the payload.

### Host to device

Each queue slot is a 64-byte wave-slot represented as 8 `uint64_t` lanes:

```text
+--------+--------+--------+--------+--------+--------+--------+-------------+
| lane 0 | lane 1 | lane 2 | lane 3 | lane 4 | lane 5 | lane 6 | lane 7      |
| 8B pay | 8B pay | 8B pay | 8B pay | 8B pay | 8B pay | 8B pay | epoch/seq   |
+--------+--------+--------+--------+--------+--------+--------+-------------+
```

- Lanes 0-6 provide 56 bytes of payload.
- Lane 7 is control metadata: SPSC queues use an epoch and MPMC queues use a sequence number.

## Examples

Build examples with `KIWI_BUILD_EXAMPLES=ON` and run them from the build tree:

```bash
cd build
./examples/queue/simple_example
./examples/invoke/invoke_example
```

## Benchmarks

Build benchmarks with `KIWI_BUILD_BENCHMARKS=ON`. Executables are placed under benchmark-specific build directories:

```bash
# queue
./benchmarks/queue/bench_queue_latency --stats
./benchmarks/queue/bench_queue_throughput
./benchmarks/queue/bench_queue_overhead
./benchmarks/queue/bench_host_to_device_latency
./benchmarks/queue/bench_mpmc_h2d_latency
./benchmarks/queue/bench_mpmc_h2d_throughput

# invoke
./benchmarks/invoke/bench_invoke_latency --stats
./benchmarks/invoke/bench_invoke_throughput
./benchmarks/invoke/bench_invoke_overhead
```

See [`benchmarks/README.md`](../benchmarks/README.md) for command-line options.

## Test Targets

The most important tests for queue correctness are:

- `test_queue_basic` — slot packing, config, and error handling.
- `test_queue_chunk` — both device-to-host queues: every message size through every push path, wraparound in 1- to 4096-line rings, relaunch, consumer index publication on every pop, backpressure, late-chunk fault injection, and MPSC stress with up to 304 producer wavefronts on rings as small as one line.
- `test_queue_concurrent` — SPSC GPU-to-CPU interaction.
- `test_queue_host_poll_state` — cached host readiness polling.
- `test_host_to_device_concurrent` — SPSC CPU-to-GPU interaction.
- `test_queue_mpmc_h2d_concurrent` — MPMC CPU-to-GPU interaction.
- `test_queue_push_lane_masked` — `push_lane_masked` on both device-to-host queues: lane order, active masks, wraparound and back-pressure, an SPSC call that finds the ring short (returns `false`, writes nothing), mixed with collective pushes, many MPSC producer waves, and the traps.
- `test_queue_push_isa` — SPSC and MPSC push kernels (including invoke through a `DeviceContext`) use no scratch, global stores and `readfirstlane` broadcasts; MPSC kernels claim with a global 64-bit fetch-add, and SPSC kernels use no atomics.
- `test_queue_hardware` and `test_queue_wavefront_atomicity` — hardware assumptions used by the queue design.

FlyDSL kernels can also invoke callbacks; see the [FlyDSL invoke guide](flydsl.md).

Run all tests with:

```bash
cd build
ctest --output-on-failure
```

## Troubleshooting

### A push returns `false` or keeps waiting

The queue is full. Increase capacity, reduce producer rate, or make sure the consumer is running and flushing progress when required. A device-to-host ring must hold the largest message (`message_lines(bytes)` lines); a message longer than the ring traps instead of waiting forever, and so does a `push_lane_masked` call whose active lanes need more lines than the ring holds.

### Host-to-device producer stalls or sees a full queue after a kernel exits

Make sure the device consumer calls `flush_consumer_index()` before returning from the kernel. This applies to SPSC and MPMC host-to-device queues.

### Unexpected empty queue or missing data

Run the hardware-assumption tests and the relevant concurrent tests. Queue-owned
host storage uses ROCr's extended-scope fine-grained pool so GPU accesses use an
uncached mapping on both MI300X and MI355X. This allows the CPU and GPU to
observe updates while a kernel remains active without requiring
`__threadfence_system()` or architecture-specific volatile stores.

### Capacity errors

Queue capacity must be a power of two. Use values such as 256, 1024, or 4096 lines (device to host) or wave-slots (host to device).

## Repository Layout

```text
include/kiwi/queue/   Public queue headers
include/kiwi/invoke/  Callback bridge built on queue
src/queue/         Host-side queue implementations
benchmarks/        Performance benchmarks
tests/             Unit, integration, and hardware-assumption tests
examples/          Minimal usage examples
```

## License

KIWI is released under the [MIT license](../LICENSE).
