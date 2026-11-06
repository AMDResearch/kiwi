# KIWI invoke from FlyDSL kernels

Run the commands below from the [KIWI repository root](../README.md).

The optional `kiwi_flydsl` Python package lets FlyDSL kernels invoke callbacks
registered on a C++ `kiwi::invoke::Context`. Callbacks, dispatch, and
`progress()` stay in C++; Python only passes the context's
`device_context()` pointer into the kernel. FlyDSL device code emits the same
16-byte chunk wire format
([`chunk.hip.hpp`](../include/kiwi/queue/detail/chunk.hip.hpp)) as C++
`DeviceContext::DeviceHandle::invoke()`, so the C++ dispatcher cannot tell
FlyDSL and C++ producers apart. No device bitcode is linked into FlyDSL
kernels.

## Requirements and build

- `KIWI_BUILD_INVOKE=ON` (the default).
- FlyDSL 0.2.4, the only tested version. The package uses FlyDSL internals,
  so importing it with another version raises `ImportError`; set
  `KIWI_FLYDSL_ALLOW_UNTESTED=1` to try it anyway with a warning.
- The package itself is pure Python. The tests and the example also need
  Python 3 development headers and pybind11, for their C++ modules.
- Build C++ code that shares a process with FlyDSL against the same HIP
  runtime that FlyDSL and PyTorch load. In `rocm/primus:v26.7`, that runtime
  is the pip ROCm SDK selected by `ROCM_PATH` rather than `/opt/rocm`.

```bash
cmake -S . -B build/flydsl -DCMAKE_BUILD_TYPE=Release \
  -DKIWI_BUILD_TESTS=ON -DKIWI_BUILD_FLYDSL=ON -DHIP_ARCHITECTURES=gfx942
cmake --build build/flydsl -j
ctest --test-dir build/flydsl --output-on-failure
PYTHONPATH=build/flydsl/python python3 -c "import kiwi_flydsl"
```

The build places the package in `build/<configuration>/python/kiwi_flydsl/`.
`KIWI_BUILD_FLYDSL` is off by default and adds no dependency to the core build.

`test_flydsl_invoke` exits with code 77 when FlyDSL, PyTorch, or a GPU is
unavailable. CTest reports that as skipped but still prints "100% tests
passed", so check for "Skipped" in the output. In Docker, the container needs
GPU access: `--device /dev/kfd --device /dev/dri --group-add video
--group-add render`.

## Usage

The application owns the C++ context and registers callbacks there. Its own
pybind11 module exposes the `device_context()` pointer that C++ kernels
receive, the callback ids, and `progress()`, as in
[`examples/common`](../examples/common/kiwi_print_ext.hip.cpp):

```cpp
class App {
public:
    App() : ctx_(/*num_endpoints=*/1, /*queue_capacity=*/1024) {
        std::function<void(int32_t, float)> on_sample = [](int32_t i, float x) {
            // runs on the thread calling progress()
        };
        cb_ = ctx_.register_callback(on_sample);
    }
    uintptr_t device_context() const {
        return reinterpret_cast<uintptr_t>(ctx_.device_context());
    }
    uint8_t callback_id() const { return cb_; }
    size_t progress() { return ctx_.progress(0); }

private:
    kiwi::invoke::Context<> ctx_;
    uint8_t cb_ = 0;
};

PYBIND11_MODULE(app_ext, m) {
    pybind11::class_<App>(m, "App")
        .def(pybind11::init<>())
        .def_property_readonly("device_context", &App::device_context)
        .def_property_readonly("callback_id", &App::callback_id)
        .def("progress", &App::progress);
}
```

The `device_context` integer is a non-owning address: keep the `App` (and its
context) alive while kernels use it, and keep calling `progress()` so the
callbacks run and the queue drains.

The FlyDSL kernel invokes the callback by id:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
import kiwi_flydsl as kiwi

@flyc.kernel
def producer(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32):
    h = ctx.get_device_handle(0)
    for i in range(n):
        h = h.invoke(cb, i, fx.Float32(fx.Int32(i)) * 0.5)

@flyc.jit
def launch(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32,
           stream: fx.Stream = fx.Stream(None)):
    producer(ctx, cb, n).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

app = app_ext.App()
launch(kiwi.Endpoints(app.device_context), app.callback_id, 10)
done = 0
while done < 10:
    done += app.progress()
```

Device handles are immutable values: `invoke` returns the updated handle, and
`try_invoke` returns `(handle, ok)`. Reassign the handle inside loops and
branches so FlyDSL carries the queue state.

## API

Host, `kiwi_flydsl.Endpoints(device_context)`:

- `device_context` is the integer address from C++ `Context::device_context()`.
- Pass it directly to a `@flyc.jit` launcher; inside the kernel it is a
  `DeviceContext`.
- It does not own the endpoints. The C++ context must outlive every kernel
  launched with it, and some host thread must call `progress()` on it.

Device:

- `DeviceContext.get_device_handle(endpoint_id=None)` loads an endpoint's
  handle. The default endpoint is `block_idx.x`. As in C++, the handle is
  wave-uniform (every field is lane 0's value, through `readfirstlane`), so
  one wavefront owns the endpoint and the whole wavefront must be active.
- `DeviceContext.save_device_handle(endpoint_id, handle)` persists queue
  progress for a later launch that reuses the endpoint.
- `DeviceHandle.invoke(callback_id, *args, arg_types=None)` retries until the
  message is queued, like C++ `while (!handle.invoke(...)) {}`.
- `DeviceHandle.try_invoke(callback_id, *args, arg_types=None)` returns
  `(handle, fx.Boolean)` without waiting when the queue is full, like C++
  `invoke()`.

Arguments are packed little-endian without padding, exactly like the C++
`ArgumentSerializer`, so a C++ callback registered with
`register_callback<Args...>` receives them when the FlyDSL argument types match
`Args...` in order. Supported types are `Boolean`, `Int8`/`Uint8`,
`Int16`/`Uint16`, `Int32`/`Uint32`, `Int64`/`Uint64`, `Float16`, `Float32`, and
`Float64`. Arguments must be FlyDSL values, not Python constants. FlyDSL
arithmetic with Python literals can widen a value, so `Int16 - 300` becomes
`Int32`; pass `arg_types=` to cast arguments to the callback's types before
they are serialized. Each invoke is one queue message of exactly the argument
bytes, at most 512 (`MAX_ARG_BYTES`, C++ `kiwi::queue::kMaxMessageBytes`),
and callback ids are 0-255 (`MAX_CALLBACKS`). Queue capacity is in 64-byte
lines, and a message of B bytes takes `kiwi::queue::message_lines(B)` of them
(one line per 32 bytes).

No C++ API changes are needed. The FlyDSL code depends on the layouts of
`DeviceContext` and `DeviceContext::DeviceHandle`, which `test_flydsl_invoke`
checks against the C++ headers.

## Example

[`examples/flydsl`](../examples/flydsl/README.md) launches a FlyDSL kernel that
invokes a C++ callback, which prints the arguments it receives. Build it with
`KIWI_BUILD_EXAMPLES=ON`.

The cross-frontend benchmark harnesses have FlyDSL frontends that run the C++
benchmarks' host code: the invoke latency and throughput harnesses in
[`benchmarks/invoke/frontends/`](../benchmarks/README.md#cross-frontend-harnesses),
and the P2P readiness harness in `apps/p2p/benchmarks/frontends/readiness/`,
whose FlyDSL kernel runs the `bench_kiwi_readiness` ping-pong and invokes a
C++ LCI PUT callback (see [`apps/README.md`](../apps/README.md)).

## Limitations

- Only the SPSC device-to-host queue is supported, not the MPMC queues.
- One full 64-lane wavefront owns an endpoint at a time. Launch blocks of 64
  threads, or ensure only one wavefront uses each endpoint.
- Get a handle before the loops and branches that invoke through it, in the
  same or an enclosing region: only its queue state is carried through
  FlyDSL control flow, not its pointers.

## Tests

`test_flydsl_invoke` (Python, registered with CTest) uses the test-only
`kiwi_invoke_test_ext` module (shared with the Triton tests), which owns a C++ context whose typed callbacks
record what they receive. Callbacks taking `std::array<uint8_t, N>` capture
the exact serialized bytes. The test covers the context and handle layouts
against the C++ headers, a typed scalar callback, unaligned mixed-type
arguments, message sizes at the chunk and line boundaries up to 512 bytes,
queue wrap under backpressure, a stress run of 16K invokes into a 16K-line
queue (drained after the kernel and concurrently), per-block endpoints,
`try_invoke` on a full queue, saved progress across launches, and argument
validation. It also checks the push's final ISA: one `global_store_dwordx4`
per invoke, no flat, scratch or buffer accesses, and `v_readfirstlane_b32`
for wave-uniform values. It exits with skip code 77 when FlyDSL or a GPU is
unavailable.
