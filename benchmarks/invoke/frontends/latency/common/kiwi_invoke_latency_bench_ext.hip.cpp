// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Python entry point to the cross-frontend invoke round-trip latency harness.
//
// run() runs the harness's host code (invoke_latency_bench.hpp) with a Python
// launcher in place of the HIP kernel launch, so a kernel written in a Python
// GPU language is measured by the same host code as the C++ frontend (cpp/).
// The defaults and payload sizes are exported from the same header. Use it
// through kiwi_invoke_latency_bench.py.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <invoke_latency_bench.hpp>
#include <py_launcher.hpp>

#include <stdexcept>
#include <string>

namespace py = pybind11;
using kiwi_bench::py_launch::addr;

namespace {

// launch(ctx, ping_cb, ack_flag, num_iterations, latencies, warmup_iterations,
//        latency_stats, stream, payload_size). The host code frees its
// buffers if the launcher raises, and the exception reaches Python.
struct PyLatencyLauncher {
    py::object fn;
    int payload_size;

    template<typename PayloadT>
    void prepare() {}

    template<typename PayloadT>
    void launch(const kiwi_bench::LatencyLaunch& l) {
        kiwi_bench::py_launch::call(fn, addr(l.ctx), l.ping_cb, addr(l.ack_flag),
                                    l.num_iterations, addr(l.latencies), l.warmup_iterations,
                                    l.latency_stats, addr(l.stream), payload_size);
    }
};

void run(const py::object& launch, int queue_capacity, int num_iterations,
         int warmup_iterations, int payload_size, bool latency_stats, bool device_ack,
         const std::string& title) {
    if (!kiwi_bench::is_latency_payload_size(payload_size))
        throw std::invalid_argument("unsupported payload size");
    kiwi_bench::LatencyParams params;
    params.queue_capacity = queue_capacity;
    params.num_iterations = num_iterations;
    params.warmup_iterations = warmup_iterations;
    params.payload_size = payload_size;
    params.latency_stats = latency_stats;
    params.device_ack = device_ack;
    PyLatencyLauncher launcher{launch, payload_size};

    py::gil_scoped_release release;
    kiwi_bench::run_latency_benchmark(params, launcher, title.c_str());
}

}  // namespace

PYBIND11_MODULE(kiwi_invoke_latency_bench_ext, m) {
    const kiwi_bench::LatencyParams d;
    m.attr("DEFAULTS") = py::dict(
        py::arg("capacity") = d.queue_capacity, py::arg("iterations") = d.num_iterations,
        py::arg("warmup") = d.warmup_iterations, py::arg("payload_size") = d.payload_size);
    m.attr("PAYLOAD_SIZES") = py::tuple(py::cast(kiwi_bench::latency_payload_sizes()));
    m.def("run", &run, py::arg("launch"), py::arg("queue_capacity"),
          py::arg("num_iterations"), py::arg("warmup_iterations"), py::arg("payload_size"),
          py::arg("latency_stats"), py::arg("device_ack"), py::arg("title"));
}
