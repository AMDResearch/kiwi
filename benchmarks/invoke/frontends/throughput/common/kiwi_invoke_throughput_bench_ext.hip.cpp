// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Python entry point to the cross-frontend invoke callback throughput harness.
//
// run() runs the harness's host code (invoke_throughput_bench.hpp) with a
// Python launcher in place of the HIP kernel launch, so a kernel written in a
// Python GPU language is measured by the same host code as the C++ frontend
// (cpp/). The defaults, poll modes, endpoint limit and launch shape are
// exported from the same header. Use it through kiwi_invoke_throughput_bench.py.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <invoke_throughput_bench.hpp>
#include <py_launcher.hpp>

#include <string>

namespace py = pybind11;
using kiwi_bench::py_launch::addr;

namespace {

// launch(ctx, callback_id, num_msgs_per_endpoint, num_endpoints,
//        waves_per_block, launch_blocks), called on the producer thread. A
// launcher error stops the consumers and is rethrown by the host code.
struct PyThroughputLauncher {
    py::object fn;

    void prepare() {}

    void launch(const kiwi_bench::ThroughputLaunch& l) {
        kiwi_bench::py_launch::call(fn, addr(l.ctx), l.callback_id, l.num_msgs_per_endpoint,
                                    l.num_endpoints, l.waves_per_block, l.launch_blocks);
    }
};

int run(const py::object& launch, int queue_capacity, int num_messages, int num_endpoints,
        int num_cpu_threads, const std::string& poll_mode, bool verbose,
        const std::string& title) {
    kiwi_bench::ThroughputParams params;
    params.queue_capacity = queue_capacity;
    params.num_messages = num_messages;
    params.num_endpoints = num_endpoints;
    params.num_cpu_threads = num_cpu_threads;
    params.poll_mode = poll_mode;
    params.verbose = verbose;
    PyThroughputLauncher launcher{launch};

    py::gil_scoped_release release;
    return kiwi_bench::run_throughput_benchmark(params, launcher, title.c_str());
}

}  // namespace

PYBIND11_MODULE(kiwi_invoke_throughput_bench_ext, m) {
    const kiwi_bench::ThroughputParams d;
    m.attr("DEFAULTS") = py::dict(
        py::arg("capacity") = d.queue_capacity, py::arg("messages") = d.num_messages,
        py::arg("endpoints") = d.num_endpoints, py::arg("cpu_threads") = d.num_cpu_threads,
        py::arg("poll_mode") = d.poll_mode);
    m.attr("POLL_MODES") = py::tuple(py::cast(kiwi_bench::throughput_poll_modes()));
    m.attr("MAX_ENDPOINTS") = kiwi_bench::kMaxThroughputEndpoints;

    m.def("launch_shape", [](int num_endpoints) {
        const auto s = kiwi_bench::throughput_launch_shape(num_endpoints);
        return py::make_tuple(s.waves_per_block, s.launch_blocks);
    }, py::arg("num_endpoints"), "(waves_per_block, launch_blocks) for num_endpoints");
    m.def("run", &run, py::arg("launch"), py::arg("queue_capacity"), py::arg("num_messages"),
          py::arg("num_endpoints"), py::arg("num_cpu_threads"), py::arg("poll_mode"),
          py::arg("verbose"), py::arg("title"));
}
