// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Python entry point to the cross-frontend kiwi-spsc queue-baseline harness.
//
// run() runs the harness's host code (qb_harness.hpp) for the KIWI SPSC
// device-to-host queue with a Python launcher in place of the HIP kernels, so
// a producer written in a Python GPU language is measured by the same proxy
// threads, ack path, timers, validation and output as the C++ frontend
// (cpp/). Use it through kiwi_queue_bench.py.

#include <pybind11/pybind11.h>

#include "qb_harness.hpp"
#include "adapters/kiwi.hpp"
#include <py_launcher.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

namespace py = pybind11;
using kiwi_bench::py_launch::addr;

namespace {

using Adapter = qb::KiwiSpscAdapter;

// Calls the launcher's methods. The proxy threads wait for every message, so
// the host code cannot unwind past a failed launch: a launcher error exits
// (options are validated before the harness starts).
struct PyLauncher {
    py::object obj;

    template <typename... Args>
    void call(const char* method, Args&&... args) {
        py::gil_scoped_acquire gil;  // for the attribute lookup
        const py::object fn = obj.attr(method);
        kiwi_bench::py_launch::call_or_exit(fn, std::forward<Args>(args)...);
    }

    // A Python kernel's residency cannot be queried here; it is not checked.
    bool fits(const qb::Options&, int) { return true; }

    // latency(chunks, consumer_index, producer_index, consumer_index_cached,
    //         log2_capacity, ack, iters, latency_ticks, stream): the handle is
    // passed by value, as the C++ kernel argument.
    void latency(const qb::LatencyLaunch<Adapter>& l) {
        const auto& h = l.handle;
        call("latency", reinterpret_cast<uintptr_t>(h.chunks),
             reinterpret_cast<uintptr_t>(h.consumer_index), h.producer_index,
             h.consumer_index_cached, uint64_t{h.log2_capacity}, addr(l.ack), l.iters,
             addr(l.latency_ticks), addr(l.stream));
    }

    // push_cost(handles, blocks, waves, waves_per_block, warmup, iters, burst,
    //           wall, cycles, calibration)
    void push_cost(const qb::PushCostLaunch<Adapter>& l) {
        call("push_cost", addr(l.handles), l.blocks, l.waves, l.waves_per_block, l.warmup,
             l.iters, l.burst, l.wall, addr(l.cycles), addr(l.calibration));
    }

    // throughput(handles, blocks, waves, waves_per_block, msgs_per_wave)
    void throughput(const qb::ThroughputLaunch<Adapter>& l) {
        call("throughput", addr(l.handles), l.blocks, l.waves, l.waves_per_block,
             l.msgs_per_wave);
    }
};

int run(const py::object& launcher, const std::string& name, const std::string& mode,
        int capacity, int queues, int waves, int waves_per_block, int threads, int batch,
        int idle_pause, bool burst, const std::string& timer, const std::string& ack_mem,
        long iters, long warmup, long idle_rounds) {
    qb::Options o;
    o.queue = Adapter::kName;
    o.name = name;
    o.mode = mode;
    o.capacity = capacity;
    o.queues = queues;
    o.waves = waves;
    o.waves_per_block = waves_per_block;
    o.threads = threads;
    o.batch = batch;
    o.idle_pause = idle_pause;
    o.burst = burst;
    o.timer = timer;
    o.ack_mem = ack_mem;
    o.iters = iters;
    o.warmup = warmup;
    o.idle_rounds = idle_rounds;
    PyLauncher l{launcher};

    py::gil_scoped_release release;
    // As the C++ frontend's main().
    check_numa_affinity();
    gpu_clock_warmup();
    int rc = 2;
    if (mode == "latency") rc = qb::run_latency<Adapter>(o, l);
    else if (mode == "push-cost") rc = qb::run_push_cost<Adapter>(o, l);
    else if (mode == "throughput") rc = qb::run_throughput<Adapter>(o, l);
    else if (mode == "idle-poll") rc = qb::run_idle_poll<Adapter>(o);
    else std::fprintf(stderr, "ERROR: unknown mode %s\n", mode.c_str());
    std::fflush(stdout);
    return rc;
}

}  // namespace

PYBIND11_MODULE(kiwi_queue_bench_ext, m) {
    m.def("run", &run, py::arg("launcher"), py::arg("name"), py::arg("mode"),
          py::arg("capacity"), py::arg("queues"), py::arg("waves"), py::arg("waves_per_block"),
          py::arg("threads"), py::arg("batch"), py::arg("idle_pause"), py::arg("burst"),
          py::arg("timer"), py::arg("ack_mem"), py::arg("iters"), py::arg("warmup"),
          py::arg("idle_rounds"));
}
