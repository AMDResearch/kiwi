// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Python entry point to the cross-frontend KIWI single-PUT readiness harness.
//
// run() runs the harness's host code (kiwi_readiness_bench.hip.hpp) with a
// Python launcher in place of the HIP kernel launch, so a protocol kernel
// written in a Python GPU language is measured by the same host code as the
// C++ frontend (cpp/). Use it through kiwi_readiness_bench.py.

#include <pybind11/pybind11.h>

#include <cstdio>
#include <string>

#include "kiwi_readiness_bench.hip.hpp"
#include "py_launcher.hpp"

namespace py = pybind11;
using kiwi_bench::py_launch::addr;

namespace {

// launch(ctx, rank, peer, source, target, source_heap, target_heap, cb_id,
//        iters, depth); ctx is the invoke DeviceContext address of the P2P
// context, through which the kernel reaches the readiness callback. The
// benchmark is collective (lci::barrier), so a launcher error exits the
// process and lets the launcher of the job abort the peer.
struct PyReadinessLauncher {
  py::object fn;

  void launch(const kiwi_readiness_bench::ReadinessLaunch &l) {
    kiwi_bench::py_launch::call_or_exit(
        fn, addr(l.gctx.invoke_device_context()), l.rank, l.peer, addr(l.source),
        addr(l.target), l.source_heap, l.target_heap, l.cb_id, l.iters, l.depth);
  }
};

int run(const py::object &launch, int warmup, int iters, int depth,
        int main_core, int proxy_core, const std::string &backend,
        int prime_iters) {
  kiwi_readiness_bench::config_t cfg{warmup, iters, depth, main_core,
                                     proxy_core};
  PyReadinessLauncher launcher{launch};
  py::gil_scoped_release release;
  int rc = kiwi_readiness_bench::run_benchmark(cfg, launcher, backend.c_str(),
                                               prime_iters);
  std::fflush(stdout);
  return rc;
}

} // namespace

PYBIND11_MODULE(kiwi_readiness_bench_ext, m) {
  const kiwi_readiness_bench::config_t d;
  m.attr("DEFAULTS") = py::dict(py::arg("warmup") = d.warmup,
                                py::arg("iters") = d.iters,
                                py::arg("depth") = d.depth,
                                py::arg("main_core") = d.main_core,
                                py::arg("proxy_core") = d.proxy_core);
  m.def("run", &run, py::arg("launch"), py::arg("warmup"), py::arg("iters"),
        py::arg("depth"), py::arg("main_core"), py::arg("proxy_core"),
        py::arg("backend"), py::arg("prime_iters"));
}
