// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

// Calling a Python kernel launcher from a cross-frontend harness's host code.
//
// The Python bridges of the harnesses (common/*_bench_ext) run the harness's
// C++ host code with the GIL released and call the frontend's launcher where
// the C++ frontend launches its kernel.

#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace kiwi_bench::py_launch {

namespace py = pybind11;

/// An address as a Python int.
inline uintptr_t addr(const volatile void* p) { return reinterpret_cast<uintptr_t>(p); }

/// Calls `fn(args...)` with the GIL held. A Python exception propagates as
/// py::error_already_set; use it where the host code unwinds cleanly.
template<typename... Args>
void call(const py::object& fn, Args&&... args) {
    py::gil_scoped_acquire gil;
    fn(std::forward<Args>(args)...);
}

/// Calls `fn(args...)` with the GIL held; on a Python exception, prints it and
/// exits the process. For host code that cannot unwind past the launch: a
/// collective benchmark whose peer would wait forever, or proxy threads that
/// wait for messages a failed launch never sends.
template<typename... Args>
void call_or_exit(const py::object& fn, Args&&... args) {
    py::gil_scoped_acquire gil;
    try {
        fn(std::forward<Args>(args)...);
    } catch (py::error_already_set& e) {
        e.restore();
        PyErr_Print();
        std::fflush(stdout);
        std::_Exit(1);
    }
}

}  // namespace kiwi_bench::py_launch
