// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Example C++ side shared by the Python kernel frontend examples.
//
// Owns a kiwi::invoke::Context with one callback that prints the arguments a
// kernel invokes it with; nothing here depends on the kernel frontend. Python
// gets the device_context() pointer to pass to the kernel, and calls
// progress() to run the callbacks.

#include <pybind11/pybind11.h>

#include <kiwi/invoke/invoke.hpp>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <functional>

namespace py = pybind11;

namespace {

class PrintContext {
public:
    explicit PrintContext(size_t queue_capacity) : ctx_(1, queue_capacity) {
        std::function<void(int32_t, float, uint64_t)> print =
            [](int32_t step, float value, uint64_t tag) {
                std::printf("[C++ callback] step=%" PRId32 " value=%.2f tag=0x%" PRIx64 "\n",
                            step, value, tag);
                std::fflush(stdout);
            };
        callback_id_ = ctx_.register_callback(print);
    }

    size_t progress(int max_callbacks) {
        py::gil_scoped_release release;
        return ctx_.progress(0, max_callbacks);
    }

    uint8_t callback_id() const { return callback_id_; }

    uintptr_t device_context() const {
        return reinterpret_cast<uintptr_t>(ctx_.device_context());
    }

private:
    kiwi::invoke::Context<> ctx_;
    uint8_t callback_id_ = 0;
};

}  // namespace

PYBIND11_MODULE(kiwi_print_ext, m) {
    py::class_<PrintContext>(m, "PrintContext")
        .def(py::init<size_t>(), py::arg("queue_capacity") = 1024)
        .def("progress", &PrintContext::progress, py::arg("max_callbacks") = -1)
        .def_property_readonly("callback_id", &PrintContext::callback_id)
        .def_property_readonly("device_context", &PrintContext::device_context);
}
