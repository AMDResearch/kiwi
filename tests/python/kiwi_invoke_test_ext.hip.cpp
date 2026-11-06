// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Test-only pybind11 module for the Python invoke frontend tests
// (test_flydsl_invoke.py, test_triton_invoke.py).
//
// Owns a kiwi::invoke::Context whose typed C++ callbacks record what the test
// kernels invoke, and exposes the C++ DeviceContext/DeviceHandle layout so the
// tests can check the constants the frontends hard-code.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <kiwi/invoke/invoke.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace {

using Queue = kiwi::queue::SPSCD2HQueue;
using InvokeContext = kiwi::invoke::Context<Queue>;
using DeviceContext = kiwi::invoke::DeviceContext<Queue>;
using Handle = DeviceContext::DeviceHandle;
using QueueHandle = Queue::DeviceHandle;

static_assert(std::is_standard_layout_v<Handle>);
static_assert(std::is_standard_layout_v<QueueHandle>);
static_assert(sizeof(DeviceContext) == 2 * sizeof(uint64_t));
static_assert(sizeof(QueueHandle::log2_capacity) == sizeof(uint32_t));

constexpr size_t kQueueOffset = offsetof(Handle, queue);

// DeviceContext's members are private, so locate them by value.
std::pair<size_t, size_t> device_context_offsets() {
    constexpr uint64_t kHandles = 0x1234500000ull;
    constexpr uint64_t kNumEndpoints = 7;
    DeviceContext probe(reinterpret_cast<Handle*>(kHandles), kNumEndpoints);
    uint64_t words[2];
    std::memcpy(words, &probe, sizeof(words));
    size_t handles = words[0] == kHandles ? 0 : 8;
    size_t num_endpoints = words[0] == kNumEndpoints ? 0 : 8;
    return {handles, num_endpoints};
}

using Records = std::vector<std::string>;
using TypedRecord = std::tuple<int32_t, float, uint64_t>;

class RecordingContext {
public:
    RecordingContext(size_t num_endpoints, size_t queue_capacity)
        : ctx_(std::make_unique<InvokeContext>(num_endpoints, queue_capacity)) {}

    // Registers a typed callback taking std::array<uint8_t, arg_bytes>, which
    // receives the argument bytes exactly as serialized.
    uint8_t add_bytes_recorder(size_t arg_bytes) {
        auto records = std::make_shared<Records>();
        uint8_t id;
        switch (arg_bytes) {
            case 0: {
                std::function<void()> fn = [records] { records->emplace_back(); };
                id = ctx_->register_callback(fn);
                break;
            }
            case 4: id = add_array_recorder<4>(records); break;
            case 8: id = add_array_recorder<8>(records); break;
            case 9: id = add_array_recorder<9>(records); break;
            case 24: id = add_array_recorder<24>(records); break;
            case 32: id = add_array_recorder<32>(records); break;
            case 33: id = add_array_recorder<33>(records); break;
            case 512: id = add_array_recorder<512>(records); break;
            default: throw std::invalid_argument("unsupported recorder size");
        }
        bytes_.resize(static_cast<size_t>(id) + 1);
        bytes_[id] = records;
        return id;
    }

    uint8_t add_typed_recorder() {
        auto records = typed_;
        std::function<void(int32_t, float, uint64_t)> fn =
            [records](int32_t i, float x, uint64_t u) { records->emplace_back(i, x, u); };
        return ctx_->register_callback(fn);
    }

    std::vector<py::bytes> records(uint8_t id) const {
        std::vector<py::bytes> out;
        if (id < bytes_.size() && bytes_[id]) {
            for (const auto& r : *bytes_[id]) out.emplace_back(r);
        }
        return out;
    }

    std::vector<TypedRecord> typed_records() const { return *typed_; }

    size_t progress(size_t endpoint_id, int max_callbacks) {
        py::gil_scoped_release release;
        return ctx_->progress(endpoint_id, max_callbacks);
    }

    uintptr_t device_context() const {
        return reinterpret_cast<uintptr_t>(ctx_->device_context());
    }

private:
    template<size_t N>
    uint8_t add_array_recorder(const std::shared_ptr<Records>& records) {
        std::function<void(std::array<uint8_t, N>)> fn = [records](std::array<uint8_t, N> a) {
            records->emplace_back(reinterpret_cast<const char*>(a.data()), N);
        };
        return ctx_->register_callback(fn);
    }

    std::unique_ptr<InvokeContext> ctx_;
    std::vector<std::shared_ptr<Records>> bytes_;
    std::shared_ptr<std::vector<TypedRecord>> typed_ =
        std::make_shared<std::vector<TypedRecord>>();
};

}  // namespace

PYBIND11_MODULE(kiwi_invoke_test_ext, m) {
    auto [handles_offset, num_endpoints_offset] = device_context_offsets();
    m.attr("CONTEXT_HANDLES_OFFSET") = handles_offset;
    m.attr("CONTEXT_NUM_ENDPOINTS_OFFSET") = num_endpoints_offset;
    m.attr("HANDLE_SIZE") = sizeof(Handle);
    m.attr("CHUNKS_OFFSET") = kQueueOffset + offsetof(QueueHandle, chunks);
    m.attr("CONSUMER_INDEX_OFFSET") = kQueueOffset + offsetof(QueueHandle, consumer_index);
    m.attr("PRODUCER_INDEX_OFFSET") = kQueueOffset + offsetof(QueueHandle, producer_index);
    m.attr("CONSUMER_INDEX_CACHED_OFFSET") =
        kQueueOffset + offsetof(QueueHandle, consumer_index_cached);
    m.attr("LOG2_CAPACITY_OFFSET") = kQueueOffset + offsetof(QueueHandle, log2_capacity);
    m.attr("CHUNK_BYTES") = sizeof(kiwi::queue::detail::Chunk);
    m.attr("CHUNK_PAYLOAD_BYTES") = kiwi::queue::detail::kChunkPayloadBytes;
    m.attr("CHUNKS_PER_LINE") = kiwi::queue::detail::kChunksPerLine;
    m.attr("MAX_ARG_BYTES") = kiwi::queue::kMaxMessageBytes;
    m.attr("MAX_CAPACITY_LINES") = kiwi::queue::kMaxCapacityLines;

    py::class_<RecordingContext>(m, "RecordingContext")
        .def(py::init<size_t, size_t>(), py::arg("num_endpoints") = 1,
             py::arg("queue_capacity") = 1024)
        .def("add_bytes_recorder", &RecordingContext::add_bytes_recorder, py::arg("arg_bytes"))
        .def("add_typed_recorder", &RecordingContext::add_typed_recorder)
        .def("records", &RecordingContext::records, py::arg("callback_id"))
        .def("typed_records", &RecordingContext::typed_records)
        .def("progress", &RecordingContext::progress,
             py::arg("endpoint_id") = 0, py::arg("max_callbacks") = -1)
        .def_property_readonly("device_context", &RecordingContext::device_context);
}
