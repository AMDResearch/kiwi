// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// pybind11 bindings for kiwi-ep, exposing a Python interface compatible with
// MORI's EpDispatchCombineOp so that MORI's bench_dispatch_combine.py can
// drive kiwi-ep for apples-to-apples comparison.
//
// Build: hipcc + pybind11 + torch + lci + kiwi_ep
//
// Usage from Python:
//   import kiwi_ep_ext
//   ctx = kiwi_ep_ext.LciContext()      # Initialize LCI runtime (requires mpirun)
//   rank = kiwi_ep_ext.get_rank()
//   nranks = kiwi_ep_ext.get_nranks()
//   op = kiwi_ep_ext.KiwiEpOp(device_id, nranks, hidden_dim, topk,
//                              num_experts_per_rank, max_num_inp_token_per_rank)
//   (out, out_w, out_s, out_idx, recv_count) = op.dispatch(input, weights, scales, indices)
//   (combine_out, _) = op.combine(expert_output, weights, indices, out=None)
//   op.reset()
//   del op
//   del ctx  # finalize LCI

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

#include <hip/hip_runtime.h>
#include <memory>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cctype>

#include "ep_context.hip.hpp"
#include "hip_check.hpp"
#include "topology.hpp"
#include "lci.hpp"

namespace py = pybind11;

// ============================================================================
// LciContext: RAII wrapper for LCI runtime initialization.
//
// Uses the EP-owned comm runtime guard, which handles the reference-counted
// lci::g_runtime_init/fina lifecycle without creating a communication device.
// ============================================================================

class LciContext {
public:
    explicit LciContext(const std::string& device_name = {})
        : guard_(device_name) {}
    ~LciContext() = default;

    LciContext(const LciContext&) = delete;
    LciContext& operator=(const LciContext&) = delete;

private:
    // ep_runtime_guard_t initializes LCI on construction and finalizes when the
    // last EP comm/runtime context is destroyed.
    kiwi::ep::ep_runtime_guard_t guard_;
};

// ============================================================================
// KiwiEpOp: Python-visible operator wrapping kiwi-ep dispatch/combine
// ============================================================================

static int dtype_to_elem_size(torch::Dtype dtype) {
    return static_cast<int>(c10::elementSize(dtype));
}

static kiwi::ep::quant_format_t quant_format_from_name(
        const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (lower == "none") {
        return kiwi::ep::quant_format_t::NONE;
    }
    if (lower == "fp8") {
        return kiwi::ep::quant_format_t::FP8;
    }
    if (lower == "fp4") {
        return kiwi::ep::quant_format_t::FP4;
    }
    throw std::runtime_error(
        "Unsupported kiwi-ep quant_format '" + name
        + "' (expected none, fp8, or fp4)");
}

static kiwi::ep::rcomp_backend_t rcomp_backend_from_name(
        const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (lower == "rdma-only" || lower == "rdma_only" || lower == "rdma") {
        return kiwi::ep::rcomp_backend_t::RDMA_ONLY;
    }
    if (lower == "hybrid") {
        return kiwi::ep::rcomp_backend_t::HYBRID;
    }
    if (lower == "auto") {
        return kiwi::ep::rcomp_backend_t::AUTO;
    }
    throw std::runtime_error(
        "Unsupported kiwi-ep rcomp_backend '" + name
        + "' (expected auto, rdma-only, or hybrid)");
}

static torch::Tensor tensor_from_ptr(void* ptr, std::vector<int64_t> shape,
                                     torch::Dtype dtype, int device_id) {
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, device_id);
    return torch::from_blob(ptr, shape, options);
}

class KiwiEpOp {
public:
    KiwiEpOp(int device_id, int world_size, int hidden_dim, int topk,
             int num_experts_per_rank, int max_num_inp_token_per_rank,
             torch::Dtype dispatch_dtype,
             torch::Dtype combine_dtype,
             int num_blocks = 4, int num_proxy_threads = 4,
             int dispatch_elem_size = -1,
             int dispatch_scale_dim = 0, int dispatch_scale_type_size = 0,
             int combine_scale_dim = 0, int combine_scale_type_size = 0,
             torch::Dtype dispatch_scale_dtype = torch::kFloat32,
             const std::string& device_name = {},
             const std::string& dispatch_quant_format = "none",
             const std::string& combine_quant_format = "none",
             int queue_capacity = 2048,
             const std::string& rcomp_backend = "auto")
        : device_id_(device_id), dispatch_dtype_(dispatch_dtype),
          dispatch_scale_dtype_(dispatch_scale_dtype)
    {
        // Set GPU device
        HIP_CHECK_THROW(hipSetDevice(device_id_));

        int actual_dispatch_elem_size =
            (dispatch_elem_size > 0) ? dispatch_elem_size : dtype_to_elem_size(dispatch_dtype);
        int actual_combine_elem_size = dtype_to_elem_size(combine_dtype);
        kiwi::ep::quant_format_t dispatch_quant_format_value =
            quant_format_from_name(dispatch_quant_format);
        kiwi::ep::quant_format_t combine_quant_format_value =
            quant_format_from_name(combine_quant_format);
        kiwi::ep::rcomp_backend_t rcomp_backend_value =
            rcomp_backend_from_name(rcomp_backend);

        if (dispatch_quant_format_value != kiwi::ep::quant_format_t::NONE
            && actual_dispatch_elem_size != 1) {
            throw std::runtime_error(
                "KiwiEpOp dispatch_quant_format fp8/fp4 requires "
                "dispatch_elem_size == 1");
        }
        if (combine_quant_format_value != kiwi::ep::quant_format_t::NONE
            && actual_combine_elem_size != 1) {
            throw std::runtime_error(
                "KiwiEpOp combine_quant_format fp8/fp4 requires "
                "a one-byte combine_dtype");
        }
        cfg_.world_size = world_size;
        cfg_.num_experts = num_experts_per_rank * world_size;
        // MORI's FP4 public config uses the logical hidden size while the
        // dispatch tensor stores two FP4 values per byte.  Keep dispatch
        // storage packed, but do not let that packed size constrain combine:
        // downstream combine inputs may be BF16 with the logical hidden size.
        cfg_.model_hidden_dim = hidden_dim;
        if (dispatch_quant_format_value == kiwi::ep::quant_format_t::FP4) {
            if ((hidden_dim & 1) != 0) {
                throw std::runtime_error(
                    "FP4 dispatch requires an even logical hidden_dim");
            }
            cfg_.dispatch_hidden_elems = hidden_dim / 2;
        } else {
            cfg_.dispatch_hidden_elems = hidden_dim;
        }
        if (combine_quant_format_value == kiwi::ep::quant_format_t::FP4) {
            if ((hidden_dim & 1) != 0) {
                throw std::runtime_error(
                    "FP4 combine requires an even logical hidden_dim");
            }
            cfg_.combine_hidden_elems = hidden_dim / 2;
        } else {
            cfg_.combine_hidden_elems = hidden_dim;
        }
        cfg_.topk = topk;
        cfg_.max_num_inp_token_per_rank = max_num_inp_token_per_rank;
        cfg_.num_blocks = num_blocks;
        cfg_.num_proxy_threads = num_proxy_threads;
        // num_devices is derived from num_proxy_threads inside ep_context_t
        // (one LCI device per proxy thread); not set here.
        cfg_.queue_capacity = queue_capacity;
        cfg_.device_name = device_name;
        cfg_.dispatch_elem_size = actual_dispatch_elem_size;
        cfg_.combine_elem_size = actual_combine_elem_size;
        cfg_.dispatch_quant_format = dispatch_quant_format_value;
        cfg_.combine_quant_format = combine_quant_format_value;
        cfg_.dispatch_scale_dim = dispatch_scale_dim;
        cfg_.dispatch_scale_type_size = dispatch_scale_type_size;
        cfg_.combine_scale_dim = combine_scale_dim;
        cfg_.combine_scale_type_size = combine_scale_type_size;
        cfg_.rcomp_backend = rcomp_backend_value;

        ctx_ = std::make_unique<kiwi::ep::ep_context_t>(cfg_);
        ctx_->spawn_proxy_threads(cfg_.num_proxy_threads);
        gctx_ = ctx_->get_device_context();

        HIP_CHECK_THROW(
            hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));

        dispatch_output_ = tensor_from_ptr(
            gctx_.d_recv_hidden,
            {cfg_.max_slots(), cfg_.dispatch_hidden_elems_effective()},
            dispatch_dtype_, device_id_);
        dispatch_weight_output_ = tensor_from_ptr(
            gctx_.d_recv_weights, {cfg_.max_slots(), cfg_.topk},
            torch::kFloat32, device_id_);
        dispatch_index_output_ = tensor_from_ptr(
            gctx_.d_recv_indices, {cfg_.max_slots(), cfg_.topk},
            torch::kInt32, device_id_);
        dispatch_recv_count_ = tensor_from_ptr(
            gctx_.device_workspace.recv_count(), {1}, torch::kInt32,
            device_id_);
        if (gctx_.d_recv_scales && cfg_.dispatch_scale_dim > 0) {
            dispatch_scale_output_ = tensor_from_ptr(
                gctx_.d_recv_scales,
                {cfg_.max_slots(), cfg_.dispatch_scale_dim},
                dispatch_scale_dtype_, device_id_);
        }
        combine_output_ = torch::empty(
            {cfg_.max_num_inp_token_per_rank,
             cfg_.combine_hidden_elems_effective()},
            torch::TensorOptions().dtype(combine_dtype)
                                  .device(torch::kCUDA, device_id_));
        combine_weight_output_ = torch::empty(
            {cfg_.max_num_inp_token_per_rank, cfg_.topk},
            torch::TensorOptions().dtype(torch::kFloat32)
                                  .device(torch::kCUDA, device_id_));

        ctx_->reset(stream_);
        ctx_->stream_sync(stream_);

        num_input_tokens_ = 0;
    }

    ~KiwiEpOp() {
        if (ctx_) {
            ctx_->join_proxy_threads();
        }
        if (stream_) {
            (void)hipStreamDestroy(stream_);
        }
    }

    // dispatch(input, weights, scales, indices)
    // Returns (output, out_weights, out_scales, out_indices, recv_count)
    py::tuple dispatch(torch::Tensor input, torch::Tensor weights,
                       py::object scales_obj, torch::Tensor indices) {
        hipStream_t stream = current_torch_stream();
        int num_tokens = input.size(0);
        num_input_tokens_ = num_tokens;

        void* scales_ptr = nullptr;
        if (!scales_obj.is_none()) {
            torch::Tensor scales = scales_obj.cast<torch::Tensor>();
            if (scales.defined() && scales.numel() > 0) {
                scales_ptr = scales.data_ptr();
            }
        }

        kiwi::ep::dispatch_enqueue(*ctx_, gctx_,
                                  input.data_ptr(),
                                  indices.data_ptr<int32_t>(),
                                  weights.data_ptr<float>(),
                                  scales_ptr,
                                  num_tokens, stream);

        // Match MORI's hot-path lifetime contract: recv_count aliases the
        // operator-owned count buffer in eager mode and graph mode.  Callers
        // that need a stable historical value must clone/snapshot it before a
        // later dispatch/reset reuses the internal buffer.
        py::object out_scales = py::none();
        if (dispatch_scale_output_.defined()) {
            out_scales = py::cast(dispatch_scale_output_);
        }

        return py::make_tuple(
            dispatch_output_, dispatch_weight_output_, out_scales,
            dispatch_index_output_, dispatch_recv_count_);
    }

    py::tuple dispatch_send(torch::Tensor input, torch::Tensor weights,
                            py::object scales, torch::Tensor indices) {
        return dispatch(input, weights, scales, indices);
    }

    void dispatch_recv() {
        // dispatch_send currently performs the full send+receive path. Keep a
        // no-op split receive entry point for MORI AsyncLL call-shape parity.
    }

    // combine(input, weights, indices, out=None)
    // Returns (output, out_weights_or_None)
    py::tuple combine(torch::Tensor input, py::object weights_obj,
                      torch::Tensor indices, py::object out_obj = py::none()) {
        hipStream_t stream = current_torch_stream();
        int num_tokens = num_input_tokens_;
        torch::Tensor output = combine_output_;
        if (!out_obj.is_none()) {
            output = out_obj.cast<torch::Tensor>();
            validate_combine_output(output, num_tokens);
        }

        const float* weights_ptr = nullptr;
        if (!weights_obj.is_none()) {
            torch::Tensor weights = weights_obj.cast<torch::Tensor>();
            weights_ptr = weights.data_ptr<float>();
        }

        kiwi::ep::combine_enqueue(*ctx_, gctx_,
                               input.data_ptr(),
                               weights_ptr,
                               nullptr,
                               indices.data_ptr<int32_t>(),
                               output.data_ptr(),
                               weights_ptr
                                   ? combine_weight_output_.data_ptr<float>()
                                   : nullptr,
                               num_tokens, stream);
        last_combine_enqueued_ = true;
        last_combine_has_weights_ = weights_ptr != nullptr;
        // Preserve a caller-owned destination until a later combine replaces
        // it. This keeps the captured pointer alive for the graph replay path.
        last_combine_output_ = output;

        py::object out_weights = py::none();
        if (weights_ptr) {
            out_weights = py::cast(combine_weight_output_);
        }
        return py::make_tuple(output, out_weights);
    }

    py::tuple combine_send(torch::Tensor input, py::object weights,
                           torch::Tensor indices,
                           py::object out_obj = py::none()) {
        return combine(input, weights, indices, out_obj);
    }

    py::tuple combine_recv() {
        if (!last_combine_enqueued_) {
            throw std::runtime_error(
                "combine_recv requires a prior combine/combine_send call");
        }
        py::object out_weights = py::none();
        if (last_combine_has_weights_) {
            out_weights = py::cast(combine_weight_output_);
        }
        return py::make_tuple(last_combine_output_, out_weights);
    }

    void reset() {
        hipStream_t stream = current_torch_stream();
        if (stream_is_capturing(stream)) {
            throw std::runtime_error(
                "KiwiEpOp.reset() is not supported during graph capture; "
                "dispatch/combine enqueue graph-safe per-round reset work");
        }
        ctx_->reset(stream);
        ctx_->stream_sync(stream);
        num_input_tokens_ = 0;
        last_combine_enqueued_ = false;
        last_combine_has_weights_ = false;
    }

    // Mirror MORI's public capacity accessors.  These describe the
    // source/receive-token address space, not dispatch's internal multicast
    // staging slots: each rank owns max_num_inp_token_per_rank source tokens,
    // so the flat cross-rank capacity is world_size times that value.
    int max_num_tokens_to_recv() const { return cfg_.max_recv_slots_total(); }
    int max_num_tokens_to_recv_per_rank() const { return cfg_.max_recv_slots_per_rank(); }
    int max_num_tokens_to_send() const {
        return cfg_.world_size * cfg_.max_num_inp_token_per_rank;
    }
    int max_num_tokens_to_send_per_rank() const { return cfg_.max_num_inp_token_per_rank; }

    // Inverse of dispatch's d_recv_src_token_pos encoding:
    // src_rank * (world_size * max_num_inp_token_per_rank) + src_token_id.
    py::tuple decode_send_flat_idx(int flat_idx) const {
        int stride = max_num_tokens_to_send();
        return py::make_tuple(flat_idx / stride, flat_idx % stride);
    }

    torch::Tensor get_registered_combine_input_buffer(torch::Dtype combine_dtype,
                                                      int hidden_dim = -1) const {
        // Do not expose arbitrary runtime/capacity shapes here.  The
        // registered combine-input buffer is shaped by the configured combine
        // row hidden elements.  If callers ask for a hidden dimension explicitly, it
        // must be that configured value.
        int requested_combine_hidden_elems =
            (hidden_dim > 0) ? hidden_dim : cfg_.combine_hidden_elems_effective();
        if (requested_combine_hidden_elems != cfg_.combine_hidden_elems_effective()) {
            throw std::runtime_error(
                "get_registered_combine_input_buffer hidden_dim must match "
                "configured combine_hidden_elems");
        }
        if (!registered_combine_public_buffer_.defined() ||
            registered_combine_public_buffer_.size(0) != cfg_.max_slots() ||
            registered_combine_public_buffer_.size(1) != requested_combine_hidden_elems ||
            registered_combine_public_buffer_.dtype() != combine_dtype) {
            registered_combine_public_buffer_ = torch::empty(
                {cfg_.max_slots(), requested_combine_hidden_elems},
                torch::TensorOptions().dtype(combine_dtype)
                                      .device(torch::kCUDA, device_id_));
        }
        return registered_combine_public_buffer_;
    }

    torch::Tensor get_dispatch_src_token_pos() const {
        int32_t total_recv = ctx_->copy_recv_count_to_host(current_torch_stream());
        return tensor_from_ptr(gctx_.d_recv_src_token_pos,
                               {total_recv},
                               torch::kInt32, device_id_);
    }

    void set_debug_timing_enabled(bool enabled) {
        ctx_->set_debug_timing_enabled(enabled);
    }

    py::dict get_debug_timing_snapshot() const {
        py::dict snapshot;
        snapshot["enabled"] = ctx_ && ctx_->debug_timing_enabled();

        std::vector<uint64_t> times(kiwi::ep::DEBUG_TIME_WORDS, 0);
        std::vector<int32_t> meta(kiwi::ep::DEBUG_META_WORDS, 0);
        if (ctx_) {
            const uint64_t* src_times = ctx_->debug_time_buf();
            if (src_times != nullptr) {
                std::memcpy(times.data(), src_times,
                            times.size() * sizeof(uint64_t));
            }
            const int32_t* src_meta = ctx_->debug_time_meta();
            if (src_meta != nullptr) {
                std::memcpy(meta.data(), src_meta,
                            meta.size() * sizeof(int32_t));
            }
        }

        int wall_clock_rate_khz = 0;
        int device = device_id_;
        if (hipGetDevice(&device) == hipSuccess) {
            (void)hipDeviceGetAttribute(&wall_clock_rate_khz,
                                        hipDeviceAttributeWallClockRate,
                                        device);
        }

        snapshot["time_ticks"] = py::cast(std::move(times));
        snapshot["meta"] = py::cast(std::move(meta));
        snapshot["wall_clock_rate_khz"] = wall_clock_rate_khz;
        return snapshot;
    }

private:
    void validate_combine_output(const torch::Tensor& output,
                                 int num_tokens) const {
        if (!output.defined()) {
            throw std::runtime_error("combine out must be a defined tensor");
        }
        if (!output.is_cuda()) {
            throw std::runtime_error("combine out must be a CUDA tensor");
        }
        if (output.get_device() != device_id_) {
            throw std::runtime_error(
                "combine out must be on the KiwiEpOp device");
        }
        if (output.dtype() != combine_output_.dtype()) {
            throw std::runtime_error(
                "combine out dtype must match the configured combine dtype");
        }
        if (output.dim() != 2 ||
            output.size(1) != cfg_.combine_hidden_elems_effective()) {
            throw std::runtime_error(
                "combine out must have shape [token_capacity, "
                "configured_combine_hidden_elems]");
        }
        if (output.size(0) < num_tokens) {
            throw std::runtime_error(
                "combine out token capacity is smaller than the current "
                "dispatch token count");
        }
        if (!output.is_contiguous()) {
            throw std::runtime_error("combine out must be contiguous");
        }
    }

    hipStream_t current_torch_stream() const {
        return c10::hip::getCurrentHIPStream(device_id_).stream();
    }

    static bool stream_is_capturing(hipStream_t stream) {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        HIP_CHECK_THROW(hipStreamIsCapturing(stream, &status));
        return status != hipStreamCaptureStatusNone;
    }

    int device_id_;
    torch::Dtype dispatch_dtype_;
    torch::Dtype dispatch_scale_dtype_;
    kiwi::ep::config_t cfg_;
    std::unique_ptr<kiwi::ep::ep_context_t> ctx_;
    kiwi::ep::ep_gcontext_t gctx_;
    hipStream_t stream_ = nullptr;
    int num_input_tokens_ = 0;
    torch::Tensor dispatch_output_;
    torch::Tensor dispatch_weight_output_;
    torch::Tensor dispatch_scale_output_;
    torch::Tensor dispatch_index_output_;
    torch::Tensor dispatch_recv_count_;
    torch::Tensor combine_output_;
    torch::Tensor combine_weight_output_;
    torch::Tensor last_combine_output_;
    bool last_combine_enqueued_ = false;
    bool last_combine_has_weights_ = false;
    mutable torch::Tensor registered_combine_public_buffer_;
};

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(kiwi_ep_ext, m) {
    m.doc() = "kiwi-ep Python bindings for MORI EP benchmark integration";

    // LCI lifecycle
    py::class_<LciContext>(m, "LciContext")
        .def(py::init<const std::string&>(),
             py::arg("device_name") = "",
             "Initialize LCI runtime, optionally binding to an HCA/NIC. Must be "
             "called before creating KiwiEpOp and requires mpirun.");

    // LCI utilities (require LciContext to be alive)
    m.def("barrier", []() { lci::barrier(); },
          "Global barrier across all LCI ranks.");
    m.def("get_rank", []() { return lci::get_rank_me(); },
          "Get this process's rank.");
    m.def("get_nranks", []() { return lci::get_rank_n(); },
          "Get total number of ranks.");
    m.def("get_hca_name_for_gpu",
          [](int gpu_id) { return kiwi::p2p::topology::get_hca_name_for_gpu(gpu_id); },
          py::arg("gpu_id"),
          "Return the auto-selected active HCA/NIC name for a HIP GPU id, or an empty string.");
    m.def("get_hca_name_for_current_gpu",
          []() { return kiwi::p2p::topology::get_hca_name_for_current_gpu(); },
          "Return the auto-selected active HCA/NIC name for the current HIP GPU, or an empty string.");
    m.def("recommended_num_blocks",
          [](int num_tokens, int world_size) {
              return kiwi::ep::recommended_num_blocks(num_tokens, world_size);
          },
          py::arg("num_tokens"), py::arg("world_size"),
          "Return the topology-aware EP block-count recommendation. "
          "Unmeasured world sizes use the legacy default.");
    m.def("recommended_runtime_defaults",
          [](int num_tokens, int world_size) {
              const auto defaults =
                  kiwi::ep::recommended_runtime_defaults(
                      num_tokens, world_size);
              py::dict result;
              result["num_blocks"] = defaults.num_blocks;
              result["rcomp_backend"] =
                  kiwi::ep::rcomp_backend_name(defaults.rcomp_backend);
              return result;
          },
          py::arg("num_tokens"), py::arg("world_size"),
          "Return the centralized EP block-count and rcomp-backend defaults.");

    // KiwiEpOp
    py::class_<KiwiEpOp>(m, "KiwiEpOp")
        .def(py::init([](int device_id, int world_size, int hidden_dim,
                         int topk, int num_experts_per_rank,
                         int max_num_inp_token_per_rank,
                         torch::Dtype dispatch_dtype,
                         torch::Dtype combine_dtype,
                         int num_blocks, int num_proxy_threads,
                         int dispatch_elem_size, int dispatch_scale_dim,
                         int dispatch_scale_type_size,
                         int combine_scale_dim, int combine_scale_type_size,
                         py::object dispatch_scale_dtype,
                         const std::string& device_name,
                         const std::string& dispatch_quant_format,
                         const std::string& combine_quant_format,
                         int queue_capacity,
                         const std::string& rcomp_backend) {
                 // Avoid materializing a torch.dtype Python object while the
                 // extension module is importing.  Some ROCm PyTorch builds do
                 // not have dtype objects fully registered until after the
                 // module import path has completed, so a torch::Dtype default
                 // argument here can fail with "unsupported scalarType" before
                 // users ever construct KiwiEpOp.  Keep the public default
                 // behavior by resolving None to float32 at construction time.
                 torch::Dtype scale_dtype = torch::kFloat32;
                 if (!dispatch_scale_dtype.is_none()) {
                     scale_dtype = dispatch_scale_dtype.cast<torch::Dtype>();
                 }
                 return new KiwiEpOp(
                     device_id, world_size, hidden_dim, topk,
                     num_experts_per_rank, max_num_inp_token_per_rank,
                     dispatch_dtype, combine_dtype, num_blocks,
                     num_proxy_threads, dispatch_elem_size,
                     dispatch_scale_dim, dispatch_scale_type_size,
                     combine_scale_dim, combine_scale_type_size,
                     scale_dtype, device_name, dispatch_quant_format,
                     combine_quant_format, queue_capacity, rcomp_backend);
             }),
             py::arg("device_id"),
             py::arg("world_size"),
             py::arg("hidden_dim"),
             py::arg("topk"),
             py::arg("num_experts_per_rank"),
             py::arg("max_num_inp_token_per_rank"),
             py::arg("dispatch_dtype"),
             py::arg("combine_dtype"),
             py::arg("num_blocks") = 4,
             py::arg("num_proxy_threads") = 4,
             py::arg("dispatch_elem_size") = -1,
             py::arg("dispatch_scale_dim") = 0,
             py::arg("dispatch_scale_type_size") = 0,
             py::arg("combine_scale_dim") = 0,
             py::arg("combine_scale_type_size") = 0,
             py::arg_v("dispatch_scale_dtype", py::none(), "torch.float32"),
             py::arg("device_name") = "",
             py::arg("dispatch_quant_format") = "none",
             py::arg("combine_quant_format") = "none",
             py::arg("queue_capacity") = 2048,
             py::arg("rcomp_backend") = "auto")
        .def("dispatch",
             [](KiwiEpOp& self, torch::Tensor input, torch::Tensor weights,
                py::object scales, torch::Tensor indices, py::kwargs) {
                 return self.dispatch(input, weights, scales, indices);
             },
             py::arg("input"), py::arg("weights"),
             py::arg("scales").none(true), py::arg("indices"),
             "Asynchronously dispatch contiguous tensors to expert ranks. "
             "MORI launch kwargs such as block_num/warp_per_block are accepted "
             "and ignored.")
        .def("dispatch_send",
             [](KiwiEpOp& self, torch::Tensor input, torch::Tensor weights,
                py::object scales, torch::Tensor indices, py::kwargs) {
                 return self.dispatch_send(input, weights, scales, indices);
             },
             py::arg("input"), py::arg("weights"),
             py::arg("scales").none(true), py::arg("indices"),
             "Alias of dispatch() for MORI interface compatibility.")
        .def("dispatch_recv",
             [](KiwiEpOp& self, py::kwargs) { self.dispatch_recv(); },
             "No-op split receive after dispatch_send().")
        .def("combine",
             [](KiwiEpOp& self, torch::Tensor input, py::object weights,
                torch::Tensor indices, py::object out, py::kwargs) {
                 return self.combine(input, weights, indices, out);
             },
             py::arg("input"), py::arg("weights").none(true),
             py::arg("indices"), py::arg("out") = py::none(),
             "Asynchronously combine contiguous expert results into an "
             "operator-owned output buffer or the supplied out tensor. MORI "
             "launch kwargs such as block_num/warp_per_block are accepted "
             "and ignored.")
        .def("combine_send",
             [](KiwiEpOp& self, torch::Tensor input, py::object weights,
                torch::Tensor indices, py::object out, py::kwargs) {
                 return self.combine_send(input, weights, indices, out);
             },
             py::arg("input"), py::arg("weights").none(true),
             py::arg("indices"), py::arg("out") = py::none(),
             "Alias of combine() for MORI interface compatibility.")
        .def("combine_recv",
             [](KiwiEpOp& self, py::kwargs) { return self.combine_recv(); },
             "Return the result tensor from the previous asynchronous "
             "combine/combine_send call.")
        .def("reset", &KiwiEpOp::reset,
             "Reset internal state for next dispatch/combine round.")
        .def("max_num_tokens_to_recv", &KiwiEpOp::max_num_tokens_to_recv)
        .def("max_num_tokens_to_recv_per_rank",
             &KiwiEpOp::max_num_tokens_to_recv_per_rank)
        .def("max_num_tokens_to_send", &KiwiEpOp::max_num_tokens_to_send)
        .def("max_num_tokens_to_send_per_rank",
             &KiwiEpOp::max_num_tokens_to_send_per_rank)
        .def("decode_send_flat_idx", &KiwiEpOp::decode_send_flat_idx,
             py::arg("flat_idx"))
        .def("get_registered_combine_input_buffer",
             &KiwiEpOp::get_registered_combine_input_buffer,
             py::arg("combine_dtype"), py::arg("hidden_dim") = -1)
        .def("get_dispatch_src_token_pos",
             &KiwiEpOp::get_dispatch_src_token_pos)
        .def("set_debug_timing_enabled",
             &KiwiEpOp::set_debug_timing_enabled,
             py::arg("enabled"),
             "Enable or disable low-level combine phase timing diagnostics.")
        .def("get_debug_timing_snapshot",
             &KiwiEpOp::get_debug_timing_snapshot,
             "Return the latest low-level combine timing words and metadata.");
}
