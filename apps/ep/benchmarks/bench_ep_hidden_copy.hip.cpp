// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/verbs.h>
#include <CLI/CLI.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <string>
#include <vector>

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(err));                   \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace {

constexpr int kWaveSize = 64;
constexpr int kWavesPerBlock = 4;

struct config_t {
    std::string mode = "all";
    int rows = 334;
    int recv_rows = -1;
    int self_rows = 43;
    int blocks = 4;
    int combine_blocks = 4;
    int waves_per_block = kWavesPerBlock;
    int topk = 8;
    int dispatch_hidden_bytes = 7168;
    int combine_hidden_bytes = 14336;
    int dispatch_slot_stride = 7472;
    int combine_slot_stride = 14416;
    int warmup = 100;
    int iters = 1000;
    int system_fences = 0;
    bool production_system_scope = false;
    bool register_ibv = false;
    bool register_dmabuf = false;
    std::string ib_dev;
};

struct ibv_registration_t {
    ibv_device** dev_list = nullptr;
    ibv_context* context = nullptr;
    ibv_pd* pd = nullptr;
    std::vector<ibv_mr*> mrs;
    std::vector<int> dmabuf_fds;

    ~ibv_registration_t() { reset(); }

    void reset() {
        for (auto* mr : mrs) {
            if (mr != nullptr) ibv_dereg_mr(mr);
        }
        mrs.clear();
        for (int fd : dmabuf_fds) {
            if (fd >= 0) close(fd);
        }
        if (pd != nullptr) ibv_dealloc_pd(pd);
        pd = nullptr;
        if (context != nullptr) ibv_close_device(context);
        context = nullptr;
        if (dev_list != nullptr) ibv_free_device_list(dev_list);
        dev_list = nullptr;
    }

    void open_device(const std::string& requested) {
        int ndev = 0;
        dev_list = ibv_get_device_list(&ndev);
        if (dev_list == nullptr || ndev <= 0) {
            fprintf(stderr, "ibv_get_device_list found no devices\n");
            std::abort();
        }

        ibv_device* selected = nullptr;
        if (!requested.empty()) {
            for (int i = 0; i < ndev; ++i) {
                if (requested == ibv_get_device_name(dev_list[i])) {
                    selected = dev_list[i];
                    break;
                }
            }
            if (selected == nullptr) {
                fprintf(stderr, "requested IB device '%s' not found\n",
                        requested.c_str());
                std::abort();
            }
        } else {
            selected = dev_list[0];
        }

        context = ibv_open_device(selected);
        if (context == nullptr) {
            fprintf(stderr, "ibv_open_device(%s) failed: %s\n",
                    ibv_get_device_name(selected), std::strerror(errno));
            std::abort();
        }
        pd = ibv_alloc_pd(context);
        if (pd == nullptr) {
            fprintf(stderr, "ibv_alloc_pd failed: %s\n", std::strerror(errno));
            std::abort();
        }
        printf("# ibv_device=%s\n", ibv_get_device_name(selected));
    }

    void register_dmabuf_buffer(void* ptr, size_t bytes, const char* name) {
        constexpr int access = IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_WRITE;
        const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        void* aligned_ptr = reinterpret_cast<void*>(
            reinterpret_cast<uintptr_t>(ptr) & ~(page_size - 1));
        const size_t page_delta = reinterpret_cast<uintptr_t>(ptr)
                                - reinterpret_cast<uintptr_t>(aligned_ptr);
        const size_t aligned_bytes = ((bytes + page_delta + page_size - 1) /
                                      page_size) * page_size;
        int dmabuf_fd = -1;
        uint64_t dmabuf_offset = 0;
        hsa_status_t status = hsa_amd_portable_export_dmabuf(
            aligned_ptr, aligned_bytes, &dmabuf_fd, &dmabuf_offset);
        if (status != HSA_STATUS_SUCCESS) {
            const char* msg = nullptr;
            hsa_status_string(status, &msg);
            fprintf(stderr, "hsa_amd_portable_export_dmabuf(%s, %p, %zu) failed: %s\n",
                    name, aligned_ptr, aligned_bytes, msg ? msg : "unknown");
            std::abort();
        }
        dmabuf_fds.push_back(dmabuf_fd);
        ibv_mr* mr = ibv_reg_dmabuf_mr(pd, dmabuf_offset, aligned_bytes,
                                       reinterpret_cast<uint64_t>(aligned_ptr),
                                       dmabuf_fd, access);
        if (mr == nullptr) {
            fprintf(stderr, "ibv_reg_dmabuf_mr(%s, fd=%d, offset=%lu, size=%zu) failed: %s\n",
                    name, dmabuf_fd, static_cast<unsigned long>(dmabuf_offset),
                    aligned_bytes, std::strerror(errno));
            std::abort();
        }
        printf("# ibv_dmabuf_registered %s ptr=%p aligned_ptr=%p bytes=%zu aligned_bytes=%zu fd=%d offset=%lu lkey=0x%x rkey=0x%x\n",
               name, ptr, aligned_ptr, bytes, aligned_bytes, dmabuf_fd,
               static_cast<unsigned long>(dmabuf_offset), mr->lkey, mr->rkey);
        mrs.push_back(mr);
    }

    void register_buffer(void* ptr, size_t bytes, const char* name) {
        constexpr int access = IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_READ |
                               IBV_ACCESS_REMOTE_WRITE;
        ibv_mr* mr = ibv_reg_mr(pd, ptr, bytes, access);
        if (mr == nullptr) {
            fprintf(stderr, "ibv_reg_mr(%s, %p, %zu) failed: %s\n",
                    name, ptr, bytes, std::strerror(errno));
            std::abort();
        }
        printf("# ibv_registered %s ptr=%p bytes=%zu lkey=0x%x rkey=0x%x\n",
               name, ptr, bytes, mr->lkey, mr->rkey);
        mrs.push_back(mr);
    }
};

struct ep_multi_warp_iter_t {
    int warps_per_item = 1;
    size_t dim_per_warp = 0;
    size_t dim_size = 0;

    __host__ __device__ ep_multi_warp_iter_t() = default;
    __host__ __device__ ep_multi_warp_iter_t(int global_warp_num,
                                             int num_items,
                                             size_t dim_size_)
        : dim_size(dim_size_) {
        if (global_warp_num <= 0 || num_items <= 0) {
            warps_per_item = 1;
            dim_per_warp = dim_size;
            return;
        }
        warps_per_item = (global_warp_num + num_items - 1) / num_items;
        if (warps_per_item < 1) warps_per_item = 1;
        dim_per_warp = (dim_size + static_cast<size_t>(warps_per_item) - 1) /
                       static_cast<size_t>(warps_per_item);
    }

    __host__ __device__ int total_parts(int num_items) const {
        return num_items * warps_per_item;
    }

    __host__ __device__ void decode(int i, int& item_id,
                                    int& in_item_part_id,
                                    size_t& dim_offset,
                                    size_t& dim_chunk) const {
        item_id = i / warps_per_item;
        in_item_part_id = i - item_id * warps_per_item;
        dim_offset = static_cast<size_t>(in_item_part_id) * dim_per_warp;
        dim_chunk = (dim_offset < dim_size)
            ? std::min(dim_size - dim_offset, dim_per_warp)
            : size_t{0};
    }
};

using u64_global_ptr_t = __attribute__((address_space(1))) uint64_t*;
using const_u64_global_ptr_t =
    __attribute__((address_space(1))) const uint64_t*;

template <bool NonTemporalStore, typename T>
__device__ inline void ep_store(T* dst, T value) {
    if constexpr (NonTemporalStore) {
        __builtin_nontemporal_store(value, dst);
    } else {
        *dst = value;
    }
}

template <bool NonTemporalStore>
__device__ inline void ep_store(uint4* dst, uint4 value) {
    if constexpr (NonTemporalStore) {
        const uint64_t* value64 = reinterpret_cast<const uint64_t*>(&value);
        auto* dst64 = (u64_global_ptr_t)(dst);
        __builtin_nontemporal_store(value64[0], dst64);
        __builtin_nontemporal_store(value64[1], dst64 + 1);
    } else {
        *dst = value;
    }
}

template <bool NonTemporalLoad, typename T>
__device__ inline T ep_load(const T* src) {
    if constexpr (NonTemporalLoad) {
        return __builtin_nontemporal_load(src);
    } else {
        return *src;
    }
}

template <>
__device__ inline uint4 ep_load<true, uint4>(const uint4* src) {
    auto* src64 = (const_u64_global_ptr_t)(src);
    uint4 out;
    uint64_t* out64 = reinterpret_cast<uint64_t*>(&out);
    out64[0] = __builtin_nontemporal_load(src64);
    out64[1] = __builtin_nontemporal_load(src64 + 1);
    return out;
}

using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));

__device__ inline void system_scope_store(uint4* dst, uint4 value) {
    const unsigned long long addr =
        reinterpret_cast<unsigned long long>(dst);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&value);
    const u32x4_t raw = {words[0], words[1], words[2], words[3]};
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1"
                 :: "v"(addr), "v"(raw) : "memory");
}

__device__ inline uint4 system_scope_load(const uint4* src) {
    const unsigned long long addr =
        reinterpret_cast<unsigned long long>(src);
    u32x4_t raw;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n\t"
                 "s_waitcnt vmcnt(0)"
                 : "=v"(raw) : "v"(addr) : "memory");
    uint4 out;
    uint32_t* words = reinterpret_cast<uint32_t*>(&out);
    words[0] = raw[0];
    words[1] = raw[1];
    words[2] = raw[2];
    words[3] = raw[3];
    return out;
}

template <bool NonTemporalStore = false, bool NonTemporalLoad = false>
__device__ inline void copy_bytes_block(const void* src, void* dst,
                                        int num_bytes,
                                        int thread_idx,
                                        int thread_count) {
    const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
    int copied = 0;

    if (((src_addr | dst_addr) & 0xf) == 0) {
        const int n_vec4 = num_bytes / static_cast<int>(sizeof(uint4));
        const uint4* src4 = reinterpret_cast<const uint4*>(src);
        uint4* dst4 = reinterpret_cast<uint4*>(dst);
        for (int i = thread_idx; i < n_vec4; i += thread_count) {
            ep_store<NonTemporalStore>(&dst4[i], ep_load<NonTemporalLoad>(&src4[i]));
        }
        copied = n_vec4 * static_cast<int>(sizeof(uint4));
    }

    if ((((src_addr + copied) | (dst_addr + copied)) & 0x7) == 0) {
        const int n64 = (num_bytes - copied) / static_cast<int>(sizeof(uint64_t));
        const uint64_t* src64 = reinterpret_cast<const uint64_t*>(
            static_cast<const char*>(src) + copied);
        uint64_t* dst64 = reinterpret_cast<uint64_t*>(
            static_cast<char*>(dst) + copied);
        for (int i = thread_idx; i < n64; i += thread_count) {
            ep_store<NonTemporalStore>(&dst64[i], ep_load<NonTemporalLoad>(&src64[i]));
        }
        copied += n64 * static_cast<int>(sizeof(uint64_t));
    }

    if ((((src_addr + copied) | (dst_addr + copied)) & 0x3) == 0) {
        const int n32 = (num_bytes - copied) / static_cast<int>(sizeof(uint32_t));
        const uint32_t* src32 = reinterpret_cast<const uint32_t*>(
            static_cast<const char*>(src) + copied);
        uint32_t* dst32 = reinterpret_cast<uint32_t*>(
            static_cast<char*>(dst) + copied);
        for (int i = thread_idx; i < n32; i += thread_count) {
            ep_store<NonTemporalStore>(&dst32[i], ep_load<NonTemporalLoad>(&src32[i]));
        }
        copied += n32 * static_cast<int>(sizeof(uint32_t));
    }

    if ((((src_addr + copied) | (dst_addr + copied)) & 0x1) == 0) {
        const int n16 = (num_bytes - copied) / static_cast<int>(sizeof(uint16_t));
        const uint16_t* src16 = reinterpret_cast<const uint16_t*>(
            static_cast<const char*>(src) + copied);
        uint16_t* dst16 = reinterpret_cast<uint16_t*>(
            static_cast<char*>(dst) + copied);
        for (int i = thread_idx; i < n16; i += thread_count) {
            ep_store<NonTemporalStore>(&dst16[i], ep_load<NonTemporalLoad>(&src16[i]));
        }
        copied += n16 * static_cast<int>(sizeof(uint16_t));
    }

    const uint8_t* src8 = reinterpret_cast<const uint8_t*>(src);
    uint8_t* dst8 = reinterpret_cast<uint8_t*>(dst);
    for (int b = copied + thread_idx; b < num_bytes; b += thread_count) {
        ep_store<NonTemporalStore>(&dst8[b], ep_load<NonTemporalLoad>(&src8[b]));
    }
}

__device__ inline bool is_remote_row(int row, int rows, int self_rows) {
    return row < rows - self_rows;
}

__device__ inline void maybe_fence(int system_fences) {
    if (system_fences > 0) {
        __builtin_amdgcn_s_waitcnt(0);
        for (int f = 0; f < system_fences; ++f) __threadfence_system();
    }
}

__global__ void dispatch_unpack_hidden_kernel(const char* slots,
                                              char* recv_hidden,
                                              int rows,
                                              int self_rows,
                                              int hidden_bytes,
                                              int slot_stride,
                                              int system_fences) {
    const int lane = threadIdx.x % kWaveSize;
    const int wf_in_blk = threadIdx.x / kWaveSize;
    const int wave_count = blockDim.x / kWaveSize;
    const int global_wave_count = gridDim.x * wave_count;
    ep_multi_warp_iter_t iter(global_wave_count, rows,
                              static_cast<size_t>(hidden_bytes));
    const int total_parts = iter.total_parts(rows);
    for (int base_part = blockIdx.x * wave_count; base_part < total_parts;
         base_part += gridDim.x * wave_count) {
        const int part_idx = base_part + wf_in_blk;
        if (part_idx >= total_parts) continue;
        int row = 0;
        int in_row_part = 0;
        size_t offset = 0;
        size_t count = 0;
        iter.decode(part_idx, row, in_row_part, offset, count);
        const char* src = slots + static_cast<size_t>(row) * slot_stride + offset;
        char* dst = recv_hidden + static_cast<size_t>(row) * hidden_bytes + offset;
        const int bytes = static_cast<int>(count);
        if (is_remote_row(row, rows, self_rows)) {
            copy_bytes_block<false, true>(src, dst, bytes, lane, kWaveSize);
        } else {
            copy_bytes_block(src, dst, bytes, lane, kWaveSize);
        }
        maybe_fence(system_fences);
    }
}

__global__ void combine_pack_hidden_kernel(const char* expert_output,
                                           char* staging_slots,
                                           int rows,
                                           int self_rows,
                                           int hidden_bytes,
                                           int slot_stride,
                                           int system_fences) {
    const int lane = threadIdx.x % kWaveSize;
    const int wf_in_blk = threadIdx.x / kWaveSize;
    const int wave_count = blockDim.x / kWaveSize;
    for (int row = blockIdx.x * wave_count + wf_in_blk; row < rows;
         row += gridDim.x * wave_count) {
        const char* src = expert_output + static_cast<size_t>(row) * hidden_bytes;
        char* dst = staging_slots + static_cast<size_t>(row) * slot_stride;
        if (is_remote_row(row, rows, self_rows)) {
            copy_bytes_block<true, false>(src, dst, hidden_bytes, lane, kWaveSize);
        } else {
            copy_bytes_block(src, dst, hidden_bytes, lane, kWaveSize);
        }
        maybe_fence(system_fences);
    }
}

__global__ void dispatch_pack_hidden_kernel(
        const char* tokens, char* staging_slots, int rows, int hidden_bytes,
        int slot_stride, bool system_scope) {
    const int lane = threadIdx.x % kWaveSize;
    const int wf_in_blk = threadIdx.x / kWaveSize;
    const int wave_count = blockDim.x / kWaveSize;
    const int raw_tokens_per_block = (rows + gridDim.x - 1) / gridDim.x;
    const int tokens_per_block =
        min(max(raw_tokens_per_block, 1), wave_count);
    const int waves_per_token = max(wave_count / tokens_per_block, 1);
    for (int base = blockIdx.x * tokens_per_block; base < rows;
         base += gridDim.x * tokens_per_block) {
        const int item = wf_in_blk / waves_per_token;
        const int part = wf_in_blk % waves_per_token;
        if (item >= min(rows - base, tokens_per_block)) continue;
        const int token = base + item;
        const int begin = (hidden_bytes * part) / waves_per_token;
        const int end = (hidden_bytes * (part + 1)) / waves_per_token;
        const int count = end - begin;
        const char* src = tokens + static_cast<size_t>(token) * hidden_bytes
                        + begin;
        char* dst = staging_slots + static_cast<size_t>(token) * slot_stride
                  + begin;
        if (system_scope && count % 16 == 0
            && ((reinterpret_cast<uintptr_t>(src)
                 | reinterpret_cast<uintptr_t>(dst)) & 0xf) == 0) {
            const uint4* src4 = reinterpret_cast<const uint4*>(src);
            uint4* dst4 = reinterpret_cast<uint4*>(dst);
            for (int i = lane; i < count / 16; i += kWaveSize) {
                system_scope_store(&dst4[i], src4[i]);
            }
        } else {
            copy_bytes_block(src, dst, count, lane, kWaveSize);
        }
    }
}

struct alignas(16) bf16_vec8_t {
    hip_bfloat16 v[8];
};

__global__ void combine_accumulate_hidden_kernel(
        const char* recv_slots, hip_bfloat16* output, int rows, int topk,
        int hidden_bytes, int slot_stride, bool system_scope) {
    const int lane = threadIdx.x % kWaveSize;
    const int wf_in_blk = threadIdx.x / kWaveSize;
    const int wave_count = blockDim.x / kWaveSize;
    const int hidden_elems = hidden_bytes / sizeof(hip_bfloat16);
    const int raw_tokens_per_block = (rows + gridDim.x - 1) / gridDim.x;
    const int tokens_per_block =
        min(max(raw_tokens_per_block, 1), wave_count);
    const int waves_per_token = max(wave_count / tokens_per_block, 1);
    for (int base = blockIdx.x * tokens_per_block; base < rows;
         base += gridDim.x * tokens_per_block) {
        const int item = wf_in_blk / waves_per_token;
        const int part = wf_in_blk % waves_per_token;
        if (item >= min(rows - base, tokens_per_block)) continue;
        const int token = base + item;
        const int elem_begin = (hidden_elems * part) / waves_per_token;
        const int elem_end = (hidden_elems * (part + 1)) / waves_per_token;
        const int nvec = (elem_end - elem_begin) / 8;
        uint4* out4 = reinterpret_cast<uint4*>(output)
                   + static_cast<size_t>(token) * hidden_elems / 8
                   + elem_begin / 8;
        for (int vi = lane; vi < nvec; vi += kWaveSize) {
            float acc[8] = {};
            for (int k = 0; k < topk; ++k) {
                const char* slot =
                    recv_slots
                    + (static_cast<size_t>(token) * topk + k) * slot_stride;
                const uint4* src4 = reinterpret_cast<const uint4*>(slot)
                                  + elem_begin / 8 + vi;
                const uint4 raw =
                    system_scope ? system_scope_load(src4) : *src4;
                const auto* values =
                    reinterpret_cast<const hip_bfloat16*>(&raw);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] += static_cast<float>(values[i]);
                }
            }
            bf16_vec8_t packed;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                packed.v[i] = hip_bfloat16(acc[i]);
            }
            out4[vi] = *reinterpret_cast<const uint4*>(&packed);
        }
    }
}

__global__ void fill_bf16_kernel(hip_bfloat16* data, size_t count,
                                 hip_bfloat16 value) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        data[i] = value;
    }
}

__global__ void combine_self_slot_copy_kernel(const char* staging_slots,
                                              char* recv_slots,
                                              int rows,
                                              int self_rows,
                                              int slot_stride,
                                              int system_fences) {
    const int lane = threadIdx.x % kWaveSize;
    const int wf_in_blk = threadIdx.x / kWaveSize;
    const int wave_count = blockDim.x / kWaveSize;
    const int first_self = rows - self_rows;
    for (int idx = blockIdx.x * wave_count + wf_in_blk; idx < self_rows;
         idx += gridDim.x * wave_count) {
        const int row = first_self + idx;
        const char* src = staging_slots + static_cast<size_t>(row) * slot_stride;
        char* dst = recv_slots + static_cast<size_t>(idx) * slot_stride;
        copy_bytes_block(src, dst, slot_stride, lane, kWaveSize);
        maybe_fence(system_fences);
    }
}

void run_one(const char* name, void (*launcher)(const config_t&, char*, char*, char*),
             const config_t& cfg, char* a, char* b, char* c, size_t bytes_per_iter) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for (int i = 0; i < cfg.warmup; ++i) launcher(cfg, a, b, c);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < cfg.iters; ++i) launcher(cfg, a, b, c);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    const double us_per_iter = static_cast<double>(ms) * 1000.0 / cfg.iters;
    const double gbps = static_cast<double>(bytes_per_iter) / us_per_iter / 1000.0;
    printf("%-24s %12.3f us %12.3f GB/s %12zu bytes/iter\n",
           name, us_per_iter, gbps, bytes_per_iter);
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

void launch_dispatch(const config_t& cfg, char* a, char* b, char*) {
    dispatch_unpack_hidden_kernel<<<
        cfg.blocks, cfg.waves_per_block * kWaveSize>>>(
        a, b, cfg.recv_rows, cfg.self_rows, cfg.dispatch_hidden_bytes,
        cfg.dispatch_slot_stride, cfg.system_fences);
}

void launch_combine_pack(const config_t& cfg, char* a, char* b, char*) {
    combine_pack_hidden_kernel<<<
        cfg.combine_blocks, cfg.waves_per_block * kWaveSize>>>(
        a, b, cfg.recv_rows, cfg.self_rows, cfg.combine_hidden_bytes,
        cfg.combine_slot_stride, cfg.system_fences);
}

void launch_dispatch_pack(const config_t& cfg, char* a, char* b, char*) {
    dispatch_pack_hidden_kernel<<<
        cfg.blocks, cfg.waves_per_block * kWaveSize>>>(
        a, b, cfg.rows, cfg.dispatch_hidden_bytes, cfg.dispatch_slot_stride,
        cfg.production_system_scope);
}

void launch_combine_accumulate(const config_t& cfg, char* a, char* b, char*) {
    combine_accumulate_hidden_kernel<<<
        cfg.combine_blocks, cfg.waves_per_block * kWaveSize>>>(
        a, reinterpret_cast<hip_bfloat16*>(b), cfg.rows, cfg.topk,
        cfg.combine_hidden_bytes, cfg.combine_slot_stride,
        cfg.production_system_scope);
}

void launch_self_copy(const config_t& cfg, char*, char* b, char* c) {
    combine_self_slot_copy_kernel<<<
        cfg.combine_blocks, cfg.waves_per_block * kWaveSize>>>(
        b, c, cfg.recv_rows, cfg.self_rows, cfg.combine_slot_stride,
        cfg.system_fences);
}

} // namespace

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{"KIWI EP hidden-copy pattern microbenchmark"};
    app.add_option("--mode", cfg.mode,
                   "Mode: all, dispatch-unpack, dispatch-pack, combine-pack, "
                   "combine-accumulate, pack-accumulate, combine-self-copy")
        ->default_val("all")
        ->check(CLI::IsMember({
            "all", "dispatch-unpack", "dispatch-pack", "combine-pack",
            "combine-accumulate", "pack-accumulate",
            "combine-self-copy"}));
    app.add_option("--rows", cfg.rows, "Input tokens/output rows")->default_val(334);
    app.add_option("--recv-rows", cfg.recv_rows,
                   "Rows for dispatch-unpack/combine-pack (-1 uses --rows)")
        ->default_val(-1);
    app.add_option("--self-rows", cfg.self_rows, "Rows local to this rank")->default_val(43);
    app.add_option("--blocks", cfg.blocks, "Dispatch-style blocks")->default_val(4);
    app.add_option("--combine-blocks", cfg.combine_blocks, "Combine-style blocks")->default_val(4);
    app.add_option("--waves-per-block", cfg.waves_per_block,
                   "Wavefronts per workgroup")
        ->default_val(kWavesPerBlock)
        ->check(CLI::IsMember({1, 2, 4, 8, 16}));
    app.add_option("--topk", cfg.topk,
                   "BF16 rows accumulated per output token")->default_val(8);
    app.add_option("--dispatch-hidden-bytes", cfg.dispatch_hidden_bytes)->default_val(7168);
    app.add_option("--combine-hidden-bytes", cfg.combine_hidden_bytes)->default_val(14336);
    app.add_option("--dispatch-slot-stride", cfg.dispatch_slot_stride)->default_val(7472);
    app.add_option("--combine-slot-stride", cfg.combine_slot_stride)->default_val(14416);
    app.add_option("--warmup", cfg.warmup)->default_val(100);
    app.add_option("--iters", cfg.iters)->default_val(1000);
    app.add_option("--system-fences", cfg.system_fences,
                   "Number of __threadfence_system() calls after each copied row/part")
        ->default_val(0);
    app.add_flag("--production-system-scope", cfg.production_system_scope,
                 "Use production-style system-scope stores/loads in the new "
                 "packing/accumulation modes");
    app.add_flag("--register-ibv", cfg.register_ibv,
                 "Register all benchmark hipMalloc buffers with ibv_reg_mr before timing");
    app.add_flag("--register-dmabuf", cfg.register_dmabuf,
                 "Export hipMalloc buffers as DMA-BUF and register with ibv_reg_dmabuf_mr before timing");
    app.add_option("--ib-dev", cfg.ib_dev,
                   "IB device for --register-ibv (default: first verbs device)")
        ->default_val("");
    CLI11_PARSE(app, argc, argv);
    if (cfg.recv_rows < 0) cfg.recv_rows = cfg.rows;

    if (cfg.register_ibv && cfg.register_dmabuf) {
        fprintf(stderr, "--register-ibv and --register-dmabuf are mutually exclusive\n");
        return 1;
    }
    if (cfg.rows <= 0 || cfg.recv_rows <= 0 || cfg.topk <= 0
        || cfg.self_rows < 0 || cfg.self_rows > cfg.recv_rows
        || cfg.combine_hidden_bytes % 16 != 0) {
        fprintf(stderr, "invalid rows/topk/hidden-size configuration\n");
        return 1;
    }
    const int registration_modes = (cfg.register_ibv ? 1 : 0) +
                                   (cfg.register_dmabuf ? 1 : 0);
    if (registration_modes > 1) {
        fprintf(stderr, "choose at most one registration mode: --register-ibv or --register-dmabuf\n");
        return 1;
    }

    const size_t dispatch_slots_bytes =
        static_cast<size_t>(std::max(cfg.rows, cfg.recv_rows))
        * cfg.dispatch_slot_stride;
    const size_t dispatch_out_bytes =
        static_cast<size_t>(cfg.recv_rows) * cfg.dispatch_hidden_bytes;
    const size_t combine_expert_bytes =
        static_cast<size_t>(std::max(cfg.rows, cfg.recv_rows))
        * cfg.combine_hidden_bytes;
    const size_t combine_slots_bytes =
        static_cast<size_t>(std::max(cfg.recv_rows, cfg.rows * cfg.topk))
        * cfg.combine_slot_stride;
    const size_t self_slots_bytes = static_cast<size_t>(cfg.self_rows) * cfg.combine_slot_stride;
    const size_t combine_output_bytes =
        static_cast<size_t>(cfg.rows) * cfg.combine_hidden_bytes;
    const size_t a_bytes = std::max(dispatch_slots_bytes, combine_slots_bytes);
    const size_t b_bytes = std::max(
        std::max(std::max(dispatch_out_bytes, combine_expert_bytes),
                 combine_slots_bytes),
        combine_output_bytes);
    const size_t c_bytes = std::max<size_t>(self_slots_bytes, 1);

    char *a = nullptr, *b = nullptr, *c = nullptr;
    HIP_CHECK(hipMalloc(&a, a_bytes));
    HIP_CHECK(hipMalloc(&b, b_bytes));
    HIP_CHECK(hipMalloc(&c, c_bytes));
    HIP_CHECK(hipMemset(a, 0xA5, a_bytes));
    HIP_CHECK(hipMemset(b, 0, b_bytes));
    HIP_CHECK(hipMemset(c, 0, c_bytes));
    const bool run_accumulation =
        cfg.mode == "combine-accumulate"
        || cfg.mode == "pack-accumulate";
    if (run_accumulation) {
        fill_bf16_kernel<<<256, 256>>>(
            reinterpret_cast<hip_bfloat16*>(a),
            a_bytes / sizeof(hip_bfloat16), hip_bfloat16(1.0f));
        HIP_CHECK(hipDeviceSynchronize());
        launch_combine_accumulate(cfg, a, b, c);
        HIP_CHECK(hipDeviceSynchronize());
        const size_t count =
            static_cast<size_t>(cfg.rows) * cfg.combine_hidden_bytes
            / sizeof(hip_bfloat16);
        std::vector<hip_bfloat16> output(count);
        HIP_CHECK(hipMemcpy(
            output.data(), b, count * sizeof(hip_bfloat16),
            hipMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            if (static_cast<float>(output[i]) != static_cast<float>(cfg.topk)) {
                fprintf(stderr,
                        "combine accumulation validation failed at %zu\n", i);
                return 1;
            }
        }
        printf("# combine_full_output_validation=pass elements=%zu\n", count);
    }
    if (cfg.mode == "dispatch-pack" || cfg.mode == "pack-accumulate") {
        launch_dispatch_pack(cfg, a, b, c);
        HIP_CHECK(hipDeviceSynchronize());
        const size_t src_bytes =
            static_cast<size_t>(cfg.rows) * cfg.dispatch_hidden_bytes;
        const size_t dst_bytes =
            static_cast<size_t>(cfg.rows) * cfg.dispatch_slot_stride;
        std::vector<char> src(src_bytes);
        std::vector<char> dst(dst_bytes);
        HIP_CHECK(hipMemcpy(
            src.data(), a, src_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(
            dst.data(), b, dst_bytes, hipMemcpyDeviceToHost));
        for (int row = 0; row < cfg.rows; ++row) {
            if (memcmp(
                    src.data()
                        + static_cast<size_t>(row)
                            * cfg.dispatch_hidden_bytes,
                    dst.data()
                        + static_cast<size_t>(row)
                            * cfg.dispatch_slot_stride,
                    cfg.dispatch_hidden_bytes) != 0) {
                fprintf(stderr,
                        "dispatch packing validation failed at row %d\n",
                        row);
                return 1;
            }
        }
        printf("# dispatch_full_output_validation=pass bytes=%zu\n",
               src_bytes);
    }

    std::unique_ptr<ibv_registration_t> ibv;
    if (cfg.register_ibv || cfg.register_dmabuf) {
        ibv = std::make_unique<ibv_registration_t>();
        ibv->open_device(cfg.ib_dev);
        if (cfg.register_dmabuf) {
            ibv->register_dmabuf_buffer(a, a_bytes, "buffer_a");
            ibv->register_dmabuf_buffer(b, b_bytes, "buffer_b");
            ibv->register_dmabuf_buffer(c, c_bytes, "buffer_c");
        } else {
            ibv->register_buffer(a, a_bytes, "buffer_a");
            ibv->register_buffer(b, b_bytes, "buffer_b");
            ibv->register_buffer(c, c_bytes, "buffer_c");
        }
        HIP_CHECK(hipDeviceSynchronize());
    }

    printf("# rows=%d recv_rows=%d self_rows=%d topk=%d dispatch_hidden=%d combine_hidden=%d dispatch_slot=%d combine_slot=%d\n",
           cfg.rows, cfg.recv_rows, cfg.self_rows, cfg.topk,
           cfg.dispatch_hidden_bytes, cfg.combine_hidden_bytes,
           cfg.dispatch_slot_stride, cfg.combine_slot_stride);
    printf("# blocks=%d combine_blocks=%d waves_per_block=%d warmup=%d iters=%d system_fences=%d production_system_scope=%s register_ibv=%s register_dmabuf=%s\n",
           cfg.blocks, cfg.combine_blocks, cfg.waves_per_block,
           cfg.warmup, cfg.iters, cfg.system_fences,
           cfg.production_system_scope ? "true" : "false",
           cfg.register_ibv ? "true" : "false",
           cfg.register_dmabuf ? "true" : "false");
    printf("%-24s %15s %17s %18s\n", "mode", "time/iter", "effective BW", "bytes");

    if (cfg.mode == "all" || cfg.mode == "dispatch-unpack") {
        run_one("dispatch-unpack-hidden", launch_dispatch, cfg, a, b, c,
                static_cast<size_t>(cfg.recv_rows)
                * cfg.dispatch_hidden_bytes);
    }
    if (cfg.mode == "dispatch-pack" || cfg.mode == "pack-accumulate") {
        run_one("dispatch-pack-hidden", launch_dispatch_pack, cfg, a, b, c,
                static_cast<size_t>(cfg.rows) * cfg.dispatch_hidden_bytes);
    }
    if (cfg.mode == "all" || cfg.mode == "combine-pack") {
        run_one("combine-pack-hidden", launch_combine_pack, cfg, a, b, c,
                static_cast<size_t>(cfg.recv_rows)
                * cfg.combine_hidden_bytes);
    }
    if (run_accumulation) {
        run_one("combine-accumulate-hidden", launch_combine_accumulate,
                cfg, a, b, c,
                static_cast<size_t>(cfg.rows) * cfg.combine_hidden_bytes
                * static_cast<size_t>(cfg.topk + 1));
    }
    if (cfg.mode == "all" || cfg.mode == "combine-self-copy") {
        run_one("combine-self-slot-copy", launch_self_copy, cfg, a, b, c,
                static_cast<size_t>(cfg.self_rows) * cfg.combine_slot_stride);
    }

    ibv.reset();
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(c));
    return 0;
}
