// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

// Shared pieces of the single-write readiness benchmarks
// (bench_kiwi_readiness, bench_lci_readiness, bench_ibverbs_readiness).
//
// Every benchmark runs the same two-rank protocol. In round i (epoch i + 1),
// rank 0 writes `depth` 8-byte values carrying the epoch into rank 1's target
// slots. Rank 1 waits for all of them and writes one 8-byte ACK back into
// slot 0 of rank 0's target. Each logical write is exactly one network write.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define READINESS_HIP_CHECK(call)                                              \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,     \
                   hipGetErrorString(err));                                    \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace readiness_bench {

// Who detects an arriving write on the receiving rank.
enum class receiver_t {
  cpu, // the posting CPU thread polls the target directly
  gpu, // a persistent GPU kernel polls the target and notifies the CPU
};

enum class memory_t { gpu, fine_gpu, cpu };

inline const char *memory_name(memory_t memory) {
  switch (memory) {
  case memory_t::gpu:
    return "gpu";
  case memory_t::fine_gpu:
    return "fine_gpu";
  case memory_t::cpu:
    return "cpu";
  }
  return "unknown";
}

inline int usage_error(const char *message) {
  std::fprintf(stderr, "error: %s\n", message);
  return 2;
}

// Source holds one value per write: source[i * depth + slot] = i + 1.
// Target holds `depth` receive slots.
//   --cpu-memory:     CPU source and CPU target.
//   CPU receiver:     GPU source, fine-grained GPU target (CPU-dereferenceable).
//   GPU receiver:     GPU source, ordinary GPU target.
struct buffers_t {
  uint64_t *source = nullptr;
  uint64_t *target = nullptr;
  size_t source_count = 0;
  int depth = 0;
  memory_t source_memory = memory_t::gpu;
  memory_t target_memory = memory_t::gpu;

  size_t source_bytes() const { return source_count * sizeof(uint64_t); }
  size_t target_bytes() const { return (size_t)depth * sizeof(uint64_t); }
};

inline buffers_t alloc_buffers(int max_iters, int depth, bool cpu_memory,
                               receiver_t receiver) {
  buffers_t b;
  b.depth = depth;
  b.source_count = (size_t)max_iters * depth;
  if (cpu_memory) {
    b.source_memory = b.target_memory = memory_t::cpu;
    if (posix_memalign(reinterpret_cast<void **>(&b.source), 4096,
                       b.source_bytes()) != 0 ||
        posix_memalign(reinterpret_cast<void **>(&b.target), 4096,
                       b.target_bytes()) != 0) {
      std::fprintf(stderr, "posix_memalign failed\n");
      std::abort();
    }
    return b;
  }
  READINESS_HIP_CHECK(hipMalloc(&b.source, b.source_bytes()));
  if (receiver == receiver_t::cpu) {
    b.target_memory = memory_t::fine_gpu;
    READINESS_HIP_CHECK(
        hipExtMallocWithFlags(reinterpret_cast<void **>(&b.target),
                              b.target_bytes(), hipDeviceMallocFinegrained));
  } else {
    READINESS_HIP_CHECK(hipMalloc(&b.target, b.target_bytes()));
  }
  return b;
}

inline void free_buffers(buffers_t &b) {
  if (b.source_memory == memory_t::cpu) {
    std::free(b.target);
    std::free(b.source);
  } else {
    READINESS_HIP_CHECK(hipFree(b.target));
    READINESS_HIP_CHECK(hipFree(b.source));
  }
  b.source = b.target = nullptr;
}

__global__ void initialize_values(uint64_t *values, int iters, int depth) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t count = (size_t)iters * depth;
  if (index < count)
    values[index] = index / depth + 1;
}

// Fill the source epochs and zero the target. Completes before returning, so
// a peer write that arrives after the following barrier cannot be erased.
inline void prepare_buffers(const buffers_t &b, int iters) {
  size_t count = (size_t)iters * b.depth;
  if (b.source_memory == memory_t::cpu) {
    for (size_t i = 0; i < count; ++i)
      b.source[i] = i / b.depth + 1;
    std::memset(b.target, 0, b.target_bytes());
    return;
  }
  int blocks = (int)((count + 255) / 256);
  initialize_values<<<blocks, 256>>>(b.source, iters, b.depth);
  READINESS_HIP_CHECK(hipGetLastError());
  READINESS_HIP_CHECK(hipMemset(b.target, 0, b.target_bytes()));
  READINESS_HIP_CHECK(hipDeviceSynchronize());
}

// After a run, every slot this rank waited on must hold exactly the final
// epoch: rank 0 waits on the ACK slot, rank 1 on all `depth` slots.
inline void validate_target(const buffers_t &b, int rank, int iters) {
  int slots = rank == 0 ? 1 : b.depth;
  for (int slot = 0; slot < slots; ++slot) {
    uint64_t value = 0;
    if (b.target_memory == memory_t::cpu) {
      value = reinterpret_cast<const volatile uint64_t *>(b.target)[slot];
    } else {
      READINESS_HIP_CHECK(hipMemcpy(&value, b.target + slot, sizeof(value),
                                    hipMemcpyDeviceToHost));
    }
    if (value != (uint64_t)iters) {
      std::fprintf(stderr,
                   "rank %d target validation failed: slot %d holds %lu, "
                   "expected %d\n",
                   rank, slot, (unsigned long)value, iters);
      std::abort();
    }
  }
}

// latency_us is half the measured round-trip time and is printed only for
// --depth 1 ping-pong. mops counts the forward writes only; the per-batch ACK
// is excluded.
inline void print_result(const std::string &fields, int depth, int iters,
                         double seconds) {
  std::printf("result %s depth=%d iters=%d", fields.c_str(), depth, iters);
  if (depth == 1)
    std::printf(" latency_us=%.6f", seconds * 1e6 / iters / 2.0);
  std::printf(" mops=%.6f\n", (double)iters * depth / seconds / 1e6);
}

// System-scope load that bypasses non-coherent caching of NIC-written data.
__device__ inline uint64_t load_system(const uint64_t *ptr) {
  const unsigned long long addr = reinterpret_cast<unsigned long long>(ptr);
  unsigned long long value;
  asm volatile("global_load_dwordx2 %0, %1, off sc0 sc1\n\t"
               "s_waitcnt vmcnt(0)"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  return value;
}

__device__ inline void store_system(uint64_t *ptr, uint64_t value) {
  const unsigned long long addr = reinterpret_cast<unsigned long long>(ptr);
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __builtin_amdgcn_s_waitcnt(0);
}

// Persistent GPU receiver for the direct LCI and ibverbs controls: wait until
// `depth` target slots reach each epoch, then publish the epoch to a pinned
// host word that the posting CPU thread polls before replying.
__global__ void poll_target(uint64_t *target, int depth, int iters,
                            uint64_t *observed) {
  for (uint64_t epoch = 1; epoch <= (uint64_t)iters; ++epoch) {
    for (int slot = threadIdx.x; slot < depth; slot += blockDim.x) {
      while (load_system(target + slot) < epoch) {
      }
    }
    __syncthreads();
    if (threadIdx.x == 0)
      store_system(observed, epoch);
    __syncthreads();
  }
}

// Pinned, GPU-writable host word used by poll_target's notification.
struct host_notify_t {
  uint64_t *host = nullptr;
  uint64_t *device = nullptr;

  void alloc() {
    READINESS_HIP_CHECK(
        hipHostMalloc(reinterpret_cast<void **>(&host), sizeof(uint64_t),
                      hipHostMallocMapped | hipHostMallocCoherent));
    *host = 0;
    READINESS_HIP_CHECK(hipHostGetDevicePointer(
        reinterpret_cast<void **>(&device), host, 0));
  }
  void release() {
    if (host)
      READINESS_HIP_CHECK(hipHostFree(host));
    host = device = nullptr;
  }
  void reset() const { __atomic_store_n(host, 0, __ATOMIC_RELEASE); }
  uint64_t load() const { return __atomic_load_n(host, __ATOMIC_ACQUIRE); }
};

// Launch the GPU receiver for this rank's run: rank 0 waits on the ACK slot,
// rank 1 on all forward slots.
inline void launch_poll_target(const buffers_t &b, int rank, int iters,
                               const host_notify_t &notify) {
  poll_target<<<1, 64>>>(b.target, rank == 0 ? 1 : b.depth, iters,
                         notify.device);
  READINESS_HIP_CHECK(hipGetLastError());
}

// CPU receiver: spin on the target, calling `progress()` between loads.
template <typename Progress>
inline void wait_target_cpu(const uint64_t *target, int depth, uint64_t epoch,
                            Progress &&progress) {
  for (int slot = 0; slot < depth; ++slot) {
    while (reinterpret_cast<const volatile uint64_t *>(target)[slot] < epoch)
      progress();
  }
}

} // namespace readiness_bench
