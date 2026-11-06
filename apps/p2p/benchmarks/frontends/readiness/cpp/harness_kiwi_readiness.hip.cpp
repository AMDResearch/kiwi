// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// C++ frontend of the cross-frontend KIWI single-PUT readiness harness.
//
// The protocol kernel and launch of the standalone bench_kiwi_readiness
// (apps/p2p/benchmarks/bench_kiwi_readiness.hip.cpp), run through the
// harness's host code (common/kiwi_readiness_bench.hip.hpp) like the Python
// frontends. Same options as the standalone benchmark; the result line says
// backend=kiwi_cpp.

#include <CLI/CLI.hpp>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "kiwi_readiness_bench.hip.hpp"

namespace {

__global__ void run_protocol(kiwi::p2p::gcontext_t gctx, int rank, int peer,
                             uint64_t *source, uint64_t *target,
                             int source_heap, int target_heap, uint8_t cb_id,
                             int iters, int depth) {
  auto handle = gctx.get_handle();
  for (uint64_t epoch = 1; epoch <= (uint64_t)iters; ++epoch) {
    if (rank == 0) {
      for (int slot = 0; slot < depth; ++slot) {
        kiwi_readiness_bench::readiness_put_t desc{reinterpret_cast<uintptr_t>(
                                 source + (size_t)(epoch - 1) * depth + slot),
                             (uintptr_t)slot * sizeof(uint64_t), source_heap,
                             target_heap, peer};
        handle.post_desc(cb_id, desc);
      }
      if (threadIdx.x == 0) {
        while (readiness_bench::load_system(target) < epoch) {
        }
      }
    } else {
      for (int slot = threadIdx.x; slot < depth; slot += blockDim.x) {
        while (readiness_bench::load_system(target + slot) < epoch) {
        }
      }
      __syncthreads();
      kiwi_readiness_bench::readiness_put_t desc{
          reinterpret_cast<uintptr_t>(source + (size_t)(epoch - 1) * depth), 0,
          source_heap, target_heap, peer};
      handle.post_desc(cb_id, desc);
    }
    __syncthreads();
  }
}

// Launches run_protocol for the harness's host code.
struct HipReadinessLauncher {
  void launch(const kiwi_readiness_bench::ReadinessLaunch &l) {
    run_protocol<<<1, 64>>>(l.gctx, l.rank, l.peer, l.source, l.target,
                            l.source_heap, l.target_heap, l.cb_id, l.iters,
                            l.depth);
    READINESS_HIP_CHECK(hipGetLastError());
  }
};

} // namespace

int main(int argc, char **argv) {
  kiwi_readiness_bench::config_t cfg;
  CLI::App app{"KIWI single-PUT GPU-polled readiness benchmark"};
  app.add_option("--warmup", cfg.warmup)
      ->default_val(1000)
      ->check(CLI::NonNegativeNumber);
  app.add_option("--iters", cfg.iters)
      ->default_val(100000)
      ->check(CLI::PositiveNumber);
  app.add_option("--depth", cfg.depth,
                 "Forward writes per acknowledged batch; use 1 for latency")
      ->default_val(1)
      ->check(CLI::PositiveNumber);
  app.add_option("--main-core", cfg.main_core,
                 "Pin the main/control thread to this CPU (-1: inherit)")
      ->default_val(-1);
  app.add_option("--proxy-core", cfg.proxy_core,
                 "Pin the KIWI proxy thread to this CPU (-1: inherit)")
      ->default_val(-1);
  CLI11_PARSE(app, argc, argv);

  HipReadinessLauncher launcher;
  return kiwi_readiness_bench::run_benchmark(cfg, launcher, "kiwi_cpp");
}

