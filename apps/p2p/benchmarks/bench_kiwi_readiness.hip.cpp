// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// KIWI single-PUT readiness benchmark.
//
// A persistent GPU kernel drives the whole ping-pong. Each logical write is one
// KIWI invoke descriptor whose host callback issues exactly one LCI PUT of an
// 8-byte registered GPU source into the peer's GPU target. The receiving GPU
// polls its target directly and posts the reply descriptor itself.

#include <CLI/CLI.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include "bench_affinity.hpp"
#include "context.hip.hpp"
#include "lci.hpp"
#include "readiness_bench_common.hip.hpp"

namespace {

using readiness_bench::receiver_t;

struct config_t {
  int warmup = 1000;
  int iters = 100000;
  int depth = 1;
  int main_core = -1;
  int proxy_core = -1;
};

struct readiness_put_t {
  uintptr_t source;
  uintptr_t remote_offset;
  int source_heap;
  int target_heap;
  int rank;
};

__global__ void run_protocol(kiwi::p2p::gcontext_t gctx, int rank, int peer,
                             uint64_t *source, uint64_t *target,
                             int source_heap, int target_heap, uint8_t cb_id,
                             int iters, int depth) {
  auto handle = gctx.get_handle();
  for (uint64_t epoch = 1; epoch <= (uint64_t)iters; ++epoch) {
    if (rank == 0) {
      for (int slot = 0; slot < depth; ++slot) {
        readiness_put_t desc{reinterpret_cast<uintptr_t>(
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
      readiness_put_t desc{
          reinterpret_cast<uintptr_t>(source + (size_t)(epoch - 1) * depth), 0,
          source_heap, target_heap, peer};
      handle.post_desc(cb_id, desc);
    }
    __syncthreads();
  }
}

// The timed region includes the single kernel launch, which is amortized over
// all iterations.
double run(kiwi::p2p::gcontext_t gctx, int rank, int peer,
           const readiness_bench::buffers_t &b, int source_heap,
           int target_heap, uint8_t cb_id, int iters) {
  readiness_bench::prepare_buffers(b, iters);
  lci::barrier();
  auto start = std::chrono::steady_clock::now();
  run_protocol<<<1, 64>>>(gctx, rank, peer, b.source, b.target, source_heap,
                          target_heap, cb_id, iters, b.depth);
  READINESS_HIP_CHECK(hipGetLastError());
  READINESS_HIP_CHECK(hipDeviceSynchronize());
  auto stop = std::chrono::steady_clock::now();
  lci::barrier();
  readiness_bench::validate_target(b, rank, iters);
  return std::chrono::duration<double>(stop - start).count();
}

} // namespace

int main(int argc, char **argv) {
  config_t cfg;
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
  if (cfg.main_core >= 0 && cfg.main_core == cfg.proxy_core)
    return readiness_bench::usage_error(
        "--main-core and --proxy-core must differ");

  kiwi::p2p::bench::pin_main_thread(-1, cfg.main_core);
  kiwi::p2p::context_t ctx(1, 1);
  int rank = lci::get_rank_me();
  if (lci::get_rank_n() != 2)
    return readiness_bench::usage_error("benchmark requires exactly two ranks");
  int peer = 1 - rank;

  auto b = readiness_bench::alloc_buffers(std::max(cfg.warmup, cfg.iters),
                                          cfg.depth, false, receiver_t::gpu);
  int source_heap = ctx.register_symmetric(b.source, b.source_bytes());
  int target_heap = ctx.register_symmetric(b.target, b.target_bytes());
  const auto *heaps = &ctx.get_heap_table();
  lci::device_t device = ctx.get_device(0);
  uint8_t cb_id = ctx.register_callback<readiness_put_t>(
      std::function<void(readiness_put_t)>([heaps,
                                            device](readiness_put_t desc) {
        const auto &source_entry = (*heaps)[desc.source_heap].device_entries[0];
        const auto &target_entry = (*heaps)[desc.target_heap].device_entries[0];
        lci::error_t error;
        do {
          error = lci::net_post_put_x(desc.rank,
                                      reinterpret_cast<void *>(desc.source),
                                      sizeof(uint64_t), source_entry.mr,
                                      desc.remote_offset,
                                      target_entry.rmrs[desc.rank])
                      .device(device)();
          if (error.is_retry())
            lci::progress_x().device(device)();
        } while (error.is_retry());
        if (!error.is_posted()) {
          std::fprintf(stderr, "readiness PUT did not post\n");
          std::abort();
        }
      }));
  ctx.spawn_proxy_threads(1, cfg.proxy_core);
  auto gctx = ctx.get_device_context();

  if (cfg.warmup > 0)
    run(gctx, rank, peer, b, source_heap, target_heap, cb_id, cfg.warmup);
  double seconds =
      run(gctx, rank, peer, b, source_heap, target_heap, cb_id, cfg.iters);
  if (rank == 0) {
    std::string fields =
        "backend=kiwi op=readiness_put source=gpu target=gpu receiver=gpu "
        "main_core=" +
        std::to_string(cfg.main_core) +
        " proxy_core=" + std::to_string(cfg.proxy_core);
    readiness_bench::print_result(fields, cfg.depth, cfg.iters, seconds);
  }

  ctx.join_proxy_threads();
  lci::barrier();
  ctx.deregister_symmetric(target_heap);
  ctx.deregister_symmetric(source_heap);
  readiness_bench::free_buffers(b);
  return 0;
}
