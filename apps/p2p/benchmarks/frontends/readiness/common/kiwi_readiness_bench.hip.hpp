// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of the cross-frontend KIWI single-PUT readiness harness.
//
// Each logical write is one KIWI invoke descriptor whose host callback issues
// exactly one LCI PUT of an 8-byte registered GPU source into the peer's GPU
// target. In round `epoch`, rank 0 writes `depth` epoch values into rank 1's
// target slots and waits for the ACK; rank 1 polls its target directly and
// posts the ACK descriptor itself. A copy of the host code of the standalone
// bench_kiwi_readiness (apps/p2p/benchmarks/bench_kiwi_readiness.hip.cpp)
// with the protocol kernel launch taken from a launcher, so every frontend of
// the harness runs the same host code: the C++ kernel (cpp/) and the Python
// kernels (flydsl/, ...) through kiwi_readiness_bench_ext.
//
//   struct Launcher {
//     // Launch the protocol kernel asynchronously: one block of 64 threads.
//     void launch(const ReadinessLaunch& l);
//   };

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

namespace kiwi_readiness_bench {

struct config_t {
  int warmup = 1000;
  int iters = 100000;
  int depth = 1;
  int main_core = -1;
  int proxy_core = -1;
};

// The invoke descriptor; frontends serialize the same 32 bytes (two u64,
// three i32 and 4 bytes of padding).
struct readiness_put_t {
  uintptr_t source;
  uintptr_t remote_offset;
  int source_heap;
  int target_heap;
  int rank;
};
static_assert(sizeof(readiness_put_t) == 32);

// Arguments of one protocol kernel launch.
struct ReadinessLaunch {
  kiwi::p2p::gcontext_t gctx;
  int rank;
  int peer;
  uint64_t *source;
  uint64_t *target;
  int source_heap;
  int target_heap;
  uint8_t cb_id;
  int iters;
  int depth;
};

// The timed region includes the single kernel launch, which is amortized over
// all iterations.
template <typename Launcher>
double run(Launcher &launcher, kiwi::p2p::gcontext_t gctx, int rank, int peer,
           const readiness_bench::buffers_t &b, int source_heap,
           int target_heap, uint8_t cb_id, int iters) {
  readiness_bench::prepare_buffers(b, iters);
  lci::barrier();
  auto start = std::chrono::steady_clock::now();
  launcher.launch(ReadinessLaunch{gctx, rank, peer, b.source, b.target,
                                  source_heap, target_heap, cb_id, iters,
                                  b.depth});
  READINESS_HIP_CHECK(hipDeviceSynchronize());
  auto stop = std::chrono::steady_clock::now();
  lci::barrier();
  readiness_bench::validate_target(b, rank, iters);
  return std::chrono::duration<double>(stop - start).count();
}

// Runs the benchmark and prints the result line with `backend=<backend>`.
// `prime_iters` > 0 first runs that many untimed rounds, for frontends that
// compile their kernel on the first launch; the C++ frontend passes 0, as
// the standalone benchmark has no such round.
template <typename Launcher>
int run_benchmark(const config_t &cfg, Launcher &launcher,
                  const char *backend, int prime_iters = 0) {
  using readiness_bench::receiver_t;
  if (cfg.main_core >= 0 && cfg.main_core == cfg.proxy_core)
    return readiness_bench::usage_error(
        "--main-core and --proxy-core must differ");

  kiwi::p2p::bench::pin_main_thread(-1, cfg.main_core);
  kiwi::p2p::context_t ctx(1, 1);
  int rank = lci::get_rank_me();
  if (lci::get_rank_n() != 2)
    return readiness_bench::usage_error("benchmark requires exactly two ranks");
  int peer = 1 - rank;

  auto b = readiness_bench::alloc_buffers(
      std::max({cfg.warmup, cfg.iters, prime_iters}), cfg.depth, false,
      receiver_t::gpu);
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

  if (prime_iters > 0)
    run(launcher, gctx, rank, peer, b, source_heap, target_heap, cb_id,
        prime_iters);
  if (cfg.warmup > 0)
    run(launcher, gctx, rank, peer, b, source_heap, target_heap, cb_id,
        cfg.warmup);
  double seconds = run(launcher, gctx, rank, peer, b, source_heap,
                       target_heap, cb_id, cfg.iters);
  if (rank == 0) {
    std::string fields =
        std::string("backend=") + backend +
        " op=readiness_put source=gpu target=gpu receiver=gpu "
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

} // namespace kiwi_readiness_bench
