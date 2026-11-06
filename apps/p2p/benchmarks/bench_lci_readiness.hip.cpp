// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Direct LCI single-PUT readiness benchmark.
//
// One lci::net_post_put_x per logical write, with no remote completion. This
// is LCI's network-layer put, the same call KIWI's readiness callback makes,
// with the same local-completion handling: no completion object, and LCI
// progress reclaims the send slot. The receiver is either the posting CPU
// thread polling the target (--receiver cpu) or a persistent GPU kernel that
// polls the target and notifies the CPU through a pinned host word
// (--receiver gpu, the matched control for KIWI).

#include <CLI/CLI.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "lci.hpp"
#include "readiness_bench_common.hip.hpp"

namespace {

using readiness_bench::receiver_t;

struct config_t {
  int warmup = 1000;
  int iters = 100000;
  int depth = 1;
  receiver_t receiver = receiver_t::gpu;
  bool cpu_memory = false;
};

struct lci_state_t {
  int peer = 0;
  lci::device_t device;
  lci::mr_t source_mr;
  lci::rmr_t peer_target;
};

void progress(const lci_state_t &s) { lci::progress_x().device(s.device)(); }

void post_one(const lci_state_t &s, uint64_t *source, size_t source_offset,
              size_t remote_slot) {
  lci::error_t error;
  do {
    error = lci::net_post_put_x(s.peer, source + source_offset,
                                sizeof(uint64_t), s.source_mr,
                                remote_slot * sizeof(uint64_t), s.peer_target)
                .device(s.device)();
    if (error.is_retry())
      progress(s);
  } while (error.is_retry());
  if (!error.is_posted()) {
    std::fprintf(stderr, "readiness PUT did not post\n");
    std::abort();
  }
}

double run(const config_t &cfg, int iters, int rank,
           const readiness_bench::buffers_t &b, const lci_state_t &s,
           const readiness_bench::host_notify_t &notify) {
  const bool gpu_receiver = cfg.receiver == receiver_t::gpu;
  readiness_bench::prepare_buffers(b, iters);
  if (gpu_receiver)
    notify.reset();
  lci::barrier_x().device(s.device)();
  if (gpu_receiver)
    readiness_bench::launch_poll_target(b, rank, iters, notify);

  const int wait_slots = rank == 0 ? 1 : cfg.depth;
  auto wait_epoch = [&](uint64_t epoch) {
    if (gpu_receiver) {
      while (notify.load() < epoch)
        progress(s);
    } else {
      readiness_bench::wait_target_cpu(b.target, wait_slots, epoch,
                                       [&] { progress(s); });
    }
  };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    uint64_t epoch = (uint64_t)i + 1;
    if (rank == 0) {
      for (int slot = 0; slot < cfg.depth; ++slot)
        post_one(s, b.source, (size_t)i * cfg.depth + slot, slot);
      wait_epoch(epoch);
    } else {
      wait_epoch(epoch);
      post_one(s, b.source, (size_t)i * cfg.depth, 0);
    }
  }
  // Reap the local completions of every posted PUT.
  lci::wait_drained_x().device(s.device)();
  if (gpu_receiver)
    READINESS_HIP_CHECK(hipDeviceSynchronize());
  auto stop = std::chrono::steady_clock::now();
  lci::barrier_x().device(s.device)();
  readiness_bench::validate_target(b, rank, iters);
  return std::chrono::duration<double>(stop - start).count();
}

} // namespace

int main(int argc, char **argv) {
  config_t cfg;
  CLI::App app{"Direct LCI single-PUT readiness benchmark"};
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
  std::string receiver = "gpu";
  app.add_option("--receiver", receiver,
                 "Who detects arriving writes: cpu (CPU polls the target) or "
                 "gpu (persistent GPU poller notifies the CPU)")
      ->default_val("gpu")
      ->check(CLI::IsMember({"cpu", "gpu"}));
  app.add_flag("--cpu-memory", cfg.cpu_memory,
               "Use CPU source and target buffers (requires --receiver cpu)");
  CLI11_PARSE(app, argc, argv);
  cfg.receiver = receiver == "cpu" ? receiver_t::cpu : receiver_t::gpu;
  if (cfg.cpu_memory && cfg.receiver != receiver_t::cpu)
    return readiness_bench::usage_error("--cpu-memory requires --receiver cpu");

  lci::g_runtime_init();
  int rank = lci::get_rank_me();
  if (lci::get_rank_n() != 2) {
    std::fprintf(stderr, "error: benchmark requires exactly two ranks\n");
    lci::g_runtime_fina();
    return 2;
  }

  auto b = readiness_bench::alloc_buffers(std::max(cfg.warmup, cfg.iters),
                                          cfg.depth, cfg.cpu_memory,
                                          cfg.receiver);
  readiness_bench::host_notify_t notify;
  if (cfg.receiver == receiver_t::gpu)
    notify.alloc();

  lci_state_t s;
  s.peer = 1 - rank;
  s.device = lci::alloc_device();
  s.source_mr =
      lci::register_memory_x(b.source, b.source_bytes()).device(s.device)();
  lci::mr_t target_mr =
      lci::register_memory_x(b.target, b.target_bytes()).device(s.device)();
  lci::rmr_t my_target = lci::get_rmr(target_mr);
  std::vector<lci::rmr_t> targets(2);
  lci::allgather_x(&my_target, targets.data(), sizeof(my_target))
      .device(s.device)();
  s.peer_target = targets[s.peer];

  if (cfg.warmup > 0)
    run(cfg, cfg.warmup, rank, b, s, notify);
  double seconds = run(cfg, cfg.iters, rank, b, s, notify);
  if (rank == 0) {
    std::string fields =
        std::string("backend=lci op=net_post_put remote_completion=off "
                    "source=") +
        readiness_bench::memory_name(b.source_memory) +
        " target=" + readiness_bench::memory_name(b.target_memory) +
        " receiver=" + (cfg.receiver == receiver_t::gpu ? "gpu" : "cpu");
    readiness_bench::print_result(fields, cfg.depth, cfg.iters, seconds);
  }

  lci::deregister_memory(&target_mr);
  lci::deregister_memory(&s.source_mr);
  lci::free_device(&s.device);
  notify.release();
  readiness_bench::free_buffers(b);
  lci::g_runtime_fina();
  return 0;
}
