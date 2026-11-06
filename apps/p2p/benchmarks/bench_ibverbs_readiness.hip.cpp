// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Raw ibverbs single-WRITE readiness benchmark.
//
// One ibv_post_send(IBV_WR_RDMA_WRITE) of an 8-byte registered source per
// logical write; no immediate data and no receive WQEs. The receiver is either
// the posting CPU thread polling the target (--receiver cpu) or a persistent
// GPU kernel that polls the target and notifies the CPU through a pinned host
// word (--receiver gpu).
//
// Each rank signals only the last WRITE of a batch and waits for that send
// completion right after posting. On the replying rank this wait overlaps the
// peer's detection of the reply.

#include <CLI/CLI.hpp>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <infiniband/verbs.h>
#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

#include "readiness_bench_common.hip.hpp"

namespace {

using readiness_bench::receiver_t;

constexpr int kPort = 1;

struct config_t {
  int warmup = 1000;
  int iters = 100000;
  int depth = 1;
  receiver_t receiver = receiver_t::gpu;
  bool cpu_memory = false;
  bool dmabuf = false;
  int gid_index = 3;
  std::string device;
};

struct peer_info_t {
  uint64_t address;
  uint32_t rkey;
  uint32_t qpn;
  uint32_t psn;
  uint16_t lid;
  uint8_t mtu;
  uint8_t gid[16];
};

// ibv_* calls that return int report the error code directly; calls that
// return pointers set errno.
[[noreturn]] void fail(const char *what, int err = errno) {
  std::fprintf(stderr, "%s failed: %s\n", what, std::strerror(err));
  std::abort();
}

struct verbs_t {
  ibv_device **devices = nullptr;
  ibv_context *context = nullptr;
  ibv_pd *pd = nullptr;
  ibv_cq *cq = nullptr;
  ibv_qp *qp = nullptr;
  ibv_mr *source_mr = nullptr;
  ibv_mr *target_mr = nullptr;
  int source_fd = -1;
  int target_fd = -1;

  ~verbs_t() {
    if (qp)
      ibv_destroy_qp(qp);
    if (target_mr)
      ibv_dereg_mr(target_mr);
    if (source_mr)
      ibv_dereg_mr(source_mr);
    if (target_fd >= 0)
      close(target_fd);
    if (source_fd >= 0)
      close(source_fd);
    if (cq)
      ibv_destroy_cq(cq);
    if (pd)
      ibv_dealloc_pd(pd);
    if (context)
      ibv_close_device(context);
    if (devices)
      ibv_free_device_list(devices);
  }
};

ibv_mr *register_dmabuf(ibv_pd *pd, void *ptr, size_t bytes, int access,
                        int *fd_out) {
  size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
  uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  void *aligned_ptr = reinterpret_cast<void *>(address & ~(page_size - 1));
  size_t delta = address - reinterpret_cast<uintptr_t>(aligned_ptr);
  size_t aligned_bytes =
      ((bytes + delta + page_size - 1) / page_size) * page_size;
  uint64_t offset = 0;
  if (hsa_amd_portable_export_dmabuf(aligned_ptr, aligned_bytes, fd_out,
                                     &offset) != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr, "hsa_amd_portable_export_dmabuf failed\n");
    std::abort();
  }
  ibv_mr *mr = ibv_reg_dmabuf_mr(pd, offset, aligned_bytes,
                                 reinterpret_cast<uintptr_t>(aligned_ptr),
                                 *fd_out, access);
  if (!mr)
    fail("ibv_reg_dmabuf_mr");
  return mr;
}

ibv_mr *register_buffer(verbs_t &v, void *ptr, size_t bytes, int access,
                        bool dmabuf, int *fd_out) {
  if (dmabuf)
    return register_dmabuf(v.pd, ptr, bytes, access, fd_out);
  ibv_mr *mr = ibv_reg_mr(v.pd, ptr, bytes, access);
  if (!mr)
    fail("ibv_reg_mr");
  return mr;
}

void connect_qp(verbs_t &v, const peer_info_t &mine, const peer_info_t &peer,
                int gid_index) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = kPort;
  attr.pkey_index = 0;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  if (int rc = ibv_modify_qp(v.qp, &attr,
                             IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                 IBV_QP_ACCESS_FLAGS))
    fail("QP INIT", rc);

  attr = {};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = (ibv_mtu)std::min(mine.mtu, peer.mtu);
  attr.dest_qp_num = peer.qpn;
  attr.rq_psn = peer.psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 1;
  std::memcpy(&attr.ah_attr.grh.dgid, peer.gid, 16);
  attr.ah_attr.grh.sgid_index = gid_index;
  attr.ah_attr.grh.hop_limit = 1;
  attr.ah_attr.dlid = peer.lid;
  attr.ah_attr.port_num = kPort;
  if (int rc = ibv_modify_qp(v.qp, &attr,
                             IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                 IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                 IBV_QP_MAX_DEST_RD_ATOMIC |
                                 IBV_QP_MIN_RNR_TIMER))
    fail("QP RTR", rc);

  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = mine.psn;
  attr.max_rd_atomic = 1;
  if (int rc = ibv_modify_qp(v.qp, &attr,
                             IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                 IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                 IBV_QP_MAX_QP_RD_ATOMIC))
    fail("QP RTS", rc);
}

void post_write(verbs_t &v, uint64_t *source, size_t source_offset,
                size_t remote_slot, const peer_info_t &peer, bool signaled) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(source + source_offset);
  sge.length = sizeof(uint64_t);
  sge.lkey = v.source_mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id = source_offset + 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = peer.address + remote_slot * sizeof(uint64_t);
  wr.wr.rdma.rkey = peer.rkey;
  ibv_send_wr *bad = nullptr;
  if (int rc = ibv_post_send(v.qp, &wr, &bad))
    fail("ibv_post_send", rc);
}

void wait_send_completion(verbs_t &v) {
  ibv_wc wc{};
  int n;
  do {
    n = ibv_poll_cq(v.cq, 1, &wc);
  } while (n == 0);
  if (n < 0)
    fail("ibv_poll_cq", EIO);
  if (wc.status != IBV_WC_SUCCESS) {
    std::fprintf(stderr, "send completion failed: %s\n",
                 ibv_wc_status_str(wc.status));
    std::abort();
  }
}

double run(const config_t &cfg, int iters, int rank, verbs_t &v,
           const peer_info_t &peer, const readiness_bench::buffers_t &b,
           const readiness_bench::host_notify_t &notify) {
  const bool gpu_receiver = cfg.receiver == receiver_t::gpu;
  readiness_bench::prepare_buffers(b, iters);
  if (gpu_receiver)
    notify.reset();
  MPI_Barrier(MPI_COMM_WORLD);
  if (gpu_receiver)
    readiness_bench::launch_poll_target(b, rank, iters, notify);

  const int wait_slots = rank == 0 ? 1 : cfg.depth;
  auto wait_epoch = [&](uint64_t epoch) {
    if (gpu_receiver) {
      while (notify.load() < epoch) {
      }
    } else {
      readiness_bench::wait_target_cpu(b.target, wait_slots, epoch, [] {});
    }
  };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    uint64_t epoch = (uint64_t)i + 1;
    if (rank == 0) {
      for (int slot = 0; slot < cfg.depth; ++slot)
        post_write(v, b.source, (size_t)i * cfg.depth + slot, slot, peer,
                   slot + 1 == cfg.depth);
      wait_send_completion(v);
      wait_epoch(epoch);
    } else {
      wait_epoch(epoch);
      post_write(v, b.source, (size_t)i * cfg.depth, 0, peer, true);
      wait_send_completion(v);
    }
  }
  if (gpu_receiver)
    READINESS_HIP_CHECK(hipDeviceSynchronize());
  auto stop = std::chrono::steady_clock::now();
  MPI_Barrier(MPI_COMM_WORLD);
  readiness_bench::validate_target(b, rank, iters);
  return std::chrono::duration<double>(stop - start).count();
}

int parse_args(int argc, char **argv, config_t &cfg) {
  CLI::App app{"Raw ibverbs single-WRITE readiness benchmark"};
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
  app.add_flag("--dmabuf", cfg.dmabuf,
               "Register GPU buffers through exported DMA-BUF file descriptors")
      ->excludes("--cpu-memory");
  app.add_option("--ib-device", cfg.device, "RDMA device, e.g. mlx5_0")
      ->required();
  app.add_option("--gid-index", cfg.gid_index,
                 "GID index: typically 3 for RoCEv2, 0 for InfiniBand")
      ->default_val(3)
      ->check(CLI::NonNegativeNumber);
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e) == 0 ? -1 : 2;
  }
  cfg.receiver = receiver == "cpu" ? receiver_t::cpu : receiver_t::gpu;
  if (cfg.cpu_memory && cfg.receiver != receiver_t::cpu)
    return readiness_bench::usage_error("--cpu-memory requires --receiver cpu");
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  config_t cfg;
  int parse_rc = parse_args(argc, argv, cfg);
  if (parse_rc == 0 && nranks != 2)
    parse_rc = readiness_bench::usage_error("benchmark requires two ranks");
  if (parse_rc != 0) {
    MPI_Finalize();
    return parse_rc < 0 ? 0 : parse_rc;
  }

  auto b = readiness_bench::alloc_buffers(std::max(cfg.warmup, cfg.iters),
                                          cfg.depth, cfg.cpu_memory,
                                          cfg.receiver);
  readiness_bench::host_notify_t notify;
  if (cfg.receiver == receiver_t::gpu)
    notify.alloc();

  {
    verbs_t v;
    int ndev = 0;
    v.devices = ibv_get_device_list(&ndev);
    ibv_device *selected = nullptr;
    for (int i = 0; i < ndev; ++i) {
      if (cfg.device == ibv_get_device_name(v.devices[i]))
        selected = v.devices[i];
    }
    if (!selected) {
      std::fprintf(stderr, "IB device %s not found\n", cfg.device.c_str());
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    v.context = ibv_open_device(selected);
    if (!v.context)
      fail("ibv_open_device");
    v.pd = ibv_alloc_pd(v.context);
    if (!v.pd)
      fail("ibv_alloc_pd");
    // Plain WRITEs consume no receive WQEs, so one CQ serves the QP.
    const int send_slots = cfg.depth + 16;
    v.cq = ibv_create_cq(v.context, send_slots, nullptr, nullptr, 0);
    if (!v.cq)
      fail("ibv_create_cq");
    ibv_qp_init_attr init{};
    init.send_cq = v.cq;
    init.recv_cq = v.cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = send_slots;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    v.qp = ibv_create_qp(v.pd, &init);
    if (!v.qp)
      fail("ibv_create_qp");
    v.source_mr = register_buffer(v, b.source, b.source_bytes(),
                                  IBV_ACCESS_LOCAL_WRITE, cfg.dmabuf,
                                  &v.source_fd);
    v.target_mr = register_buffer(
        v, b.target, b.target_bytes(),
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE, cfg.dmabuf,
        &v.target_fd);

    ibv_port_attr port{};
    if (int rc = ibv_query_port(v.context, kPort, &port))
      fail("ibv_query_port", rc);
    ibv_gid gid{};
    if (int rc = ibv_query_gid(v.context, kPort, cfg.gid_index, &gid))
      fail("ibv_query_gid", rc);
    peer_info_t mine{};
    mine.address = reinterpret_cast<uintptr_t>(b.target);
    mine.rkey = v.target_mr->rkey;
    mine.qpn = v.qp->qp_num;
    mine.psn = 0x123400 + (uint32_t)rank; // arbitrary distinct start PSNs
    mine.lid = port.lid;
    mine.mtu = (uint8_t)port.active_mtu;
    std::memcpy(mine.gid, &gid, 16);
    peer_info_t all[2]{};
    MPI_Allgather(&mine, sizeof(mine), MPI_BYTE, all, sizeof(mine), MPI_BYTE,
                  MPI_COMM_WORLD);
    const peer_info_t &peer = all[1 - rank];
    connect_qp(v, mine, peer, cfg.gid_index);

    if (cfg.warmup > 0)
      run(cfg, cfg.warmup, rank, v, peer, b, notify);
    double seconds = run(cfg, cfg.iters, rank, v, peer, b, notify);
    if (rank == 0) {
      std::string fields =
          std::string("backend=ibverbs op=IBV_WR_RDMA_WRITE source=") +
          readiness_bench::memory_name(b.source_memory) +
          " target=" + readiness_bench::memory_name(b.target_memory) +
          " receiver=" + (cfg.receiver == receiver_t::gpu ? "gpu" : "cpu") +
          " registration=" + (cfg.dmabuf ? "dmabuf" : "direct");
      readiness_bench::print_result(fields, cfg.depth, cfg.iters, seconds);
    }
  } // verbs resources are released before their buffers

  notify.release();
  readiness_bench::free_buffers(b);
  MPI_Finalize();
  return 0;
}
