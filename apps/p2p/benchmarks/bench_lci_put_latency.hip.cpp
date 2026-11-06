// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// CPU-driven LCI put ping-pong latency benchmark.
//
// This is a transport-only baseline for bench_latency: it uses the same LCI
// put + local/remote completion protocol, but issues LCI calls directly from
// the CPU instead of routing them through a GPU->CPU invoke proxy callback.

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <CLI/CLI.hpp>
#include "comp_counter.hip.hpp"
#include "lci.hpp"

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(err));                    \
            abort();                                                      \
        }                                                                 \
    } while (0)

struct config_t {
    int warmup_iters = 10000;
    int measure_iters = 100000;
    size_t msg_size = 8;
    bool use_cpu_mem = false;
    bool scalar_host_value = false;
};

static void post_put_retry(int peer, void* buf, size_t size,
                           lci::comp_t comp, lci::rcomp_t rcomp,
                           lci::device_t device, lci::mr_t mr,
                           lci::rmr_t peer_rmr) {
    lci::status_t status;
    do {
        status = lci::post_put_x(peer, buf, size, comp, 0, peer_rmr)
            .device(device)
            .mr(mr)
            .remote_comp(rcomp)
            .allow_done(false)();
        lci::progress_x().device(device)();
    } while (status.is_retry());
}

static void post_scalar_retry(int peer, uint64_t* value,
                              lci::comp_t comp, lci::device_t device,
                              lci::rmr_t peer_rmr) {
    lci::status_t status;
    do {
        status = lci::post_put_x(peer, value, sizeof(*value), comp, 0, peer_rmr)
            .device(device)
            .allow_done(false)();
        lci::progress_x().device(device)();
    } while (status.is_retry());
}

static void wait_counter(lci::device_t device, lci::comp_t comp,
                         uint64_t target) {
    while (kiwi::p2p::get_counter(comp) < target) {
        lci::progress_x().device(device)();
    }
}

static void wait_signal(lci::device_t device, volatile uint64_t* signal,
                        uint64_t target) {
    while (*signal < target) {
        lci::progress_x().device(device)();
    }
}

static void run_ping_pong(int niters, int rank, int peer, bool loopback,
                          void* buf, size_t size, lci::comp_t comp,
                          lci::rcomp_t rcomp, lci::device_t device,
                          lci::mr_t mr, lci::rmr_t peer_rmr) {
    for (int i = 0; i < niters; i++) {
        if (loopback) {
            post_put_retry(peer, buf, size, comp, rcomp, device, mr, peer_rmr);
            wait_counter(device, comp, 2ULL * (i + 1));
        } else if (rank == 0) {
            post_put_retry(peer, buf, size, comp, rcomp, device, mr, peer_rmr);
            wait_counter(device, comp, 2ULL * (i + 1));
        } else {
            wait_counter(device, comp, 2ULL * i + 1);
            post_put_retry(peer, buf, size, comp, rcomp, device, mr, peer_rmr);
        }
    }
    if (!loopback && rank == 1) {
        wait_counter(device, comp, 2ULL * niters);
    }
}

static void run_scalar_ping_pong(int niters, int rank, int peer, bool loopback,
                                 uint64_t* value, volatile uint64_t* signal,
                                 lci::comp_t comp, lci::device_t device,
                                 lci::rmr_t peer_rmr) {
    for (int i = 0; i < niters; i++) {
        uint64_t expected = (uint64_t)(i + 1);
        *value = expected;
        if (loopback || rank == 0) {
            post_scalar_retry(peer, value, comp, device, peer_rmr);
            wait_counter(device, comp, expected);
            wait_signal(device, signal, expected);
        } else {
            wait_signal(device, signal, expected);
            post_scalar_retry(peer, value, comp, device, peer_rmr);
            wait_counter(device, comp, expected);
        }
    }
}

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{"CPU-driven LCI put ping-pong latency benchmark"};
    app.add_option("--warmup", cfg.warmup_iters, "Warmup iterations")->default_val(10000);
    app.add_option("--iters", cfg.measure_iters, "Measurement iterations")->default_val(100000);
    app.add_option("--size", cfg.msg_size, "Message size in bytes")->default_val(8);
    app.add_flag("--cpu-mem", cfg.use_cpu_mem, "Use CPU memory instead of GPU memory");
    app.add_flag("--scalar-host-value", cfg.scalar_host_value,
                 "Match KIWI post_p proxy path: put a host uint64_t value and poll target memory");
    CLI11_PARSE(app, argc, argv);

    if (cfg.scalar_host_value && cfg.use_cpu_mem) {
        fprintf(stderr,
                "--scalar-host-value cannot be combined with --cpu-mem: "
                "the scalar baseline must keep the receive/signal slot in GPU memory\n");
        return 1;
    }
    lci::g_runtime_init();

    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    if (nranks != 1 && nranks != 2) {
        fprintf(stderr, "bench_lci_put_latency requires 1 or 2 ranks (got %d)\n", nranks);
        return 1;
    }
    bool loopback = (nranks == 1);
    int peer = loopback ? rank : 1 - rank;

    if (cfg.scalar_host_value) {
        // Match the host-side LCI operation used by KIWI post_p: the source
        // scalar is host memory, while the symmetric receive/signal slot stays
        // in GPU memory.
        cfg.msg_size = sizeof(uint64_t);
    }

    size_t alloc_size = std::max<size_t>(cfg.msg_size, sizeof(uint64_t));
    void* buf;
    if (cfg.scalar_host_value) {
        HIP_CHECK(hipExtMallocWithFlags(&buf, alloc_size,
                                        hipDeviceMallocFinegrained));
        HIP_CHECK(hipMemset(buf, 0, alloc_size));
    } else if (cfg.use_cpu_mem) {
        buf = calloc(1, alloc_size);
        if (!buf) {
            fprintf(stderr, "calloc failed\n");
            return 1;
        }
    } else {
        HIP_CHECK(hipMalloc(&buf, alloc_size));
        HIP_CHECK(hipMemset(buf, 0, alloc_size));
    }

    size_t npackets = lci::get_default_packet_pool().get_attr_npackets();
    size_t max_nrecvs = std::min(npackets / 4, 4096UL);
    size_t max_nsends = std::min(npackets / 4 / (size_t)nranks, 64UL);
    max_nsends = std::max(max_nsends, 8UL);
    lci::device_t device = lci::alloc_device_x()
        .net_max_sends(max_nsends)
        .net_max_recvs(max_nrecvs)();

    lci::mr_t mr = lci::register_memory_x(buf, alloc_size).device(device)();
    lci::rmr_t my_rmr = lci::get_rmr(mr);
    std::vector<lci::rmr_t> rmrs(nranks);
    lci::allgather_x(&my_rmr, rmrs.data(), sizeof(lci::rmr_t))
        .device(device)();
    lci::rmr_t peer_rmr = rmrs[peer];

    lci::comp_t comp = kiwi::p2p::alloc_counter(0);
    lci::rcomp_t rcomp = lci::register_rcomp(comp);

    lci::barrier_x().device(device)();

    uint64_t scalar_value_storage = 0;
    uint64_t* scalar_value =
        cfg.scalar_host_value ? &scalar_value_storage : reinterpret_cast<uint64_t*>(buf);
    uint64_t* scalar_signal_storage = reinterpret_cast<uint64_t*>(buf);
    volatile uint64_t* scalar_signal = scalar_signal_storage;
    kiwi::p2p::set_counter(comp, 0);
    if (cfg.scalar_host_value) {
        *scalar_signal_storage = 0;
        run_scalar_ping_pong(cfg.warmup_iters, rank, peer, loopback,
                             scalar_value, scalar_signal,
                             comp, device, peer_rmr);
    } else {
        run_ping_pong(cfg.warmup_iters, rank, peer, loopback, buf, cfg.msg_size,
                      comp, rcomp, device, mr, peer_rmr);
    }

    kiwi::p2p::set_counter(comp, 0);
    if (cfg.scalar_host_value) {
        *scalar_signal_storage = 0;
    }
    lci::barrier_x().device(device)();
    auto start = std::chrono::high_resolution_clock::now();
    if (cfg.scalar_host_value) {
        run_scalar_ping_pong(cfg.measure_iters, rank, peer, loopback,
                             scalar_value, scalar_signal,
                             comp, device, peer_rmr);
    } else {
        run_ping_pong(cfg.measure_iters, rank, peer, loopback, buf, cfg.msg_size,
                      comp, rcomp, device, mr, peer_rmr);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    lci::barrier_x().device(device)();

    if (rank == 0) {
        double elapsed_us =
            std::chrono::duration<double, std::micro>(stop - start).count();
        double rtt_us = elapsed_us / cfg.measure_iters;
        double latency_us = loopback ? rtt_us : rtt_us / 2.0;
        printf("# CPU-driven LCI put %s latency benchmark\n",
               loopback ? "loopback" : "ping-pong");
        printf("# warmup=%d, iters=%d, size=%zu, mem=%s, mode=%s, signal_poll=%s\n",
               cfg.warmup_iters, cfg.measure_iters, cfg.msg_size,
               cfg.use_cpu_mem ? "cpu" : "gpu",
               cfg.scalar_host_value ? "scalar-host-value" : "put",
               cfg.scalar_host_value ? "host-load" : "n/a");
        if (cfg.scalar_host_value) {
            printf("# scalar_source=host signal_mem=fine-grained-device\n");
        }
        printf("%-16s %12.2f us\n", "rtt", rtt_us);
        printf("%-16s %12.2f us\n", "latency", latency_us);
    }

    lci::deregister_rcomp(rcomp);
    kiwi::p2p::free_counter(comp);
    lci::deregister_memory(&mr);
    lci::free_device(&device);
    lci::barrier();
    if (cfg.use_cpu_mem)
        free(buf);
    else
        HIP_CHECK(hipFree(buf));
    lci::g_runtime_fina();

    return 0;
}
