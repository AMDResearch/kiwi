// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// CPU-only LCI loopback benchmark: CPU calls lci::post_put to self and
// polls lci::progress until completion. No GPU or queue involved.
// Measures pure LCI put + completion overhead (local + remote completion).

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <CLI/CLI.hpp>
#include "comp_counter.hip.hpp"
#include "lci.hpp"

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(err));                     \
            abort();                                                      \
        }                                                                 \
    } while (0)

struct config_t {
    int warmup_iters = 1000;
    int measure_iters = 100000;
    size_t msg_size = 8;
    bool use_cpu_mem = false;
};

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{"CPU-only LCI loopback benchmark (no GPU/queue)"};
    app.add_option("--warmup", cfg.warmup_iters, "Warmup iterations")->default_val(1000);
    app.add_option("--iters", cfg.measure_iters, "Measurement iterations")->default_val(100000);
    app.add_option("--size", cfg.msg_size, "Message size in bytes")->default_val(8);
    app.add_flag("--cpu-mem", cfg.use_cpu_mem, "Use CPU memory instead of GPU memory");
    CLI11_PARSE(app, argc, argv);

    lci::g_runtime_init();

    int rank = lci::get_rank_me();

    // Allocate and register memory
    void* buf;
    if (cfg.use_cpu_mem) {
        buf = calloc(1, cfg.msg_size);
        if (!buf) { fprintf(stderr, "calloc failed\n"); abort(); }
    } else {
        HIP_CHECK(hipMalloc(&buf, cfg.msg_size));
        HIP_CHECK(hipMemset(buf, 0, cfg.msg_size));
    }

    lci::mr_t mr = lci::register_memory(buf, cfg.msg_size);
    lci::rmr_t my_rmr = lci::get_rmr(mr);

    // Allocate a completion counter in managed memory (same as GPU path uses)
    lci::comp_t comp = kiwi::p2p::alloc_counter(0);
    lci::rcomp_t rcomp = lci::register_rcomp(comp);

    lci::device_t device = lci::alloc_device();

    lci::barrier();

    // Warmup
    kiwi::p2p::set_counter(comp, 0);
    for (int i = 0; i < cfg.warmup_iters; i++) {
        lci::status_t status;
        do {
            status = lci::post_put_x(rank, buf, cfg.msg_size, comp, 0, my_rmr)
                .device(device)
                .mr(mr)
                .remote_comp(rcomp)
                .allow_done(false)();
            lci::progress_x().device(device)();
        } while (status.is_retry());

        while (kiwi::p2p::get_counter(comp) < (uint64_t)2 * (i + 1)) {
            lci::progress_x().device(device)();
        }
    }

    // Measurement
    kiwi::p2p::set_counter(comp, 0);

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < cfg.measure_iters; i++) {
        lci::status_t status;
        do {
            status = lci::post_put_x(rank, buf, cfg.msg_size, comp, 0, my_rmr)
                .device(device)
                .mr(mr)
                .remote_comp(rcomp)
                .allow_done(false)();
            lci::progress_x().device(device)();
        } while (status.is_retry());

        while (kiwi::p2p::get_counter(comp) < (uint64_t)2 * (i + 1)) {
            lci::progress_x().device(device)();
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_us = std::chrono::duration<double, std::micro>(wall_end - wall_start).count();
    double avg_us = wall_us / cfg.measure_iters;

    printf("# CPU-only LCI loopback (no GPU/queue)\n");
    printf("# warmup=%d, iters=%d, size=%zu, mem=%s\n",
           cfg.warmup_iters, cfg.measure_iters, cfg.msg_size,
           cfg.use_cpu_mem ? "cpu" : "gpu");
    printf("%-20s %12.2f us\n", "avg latency", avg_us);

    // Cleanup
    lci::free_device(&device);
    lci::barrier();
    kiwi::p2p::free_counter(comp);
    lci::deregister_memory(&mr);
    if (cfg.use_cpu_mem)
        free(buf);
    else
        HIP_CHECK(hipFree(buf));
    lci::g_runtime_fina();

    return 0;
}
