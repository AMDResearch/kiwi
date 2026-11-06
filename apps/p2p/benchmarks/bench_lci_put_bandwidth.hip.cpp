// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Direct LCI CPU-threaded post_put bandwidth benchmark.
//
// This is a transport-only bandwidth baseline for bench_bandwidth.  It keeps
// the same two-rank unidirectional, bounded-window shape, but it issues LCI
// calls directly from CPU threads instead of routing requests through KIWI or
// invoke proxy callbacks.
//
// Semantics:
// - exactly two ranks, rank 0 sends data to rank 1;
// - sender CPU threads emulate KIWI bandwidth workgroups;
// - by default, --devices -1 allocates one LCI device per thread;
// - total --depth is a bounded outstanding data-PUT window split across
//   threads;
// - data PUTs use strided remote offsets over depth * message_size;
// - rank 1 runs one receiver/progress thread per sender thread, waits on a
//   per-thread remote-completion counter, then sends a small ACK PUT back;
// - rank 0 waits for both the per-thread ACK and local completions before slot
//   reuse.

#include <hip/hip_runtime.h>
#include <CLI/CLI.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sched.h>
#include <thread>
#include <vector>

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
    int num_threads = 8;
    int num_devices = -1;  // -1 = num_threads
    int depth = 1024;      // total outstanding data PUTs per iteration
    int warmup_iters = 100;
    int measure_iters = 1000;
    size_t msg_size = 8192;
    bool use_cpu_mem = false;
    int main_core = -1;
    int thread_core = -1;
    bool verbose_affinity = false;
};

struct local_comp_t : public lci::comp_impl_t {
    std::atomic<uint64_t> count{0};

    void signal(lci::status_t) override {
        count.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t get() const { return count.load(std::memory_order_relaxed); }
};

static int g_rank_for_logs = -1;
static std::vector<int> g_initial_allowed_cpus;

static std::vector<int> allowed_cpus() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        fprintf(stderr, "ERROR: sched_getaffinity failed: %s\n",
                std::strerror(errno));
        abort();
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    }
    if (cpus.empty()) {
        fprintf(stderr, "ERROR: empty CPU affinity mask\n");
        abort();
    }
    return cpus;
}

static void pin_current_thread_to_cpu(int cpu, const char* role, int logical,
                                      bool verbose) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        fprintf(stderr,
                "ERROR: invalid CPU core for rank=%d role=%s logical=%d cpu=%d\n",
                g_rank_for_logs, role, logical, cpu);
        abort();
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        fprintf(stderr,
                "ERROR: failed to pin rank=%d role=%s logical=%d cpu=%d: %s\n",
                g_rank_for_logs, role, logical, cpu, std::strerror(errno));
        abort();
    }

    if (verbose) {
        fprintf(stderr,
                "# affinity rank=%d role=%s logical=%d requested_cpu=%d current_cpu=%d\n",
                g_rank_for_logs, role, logical, cpu, sched_getcpu());
    }
}

static void pin_main_thread(int main_core) {
    if (main_core < 0) return;
    pin_current_thread_to_cpu(main_core, "main/control", 0, true);
}

static void pin_worker_thread(int logical, const char* role,
                              const config_t& cfg) {
    int cpu = cfg.thread_core >= 0
                  ? cfg.thread_core + logical
                  : g_initial_allowed_cpus[(size_t)logical %
                                           g_initial_allowed_cpus.size()];
    pin_current_thread_to_cpu(cpu, role, logical,
                              cfg.verbose_affinity || cfg.thread_core >= 0);
}

static void post_put_retry(int peer, void* src, size_t size, lci::comp_t comp,
                           size_t remote_disp, lci::rmr_t rmr,
                           lci::device_t device, lci::mr_t mr,
                           lci::rcomp_t rcomp) {
    lci::status_t status;
    do {
        status = lci::post_put_x(peer, src, size, comp, remote_disp, rmr)
                     .device(device)
                     .mr(mr)
                     .remote_comp(rcomp)
                     .allow_done(false)();
        if (status.is_retry()) lci::progress_x().device(device)();
    } while (status.is_retry());
}

static void post_put_retry_no_rcomp(int peer, void* src, size_t size,
                                    lci::comp_t comp, size_t remote_disp,
                                    lci::rmr_t rmr, lci::device_t device,
                                    lci::mr_t mr) {
    lci::status_t status;
    do {
        status = lci::post_put_x(peer, src, size, comp, remote_disp, rmr)
                     .device(device)
                     .mr(mr)
                     .allow_done(false)();
        if (status.is_retry()) lci::progress_x().device(device)();
    } while (status.is_retry());
}

static void exchange_rmr_per_device(const std::vector<lci::device_t>& devices,
                                    const std::vector<lci::mr_t>& mrs,
                                    int nranks, int peer,
                                    std::vector<lci::rmr_t>* peer_rmrs) {
    peer_rmrs->resize(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        lci::rmr_t my_rmr = lci::get_rmr(mrs[i]);
        std::vector<lci::rmr_t> all_rmrs((size_t)nranks);
        lci::allgather_x(&my_rmr, all_rmrs.data(), sizeof(lci::rmr_t))
            .device(devices[i])();
        (*peer_rmrs)[i] = all_rmrs[(size_t)peer];
    }
}

static double run_once(const config_t& cfg, int rank, int peer,
                       int num_devices, int depth_per_thread, void* data_buf,
                       volatile uint64_t* ack_buf,
                       const std::vector<lci::device_t>& devices,
                       const std::vector<lci::mr_t>& data_mrs,
                       const std::vector<lci::rmr_t>& peer_data_rmrs,
                       const std::vector<lci::mr_t>& ack_mrs,
                       const std::vector<lci::rmr_t>& peer_ack_rmrs,
                       const std::vector<lci::comp_t>& remote_comps,
                       const std::vector<lci::rcomp_t>& peer_data_rcomps,
                       int iters, bool print_result) {
    auto* ack_slots = const_cast<uint64_t*>(ack_buf);
    for (int t = 0; t < cfg.num_threads; ++t) {
        kiwi::p2p::set_counter(remote_comps[(size_t)t], 0);
        ack_slots[t] = 0;
    }

    lci::barrier();
    auto start = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::vector<std::thread> threads;
        threads.reserve((size_t)cfg.num_threads);
        for (int t = 0; t < cfg.num_threads; ++t) {
            threads.emplace_back([&, t] {
                pin_worker_thread(t, "sender", cfg);
                int device_index = t % num_devices;
                lci::device_t device = devices[(size_t)device_index];
                local_comp_t local_impl;
                lci::comp_t local_comp(&local_impl);
                uint64_t posted = 0;

                for (int iter = 0; iter < iters; ++iter) {
                    for (int d = 0; d < depth_per_thread; ++d) {
                        size_t slot = (size_t)t +
                                      (size_t)d * (size_t)cfg.num_threads;
                        size_t offset = slot * cfg.msg_size;
                        void* src = static_cast<char*>(data_buf) + offset;
                        post_put_retry(peer, src, cfg.msg_size, local_comp,
                                       offset, peer_data_rmrs[(size_t)device_index],
                                       device, data_mrs[(size_t)device_index],
                                       peer_data_rcomps[(size_t)t]);
                        ++posted;
                    }

                    uint64_t want_ack = (uint64_t)(iter + 1);
                    while (ack_buf[t] < want_ack || local_impl.get() < posted) {
                        lci::progress_x().device(device)();
                    }
                }
            });
        }
        for (auto& thread : threads) thread.join();
    } else {
        std::vector<std::thread> threads;
        threads.reserve((size_t)cfg.num_threads);
        for (int t = 0; t < cfg.num_threads; ++t) {
            threads.emplace_back([&, t] {
                pin_worker_thread(t, "receiver-progress", cfg);
                int device_index = t % num_devices;
                lci::device_t device = devices[(size_t)device_index];
                local_comp_t ack_impl;
                lci::comp_t ack_comp(&ack_impl);

                for (int iter = 0; iter < iters; ++iter) {
                    uint64_t want_remote =
                        (uint64_t)(iter + 1) * (uint64_t)depth_per_thread;
                    while (kiwi::p2p::get_counter(remote_comps[(size_t)t]) <
                           want_remote) {
                        lci::progress_x().device(device)();
                    }

                    uint64_t ack_value = (uint64_t)(iter + 1);
                    ack_slots[t] = ack_value;
                    post_put_retry_no_rcomp(
                        peer, ack_slots + t, sizeof(uint64_t), ack_comp,
                        (size_t)t * sizeof(uint64_t),
                        peer_ack_rmrs[(size_t)device_index], device,
                        ack_mrs[(size_t)device_index]);
                    while (ack_impl.get() < (uint64_t)(iter + 1)) {
                        lci::progress_x().device(device)();
                    }
                }
            });
        }
        for (auto& thread : threads) thread.join();
    }

    lci::barrier();
    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    if (rank == 0 && print_result) {
        uint64_t total_msgs = (uint64_t)iters * (uint64_t)cfg.depth;
        double seconds = elapsed_ms / 1e3;
        double mops = (double)total_msgs / seconds / 1e6;
        double gbps =
            (double)total_msgs * (double)cfg.msg_size / seconds / 1e9;
        printf("result size=%zu threads=%d devices=%d depth=%d "
               "depth_per_thread=%d iters=%d messages=%llu time_ms=%.3f "
               "mops=%.3f bw_gbs=%.3f transport=direct-lci-cpu-threaded "
               "op=post_put mode=remote-completion+ack mem=%s\n",
               cfg.msg_size, cfg.num_threads, num_devices, cfg.depth,
               depth_per_thread, iters, (unsigned long long)total_msgs,
               elapsed_ms, mops, gbps, cfg.use_cpu_mem ? "cpu" : "gpu");
        fflush(stdout);
    }
    return elapsed_ms;
}

int main(int argc, char** argv) {
    config_t cfg;
    CLI::App app{"Direct LCI CPU-threaded post_put bandwidth benchmark"};
    app.add_option("--threads", cfg.num_threads,
                   "CPU posting/progress threads")
        ->default_val(8);
    app.add_option("--devices", cfg.num_devices,
                   "LCI devices (-1 = threads)")
        ->default_val(-1);
    app.add_option("--depth", cfg.depth,
                   "Total outstanding data PUTs per iteration")
        ->default_val(1024);
    app.add_option("--warmup", cfg.warmup_iters, "Warmup iterations")
        ->default_val(100);
    app.add_option("--iters", cfg.measure_iters, "Measurement iterations")
        ->default_val(1000);
    app.add_option("--msg-size", cfg.msg_size,
                   "Message size in bytes")
        ->default_val(8192);
    app.add_flag("--cpu-mem", cfg.use_cpu_mem,
                 "Use CPU data buffer instead of GPU data buffer");
    app.add_option("--main-core", cfg.main_core,
                   "Pin the main/control thread to this CPU core")
        ->default_val(-1);
    app.add_option("--thread-core", cfg.thread_core,
                   "Pin posting/progress thread 0 to this CPU core "
                   "(additional threads use consecutive cores)")
        ->default_val(-1);
    app.add_flag("--verbose-affinity", cfg.verbose_affinity,
                 "Log worker-thread CPU affinity even without --thread-core");
    CLI11_PARSE(app, argc, argv);

    if (cfg.num_threads <= 0 || cfg.depth <= 0 || cfg.measure_iters <= 0 ||
        cfg.warmup_iters < 0 || cfg.msg_size == 0) {
        fprintf(stderr,
                "threads, depth, iters, and size must be positive; warmup "
                "must be non-negative\n");
        return 1;
    }
    int num_devices = cfg.num_devices == -1 ? cfg.num_threads : cfg.num_devices;
    if (num_devices <= 0) {
        fprintf(stderr, "devices must be positive or -1\n");
        return 1;
    }
    if (cfg.depth % cfg.num_threads != 0) {
        fprintf(stderr, "depth (%d) must be divisible by threads (%d)\n",
                cfg.depth, cfg.num_threads);
        return 1;
    }
    if (cfg.msg_size > std::numeric_limits<size_t>::max() /
                           (size_t)cfg.depth) {
        fprintf(stderr, "depth * size overflows size_t\n");
        return 1;
    }
    int depth_per_thread = cfg.depth / cfg.num_threads;

    lci::g_runtime_init();
    int rank = lci::get_rank_me();
    int nranks = lci::get_rank_n();
    g_rank_for_logs = rank;
    if (nranks != 2) {
        fprintf(stderr,
                "bench_lci_put_bandwidth requires exactly 2 ranks (got %d)\n",
                nranks);
        lci::g_runtime_fina();
        return 1;
    }
    int peer = 1 - rank;

    g_initial_allowed_cpus = allowed_cpus();
    pin_main_thread(cfg.main_core);

    size_t data_bytes = (size_t)cfg.depth * cfg.msg_size;
    void* data_buf = nullptr;
    if (cfg.use_cpu_mem) {
        data_buf = std::calloc(1, data_bytes);
        if (!data_buf) {
            fprintf(stderr, "calloc data buffer failed\n");
            lci::g_runtime_fina();
            return 1;
        }
    } else {
        HIP_CHECK(hipMalloc(&data_buf, data_bytes));
        HIP_CHECK(hipMemset(data_buf, 0, data_bytes));
    }

    size_t ack_bytes = (size_t)cfg.num_threads * sizeof(uint64_t);
    auto* ack_storage = static_cast<uint64_t*>(std::calloc(1, ack_bytes));
    if (!ack_storage) {
        fprintf(stderr, "calloc ACK buffer failed\n");
        if (cfg.use_cpu_mem)
            std::free(data_buf);
        else
            HIP_CHECK(hipFree(data_buf));
        lci::g_runtime_fina();
        return 1;
    }
    volatile uint64_t* ack_buf = ack_storage;

    size_t npackets = lci::get_default_packet_pool().get_attr_npackets();
    size_t max_nrecvs = std::min(npackets / 4 / (size_t)num_devices, 4096UL);
    size_t max_nsends = std::max(
        std::min(npackets / 4 / (size_t)nranks / (size_t)num_devices,
                 (size_t)std::max(depth_per_thread + 8, 64)),
        8UL);

    std::vector<lci::device_t> devices((size_t)num_devices);
    for (int i = 0; i < num_devices; ++i) {
        devices[(size_t)i] = lci::alloc_device_x()
                                  .net_max_sends(max_nsends)
                                  .net_max_recvs(max_nrecvs)();
    }

    std::vector<lci::mr_t> data_mrs((size_t)num_devices);
    std::vector<lci::mr_t> ack_mrs((size_t)num_devices);
    for (int i = 0; i < num_devices; ++i) {
        data_mrs[(size_t)i] = lci::register_memory_x(data_buf, data_bytes)
                                  .device(devices[(size_t)i])();
        ack_mrs[(size_t)i] = lci::register_memory_x(ack_storage, ack_bytes)
                                 .device(devices[(size_t)i])();
    }

    std::vector<lci::rmr_t> peer_data_rmrs;
    std::vector<lci::rmr_t> peer_ack_rmrs;
    exchange_rmr_per_device(devices, data_mrs, nranks, peer, &peer_data_rmrs);
    exchange_rmr_per_device(devices, ack_mrs, nranks, peer, &peer_ack_rmrs);

    std::vector<lci::comp_t> remote_comps((size_t)cfg.num_threads);
    std::vector<lci::rcomp_t> my_data_rcomps((size_t)cfg.num_threads);
    std::vector<lci::rcomp_t> all_data_rcomps((size_t)nranks *
                                              (size_t)cfg.num_threads);
    for (int t = 0; t < cfg.num_threads; ++t) {
        remote_comps[(size_t)t] = kiwi::p2p::alloc_counter(0);
        my_data_rcomps[(size_t)t] =
            lci::register_rcomp(remote_comps[(size_t)t]);
    }
    lci::allgather_x(my_data_rcomps.data(), all_data_rcomps.data(),
                     (size_t)cfg.num_threads * sizeof(lci::rcomp_t))
        .device(devices[0])();
    std::vector<lci::rcomp_t> peer_data_rcomps((size_t)cfg.num_threads);
    for (int t = 0; t < cfg.num_threads; ++t) {
        peer_data_rcomps[(size_t)t] =
            all_data_rcomps[(size_t)peer * (size_t)cfg.num_threads +
                            (size_t)t];
    }

    if (rank == 0) {
        printf("# Direct LCI CPU-threaded post_put bandwidth benchmark\n");
        printf("# no KIWI/invoke: CPU threads call lci::post_put_x directly\n");
        printf("# flow: rank0 bounded-window data PUTs with remote completion; "
               "rank1 progress threads ACK each iteration with a PUT; rank0 "
               "waits for ACK plus local completions before slot reuse\n");
        printf("# threads=%d devices=%d depth=%d depth_per_thread=%d "
               "warmup=%d iters=%d size=%zu mem=%s data_bytes=%zu "
               "max_nsends=%zu max_nrecvs=%zu initial_allowed_cpus=%zu "
               "main_core=%d thread_core=%d\n",
               cfg.num_threads, num_devices, cfg.depth, depth_per_thread,
               cfg.warmup_iters, cfg.measure_iters, cfg.msg_size,
               cfg.use_cpu_mem ? "cpu" : "gpu", data_bytes, max_nsends,
               max_nrecvs, g_initial_allowed_cpus.size(), cfg.main_core,
               cfg.thread_core);
        fflush(stdout);
    }

    if (cfg.warmup_iters > 0) {
        run_once(cfg, rank, peer, num_devices, depth_per_thread, data_buf,
                 ack_buf, devices, data_mrs, peer_data_rmrs, ack_mrs,
                 peer_ack_rmrs, remote_comps, peer_data_rcomps,
                 cfg.warmup_iters, false);
    }
    run_once(cfg, rank, peer, num_devices, depth_per_thread, data_buf, ack_buf,
             devices, data_mrs, peer_data_rmrs, ack_mrs, peer_ack_rmrs,
             remote_comps, peer_data_rcomps, cfg.measure_iters, true);

    lci::barrier();
    for (auto comp : remote_comps) kiwi::p2p::free_counter(comp);
    for (auto& mr : ack_mrs) lci::deregister_memory(&mr);
    for (auto& mr : data_mrs) lci::deregister_memory(&mr);
    for (auto& device : devices) lci::free_device(&device);
    lci::barrier();

    std::free(ack_storage);
    if (cfg.use_cpu_mem)
        std::free(data_buf);
    else
        HIP_CHECK(hipFree(data_buf));
    lci::g_runtime_fina();
    return 0;
}
