// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "lci.hpp"

namespace kiwi::p2p::bench {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    asm volatile("" ::: "memory");
#endif
}

inline std::vector<int> allowed_cpus() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        std::fprintf(stderr, "ERROR: sched_getaffinity failed: %s\n",
                     std::strerror(errno));
        std::abort();
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    }
    if (cpus.empty()) {
        std::fprintf(stderr, "ERROR: empty CPU affinity mask\n");
        std::abort();
    }
    return cpus;
}

inline std::string cpu_list_to_string(const std::vector<int>& cpus) {
    if (cpus.empty()) return "<empty>";

    std::string out;
    for (size_t i = 0; i < cpus.size();) {
        const int begin = cpus[i];
        int end = begin;
        while (i + 1 < cpus.size() && cpus[i + 1] == end + 1) {
            ++i;
            end = cpus[i];
        }

        if (!out.empty()) out += ",";
        out += std::to_string(begin);
        if (end != begin) {
            out += "-";
            out += std::to_string(end);
        }
        ++i;
    }
    return out;
}

struct worker_affinity_plan_t {
    std::vector<int> allowed_cpus;
    int group_index = 0;
    int group_size = 1;
    bool auto_pin = true;
};

inline worker_affinity_plan_t make_worker_affinity_plan(
    int rank, int nranks, int num_workers) {
    worker_affinity_plan_t plan{};
    plan.allowed_cpus = allowed_cpus();

    constexpr size_t kMaskStringBytes = 1024;
    char host_buf[256] = {0};
    if (gethostname(host_buf, sizeof(host_buf) - 1) != 0) {
        std::snprintf(host_buf, sizeof(host_buf), "host-unknown");
    }
    // Ranks on different nodes can have identical CPU masks but do not share
    // physical CPUs.  Include hostname in the grouping key so multi-node jobs
    // only divide each node-local inherited mask among ranks on that node.
    const std::string my_key = std::string(host_buf) + "|" +
                               cpu_list_to_string(plan.allowed_cpus);
    std::vector<char> send(kMaskStringBytes, 0);
    std::snprintf(send.data(), send.size(), "%s", my_key.c_str());
    std::vector<char> all((size_t)nranks * kMaskStringBytes, 0);
    lci::allgather(send.data(), all.data(), kMaskStringBytes);

    const char* my = all.data() + (size_t)rank * kMaskStringBytes;
    int same_before = 0;
    int same_total = 0;
    for (int r = 0; r < nranks; ++r) {
        const char* other = all.data() + (size_t)r * kMaskStringBytes;
        if (std::strncmp(my, other, kMaskStringBytes) == 0) {
            if (r < rank) ++same_before;
            ++same_total;
        }
    }

    plan.group_index = same_before;
    plan.group_size = std::max(same_total, 1);
    const size_t requested_workers_in_mask =
        (size_t)num_workers * (size_t)plan.group_size;
    if (requested_workers_in_mask > plan.allowed_cpus.size()) {
        plan.auto_pin = false;
    }
    return plan;
}

inline int worker_cpu(const worker_affinity_plan_t& plan, int logical,
                      int num_workers) {
    const size_t ncpus = plan.allowed_cpus.size();
    if (ncpus == 0) {
        std::fprintf(stderr, "ERROR: empty initial CPU affinity mask\n");
        std::abort();
    }
    const size_t per_rank_stride =
        static_cast<size_t>(std::max(num_workers, 1));
    const size_t offset =
        static_cast<size_t>(std::max(plan.group_index, 0)) * per_rank_stride;
    return plan.allowed_cpus[(offset + static_cast<size_t>(logical)) % ncpus];
}

inline std::string worker_cpu_plan_string(const worker_affinity_plan_t& plan,
                                          int thread_core,
                                          int num_workers) {
    if (thread_core < 0 && !plan.auto_pin) {
        return "inherited-mask(no-auto-pin:workers-exceed-mask)";
    }
    std::string out = thread_core >= 0 ? "explicit" : "auto-rank-in-mask";
    out += "[";
    for (int logical = 0; logical < num_workers; ++logical) {
        if (logical != 0) out += ",";
        const int cpu = thread_core >= 0
                            ? thread_core + logical
                            : worker_cpu(plan, logical, num_workers);
        out += std::to_string(cpu);
    }
    out += "]";
    return out;
}

inline void pin_current_thread_to_cpu(int rank_for_logs, int cpu,
                                      const char* role, int logical,
                                      bool verbose) {
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        std::fprintf(stderr,
                     "ERROR: invalid CPU core for rank=%d role=%s logical=%d cpu=%d\n",
                     rank_for_logs, role, logical, cpu);
        std::abort();
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        std::fprintf(stderr,
                     "ERROR: failed to pin rank=%d role=%s logical=%d cpu=%d: %s\n",
                     rank_for_logs, role, logical, cpu, std::strerror(errno));
        std::abort();
    }

    if (verbose) {
        std::fprintf(stderr,
                     "# affinity rank=%d role=%s logical=%d requested_cpu=%d current_cpu=%d\n",
                     rank_for_logs, role, logical, cpu, sched_getcpu());
    }
}

inline void pin_main_thread(int rank_for_logs, int main_core) {
    if (main_core < 0) return;
    pin_current_thread_to_cpu(rank_for_logs, main_core, "main/control", 0,
                              true);
}

inline void pin_worker_thread(int rank_for_logs,
                              const worker_affinity_plan_t& plan,
                              int logical, int num_workers, int thread_core,
                              bool verbose_affinity) {
    if (thread_core < 0 && !plan.auto_pin) {
        if (verbose_affinity) {
            std::fprintf(stderr,
                         "# affinity rank=%d role=lci-post logical=%d mode=inherited-mask current_cpu=%d\n",
                         rank_for_logs, logical, sched_getcpu());
        }
        return;
    }

    const int cpu = thread_core >= 0 ? thread_core + logical
                                     : worker_cpu(plan, logical, num_workers);
    pin_current_thread_to_cpu(rank_for_logs, cpu, "lci-post", logical,
                              verbose_affinity || thread_core >= 0);
}

}  // namespace kiwi::p2p::bench
