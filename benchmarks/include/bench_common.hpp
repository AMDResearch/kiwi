// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <pthread.h>
#include <vector>

// ---------------------------------------------------------------------------
// Scratch guard for push/invoke kernels
// ---------------------------------------------------------------------------
// A payload placed in scratch makes each push wait on the previous push's
// host-write acknowledgement (scratch and host stores share vmcnt), which
// roughly halves message rate. Every push path keeps its payload in
// registers, LDS or global memory; fail loudly if a kernel regresses.
template<typename Kernel>
inline void require_no_scratch(Kernel kernel, const char* name) {
    hipFuncAttributes attr{};
    hipError_t err = hipFuncGetAttributes(
        &attr, reinterpret_cast<const void*>(kernel));
    if (err != hipSuccess) {
        fprintf(stderr, "hipFuncGetAttributes(%s) failed: %s\n", name,
                hipGetErrorString(err));
        exit(1);
    }
    if (attr.localSizeBytes != 0) {
        fprintf(stderr, "%s uses %zu bytes of scratch per lane; "
                "its push payload must stay in registers or memory\n",
                name, attr.localSizeBytes);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// GPU clock frequency query via sysfs
// ---------------------------------------------------------------------------
struct GpuClockInfo { uint32_t current_mhz; uint32_t max_mhz; };

inline GpuClockInfo get_gpu_clock(int device) {
    GpuClockInfo info = {0, 0};
    char pci_id[64];
    if (hipDeviceGetPCIBusId(pci_id, sizeof(pci_id), device) != hipSuccess)
        return info;
    for (char* p = pci_id; *p; p++)
        if (*p >= 'A' && *p <= 'F') *p += 32;

    char path[256];
    snprintf(path, sizeof(path),
             "/sys/bus/pci/devices/%s/pp_dpm_sclk", pci_id);
    FILE* f = fopen(path, "r");
    if (!f) return info;

    char line[64];
    while (fgets(line, sizeof(line), f)) {
        unsigned idx, mhz;
        if (sscanf(line, "%u: %uMhz", &idx, &mhz) == 2) {
            if (mhz > info.max_mhz) info.max_mhz = mhz;
            if (strstr(line, "*")) info.current_mhz = mhz;
        }
    }
    fclose(f);
    return info;
}

// ---------------------------------------------------------------------------
// GPU clock warmup: MI300X idles at 131 MHz, needs ~3s to reach 2100 MHz.
// Polls sysfs to detect when clock reaches 90% of max, skips if already warm.
// ---------------------------------------------------------------------------
__global__ void gpu_clock_warmup_kernel(int niters) {
    uint64_t x = threadIdx.x;
    for (int i = 0; i < niters; i++)
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    if (x == 0) asm volatile("");
}

inline void gpu_clock_warmup() {
    int device;
    (void)hipGetDevice(&device);
    GpuClockInfo clk = get_gpu_clock(device);

    if (clk.max_mhz == 0) {
        fprintf(stderr, "# GPU clock: sysfs unavailable, blind warmup (~3s)\n");
        gpu_clock_warmup_kernel<<<1, 64>>>(40000000);
        (void)hipDeviceSynchronize();
        return;
    }

    uint32_t target = clk.max_mhz * 90 / 100;

    if (clk.current_mhz >= target) {
        fprintf(stderr, "# GPU %d clock: %u/%u MHz, warmup skipped\n",
                device, clk.current_mhz, clk.max_mhz);
        return;
    }

    fprintf(stderr, "# GPU %d clock: %u MHz (max %u), warming up...\n",
            device, clk.current_mhz, clk.max_mhz);

    hipStream_t stream;
    (void)hipStreamCreate(&stream);
    gpu_clock_warmup_kernel<<<1, 64, 0, stream>>>(80000000);

    for (int i = 0; i < 100; i++) {
        usleep(100000);
        clk = get_gpu_clock(device);
        if (clk.current_mhz >= target) {
            fprintf(stderr, "# GPU %d clock: %u MHz reached in %.1fs\n",
                    device, clk.current_mhz, (i + 1) * 0.1);
            break;
        }
    }

    (void)hipStreamSynchronize(stream);
    (void)hipStreamDestroy(stream);

    if (clk.current_mhz < target)
        fprintf(stderr, "# Warning: GPU %d stuck at %u MHz (target %u)\n",
                device, clk.current_mhz, target);
}

// Check CPU-GPU NUMA affinity and warn if mismatched or unpinned.
// Call after hipSetDevice / HIP context creation.
inline void check_numa_affinity() {
    int device;
    if (hipGetDevice(&device) != hipSuccess) return;

    // Get GPU PCI bus ID (format: "0000:0c:00.0")
    char pciBusId[64];
    if (hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device) != hipSuccess) return;
    for (char* p = pciBusId; *p; p++) *p = tolower(*p);

    // Read GPU NUMA node from sysfs
    char path[256];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", pciBusId);
    FILE* f = fopen(path, "r");
    if (!f) return;
    int gpu_numa = -1;
    if (fscanf(f, "%d", &gpu_numa) != 1) { fclose(f); return; }
    fclose(f);
    if (gpu_numa < 0) return;

    // Check if current CPU is on the GPU's NUMA node
    int cpu = sched_getcpu();
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpu%d", gpu_numa, cpu);
    struct stat st;
    bool on_correct_numa = (stat(path, &st) == 0);

    // Check if process is pinned (affinity mask restricted)
    cpu_set_t mask;
    bool pinned = false;
    int affinity_count = 0;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        affinity_count = CPU_COUNT(&mask);
        // Consider "pinned" if affinity is restricted to fewer than all CPUs
        long total_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        pinned = (affinity_count < total_cpus);
    }

    if (!on_correct_numa) {
        fprintf(stderr,
            "WARNING: CPU %d is not on GPU %d's NUMA node %d. "
            "Cross-NUMA adds ~2 us latency.\n"
            "  Fix: taskset -c <core-on-node-%d> ./benchmark\n\n",
            cpu, device, gpu_numa, gpu_numa);
    } else if (!pinned) {
        fprintf(stderr,
            "WARNING: Process not pinned — may migrate to wrong NUMA node during run.\n"
            "  Fix: taskset -c %d ./benchmark\n\n",
            cpu);
    }
}

inline std::vector<int> worker_cpu_plan(int num_workers) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        fprintf(stderr, "ERROR: failed to read worker CPU affinity: %s\n",
                strerror(errno));
        std::abort();
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
    if (static_cast<int>(cpus.size()) < num_workers) {
        fprintf(stderr,
                "ERROR: throughput benchmark needs %d distinct worker CPUs, "
                "but its affinity mask contains %zu\n",
                num_workers, cpus.size());
        std::abort();
    }
    cpus.resize(static_cast<size_t>(num_workers));
    return cpus;
}

inline int coordinator_cpu_plan(const std::vector<int>& worker_cpus) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        fprintf(stderr, "ERROR: failed to read coordinator CPU affinity: %s\n",
                strerror(errno));
        std::abort();
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        bool used_by_worker = false;
        for (int worker_cpu : worker_cpus) {
            if (worker_cpu == cpu) {
                used_by_worker = true;
                break;
            }
        }
        if (!used_by_worker) return cpu;
    }
    return -1;
}

inline void pin_current_thread_to_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    int result = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    if (result != 0) {
        fprintf(stderr, "ERROR: failed to pin worker to CPU %d: %s\n",
                cpu, strerror(result));
        std::abort();
    }
}

inline void print_worker_cpu_plan(const std::vector<int>& cpus) {
    fprintf(stderr, "# worker_cpus:");
    for (int cpu : cpus) fprintf(stderr, " %d", cpu);
    fprintf(stderr, "\n");
}
