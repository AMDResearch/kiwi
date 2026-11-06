// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <unistd.h>

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,  \
                         __LINE__, hipGetErrorString(err));               \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

#undef MPI_CHECK
#define MPI_CHECK(call)                                                   \
    do {                                                                  \
        int err = (call);                                                 \
        if (err != MPI_SUCCESS) {                                         \
            char message[MPI_MAX_ERROR_STRING] = {};                      \
            int length = 0;                                               \
            MPI_Error_string(err, message, &length);                      \
            std::fprintf(stderr, "MPI error at %s:%d: %.*s\n",           \
                         __FILE__, __LINE__, length, message);            \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace ep_copy_utils {

constexpr size_t kHostnameBytes = 256;
constexpr size_t kPciBusIdBytes = 64;

struct bootstrap_t {
    bool mpi_initialized = false;
    int rank = 0;
    int peer_rank = -1;
    int device = 0;
};

struct peer_topology_info_t {
    std::array<char, kHostnameBytes> hostname{};
    std::array<char, kPciBusIdBytes> pci_bus_id{};
};

struct peer_ipc_handles_t {
    hipIpcMemHandle_t dst_handle{};
    hipIpcMemHandle_t ready_handle{};
};

struct peer_mapping_t {
    char* dst = nullptr;
    uint32_t* ready = nullptr;
};

inline bool parse_env_int(const char* name, int* value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return false;

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0'
        || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

inline int detect_local_rank(int fallback) {
    constexpr const char* kNames[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "PMI_LOCAL_RANK",
        "LOCAL_RANK",
    };
    for (const char* name : kNames) {
        int value = 0;
        if (parse_env_int(name, &value)) return value;
    }
    return fallback;
}

inline int select_hip_device(int requested, int local_rank) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "bench_ep_copy requires at least one HIP device\n");
        std::abort();
    }

    const int device = requested >= 0 ? requested : local_rank % device_count;
    if (device < 0 || device >= device_count) {
        std::fprintf(stderr,
                     "invalid --hip-device %d (visible devices=%d)\n",
                     device, device_count);
        std::abort();
    }
    HIP_CHECK(hipSetDevice(device));
    return device;
}

inline void require_peer_topology(int device, int rank) {
    peer_topology_info_t local{};
    if (gethostname(local.hostname.data(), local.hostname.size()) != 0) {
        std::fprintf(stderr, "rank %d: gethostname failed: %s\n",
                     rank, std::strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    local.hostname.back() = '\0';
    HIP_CHECK(hipDeviceGetPCIBusId(
        local.pci_bus_id.data(), static_cast<int>(local.pci_bus_id.size()),
        device));
    local.pci_bus_id.back() = '\0';

    std::array<peer_topology_info_t, 2> peers{};
    MPI_CHECK(MPI_Allgather(
        &local, static_cast<int>(sizeof(local)), MPI_BYTE, peers.data(),
        static_cast<int>(sizeof(local)), MPI_BYTE, MPI_COMM_WORLD));

    const char* host0 = peers[0].hostname.data();
    const char* host1 = peers[1].hostname.data();
    const char* bdf0 = peers[0].pci_bus_id.data();
    const char* bdf1 = peers[1].pci_bus_id.data();
    if (std::strcmp(host0, host1) != 0) {
        std::fprintf(stderr,
                     "peer memory target requires both MPI ranks on one host "
                     "(rank 0=%s rank 1=%s)\n",
                     host0, host1);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    if (std::strcmp(bdf0, bdf1) == 0) {
        std::fprintf(stderr,
                     "peer memory target requires distinct GPUs "
                     "(both ranks selected %s)\n",
                     bdf0);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

inline bootstrap_t bootstrap(const std::string& memory_target,
                             int requested_device, int* argc, char*** argv) {
    bootstrap_t result;
    if (memory_target == "peer") {
        MPI_CHECK(MPI_Init(argc, argv));
        result.mpi_initialized = true;

        int world_size = 0;
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &result.rank));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
        if (world_size != 2) {
            if (result.rank == 0) {
                std::fprintf(
                    stderr,
                    "peer memory target requires exactly two MPI ranks "
                    "(got %d)\n",
                    world_size);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            std::abort();
        }
        result.peer_rank = (result.rank + 1) % world_size;
    }

    const int local_rank = detect_local_rank(result.rank);
    result.device = select_hip_device(requested_device, local_rank);
    if (result.mpi_initialized) {
        require_peer_topology(result.device, result.rank);
    }
    return result;
}

inline void shutdown(const bootstrap_t& bootstrap) {
    if (bootstrap.mpi_initialized) MPI_CHECK(MPI_Finalize());
}

inline peer_mapping_t open_peer_mappings(char* d_dst,
                                         uint32_t* local_ready,
                                         int peer_rank) {
    peer_ipc_handles_t local{};
    if (d_dst != nullptr) {
        HIP_CHECK(hipIpcGetMemHandle(&local.dst_handle, d_dst));
    }
    if (local_ready != nullptr) {
        HIP_CHECK(hipIpcGetMemHandle(&local.ready_handle, local_ready));
    }

    std::array<peer_ipc_handles_t, 2> all{};
    MPI_CHECK(MPI_Allgather(
        &local, static_cast<int>(sizeof(local)), MPI_BYTE, all.data(),
        static_cast<int>(sizeof(local)), MPI_BYTE, MPI_COMM_WORLD));

    peer_mapping_t mapping;
    if (d_dst != nullptr) {
        void* dst = nullptr;
        HIP_CHECK(hipIpcOpenMemHandle(
            &dst, all[static_cast<size_t>(peer_rank)].dst_handle,
            hipIpcMemLazyEnablePeerAccess));
        mapping.dst = static_cast<char*>(dst);
    }
    if (local_ready != nullptr) {
        void* ready = nullptr;
        HIP_CHECK(hipIpcOpenMemHandle(
            &ready, all[static_cast<size_t>(peer_rank)].ready_handle,
            hipIpcMemLazyEnablePeerAccess));
        mapping.ready = static_cast<uint32_t*>(ready);
    }
    return mapping;
}

inline void close_peer_mappings(peer_mapping_t& mapping) {
    if (mapping.ready != nullptr) {
        HIP_CHECK(hipIpcCloseMemHandle(mapping.ready));
        mapping.ready = nullptr;
    }
    if (mapping.dst != nullptr) {
        HIP_CHECK(hipIpcCloseMemHandle(mapping.dst));
        mapping.dst = nullptr;
    }
}

}  // namespace ep_copy_utils
