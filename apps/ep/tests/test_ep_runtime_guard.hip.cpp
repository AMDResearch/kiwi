// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "comm/context.hip.hpp"

#include "lci.hpp"

#include <cstdio>
#include <cstdlib>

int main() {
    kiwi::ep::ep_runtime_guard_t runtime_guard;

    const size_t packet_size =
        lci::get_default_packet_pool().get_attr_packet_size();
    const char* packet_size_env = std::getenv("LCI_ATTR_PACKET_SIZE");
    const size_t expected_packet_size = packet_size_env == nullptr
        ? 512
        : std::strtoull(packet_size_env, nullptr, 10);
    if (packet_size != expected_packet_size) {
        std::fprintf(stderr, "expected %zu-byte control packets, got %zu\n",
                     expected_packet_size, packet_size);
        return 1;
    }
    if (lci::get_max_bcopy_size() < 256) {
        std::fprintf(stderr,
                     "control packets cannot carry a 256-byte hostname\n");
        return 1;
    }

    const size_t packet_count =
        lci::get_default_packet_pool().get_attr_npackets();
    const char* packet_count_env = std::getenv("LCI_ATTR_NPACKETS");
    const size_t expected_packet_count = packet_count_env == nullptr
        ? 8192
        : std::strtoull(packet_count_env, nullptr, 10);
    if (packet_count != expected_packet_count) {
        std::fprintf(stderr, "expected %zu control packets, got %zu\n",
                     expected_packet_count, packet_count);
        return 1;
    }
    const size_t max_recvs = lci::get_default_device().get_attr_net_max_recvs();
    const char* max_recvs_env = std::getenv("LCI_ATTR_NET_MAX_RECVS");
    const size_t expected_max_recvs = max_recvs_env == nullptr
        ? 4096
        : std::strtoull(max_recvs_env, nullptr, 10);
    if (max_recvs != expected_max_recvs) {
        std::fprintf(stderr, "expected %zu control receives, got %zu\n",
                     expected_max_recvs, max_recvs);
        return 1;
    }

    lci::barrier();

    lci::device_t device = lci::alloc_device_x()
        .net_max_sends(8)
        .net_max_recvs(0)
        .shm_enable(false)
        .packet_pool(lci::packet_pool_t())();
    const int device_uid = device.get_attr_uid();
    lci::free_device(&device);
    if (device_uid != 1) {
        std::fprintf(stderr,
                     "expected first non-default device UID 1, got %d\n",
                     device_uid);
        return 1;
    }

    return 0;
}
