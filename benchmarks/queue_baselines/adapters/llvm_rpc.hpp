// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file llvm_rpc.hpp
 * @brief LLVM libc GPU RPC adapter (libc/shared/rpc.h, unmodified).
 *
 * One "queue" is one rpc::Client / rpc::Server pair over a shared buffer of
 * `capacity` ports. A push is Client::open<opcode>() + Port::send() + close,
 * i.e. a send-only RPC; the consumer is Server::try_open() + Port::recv() +
 * close, which hands the port back to the client. This is the upstream
 * fire-and-forget pattern (e.g. used for asynchronous printf-style calls).
 *
 * Deviations from the LLVM offload runtime, both in the baseline's favor:
 *  - The doorbell pointer is left null, so no s_sendmsg interrupt is raised
 *    and the host server busy-polls instead of sleeping on an HSA signal.
 *  - Only lane 0 opens the port, so a message writes one 64B lane buffer
 *    instead of one buffer per active lane (4 KiB for a full wave).
 */

#pragma once

#include "../qb_common.hpp"
#include "llvm_rpc_server.hpp"

#include <rpc.h>

#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace qb {

struct LlvmRpcAdapter {
    static constexpr const char* kName = "llvm-rpc";
    static constexpr bool kMultiProducer = true;
    // Client::open() scans ports from 0 and Server::try_open() serves the
    // lowest ready port, so per-producer order is not preserved.
    static constexpr bool kFifo = false;
    static constexpr int kDescriptorBytes = sizeof(rpc::Buffer);
    static constexpr uint32_t kOpcode = 0x7100;
    // Packet stride: the server indexes packets by the client wave size.
    static constexpr uint32_t kLaneSize = 64;
    using Handle = rpc::Client*;

    struct Channel {
        void* buffer = nullptr;               // fine-grained host memory
        rpc::Client* client = nullptr;        // device memory (holds port locks)
        std::unique_ptr<LlvmRpcServer> server;  // host (plain C++ TU)
    };

    LlvmRpcAdapter(int num_queues, int capacity) {
        const uint32_t ports = static_cast<uint32_t>(capacity);
        if (ports > rpc::MAX_PORT_COUNT)
            throw std::runtime_error("llvm-rpc: capacity exceeds MAX_PORT_COUNT");
        const uint64_t bytes = rpc::Server::allocation_size(kLaneSize, ports);
        for (int q = 0; q < num_queues; ++q) {
            Channel c;
            QB_HIP_CHECK(hipHostMalloc(&c.buffer, bytes, hipHostMallocCoherent));
            std::memset(c.buffer, 0, bytes);  // null doorbell => no interrupts

            // Build the client on the host, then copy its bytes to device
            // memory, as the offload plugin does for __llvm_rpc_client.
            alignas(rpc::Client) unsigned char staging[sizeof(rpc::Client)];
            new (staging) rpc::Client(ports, c.buffer);
            QB_HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&c.client),
                                   sizeof(rpc::Client)));
            QB_HIP_CHECK(hipMemcpy(c.client, staging, sizeof(rpc::Client),
                                   hipMemcpyHostToDevice));

            c.server = std::make_unique<LlvmRpcServer>(ports, kLaneSize, c.buffer);
            channels_.push_back(std::move(c));
        }
    }

    ~LlvmRpcAdapter() {
        for (auto& c : channels_) {
            (void)hipFree(c.client);
            (void)hipHostFree(c.buffer);
        }
    }

    LlvmRpcAdapter(const LlvmRpcAdapter&) = delete;
    LlvmRpcAdapter& operator=(const LlvmRpcAdapter&) = delete;

    Handle handle(int queue) { return channels_[queue].client; }

    // rpc.h itself declares the mailbox, header and packet pointers it loads
    // from the Client object as global (RPC_GLOBAL), so marking the client
    // pointer covers every access.
    __device__ static void globalize(Handle& client) { client = as_global(client); }

    __device__ static void push(Handle& client, const Msg& m) {
        if (lane_id() == 0) {
            rpc::Client::Port port = client->open<kOpcode>();
            // On the GPU, rpc.h passes the lane buffer as RPC_GLOBAL Buffer*.
            port.send([&](auto* b, uint32_t) {
                b->data[0] = msg_word0(m);
                b->data[1] = m.tag;
#pragma unroll
                for (int w = 2; w < 8; ++w) b->data[w] = filler_word(m, w);
            });
        }  // ~Port() closes the port
    }

    bool pop(int queue, Msg& out) {
        uint64_t data[8];
        if (!channels_[queue].server->try_recv(data)) return false;
        out = msg_from_words(data[0], data[1]);
        return true;
    }

    static std::string notes() {
        return "per-port mailbox; lane-0 device fetch_or port lock, 64B buffer, "
               "system release fence + outbox flip; server scans all ports";
    }

    std::vector<Channel> channels_;
};

}  // namespace qb
