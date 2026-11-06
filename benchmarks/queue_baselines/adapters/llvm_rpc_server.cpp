// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "llvm_rpc_server.hpp"

#include <rpc.h>

#include <cstring>

namespace qb {

struct LlvmRpcServer::Impl {
    Impl(uint32_t port_count, uint32_t lane_size, void* buffer)
        : server(port_count, buffer), lane_size(lane_size) {}
    rpc::Server server;
    uint32_t lane_size;
};

LlvmRpcServer::LlvmRpcServer(uint32_t port_count, uint32_t lane_size, void* buffer)
    : impl_(std::make_unique<Impl>(port_count, lane_size, buffer)) {}

LlvmRpcServer::~LlvmRpcServer() = default;

bool LlvmRpcServer::try_recv(uint64_t (&data)[8]) {
    // Same call the LLVM offload server makes: scan from port 0.
    auto port = impl_->server.try_open(impl_->lane_size);
    if (!port) return false;
    port->recv([&](rpc::Buffer* b, uint32_t) {
        std::memcpy(data, b->data, sizeof(b->data));
    });
    return true;
}

}  // namespace qb
