// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file llvm_rpc_server.hpp
 * @brief Host-side LLVM libc RPC server wrapper.
 *
 * LLVM builds its RPC server as plain C++ (offload plugin), not HIP: in a HIP
 * translation unit the host pass of rpc_util.h's sleep_briefly() resolves to
 * the AMDGPU s_sleep builtin. The server therefore lives in its own C++ file.
 */

#pragma once

#include <cstdint>
#include <memory>

namespace qb {

class LlvmRpcServer {
public:
    LlvmRpcServer(uint32_t port_count, uint32_t lane_size, void* buffer);
    ~LlvmRpcServer();
    LlvmRpcServer(const LlvmRpcServer&) = delete;
    LlvmRpcServer& operator=(const LlvmRpcServer&) = delete;

    /// Server::try_open() + Port::recv() of the first 64B lane buffer +
    /// Port close (returns the port to the client). False if no port is ready.
    bool try_recv(uint64_t (&data)[8]);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qb
