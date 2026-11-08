// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// MI300X with ConnectX-7 NICs.

#include "ep_tuning.hpp"

namespace kiwi::ep {

namespace {

// {world_size, max_num_tokens, num_blocks, rcomp_backend}
constexpr runtime_tuning_rule_t kRules[] = {
    {8, 1, 12, rcomp_backend_t::RDMA_ONLY},
    {8, 2, 20, rcomp_backend_t::RDMA_ONLY},
    {8, 4, 32, rcomp_backend_t::RDMA_ONLY},
    {8, 8, 32, rcomp_backend_t::HYBRID},
    {8, 16, 32, rcomp_backend_t::HYBRID},
    {8, 32, 32, rcomp_backend_t::HYBRID},
    {8, 64, 64, rcomp_backend_t::HYBRID},

    {16, 1, 12, rcomp_backend_t::RDMA_ONLY},
    {16, 2, 20, rcomp_backend_t::RDMA_ONLY},
    {16, 4, 32, rcomp_backend_t::HYBRID},
    {16, 8, 32, rcomp_backend_t::HYBRID},
    {16, 16, 32, rcomp_backend_t::HYBRID},
    {16, 32, 32, rcomp_backend_t::HYBRID},
    {16, 64, 64, rcomp_backend_t::HYBRID},

    {32, 1, 12, rcomp_backend_t::RDMA_ONLY},
    {32, 2, 20, rcomp_backend_t::RDMA_ONLY},
    {32, 4, 32, rcomp_backend_t::HYBRID},
    {32, 8, 32, rcomp_backend_t::HYBRID},
    {32, 16, 32, rcomp_backend_t::HYBRID},
    {32, 32, 32, rcomp_backend_t::HYBRID},
    {32, 64, 64, rcomp_backend_t::HYBRID},

    {64, 1, 12, rcomp_backend_t::RDMA_ONLY},
    {64, 2, 20, rcomp_backend_t::RDMA_ONLY},
    {64, 4, 32, rcomp_backend_t::HYBRID},
    {64, 8, 32, rcomp_backend_t::HYBRID},
    {64, 16, 32, rcomp_backend_t::HYBRID},
    {64, 32, 32, rcomp_backend_t::HYBRID},
    {64, 64, 64, rcomp_backend_t::HYBRID},
};

} // namespace

extern const runtime_tuning_table_t kGfx942Cx7Tuning = {
    "gfx942", "cx7", kRules, sizeof(kRules) / sizeof(kRules[0]),
};

} // namespace kiwi::ep
