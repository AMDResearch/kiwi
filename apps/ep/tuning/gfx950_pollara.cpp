// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// MI355X with Pollara NICs: BF16, hidden 7168, top-k 8, eight ranks per node.

#include "ep_tuning.hpp"

namespace kiwi::ep {

namespace {

// {world_size, max_num_tokens, num_blocks, rcomp_backend}
constexpr runtime_tuning_rule_t kRules[] = {
    {8, 1, 12, rcomp_backend_t::HYBRID},
    {8, 2, 12, rcomp_backend_t::HYBRID},
    {8, 4, 16, rcomp_backend_t::HYBRID},
    {8, 8, 32, rcomp_backend_t::HYBRID},
    {8, 16, 48, rcomp_backend_t::HYBRID},
    {8, 32, 48, rcomp_backend_t::HYBRID},
    {8, 64, 64, rcomp_backend_t::HYBRID},

    {16, 1, 16, rcomp_backend_t::HYBRID},
    {16, 2, 24, rcomp_backend_t::HYBRID},
    {16, 4, 24, rcomp_backend_t::HYBRID},
    {16, 8, 48, rcomp_backend_t::HYBRID},
    {16, 16, 64, rcomp_backend_t::HYBRID},
    {16, 32, 64, rcomp_backend_t::HYBRID},
    {16, 64, 64, rcomp_backend_t::HYBRID},

    {32, 1, 12, rcomp_backend_t::HYBRID},
    {32, 2, 20, rcomp_backend_t::HYBRID},
    {32, 4, 24, rcomp_backend_t::HYBRID},
    {32, 8, 24, rcomp_backend_t::HYBRID},
    {32, 16, 24, rcomp_backend_t::HYBRID},
    {32, 32, 32, rcomp_backend_t::HYBRID},
    {32, 64, 64, rcomp_backend_t::HYBRID},
};

} // namespace

extern const runtime_tuning_table_t kGfx950PollaraTuning = {
    "gfx950", "pollara", kRules, sizeof(kRules) / sizeof(kRules[0]),
};

} // namespace kiwi::ep
