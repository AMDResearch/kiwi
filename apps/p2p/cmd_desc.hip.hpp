// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstddef>

namespace kiwi::p2p {
namespace internal {

// POD-only structs for GPU-CPU transfer through invoke.
// Fields ordered: 8-byte first, then 4-byte packed, to eliminate padding.
// Memory regions resolved via symmetric heap_id on CPU side.
struct cmd_desc_p_t {
    uint64_t value;
    uintptr_t remote_disp;
    int rank;
    int heap_id;
};

struct cmd_desc_put_t {
    uintptr_t local_buffer;
    uint64_t size;
    uintptr_t remote_disp;
    uintptr_t ready_remote_disp;
    int rank;
    int data_heap_id;
    int ready_heap_id;
};

// Fixed-shape barrier descriptor. comp is lci::comp_t::p_impl.
struct cmd_desc_barrier_t {
    uintptr_t comp;
};

// Fixed-shape all2all descriptor. The vectors are CPU VAs, typically into
// pinned/device-mapped host workspace, and comp is lci::comp_t::p_impl.
struct cmd_desc_all2all_t {
    uintptr_t send_buf;
    uintptr_t recv_buf;
    uintptr_t comp;
};

static_assert(sizeof(cmd_desc_p_t) == 24, "cmd_desc_p_t must be 24 bytes (one one-line invoke message)");
static_assert(sizeof(cmd_desc_put_t) == 48,
              "cmd_desc_put_t must be 48 bytes (one two-line invoke message)");
static_assert(sizeof(cmd_desc_barrier_t) == sizeof(uintptr_t),
              "cmd_desc_barrier_t should only carry the completion object pointer");
static_assert(sizeof(cmd_desc_all2all_t) <= 48,
              "cmd_desc_all2all_t must fit in one invoke message");

} // namespace internal
} // namespace kiwi::p2p
