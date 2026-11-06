// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ep_config.hpp"
#include "comm/active_queue_state.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

void require_impl(bool condition, const char* expression,
                  const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "%s:%d: requirement failed: %s\n",
                     file, line, expression);
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

#define REQUIRE(condition) \
    require_impl((condition), #condition, __FILE__, __LINE__)

int main() {
    using kiwi::ep::recommended_num_blocks;
    using kiwi::ep::recommended_runtime_defaults;
    using kiwi::ep::rcomp_backend_t;

    const auto require_defaults = [](
            int num_tokens, int world_size, int num_blocks,
            rcomp_backend_t rcomp_backend) {
        const auto defaults =
            recommended_runtime_defaults(num_tokens, world_size);
        REQUIRE(defaults.num_blocks == num_blocks);
        REQUIRE(defaults.rcomp_backend == rcomp_backend);
        REQUIRE(recommended_num_blocks(num_tokens, world_size) == num_blocks);
    };

    bool rejected_zero = false;
    try {
        (void)recommended_num_blocks(0, 8);
    } catch (const std::invalid_argument&) {
        rejected_zero = true;
    }
    REQUIRE(rejected_zero);
    bool rejected_negative = false;
    try {
        (void)recommended_num_blocks(-1, 8);
    } catch (const std::invalid_argument&) {
        rejected_negative = true;
    }
    REQUIRE(rejected_negative);

    require_defaults(1, 8, 12, rcomp_backend_t::RDMA_ONLY);
    require_defaults(2, 8, 20, rcomp_backend_t::RDMA_ONLY);
    require_defaults(3, 8, 32, rcomp_backend_t::RDMA_ONLY);
    require_defaults(4, 8, 32, rcomp_backend_t::RDMA_ONLY);
    require_defaults(7, 8, 32, rcomp_backend_t::RDMA_ONLY);
    require_defaults(8, 8, 32, rcomp_backend_t::HYBRID);
    require_defaults(16, 8, 32, rcomp_backend_t::HYBRID);
    require_defaults(32, 8, 32, rcomp_backend_t::HYBRID);
    require_defaults(33, 8, 64, rcomp_backend_t::HYBRID);
    require_defaults(64, 8, 64, rcomp_backend_t::HYBRID);
    require_defaults(65, 8, 256, rcomp_backend_t::HYBRID);
    require_defaults(512, 8, 256, rcomp_backend_t::HYBRID);
    require_defaults(4096, 8, 256, rcomp_backend_t::HYBRID);

    require_defaults(1, 16, 12, rcomp_backend_t::RDMA_ONLY);
    require_defaults(2, 16, 20, rcomp_backend_t::RDMA_ONLY);
    require_defaults(3, 16, 32, rcomp_backend_t::RDMA_ONLY);
    require_defaults(4, 16, 32, rcomp_backend_t::HYBRID);
    require_defaults(8, 16, 32, rcomp_backend_t::HYBRID);
    require_defaults(16, 16, 32, rcomp_backend_t::HYBRID);
    require_defaults(32, 16, 32, rcomp_backend_t::HYBRID);
    require_defaults(33, 16, 64, rcomp_backend_t::HYBRID);
    require_defaults(64, 16, 64, rcomp_backend_t::HYBRID);
    require_defaults(65, 16, 256, rcomp_backend_t::HYBRID);
    require_defaults(512, 16, 256, rcomp_backend_t::HYBRID);
    require_defaults(4096, 16, 256, rcomp_backend_t::HYBRID);

    require_defaults(1, 2, 12, rcomp_backend_t::RDMA_ONLY);
    require_defaults(2, 32, 20, rcomp_backend_t::RDMA_ONLY);
    require_defaults(3, 4, 32, rcomp_backend_t::RDMA_ONLY);
    require_defaults(4, 4, 32, rcomp_backend_t::HYBRID);
    require_defaults(64, 4, 64, rcomp_backend_t::HYBRID);
    require_defaults(65, 32, 256, rcomp_backend_t::HYBRID);

    kiwi::ep::config_t automatic;
    automatic.world_size = 16;
    automatic.max_num_inp_token_per_rank = 4096;
    automatic.num_blocks = 0;
    automatic.rcomp_backend = rcomp_backend_t::AUTO;

    REQUIRE(automatic.uses_automatic_num_blocks());
    REQUIRE(automatic.provisioned_num_blocks() == 256);
    REQUIRE(automatic.num_queues() == 256 * kiwi::ep::EP_WAVES_PER_BLOCK);
    REQUIRE(automatic.rcomp_backend == rcomp_backend_t::AUTO);
    REQUIRE(automatic.rcomp_ipc_resources_required());
    REQUIRE(automatic.launch_policy_for(0).num_blocks == 12);
    REQUIRE(automatic.launch_policy_for(0).rcomp_backend
            == rcomp_backend_t::RDMA_ONLY);
    REQUIRE(automatic.launch_policy_for(1).num_blocks == 12);
    REQUIRE(automatic.launch_policy_for(1).rcomp_backend
            == rcomp_backend_t::RDMA_ONLY);
    REQUIRE(automatic.launch_policy_for(64).num_blocks == 64);
    REQUIRE(automatic.launch_policy_for(64).rcomp_backend
            == rcomp_backend_t::HYBRID);
    REQUIRE(automatic.launch_policy_for(4096).num_blocks == 256);
    REQUIRE(automatic.launch_policy_for(4096).rcomp_backend
            == rcomp_backend_t::HYBRID);

    kiwi::ep::config_t fixed = automatic;
    fixed.num_blocks = 7;
    fixed.rcomp_backend = rcomp_backend_t::RDMA_ONLY;
    REQUIRE(!fixed.uses_automatic_num_blocks());
    REQUIRE(fixed.provisioned_num_blocks() == 7);
    REQUIRE(fixed.launch_policy_for(4096).num_blocks == 7);
    REQUIRE(fixed.launch_policy_for(4096).rcomp_backend
            == rcomp_backend_t::RDMA_ONLY);
    REQUIRE(!fixed.rcomp_ipc_resources_required());

    kiwi::ep::config_t automatic_small_capacity;
    automatic_small_capacity.world_size = 16;
    automatic_small_capacity.max_num_inp_token_per_rank = 2;
    automatic_small_capacity.num_blocks = 0;
    automatic_small_capacity.rcomp_backend = rcomp_backend_t::AUTO;
    REQUIRE(automatic_small_capacity.provisioned_num_blocks() == 20);
    REQUIRE(!automatic_small_capacity.rcomp_ipc_resources_required());

    REQUIRE(kiwi::ep::internal::kEpMaxCallbacksPerEndpointSweep == 4);

    kiwi::ep::internal::active_queue_request_t requests(64);
    kiwi::ep::internal::proxy_active_queue_state_t proxy_state(64);
    const auto initial_request = requests.load();
    REQUIRE(initial_request.active_count == 64);
    REQUIRE(proxy_state.polling_count() == 64);

    requests.publish(64);
    const auto repeated_initial_request = requests.load();
    REQUIRE(repeated_initial_request.active_count == 64);
    REQUIRE(repeated_initial_request.generation == initial_request.generation);

    requests.publish(8);
    const auto first_shrink_request = requests.load();
    proxy_state.observe(first_shrink_request);
    REQUIRE(proxy_state.has_retiring_queues());
    REQUIRE(proxy_state.is_retiring_queue(8));
    REQUIRE(proxy_state.is_retiring_queue(63));
    REQUIRE(!proxy_state.is_retiring_queue(7));
    // A nonempty retiring range is not eligible for completion.
    REQUIRE(proxy_state.polling_count() == 64);

    requests.publish(48);
    const auto newer_request = requests.load();
    proxy_state.complete_shrink_if_current(newer_request.generation);
    REQUIRE(proxy_state.polling_count() == 64);
    proxy_state.observe(newer_request);
    proxy_state.complete_shrink_if_current(newer_request.generation);
    REQUIRE(proxy_state.polling_count() == 48);

    requests.publish(8);
    const auto shrink_request = requests.load();
    proxy_state.observe(shrink_request);
    requests.publish(8);
    const auto repeated_shrink_request = requests.load();
    REQUIRE(repeated_shrink_request.generation == shrink_request.generation);
    proxy_state.complete_shrink_if_current(
        repeated_shrink_request.generation);
    REQUIRE(proxy_state.polling_count() == 8);

    requests.publish(64);
    const auto growth_request = requests.load();
    proxy_state.observe(growth_request);
    REQUIRE(proxy_state.polling_count() == 64);
    REQUIRE(!proxy_state.has_retiring_queues());

    requests.publish(8);
    const auto first_aba_shrink = requests.load();
    proxy_state.observe(first_aba_shrink);
    requests.publish(64);
    requests.publish(8);
    const auto second_aba_shrink = requests.load();
    proxy_state.complete_shrink_if_current(second_aba_shrink.generation);
    REQUIRE(proxy_state.polling_count() == 64);
    proxy_state.observe(second_aba_shrink);
    proxy_state.complete_shrink_if_current(second_aba_shrink.generation);
    REQUIRE(proxy_state.polling_count() == 8);

    bool rejected_nonpositive_proxy_count = false;
    try {
        (void)kiwi::ep::internal::proxy_queue_owner(0, 0);
    } catch (const std::invalid_argument&) {
        rejected_nonpositive_proxy_count = true;
    }
    REQUIRE(rejected_nonpositive_proxy_count);

    for (int endpoint_id = 0; endpoint_id < 64; ++endpoint_id) {
        REQUIRE(kiwi::ep::internal::proxy_queue_owner(endpoint_id, 1) == 0);
    }

    constexpr int kNumProxies = 4;
    constexpr int kEndpointsPerBlock = kiwi::ep::EP_WAVES_PER_BLOCK;
    constexpr int kNumBlocks = 12;
    REQUIRE(kEndpointsPerBlock == 16);
    constexpr std::array<int, kEndpointsPerBlock> expected_owners{
        0, 1, 2, 3,
        1, 2, 3, 0,
        2, 3, 0, 1,
        3, 0, 1, 2,
    };
    for (int endpoint_id = 0; endpoint_id < kEndpointsPerBlock;
         ++endpoint_id) {
        REQUIRE(kiwi::ep::internal::proxy_queue_owner(
                    endpoint_id, kNumProxies)
                == expected_owners[static_cast<size_t>(endpoint_id)]);
    }

    std::array<int, kNumProxies> wave_zero_owner_counts{};
    for (int block = 0; block < kNumBlocks; ++block) {
        const int endpoint_id = block * kEndpointsPerBlock;
        const int owner = kiwi::ep::internal::proxy_queue_owner(
            endpoint_id, kNumProxies);
        ++wave_zero_owner_counts[static_cast<size_t>(owner)];
    }
    for (int count : wave_zero_owner_counts) {
        REQUIRE(count == 3);
    }

    constexpr int kEndpointRange = kNumBlocks * kEndpointsPerBlock;
    for (int group_start = 0; group_start < kEndpointRange;
         group_start += kNumProxies) {
        std::array<int, kNumProxies> group_owner_counts{};
        for (int offset = 0; offset < kNumProxies; ++offset) {
            const int owner = kiwi::ep::internal::proxy_queue_owner(
                group_start + offset, kNumProxies);
            ++group_owner_counts[static_cast<size_t>(owner)];
        }
        for (int count : group_owner_counts) {
            REQUIRE(count == 1);
        }
    }

    std::array<int, kNumProxies> endpoint_counts{};
    for (int proxy_thread = 0; proxy_thread < kNumProxies; ++proxy_thread) {
        int previous_endpoint = -1;
        for (int endpoint = 0; endpoint < kEndpointRange; ++endpoint) {
            if (kiwi::ep::internal::proxy_queue_owner(
                    endpoint, kNumProxies)
                != proxy_thread) {
                continue;
            }
            REQUIRE(endpoint > previous_endpoint);
            previous_endpoint = endpoint;
            ++endpoint_counts[static_cast<size_t>(proxy_thread)];
        }
    }
    for (int count : endpoint_counts) {
        REQUIRE(count == 48);
    }

    std::puts("test_ep_block_count: PASSED");
    return 0;
}
