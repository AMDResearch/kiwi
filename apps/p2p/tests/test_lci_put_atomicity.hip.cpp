// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Two-rank hardware diagnostic for atomic visibility of aligned 8-byte LCI
// puts into registered host memory. Run across two nodes with the intended
// network provider; a same-node run may exercise a shared-memory transport.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <CLI/CLI.hpp>
#include "comp_counter.hip.hpp"
#include "lci.hpp"

namespace {

constexpr uint64_t kFailureSentinel = 0xdeadc0debad0f00dULL;

enum class transition_t {
    EXPECTED,
    PEER_FAILED,
    INVALID,
};

void post_put_retry(int peer, uint64_t* word, lci::comp_t comp,
                    lci::device_t device, lci::mr_t mr,
                    lci::rmr_t peer_rmr) {
    lci::status_t status;
    do {
        status = lci::post_put_x(
            peer, word, sizeof(*word), comp, 0, peer_rmr)
            .device(device)
            .mr(mr)
            .allow_done(false)();
        lci::progress_x().device(device)();
    } while (status.is_retry());
}

void wait_counter(lci::device_t device, lci::comp_t comp, uint64_t target) {
    while (kiwi::p2p::get_counter(comp) < target) {
        lci::progress_x().device(device)();
    }
}

transition_t wait_transition(lci::device_t device, uint64_t* word,
                             uint64_t old_value, uint64_t expected_value) {
    while (true) {
        const uint64_t observed = __atomic_load_n(word, __ATOMIC_ACQUIRE);
        if (observed == expected_value) {
            return transition_t::EXPECTED;
        }
        if (observed == kFailureSentinel) {
            return transition_t::PEER_FAILED;
        }
        if (observed != old_value) {
            fprintf(stderr,
                    "8-byte LCI put atomicity test observed torn/unexpected "
                    "value 0x%016llx (old=0x%016llx expected=0x%016llx)\n",
                    static_cast<unsigned long long>(observed),
                    static_cast<unsigned long long>(old_value),
                    static_cast<unsigned long long>(expected_value));
            return transition_t::INVALID;
        }
        lci::progress_x().device(device)();
    }
}

void propagate_failure(int peer, uint64_t* word, lci::comp_t comp,
                       uint64_t& local_completion_target,
                       lci::device_t device, lci::mr_t mr,
                       lci::rmr_t peer_rmr) {
    __atomic_store_n(word, kFailureSentinel, __ATOMIC_RELEASE);
    post_put_retry(peer, word, comp, device, mr, peer_rmr);
    wait_counter(device, comp, ++local_completion_target);
}

bool run_test(int niters, int rank, int peer, uint64_t* word,
              lci::comp_t comp, lci::device_t device, lci::mr_t mr,
              lci::rmr_t peer_rmr) {
    uint64_t local_completion_target = 0;
    uint64_t prior_rank1_value = 0;
    for (int i = 0; i < niters; ++i) {
        const uint64_t value = (i & 1) == 0
            ? 0xaaaaaaaa55555555ULL
            : 0x55555555aaaaaaaaULL;
        const uint64_t ack = value ^ 0xffff00000000ffffULL;
        if (rank == 0) {
            __atomic_store_n(word, value, __ATOMIC_RELEASE);
            post_put_retry(peer, word, comp, device, mr, peer_rmr);
            wait_counter(device, comp, ++local_completion_target);
            const transition_t transition =
                wait_transition(device, word, value, ack);
            if (transition == transition_t::INVALID) {
                propagate_failure(peer, word, comp, local_completion_target,
                                  device, mr, peer_rmr);
                return false;
            }
            if (transition == transition_t::PEER_FAILED) {
                return false;
            }
        } else {
            const transition_t transition =
                wait_transition(device, word, prior_rank1_value, value);
            if (transition == transition_t::INVALID) {
                propagate_failure(peer, word, comp, local_completion_target,
                                  device, mr, peer_rmr);
                return false;
            }
            if (transition == transition_t::PEER_FAILED) {
                return false;
            }
            __atomic_store_n(word, ack, __ATOMIC_RELEASE);
            post_put_retry(peer, word, comp, device, mr, peer_rmr);
            wait_counter(device, comp, ++local_completion_target);
            prior_rank1_value = ack;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    int niters = 100000;
    CLI::App app{
        "Test atomic visibility of aligned 8-byte LCI puts to host memory"};
    app.add_option("--iters", niters, "Test iterations")->default_val(100000);
    CLI11_PARSE(app, argc, argv);

    lci::g_runtime_init();

    const int rank = lci::get_rank_me();
    const int nranks = lci::get_rank_n();
    if (nranks != 2) {
        if (rank == 0) {
            fprintf(stderr,
                    "test_lci_put_atomicity requires exactly 2 ranks\n");
        }
        lci::g_runtime_fina();
        return 1;
    }
    const int peer = 1 - rank;

    auto* word = static_cast<uint64_t*>(calloc(1, sizeof(uint64_t)));
    if (!word) {
        fprintf(stderr, "calloc failed\n");
        lci::g_runtime_fina();
        return 1;
    }

    const size_t npackets =
        lci::get_default_packet_pool().get_attr_npackets();
    const size_t max_nrecvs = std::min(npackets / 4, 4096UL);
    size_t max_nsends = std::min(npackets / 8, 64UL);
    max_nsends = std::max(max_nsends, 8UL);
    lci::device_t device = lci::alloc_device_x()
        .net_max_sends(max_nsends)
        .net_max_recvs(max_nrecvs)();

    lci::mr_t mr =
        lci::register_memory_x(word, sizeof(*word)).device(device)();
    lci::rmr_t my_rmr = lci::get_rmr(mr);
    std::vector<lci::rmr_t> rmrs(nranks);
    lci::allgather_x(&my_rmr, rmrs.data(), sizeof(lci::rmr_t))
        .device(device)();
    lci::comp_t comp = kiwi::p2p::alloc_counter(0);

    __atomic_store_n(word, 0ULL, __ATOMIC_RELEASE);
    lci::barrier_x().device(device)();
    const bool pass =
        run_test(niters, rank, peer, word, comp, device, mr, rmrs[peer]);
    lci::barrier_x().device(device)();

    printf("RESULT test=lci_put_u64_atomicity rank=%d iters=%d status=%s\n",
           rank, niters, pass ? "pass" : "fail");
    if (rank == 0) {
        printf("SUMMARY test=lci_put_u64_atomicity ranks=2 iters=%d "
               "status=%s source_mem=host target_mem=host\n",
               niters, pass ? "pass" : "fail");
    }

    kiwi::p2p::free_counter(comp);
    lci::deregister_memory(&mr);
    lci::free_device(&device);
    lci::barrier();
    free(word);
    lci::g_runtime_fina();
    return pass ? 0 : 1;
}
