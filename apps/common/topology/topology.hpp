// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kiwi::p2p {
namespace topology {

// CPUs for the proxy threads serving the current HIP device: its PCI-local
// CPUs within the calling thread's affinity mask, as NCCL and MORI do. Empty
// when they do not intersect or cannot be read; proxies then keep the mask
// they inherit. Call on the thread that selected the device, since
// hipSetDevice() is per thread. KIWI_PROXY_AFFINITY_DEBUG=1 prints the set.
std::vector<int> proxy_thread_cpu_set();

// Restrict the calling thread to cpus and leave placement within them to the
// scheduler. Pinning each proxy to a CPU chosen independently per rank lets
// ranks collide on one CPU; a shared set cannot. No-op for an empty set.
void bind_current_thread_to_cpus(const std::vector<int>& cpus);

// Return the nearest active HCA/NIC name for a HIP GPU id.  The mapping is
// computed across all visible GPUs and balances exact ties so ranks do not all
// bind to the same NIC when several equally-near HCAs are available.
std::string get_hca_name_for_gpu(int gpu_id);
std::string get_hca_name_for_current_gpu();
std::string get_requested_or_auto_hca_name_for_current_gpu();

} // namespace topology
} // namespace kiwi::p2p
