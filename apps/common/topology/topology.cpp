// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "topology.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cctype>
#include <filesystem>
#include <limits>
#include <numeric>
#include <system_error>
#include "lci.hpp"
#ifdef __linux__
#include <dirent.h>
#include <errno.h>
#include <sched.h>
#include <unistd.h>
#endif

#undef HIP_CHECK
#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t err = (call);                                          \
        if (err != hipSuccess) {                                          \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                    __LINE__, hipGetErrorString(err));                     \
            abort();                                                      \
        }                                                                 \
    } while (0)

namespace kiwi::p2p {
namespace topology {

namespace {

#ifdef __linux__

bool proxy_affinity_debug_enabled() {
    const char* value = std::getenv("KIWI_PROXY_AFFINITY_DEBUG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::vector<int> cpus_in_set(const cpu_set_t& set) {
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

std::string summarize_cpus(const std::vector<int>& cpus) {
    std::string out = "[";
    for (size_t i = 0; i < cpus.size();) {
        const int begin = cpus[i];
        int end = begin;
        while (++i < cpus.size() && cpus[i] == end + 1) {
            end = cpus[i];
        }
        if (out.size() > 1) {
            out += ",";
        }
        out += std::to_string(begin);
        if (end != begin) {
            out += "-" + std::to_string(end);
        }
    }
    return out + "]";
}

// Linux CPU ids local to the current HIP device, parsed from its PCI sysfs
// local_cpulist (for example "56-111,168-223"). KFD's cpu_core_id_base is a
// hardware (APIC) id, not a Linux CPU id, so it is not used. Empty if the
// device or the file cannot be read.
std::vector<int> current_gpu_local_cpus() {
    int hip_device = -1;
    char bus_id[64] = {};
    if (hipGetDevice(&hip_device) != hipSuccess
        || hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), hip_device)
               != hipSuccess) {
        return {};
    }
    std::string path = std::string("/sys/bus/pci/devices/") + bus_id;
    std::transform(path.begin(), path.end(), path.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::ifstream in(path + "/local_cpulist");
    std::vector<int> cpus;
    std::string range;
    while (std::getline(in, range, ',')) {
        int first = -1;
        int last = -1;
        const int fields = std::sscanf(range.c_str(), "%d-%d", &first, &last);
        if (fields == 1) {
            last = first;
        }
        if (fields < 1 || first < 0 || last < first) {
            return {};
        }
        for (int cpu = first; cpu <= last; ++cpu) {
            cpus.push_back(cpu);
        }
    }
    return cpus;
}

#endif

} // namespace

std::vector<int> proxy_thread_cpu_set() {
#ifdef __linux__
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return {};
    }
    const std::vector<int> local = current_gpu_local_cpus();
    std::vector<int> cpus;
    for (int cpu : local) {
        if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
            cpus.push_back(cpu);
        }
    }
    std::sort(cpus.begin(), cpus.end());
    cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
    if (proxy_affinity_debug_enabled()) {
        std::fprintf(stderr,
                     "KIWI_PROXY_AFFINITY_DEBUG lci_rank=%d gpu_local=%s "
                     "allowed=%s proxy_set=%s\n",
                     lci::get_rank_me(), summarize_cpus(local).c_str(),
                     summarize_cpus(cpus_in_set(allowed)).c_str(),
                     cpus.empty() ? "inherited" : summarize_cpus(cpus).c_str());
    }
    return cpus;
#else
    return {};
#endif
}

void bind_current_thread_to_cpus(const std::vector<int>& cpus) {
#ifdef __linux__
    if (cpus.empty()) {
        return;
    }
    cpu_set_t target;
    CPU_ZERO(&target);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &target);
        }
    }
    if (sched_setaffinity(0, sizeof(target), &target) != 0) {
        std::fprintf(stderr,
                     "ERROR: failed to bind proxy thread to CPUs %s: %s\n",
                     summarize_cpus(cpus).c_str(), std::strerror(errno));
        std::abort();
    }
    if (proxy_affinity_debug_enabled()) {
        cpu_set_t actual;
        CPU_ZERO(&actual);
        sched_getaffinity(0, sizeof(actual), &actual);
        std::fprintf(stderr,
                     "KIWI_PROXY_AFFINITY_DEBUG lci_rank=%d tid=%ld bound=%s\n",
                     lci::get_rank_me(), static_cast<long>(gettid()),
                     summarize_cpus(cpus_in_set(actual)).c_str());
    }
#else
    (void) cpus;
#endif
}

namespace nic_selection_detail {

struct pci_location_t {
    std::string bus_id;
    std::filesystem::path canonical_path;
    int numa_node = -1;

    bool is_valid() const { return !bus_id.empty() && !canonical_path.empty(); }
};

struct hca_info_t {
    std::string name;
    pci_location_t pci;
    double total_gbps = 0.0;
};

struct hca_rank_key_t {
    double total_gbps = 0.0;
    bool same_numa = false;
    int hops = std::numeric_limits<int>::max();
};

inline std::string trim_copy(const std::string& value) {
    size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin]))) {
        begin++;
    }
    size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(begin, end - begin);
}

inline std::string lowercase_ascii(std::string value) {
    for (char& c : value) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return value;
}

inline std::string read_first_line(const std::filesystem::path& path) {
    std::ifstream f(path);
    std::string line;
    if (!f.is_open() || !std::getline(f, line)) return {};
    return trim_copy(line);
}

inline int read_int_file(const std::filesystem::path& path, int fallback = -1) {
    std::ifstream f(path);
    int value = fallback;
    if (!(f >> value)) return fallback;
    return value;
}

inline double read_rate_gbps(const std::filesystem::path& path) {
    std::ifstream f(path);
    double value = 0.0;
    if (!(f >> value)) return 0.0;
    return value;
}

inline std::filesystem::path canonical_or_empty(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::path resolved = std::filesystem::canonical(path, ec);
    if (ec) return {};
    return resolved;
}

inline bool is_bdf_like(const std::string& name) {
    int colon_count = 0;
    int dot_count = 0;
    for (char c : name) {
        if (c == ':') colon_count++;
        else if (c == '.') dot_count++;
    }
    return (colon_count == 2 || colon_count == 1) && dot_count == 1;
}

inline std::vector<std::string> expand_bus_id_candidates(
        const std::string& raw_bus_id) {
    std::string bus_id = lowercase_ascii(trim_copy(raw_bus_id));
    std::vector<std::string> candidates;
    if (!bus_id.empty()) candidates.push_back(bus_id);
    if (!bus_id.empty() && is_bdf_like(bus_id) &&
        bus_id.find(':') == bus_id.rfind(':')) {
        candidates.push_back("0000:" + bus_id);
    }
    return candidates;
}

inline pci_location_t resolve_pci_location(const std::string& raw_bus_id) {
    for (const std::string& candidate : expand_bus_id_candidates(raw_bus_id)) {
        std::filesystem::path dev_path =
            canonical_or_empty(std::filesystem::path("/sys/bus/pci/devices") /
                               candidate);
        if (dev_path.empty()) continue;
        return {candidate, dev_path,
                read_int_file(std::filesystem::path("/sys/bus/pci/devices") /
                                  candidate / "numa_node",
                              -1)};
    }
    return {};
}

inline bool is_active_port(const std::filesystem::path& port_path) {
    std::string state = read_first_line(port_path / "state");
    return state.find("ACTIVE") != std::string::npos;
}

inline std::vector<std::string> split_path_components(
        const std::filesystem::path& path) {
    std::vector<std::string> parts;
    for (const auto& part : path) parts.push_back(part.string());
    return parts;
}

inline int pcie_hops(const std::filesystem::path& lhs,
                     const std::filesystem::path& rhs) {
    if (lhs.empty() || rhs.empty()) return std::numeric_limits<int>::max();

    std::vector<std::string> lhs_parts = split_path_components(lhs);
    std::vector<std::string> rhs_parts = split_path_components(rhs);
    size_t common = 0;
    while (common < lhs_parts.size() && common < rhs_parts.size() &&
           lhs_parts[common] == rhs_parts[common]) {
        common++;
    }
    size_t hops = (lhs_parts.size() - common) + (rhs_parts.size() - common);
    if (hops > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(hops);
}

inline pci_location_t get_gpu_pci_location(int gpu_id) {
    char pci_id[64] = {};
    if (hipDeviceGetPCIBusId(pci_id, sizeof(pci_id), gpu_id) != hipSuccess) {
        return {};
    }
    return resolve_pci_location(pci_id);
}

inline std::vector<hca_info_t> enumerate_active_hcas() {
    std::vector<hca_info_t> hcas;
    std::filesystem::path hca_root("/sys/class/infiniband");
    std::error_code root_ec;
    if (!std::filesystem::exists(hca_root, root_ec)) return hcas;

    std::error_code list_ec;
    for (const auto& entry : std::filesystem::directory_iterator(hca_root, list_ec)) {
        if (list_ec) break;

        hca_info_t hca;
        hca.name = entry.path().filename().string();

        std::filesystem::path dev_path = canonical_or_empty(entry.path() / "device");
        if (dev_path.empty()) continue;

        hca.pci = resolve_pci_location(dev_path.filename().string());
        if (hca.pci.bus_id.empty()) {
            hca.pci.bus_id = lowercase_ascii(dev_path.filename().string());
            hca.pci.canonical_path = dev_path;
            hca.pci.numa_node =
                read_int_file(std::filesystem::path("/sys/bus/pci/devices") /
                                  hca.pci.bus_id / "numa_node",
                              -1);
        }

        std::filesystem::path ports_dir = entry.path() / "ports";
        bool has_active_port = false;
        std::error_code ports_ec;
        for (const auto& port_entry : std::filesystem::directory_iterator(
                 ports_dir, ports_ec)) {
            if (ports_ec) break;
            if (!is_active_port(port_entry.path())) continue;
            has_active_port = true;
            hca.total_gbps += read_rate_gbps(port_entry.path() / "rate");
        }

        if (!has_active_port) continue;
        hcas.push_back(std::move(hca));
    }

    std::sort(hcas.begin(), hcas.end(),
              [](const hca_info_t& a, const hca_info_t& b) {
                  return a.name < b.name;
              });
    return hcas;
}

inline hca_rank_key_t rank_hca_for_gpu(const hca_info_t& hca,
                                       const pci_location_t& gpu) {
    hca_rank_key_t key;
    key.total_gbps = hca.total_gbps;
    key.same_numa = gpu.numa_node >= 0 && hca.pci.numa_node == gpu.numa_node;
    key.hops = pcie_hops(gpu.canonical_path, hca.pci.canonical_path);
    return key;
}

inline bool same_rank_key(const hca_rank_key_t& a, const hca_rank_key_t& b) {
    return a.total_gbps == b.total_gbps && a.same_numa == b.same_numa &&
           a.hops == b.hops;
}

inline bool better_rank_key(const hca_rank_key_t& a, const hca_rank_key_t& b) {
    if (a.total_gbps != b.total_gbps) return a.total_gbps > b.total_gbps;
    if (a.same_numa != b.same_numa) return a.same_numa && !b.same_numa;
    if (a.hops != b.hops) return a.hops < b.hops;
    return false;
}

inline std::vector<std::string> build_gpu_hca_map() {
    int num_gpus = 0;
    if (hipGetDeviceCount(&num_gpus) != hipSuccess || num_gpus <= 0) return {};

    std::vector<hca_info_t> hcas = enumerate_active_hcas();
    std::vector<std::string> mapping(num_gpus);
    if (hcas.empty()) return mapping;

    std::vector<int> assigned_count(hcas.size(), 0);
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        pci_location_t gpu_pci = get_gpu_pci_location(gpu);
        if (!gpu_pci.is_valid()) {
            mapping[gpu] = hcas[gpu % hcas.size()].name;
            assigned_count[gpu % hcas.size()]++;
            continue;
        }

        std::vector<int> candidate_indices(hcas.size());
        std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
        std::sort(candidate_indices.begin(), candidate_indices.end(),
                  [&](int lhs, int rhs) {
                      hca_rank_key_t lhs_key = rank_hca_for_gpu(hcas[lhs], gpu_pci);
                      hca_rank_key_t rhs_key = rank_hca_for_gpu(hcas[rhs], gpu_pci);
                      if (better_rank_key(lhs_key, rhs_key)) return true;
                      if (better_rank_key(rhs_key, lhs_key)) return false;
                      return hcas[lhs].name < hcas[rhs].name;
                  });

        int selected = candidate_indices.front();
        hca_rank_key_t best_key = rank_hca_for_gpu(hcas[selected], gpu_pci);
        for (int idx : candidate_indices) {
            hca_rank_key_t key = rank_hca_for_gpu(hcas[idx], gpu_pci);
            if (!same_rank_key(key, best_key)) break;
            if (assigned_count[idx] < assigned_count[selected] ||
                (assigned_count[idx] == assigned_count[selected] &&
                 hcas[idx].name < hcas[selected].name)) {
                selected = idx;
            }
        }

        assigned_count[selected]++;
        mapping[gpu] = hcas[selected].name;
    }
    return mapping;
}

} // namespace nic_selection_detail

// Return the nearest active HCA/NIC name for a HIP GPU id.  The mapping is
// computed across all visible GPUs and balances exact ties so ranks do not all
// bind to the same NIC when several equally-near HCAs are available.
std::string get_hca_name_for_gpu(int gpu_id) {
    static const std::vector<std::string> gpu_to_hca =
        nic_selection_detail::build_gpu_hca_map();
    if (gpu_id < 0 || gpu_id >= static_cast<int>(gpu_to_hca.size())) return {};
    return gpu_to_hca[gpu_id];
}

std::string get_hca_name_for_current_gpu() {
    int gpu_id = -1;
    if (hipGetDevice(&gpu_id) != hipSuccess) return {};
    return get_hca_name_for_gpu(gpu_id);
}

std::string get_requested_or_auto_hca_name_for_current_gpu() {
    const char* requested = std::getenv("KIWI_LCI_DEVICE_NAME");
    if (requested != nullptr && requested[0] != '\0') return requested;
    requested = std::getenv("LCI_ATTR_DEVICE_NAME");
    if (requested != nullptr && requested[0] != '\0') return requested;
    return get_hca_name_for_current_gpu();
}


} // namespace topology
} // namespace kiwi::p2p
