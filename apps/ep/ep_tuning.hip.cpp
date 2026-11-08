// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ep_tuning.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <fstream>

namespace kiwi::ep {

namespace {

std::string read_lowercase_token(const std::string& path) {
    std::ifstream file(path);
    std::string value;
    file >> value;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return value;
}

} // namespace

extern const runtime_tuning_table_t kGfx942Cx7Tuning;
extern const runtime_tuning_table_t kGfx950PollaraTuning;

const runtime_tuning_table_t* const kRuntimeTuningTables[] = {
    &kGfx942Cx7Tuning,
    &kGfx950PollaraTuning,
};
const size_t kNumRuntimeTuningTables =
    sizeof(kRuntimeTuningTables) / sizeof(kRuntimeTuningTables[0]);

const runtime_tuning_table_t* find_runtime_tuning_table(
        std::string_view gpu_arch, std::string_view nic_family) {
    for (const runtime_tuning_table_t* table : kRuntimeTuningTables) {
        if (gpu_arch == table->gpu_arch && nic_family == table->nic_family) {
            return table;
        }
    }
    return nullptr;
}

std::string current_gpu_arch() {
    int device = -1;
    hipDeviceProp_t props{};
    if (hipGetDevice(&device) != hipSuccess
        || hipGetDeviceProperties(&props, device) != hipSuccess) {
        return {};
    }
    const std::string arch = props.gcnArchName;
    return arch.substr(0, arch.find(':'));
}

std::string nic_family_for_hca(const std::string& hca_name,
                               const std::string& sysfs_root) {
    if (hca_name.empty()) {
        return {};
    }
    std::string prefix = hca_name.substr(0, hca_name.find('_'));
    std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (prefix == "ionic") {
        return "pollara";
    }
    if (prefix == "mlx5") {
        const std::string device_dir =
            sysfs_root + "/" + hca_name + "/device/";
        if (read_lowercase_token(device_dir + "vendor") == "0x15b3"
            && read_lowercase_token(device_dir + "device") == "0x1021") {
            return "cx7";
        }
    }
    return prefix;
}

} // namespace kiwi::ep
