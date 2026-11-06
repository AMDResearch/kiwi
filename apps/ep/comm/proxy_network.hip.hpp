// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "lci.hpp"

namespace kiwi::ep::internal {

void proxy_network_progress();

void proxy_network_record_posted_write(lci::error_t error);
void proxy_network_record_injected_write(lci::error_t error);

} // namespace kiwi::ep::internal
