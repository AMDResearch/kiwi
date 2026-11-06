// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * @file cuda_runtime.h
 * @brief Build shim for UCCL-EP's vendored MSCCL++ FIFO headers on ROCm.
 *
 * hipify maps cuda_runtime.h to hip/hip_runtime.h; see cuda.h for the names.
 */

#pragma once

#include "cuda.h"
