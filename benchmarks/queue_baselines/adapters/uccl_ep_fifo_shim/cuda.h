// Copyright (c) 2016- Facebook, Inc (Adam Paszke) and other PyTorch contributors.
// Modifications Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT AND BSD-3-Clause
//
// The CUDA-to-HIP name mappings below are taken from PyTorch
// (https://github.com/pytorch/pytorch), torch/utils/hipify/
// cuda_to_hip_mappings.py, Copyright (c) 2016- Facebook, Inc and other
// PyTorch contributors, BSD-3-Clause license. See NOTICES.txt.

/**
 * @file cuda.h
 * @brief Build shim for UCCL-EP's vendored MSCCL++ FIFO headers on ROCm.
 *
 * UCCL-EP builds on ROCm through torch's hipify (ep/setup.py, CUDAExtension),
 * which rewrites CUDA names in ep/include/fifo_util.hpp before compiling. This
 * header performs the same renames at preprocessing time, so the fetched
 * upstream headers are compiled byte-for-byte unmodified. Every mapping below
 * is the one in torch/utils/hipify/cuda_to_hip_mappings.py. Only names used by
 * fifo_util.hpp are mapped. Used only by adapters/uccl_ep_fifo.hpp.
 */

#pragma once

#include <hip/hip_runtime.h>

#define CUresult hipError_t
#define CUDA_SUCCESS hipSuccess
#define cuGetErrorString hipDrvGetErrorString
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorContextIsDestroyed hipErrorContextIsDestroyed
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaMalloc hipMalloc
#define cudaMemset hipMemset
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocMapped hipHostMallocMapped
