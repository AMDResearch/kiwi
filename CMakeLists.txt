# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Select the HIP architecture before compiler detection.
if(NOT DEFINED HIP_ARCHITECTURES)
    if(DEFINED CMAKE_HIP_ARCHITECTURES)
        set(HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    else()
        execute_process(
            COMMAND bash -c "rocminfo | grep -o 'gfx[0-9]*' | head -1"
            OUTPUT_VARIABLE ROCM_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(ROCM_ARCH)
            set(HIP_ARCHITECTURES "${ROCM_ARCH}")
        else()
            set(HIP_ARCHITECTURES "gfx942")
        endif()
    endif()
endif()
set(CMAKE_HIP_ARCHITECTURES "${HIP_ARCHITECTURES}")

project(kiwi LANGUAGES CXX HIP)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Core and app static libraries may be linked into the EP Python module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH $ENV{HIP_PATH})
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()
include_directories(${ROCM_PATH}/include)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompileOptions)
include(FetchContent)

option(KIWI_BUILD_INVOKE "Build the invocation library" ON)
option(KIWI_BUILD_TESTS "Build tests for the core and enabled applications" OFF)
option(KIWI_BUILD_BENCHMARKS "Build benchmarks for the core and enabled applications" OFF)
option(KIWI_BUILD_QUEUE_BASELINES "Build GPU->CPU queue baseline microbenchmarks (fetches third-party headers)" OFF)
option(KIWI_BUILD_EXAMPLES "Build core examples" OFF)
option(KIWI_BUILD_P2P "Build the P2P application" OFF)
option(KIWI_BUILD_EP "Build the expert-parallel application" OFF)
option(KIWI_BUILD_PYTHON "Build EP Python bindings" OFF)
option(KIWI_BUILD_FLYDSL "Build the FlyDSL invoke frontend (Python package kiwi_flydsl)" OFF)
option(KIWI_BUILD_TRITON "Build the Triton invoke frontend (Python module kiwi_triton)" OFF)

if((KIWI_BUILD_P2P OR KIWI_BUILD_EP) AND NOT KIWI_BUILD_INVOKE)
    message(FATAL_ERROR "KIWI applications require KIWI_BUILD_INVOKE=ON")
endif()
if(KIWI_BUILD_FLYDSL AND NOT KIWI_BUILD_INVOKE)
    message(FATAL_ERROR "KIWI_BUILD_FLYDSL requires KIWI_BUILD_INVOKE=ON")
endif()
if(KIWI_BUILD_TRITON AND NOT KIWI_BUILD_INVOKE)
    message(FATAL_ERROR "KIWI_BUILD_TRITON requires KIWI_BUILD_INVOKE=ON")
endif()
if(KIWI_BUILD_QUEUE_BASELINES AND NOT KIWI_BUILD_BENCHMARKS)
    message(FATAL_ERROR "KIWI_BUILD_QUEUE_BASELINES requires KIWI_BUILD_BENCHMARKS=ON")
endif()
if(KIWI_BUILD_QUEUE_BASELINES AND NOT KIWI_BUILD_INVOKE)
    message(FATAL_ERROR "KIWI_BUILD_QUEUE_BASELINES requires KIWI_BUILD_INVOKE=ON (kiwi-invoke)")
endif()
if(KIWI_BUILD_PYTHON AND NOT KIWI_BUILD_EP)
    message(FATAL_ERROR "KIWI_BUILD_PYTHON requires KIWI_BUILD_EP=ON")
endif()

# CLI11 is shared by core benchmarks and application test/benchmark harnesses.
if(KIWI_BUILD_BENCHMARKS OR
   ((KIWI_BUILD_P2P OR KIWI_BUILD_EP) AND KIWI_BUILD_TESTS))
    FetchContent_Declare(
        cli11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.4.2
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(cli11)
endif()

# Python kernel frontends. Their packages, the Python bridges of the
# cross-frontend benchmark harnesses (benchmarks/*/frontends/,
# apps/p2p/benchmarks/frontends/), and the frontend examples' C++ extension
# (examples/common/) are built into KIWI_PYTHON_BINARY_DIR; put it on
# PYTHONPATH.
if(KIWI_BUILD_FLYDSL OR KIWI_BUILD_TRITON)
    set(KIWI_PYTHON_FRONTENDS ON)
else()
    set(KIWI_PYTHON_FRONTENDS OFF)
endif()
set(KIWI_PYTHON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")

# Discover all enabled tests from the root build directory.
enable_testing()
add_subdirectory(src)
if(KIWI_PYTHON_FRONTENDS)
    add_subdirectory(python)
endif()

if(KIWI_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(googletest)
    add_subdirectory(tests)
endif()
if(KIWI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if(KIWI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(KIWI_BUILD_P2P OR KIWI_BUILD_EP)
    add_subdirectory(apps)
endif()

install(DIRECTORY include/kiwi/queue DESTINATION include/kiwi)
if(KIWI_BUILD_INVOKE)
    install(DIRECTORY include/kiwi/invoke DESTINATION include/kiwi)
endif()
