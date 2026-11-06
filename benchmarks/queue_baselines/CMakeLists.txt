# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# GPU->CPU queue baseline microbenchmarks (opt-in: KIWI_BUILD_QUEUE_BASELINES).
#
# Third-party queue code is fetched at pinned revisions with verified hashes
# and used unmodified; nothing is vendored into this repository.

include(FetchContent)

# MSCCL++ PortChannel FIFO (device header only). main as of 2026-10-16 (after
# v0.9.0, so it includes "Proxy FIFO latency improvement", microsoft/mscclpp#864).
set(KIWI_QB_MSCCLPP_COMMIT 07ce0f3e1f4eb22535fc215fc0eca18ac0efb5f4)
FetchContent_Declare(
    qb_mscclpp
    URL https://github.com/microsoft/mscclpp/archive/${KIWI_QB_MSCCLPP_COMMIT}.tar.gz
    URL_HASH SHA256=1503960f55813fdf734d4aa5861616588e03ba93ca94dfc0cc4cd47ed7486635
    # Headers only: SOURCE_SUBDIR names a directory that does not exist, so
    # MakeAvailable does not configure the MSCCL++ build.
    SOURCE_SUBDIR kiwi-headers-only
)
FetchContent_MakeAvailable(qb_mscclpp)

# LLVM libc GPU RPC (header-only libc/shared/rpc.h), main as of 2026-10-16.
set(KIWI_QB_LLVM_COMMIT cca9d7b47fe49330cf78402106ddc67a0681a084)
set(KIWI_QB_LLVM_RPC_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/llvm_rpc/${KIWI_QB_LLVM_COMMIT})
foreach(entry
        "rpc.h|9060cbcee282516804aa256c8d8ba7d9eee926f9dd2fefd4c6f309f77198b087"
        "rpc_util.h|fe6afa94c590deba4b3b0f96fe8fab4eafe1e142c6c5c3f75ce0999c2b6fce7f")
    string(REPLACE "|" ";" entry "${entry}")
    list(GET entry 0 file)
    list(GET entry 1 hash)
    if(NOT EXISTS ${KIWI_QB_LLVM_RPC_DIR}/${file})
        file(DOWNLOAD
            https://raw.githubusercontent.com/llvm/llvm-project/${KIWI_QB_LLVM_COMMIT}/libc/shared/${file}
            ${KIWI_QB_LLVM_RPC_DIR}/${file}
            EXPECTED_HASH SHA256=${hash}
            TLS_VERIFY ON)
    endif()
endforeach()

# UCCL-EP GPU->CPU command FIFO (device headers only; a vendored intermediate
# MSCCL++ FIFO). fifo_util.hpp includes <cuda.h>; UCCL builds for ROCm via
# hipify, which adapters/uccl_ep_fifo_shim reproduces with macros. main as of
# 2026-10-16.
set(KIWI_QB_UCCL_COMMIT 3ec055f3f9a6d1cdd48901abd4d430abc3530445)
set(KIWI_QB_UCCL_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/uccl_ep/${KIWI_QB_UCCL_COMMIT})
foreach(entry
        "fifo_device.hpp|f62efc80ea6327b0353de3e84be73f75e036688302b2359e02eb972bc018b7b4"
        "fifo_util.hpp|277707320cfe67da53025f68a6053a840a6e43b2c3640719a6e1fc0e6ad30553")
    string(REPLACE "|" ";" entry "${entry}")
    list(GET entry 0 file)
    list(GET entry 1 hash)
    if(NOT EXISTS ${KIWI_QB_UCCL_DIR}/uccl_ep/${file})
        file(DOWNLOAD
            https://raw.githubusercontent.com/uccl-project/uccl/${KIWI_QB_UCCL_COMMIT}/ep/include/${file}
            ${KIWI_QB_UCCL_DIR}/uccl_ep/${file}
            EXPECTED_HASH SHA256=${hash}
            TLS_VERIFY ON)
    endif()
endforeach()

# RCCL GIN proxy (device headers only, ROCm/rocm-systems projects/rccl). RCCL
# builds these headers from CUDA-style sources: it runs hipify-perl
# -experimental on each one (src/CMakeLists.txt), copies hip_compat.h as is,
# names the hipified nccl_device/core.h core_tmp.h (its unique-basename
# rule), and installs src/nccl.h.in as nccl.h (configure_file, no
# substitutions). The same steps are reproduced here. develop as of 2026-10-16.
set(KIWI_QB_RCCL_COMMIT 5909ad99a98c44a9f4e65ba7c416b3be8330e355)
set(KIWI_QB_RCCL_SRC ${CMAKE_CURRENT_BINARY_DIR}/_deps/rccl/${KIWI_QB_RCCL_COMMIT}/src)
set(KIWI_QB_RCCL_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/rccl/${KIWI_QB_RCCL_COMMIT}/include)
set(rccl_headers
    "src/nccl.h.in|a12ce01019581bee2e03556147da62549a84c092f85f7e046fb7f46b4a0bea52"
    "src/include/nccl_device/gin/gin_device_api.h|5a4568ba19f962ec8cd828a7896035ccfb9bcfbf68d459509816095db97dc548"
    "src/include/nccl_device/gin/gin_device_common.h|3e81957cf4a18484394015e6d2cff349ae1085a9a0a6b127c733073875a19107"
    "src/include/nccl_device/gin/gin_device_host_common.h|a838a5cd829181e7181f3d307bbb47eaf48e8e3caf85b0cdc0283b949c0824a7"
    "src/include/nccl_device/gin/proxy/gin_proxy.h|250b346ade8d702485850a07010b1ce0853cff66dbe66014e79e2f58d8882aa7"
    "src/include/nccl_device/gin/proxy/gin_proxy_device_host_common.h|e8e612da04746d7eecd5ad9e22c6034b8edd359ffc01116320b18615f6dc5f7a"
    "src/include/nccl_device/coop.h|0c9b0cb0ee425dc163484958b5c2fe2c60cc0cef17b83d0dfd1a7ec8e073506a"
    "src/include/nccl_device/core.h|42e9f43ededec9038009e6d21105cdef6296f2aa7ee896d011a1aa2b464f522f"
    "src/include/nccl_device/hip_compat.h|2c2a2f60a1d7dbe1f392a9aa5037a95e94370e4e18d186055b456f4e06dbfced"
    "src/include/nccl_device/net_device.h|1d3b32f8c56e4cc49625ded3e8e59ce4c6008ea081b6ecb47c8dcc428e7b30f9"
    "src/include/nccl_device/rccl_ptr.h|038c5a3b8f8e0efc3c39a4f0908b989aaa4811c637e650cbd05d1af1b8a4a0a5"
    "src/include/nccl_device/utility.h|b8a8371d76d2d4435342c50657e1ad5211921d723dce22c201fb0082adcecf0a")
find_program(KIWI_QB_HIPIFY_PERL hipify-perl
    HINTS ${ROCM_PATH}/bin $ENV{ROCM_PATH}/bin /opt/rocm/bin REQUIRED)
foreach(entry ${rccl_headers})
    string(REPLACE "|" ";" entry "${entry}")
    list(GET entry 0 file)
    list(GET entry 1 hash)
    set(src ${KIWI_QB_RCCL_SRC}/${file})
    if(NOT EXISTS ${src})
        file(DOWNLOAD
            https://raw.githubusercontent.com/ROCm/rocm-systems/${KIWI_QB_RCCL_COMMIT}/projects/rccl/${file}
            ${src}
            EXPECTED_HASH SHA256=${hash}
            TLS_VERIFY ON)
    endif()
    if(file STREQUAL "src/nccl.h.in")
        set(dst ${KIWI_QB_RCCL_DIR}/nccl.h)
    else()
        string(REPLACE "src/include/" "" rel ${file})
        if(rel STREQUAL "nccl_device/core.h")
            set(rel nccl_device/core_tmp.h)
        endif()
        set(dst ${KIWI_QB_RCCL_DIR}/${rel})
    endif()
    if(NOT EXISTS ${dst})
        get_filename_component(dst_dir ${dst} DIRECTORY)
        file(MAKE_DIRECTORY ${dst_dir})
        if(file MATCHES "nccl.h.in$|hip_compat.h$")
            configure_file(${src} ${dst} COPYONLY)
        else()
            execute_process(
                COMMAND ${KIWI_QB_HIPIFY_PERL} -experimental -quiet-warnings ${src} -o ${dst}
                RESULT_VARIABLE rc)
            if(NOT rc EQUAL 0)
                message(FATAL_ERROR "hipify-perl failed on ${file}")
            endif()
        endif()
    endif()
endforeach()

set(source bench_queue_baselines.hip.cpp)
set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
add_executable(bench_queue_baselines ${source} adapters/llvm_rpc_server.cpp)
target_link_libraries(bench_queue_baselines PRIVATE kiwi_invoke CLI11::CLI11 pthread)
target_include_directories(bench_queue_baselines PRIVATE
    ${BENCH_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bench_queue_baselines SYSTEM PRIVATE
    ${qb_mscclpp_SOURCE_DIR}/include
    ${KIWI_QB_LLVM_RPC_DIR}
    ${KIWI_QB_UCCL_DIR}
    ${KIWI_QB_RCCL_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/adapters/uccl_ep_fifo_shim)
set_hip_compile_options(bench_queue_baselines)
set_common_definitions(bench_queue_baselines)
