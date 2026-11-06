# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# invoke examples

add_invoke_example(invoke_example invoke_example.hip.cpp)

option(KIWI_BUILD_INVOKE_SDMA_PING_PONG
       "Build invoke ROCm-XIO host-SDMA ping-pong example" ON)

if(KIWI_BUILD_INVOKE_SDMA_PING_PONG)
    include(FetchContent)

    # Keep the fetched dependency focused on the library needed by this example.
    set(BUILD_CLIENTS OFF CACHE BOOL "Build ROCm-XIO client tools" FORCE)
    set(GDA_BNXT OFF CACHE BOOL "Build ROCm-XIO BNXT provider" FORCE)
    set(GDA_MLX5 OFF CACHE BOOL "Build ROCm-XIO MLX5 provider" FORCE)
    set(GDA_IONIC OFF CACHE BOOL "Build ROCm-XIO IONIC provider" FORCE)
    set(GDA_ERNIC OFF CACHE BOOL "Build ROCm-XIO ERNIC provider" FORCE)

    FetchContent_Declare(
        rocm_xio
        GIT_REPOSITORY https://github.com/ROCm/rocm-xio.git
        GIT_TAG be4e23489924d5b4e44668091ea505fdda1171d7
    )

    # ROCm-XIO revisions used for this benchmark may still assume they are the
    # top-level CMake project. Patch the fetched build script in the build tree
    # so FetchContent/add_subdirectory resolves its own source paths.
    FetchContent_GetProperties(rocm_xio)
    if(NOT rocm_xio_POPULATED)
        FetchContent_Populate(rocm_xio)

        set(_rocm_xio_add_source "${rocm_xio_SOURCE_DIR}")
        file(RELATIVE_PATH _rocm_xio_source_rel
             "${CMAKE_BINARY_DIR}" "${rocm_xio_SOURCE_DIR}")
        if(_rocm_xio_source_rel MATCHES "^[.][.]")
            set(_rocm_xio_add_source
                "${CMAKE_BINARY_DIR}/_deps/rocm_xio_fetchcontent_source")
            file(REMOVE_RECURSE "${_rocm_xio_add_source}")
            file(COPY "${rocm_xio_SOURCE_DIR}/" DESTINATION "${_rocm_xio_add_source}")
        endif()

        file(GLOB_RECURSE _rocm_xio_cmake_files CONFIGURE_DEPENDS
             "${_rocm_xio_add_source}/CMakeLists.txt"
             "${_rocm_xio_add_source}/cmake/*.cmake")
        foreach(_rocm_xio_cmake_file IN LISTS _rocm_xio_cmake_files)
            file(READ "${_rocm_xio_cmake_file}" _rocm_xio_cmake)
            string(REPLACE "\${CMAKE_SOURCE_DIR}" "\${CMAKE_CURRENT_SOURCE_DIR}"
                   _rocm_xio_cmake "${_rocm_xio_cmake}")
            file(WRITE "${_rocm_xio_cmake_file}" "${_rocm_xio_cmake}")
        endforeach()
        add_subdirectory("${_rocm_xio_add_source}" "${rocm_xio_BINARY_DIR}")
    endif()

    if(TARGET rocm-xio AND NOT TARGET rocm-xio::rocm-xio)
        add_library(rocm-xio::rocm-xio ALIAS rocm-xio)
    endif()

    find_package(MPI REQUIRED)
    find_package(hsa-runtime64 CONFIG REQUIRED)

    set_source_files_properties(invoke_sdma_ping_pong.hip.cpp PROPERTIES LANGUAGE HIP)
    add_executable(invoke_sdma_ping_pong invoke_sdma_ping_pong.hip.cpp)
    file(READ "${_rocm_xio_add_source}/src/endpoints/sdma-ep/sdma-host-queue.hip"
         _rocm_xio_sdma_host_queue_source)
    if(_rocm_xio_sdma_host_queue_source MATCHES
       "SdmaQueueHostHandle::put_signal[ \t\r\n]*\\([ \t\r\n]*void\\*[ \t\r\n]*src[ \t\r\n]*,[ \t\r\n]*void\\*[ \t\r\n]*dst")
        target_compile_definitions(invoke_sdma_ping_pong PRIVATE
            KIWI_QUEUE_ROCM_XIO_HOST_PUT_SIGNAL_SRC_DST=1)
        message(STATUS "ROCm-XIO host put_signal order detected as src,dst,size,flag,value")
    elseif(_rocm_xio_sdma_host_queue_source MATCHES
       "SdmaQueueHostHandle::put_signal[ \t\r\n]*\\([ \t\r\n]*void\\*[ \t\r\n]*dst[ \t\r\n]*,[ \t\r\n]*void\\*[ \t\r\n]*src")
        target_compile_definitions(invoke_sdma_ping_pong PRIVATE
            KIWI_QUEUE_ROCM_XIO_HOST_PUT_SIGNAL_SRC_DST=0)
        message(STATUS "ROCm-XIO host put_signal order detected as dst,src,size,flag,value")
    else()
        message(FATAL_ERROR
            "Could not detect ROCm-XIO SdmaQueueHostHandle::put_signal argument order")
    endif()
    target_link_libraries(invoke_sdma_ping_pong PRIVATE
        kiwi_invoke
        rocm-xio::rocm-xio
        hsa-runtime64::hsa-runtime64
        MPI::MPI_CXX
        pthread)
    target_include_directories(invoke_sdma_ping_pong PRIVATE
        "${_rocm_xio_add_source}/src"
        "${_rocm_xio_add_source}/src/include")
    set_hip_compile_options(invoke_sdma_ping_pong)
    set_common_definitions(invoke_sdma_ping_pong)
    target_link_options(invoke_sdma_ping_pong PRIVATE -fgpu-rdc)
endif()
