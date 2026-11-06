# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Header-only interface library
add_library(kiwi_queue_headers INTERFACE)
target_include_directories(kiwi_queue_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>
)

# Combined library (compile host code with hipcc)
# Rename .cpp to .hip.cpp so it's compiled with hipcc (device push() is now inline in header)
configure_file(device_to_host_queue.cpp device_to_host_queue_host.hip.cpp COPYONLY)
configure_file(mpsc_device_to_host_queue.cpp mpsc_device_to_host_queue_host.hip.cpp COPYONLY)
configure_file(host_to_device_queue.cpp host_to_device_queue_host.hip.cpp COPYONLY)
configure_file(mpmc_host_to_device_queue.cpp mpmc_host_to_device_queue_host.hip.cpp COPYONLY)
configure_file(host_memory.cpp host_memory_host.hip.cpp COPYONLY)

set_source_files_properties(
    device_to_host_queue_host.hip.cpp
    mpsc_device_to_host_queue_host.hip.cpp
    host_to_device_queue_host.hip.cpp
    mpmc_host_to_device_queue_host.hip.cpp
    host_memory_host.hip.cpp
    PROPERTIES LANGUAGE HIP
)

add_library(kiwi_queue_impl STATIC
    device_to_host_queue_host.hip.cpp
    mpsc_device_to_host_queue_host.hip.cpp
    host_to_device_queue_host.hip.cpp
    mpmc_host_to_device_queue_host.hip.cpp
    host_memory_host.hip.cpp
)

# configure_file() copies the sources into the build tree; keep the private
# chunk_consumer.hpp reachable from there.
target_include_directories(kiwi_queue_impl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(kiwi_queue_impl
    PUBLIC kiwi_queue_headers
    PRIVATE hsa-runtime64
)
set_hip_compile_options(kiwi_queue_impl)
set_common_definitions(kiwi_queue_impl)

# Main library target (interface to kiwi_queue_impl)
add_library(kiwi_queue INTERFACE)
add_library(KIWI::queue ALIAS kiwi_queue)
target_link_libraries(kiwi_queue INTERFACE kiwi_queue_impl kiwi_queue_headers)

# Export targets
install(TARGETS kiwi_queue kiwi_queue_impl kiwi_queue_headers
    EXPORT kiwiQueueTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
