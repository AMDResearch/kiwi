# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# kiwi_p2p library
add_library(kiwi_p2p STATIC
    context.hip.cpp
)
set_source_files_properties(context.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(kiwi_p2p PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(kiwi_p2p PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(kiwi_p2p PUBLIC kiwi_invoke LCI kiwi_topology)

if(KIWI_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(KIWI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
