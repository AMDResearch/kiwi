# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_library(kiwi_topology STATIC topology.cpp)
set_source_files_properties(topology.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(kiwi_topology PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(kiwi_topology PUBLIC LCI)
