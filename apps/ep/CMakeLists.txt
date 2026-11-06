# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(KIWI_EP_SOURCES
    comm/context.hip.cpp
    all2all.hip.cpp
    ep_context.hip.cpp
    dispatch.hip.cpp
    combine.hip.cpp
    comm/transport_callbacks.hip.cpp
)

add_library(kiwi_ep STATIC ${KIWI_EP_SOURCES})

set_source_files_properties(
    ${KIWI_EP_SOURCES}
    PROPERTIES LANGUAGE HIP
)

target_include_directories(kiwi_ep PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(kiwi_ep PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(kiwi_ep PUBLIC kiwi_invoke LCI kiwi_topology)

if(KIWI_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(KIWI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
if(KIWI_BUILD_PYTHON)
    add_subdirectory(python)
endif()
