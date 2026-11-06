# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_subdirectory(queue)

# The invocation layer is header-only and depends only on the queue.
if(KIWI_BUILD_INVOKE)
    add_library(kiwi_invoke INTERFACE)
    add_library(KIWI::invoke ALIAS kiwi_invoke)
    target_include_directories(kiwi_invoke INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(kiwi_invoke INTERFACE kiwi_queue)
    install(TARGETS kiwi_invoke
        EXPORT kiwiInvokeTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
endif()
