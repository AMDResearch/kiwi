# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Compile options for HIP targets
# Shared by core and application targets.

function(set_hip_compile_options target)
    set_target_properties(${target} PROPERTIES
        HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
    )
endfunction()

function(set_common_definitions target)
    if(KIWI_INVOKE_USE_RINGBUFFER)
        target_compile_definitions(${target} PUBLIC KIWI_INVOKE_USE_RINGBUFFER)
    endif()
endfunction()
