# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Example helper functions
function(add_hip_example example_name source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)

    add_executable(${example_name} ${source})
    target_link_libraries(${example_name} PRIVATE kiwi_queue)
    set_hip_compile_options(${example_name})
    set_common_definitions(${example_name})
endfunction()

function(add_invoke_example example_name source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)

    add_executable(${example_name} ${source})
    target_link_libraries(${example_name} PRIVATE kiwi_invoke)
    set_hip_compile_options(${example_name})
    set_common_definitions(${example_name})
endfunction()

# queue examples
add_subdirectory(queue)

# invoke examples (conditional)
if(KIWI_BUILD_INVOKE)
    add_subdirectory(invoke)
endif()

# C++ extension shared by the Python kernel frontend examples
if(KIWI_PYTHON_FRONTENDS)
    add_subdirectory(common)
endif()

if(KIWI_BUILD_FLYDSL)
    add_subdirectory(flydsl)
endif()

if(KIWI_BUILD_TRITON)
    add_subdirectory(triton)
endif()
