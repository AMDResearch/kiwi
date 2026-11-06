# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Test helper function
function(add_hip_test test_name source)
    if(source MATCHES "\\.hip\\.cpp$")
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    endif()

    add_executable(${test_name} ${source})
    target_link_libraries(${test_name} PRIVATE
        kiwi_queue
        GTest::gtest
        GTest::gtest_main
    )

    if(source MATCHES "\\.hip\\.cpp$")
        set_hip_compile_options(${test_name})
    else()
        # For C++ files, add HIP definitions and includes
        target_include_directories(${test_name} PRIVATE ${ROCM_PATH}/include)
        target_compile_definitions(${test_name} PRIVATE
            __HIP_PLATFORM_AMD__=1
            __host__=
            __device__=
        )
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# KIWI Invoke test helper function
function(add_invoke_test test_name source)
    if(source MATCHES "\\.hip\\.cpp$")
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    endif()

    add_executable(${test_name} ${source})
    target_link_libraries(${test_name} PRIVATE
        kiwi_invoke
        GTest::gtest
        GTest::gtest_main
    )

    if(source MATCHES "\\.hip\\.cpp$")
        set_hip_compile_options(${test_name})
    else()
        # For C++ files, add HIP definitions and includes
        target_include_directories(${test_name} PRIVATE ${ROCM_PATH}/include)
        target_compile_definitions(${test_name} PRIVATE
            __HIP_PLATFORM_AMD__=1
            __host__=
            __device__=
        )
    endif()

    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Standalone HIP test (own main(), no gtest)
function(add_hip_standalone_test test_name source)
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    add_executable(${test_name} ${source})
    target_link_libraries(${test_name} PRIVATE kiwi_queue)
    set_hip_compile_options(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# queue tests
add_subdirectory(queue)

# invoke tests (conditional)
if(KIWI_BUILD_INVOKE)
    add_subdirectory(invoke)
endif()

if(KIWI_PYTHON_FRONTENDS)
    add_subdirectory(python)
endif()
