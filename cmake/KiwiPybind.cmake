# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# kiwi_add_pybind_module(<target> <source> [OUTPUT_DIRECTORY <dir>])
#
# Builds a pybind11 extension module named <target> that links kiwi_invoke,
# placed in <dir> (default: the current binary directory). Use the Python
# environment of the kernel frontend so the module links the same HIP runtime
# as the frontend and PyTorch.
function(kiwi_add_pybind_module target source)
    cmake_parse_arguments(ARG "" "OUTPUT_DIRECTORY" "" ${ARGN})
    if(NOT ARG_OUTPUT_DIRECTORY)
        set(ARG_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_pybind11_dir)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
    endif()
    find_package(pybind11 CONFIG REQUIRED)

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
                "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
        OUTPUT_VARIABLE _ext_suffix
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    add_library(${target} MODULE ${source})
    set_target_properties(${target} PROPERTIES
        PREFIX ""
        SUFFIX "${_ext_suffix}"
        LIBRARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
        CXX_VISIBILITY_PRESET hidden
        HIP_VISIBILITY_PRESET hidden
    )
    target_link_libraries(${target} PRIVATE kiwi_invoke pybind11::module)
    set_hip_compile_options(${target})
endfunction()
