# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Python invoke frontend tests. Each script exits 77 when its frontend,
# PyTorch, or a GPU is unavailable, which CTest reports as skipped.
#
# kiwi_invoke_test_ext owns the C++ invoke context and callbacks the test
# kernels invoke, and exports the C++ layout the frontends hard-code.

include(KiwiPybind)
kiwi_add_pybind_module(kiwi_invoke_test_ext kiwi_invoke_test_ext.hip.cpp)

set(_frontend_tests)
if(KIWI_BUILD_FLYDSL)
    add_test(NAME test_flydsl_invoke
        COMMAND ${KIWI_FLYDSL_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test_flydsl_invoke.py -v)
    list(APPEND _frontend_tests test_flydsl_invoke)
endif()
if(KIWI_BUILD_TRITON)
    add_test(NAME test_triton_invoke
        COMMAND ${KIWI_TRITON_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test_triton_invoke.py -v)
    list(APPEND _frontend_tests test_triton_invoke)
endif()
set_tests_properties(${_frontend_tests} PROPERTIES
    ENVIRONMENT "PYTHONPATH=${KIWI_PYTHON_BINARY_DIR}:${CMAKE_CURRENT_BINARY_DIR}:$ENV{PYTHONPATH}"
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
    TIMEOUT 600)
