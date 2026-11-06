# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Optional Python packages:
#   KIWI_BUILD_FLYDSL      pure-Python FlyDSL invoke frontend (kiwi_flydsl)
#   KIWI_BUILD_TRITON      Triton invoke frontend (kiwi_triton)

if(KIWI_BUILD_FLYDSL)
    # FlyDSL invoke frontend (pure-Python package kiwi_flydsl).
    #
    # Build with: cmake -DKIWI_BUILD_FLYDSL=ON ...
    # The importable package is assembled in ${CMAKE_CURRENT_BINARY_DIR}; put that
    # directory on PYTHONPATH in the Python environment that provides FlyDSL.

    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(_kiwi_flydsl_sources
        kiwi_flydsl/__init__.py
        kiwi_flydsl/_layout.py
        kiwi_flydsl/_types.py
        kiwi_flydsl/device.py
        kiwi_flydsl/host.py
    )
    set(_kiwi_flydsl_outputs)
    foreach(_src IN LISTS _kiwi_flydsl_sources)
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/${_src}")
        add_custom_command(
            OUTPUT "${_out}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/${_src}" "${_out}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}"
            COMMENT "Copying ${_src}"
        )
        list(APPEND _kiwi_flydsl_outputs "${_out}")
    endforeach()
    add_custom_target(kiwi_flydsl ALL DEPENDS ${_kiwi_flydsl_outputs})

    set(KIWI_FLYDSL_PYTHON "${Python3_EXECUTABLE}" PARENT_SCOPE)
    set(KIWI_FLYDSL_PYTHONPATH "${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
endif()

if(KIWI_BUILD_TRITON)
    # Triton invoke frontend: the module kiwi_triton.py, the Triton device
    # side of KIWI-Invoke, copied to ${CMAKE_CURRENT_BINARY_DIR}; put that
    # directory on PYTHONPATH in the Python environment that provides Triton.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    configure_file(kiwi_triton.py kiwi_triton.py COPYONLY)
    set(KIWI_TRITON_PYTHON "${Python3_EXECUTABLE}" PARENT_SCOPE)
endif()
