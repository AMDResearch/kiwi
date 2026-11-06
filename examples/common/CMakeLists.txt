# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# C++ print extension shared by the Python kernel frontend examples: a
# kiwi::invoke::Context with one callback that prints its arguments. It does
# not depend on the frontend and is placed with the frontend modules in
# KIWI_PYTHON_BINARY_DIR.

include(KiwiPybind)
kiwi_add_pybind_module(kiwi_print_ext kiwi_print_ext.hip.cpp
    OUTPUT_DIRECTORY "${KIWI_PYTHON_BINARY_DIR}")
