# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Triton invoke example: a Triton kernel invokes a C++ callback that prints its
# arguments. The C++ side is ../common's kiwi_print_ext.

configure_file(triton_print_example.py triton_print_example.py COPYONLY)
