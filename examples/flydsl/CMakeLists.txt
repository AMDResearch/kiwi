# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# FlyDSL invoke example: a FlyDSL kernel invokes a C++ callback that prints its
# arguments. The C++ side is ../common's kiwi_print_ext.

configure_file(flydsl_print_example.py flydsl_print_example.py COPYONLY)
