# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Checks the gfx942 ISA of the push kernels in isa_push_paths.hip.cpp:
#
#   cmake -DASM=<isa_push_paths.s> -P check_push_isa.cmake
#
# Every kernel must use no scratch (a payload in scratch makes each push wait
# for the previous push's host write), store the ring with
# global_store_dwordx4 and no flat/buffer/scratch accesses, and broadcast
# wave-uniform values with v_readfirstlane_b32, not ds_bpermute (__shfl).
# The invoke kernel may load its handle from DeviceContext with one
# flat_load. No kernel may compare two 64-bit VGPR pairs (v_cmp_*_u64 with two
# v[a:b] operands): the queue indices are wave-uniform, so such a compare means
# the handle sits in VGPRs and every push runs a divergent capacity check.
# DeviceContext::get_device_handle() must return a wave-uniform handle.
# The SPSC invoke loop (a constant-size producer loop) may not use any 64-bit
# VALU compare: its credit check runs in 32 bits on the scalar unit (a 64-bit
# unsigned compare has no SALU form and costs a VALU-to-vcc round trip).
# MPSC kernels must also claim lines with a global (not flat) 64-bit
# fetch-add and wait for credit with s_sleep.

file(READ "${ASM}" isa)
set(failures 0)
foreach(kernel
        spsc_push_reg spsc_push_shared spsc_push_global spsc_push_lanes spsc_invoke_context
        spsc_invoke_loop
        mpsc_push_reg mpsc_push_shared mpsc_push_global mpsc_push_lanes mpsc_invoke_context)
    string(FIND "${isa}" "\n${kernel}:" begin)
    string(FIND "${isa}" ".amdhsa_kernel ${kernel}\n" desc)
    if(begin EQUAL -1 OR desc EQUAL -1)
        message(SEND_ERROR "${kernel}: not found in ${ASM}")
        math(EXPR failures "${failures} + 1")
        continue()
    endif()
    string(SUBSTRING "${isa}" ${begin} -1 body)
    string(FIND "${body}" ".Lfunc_end" end)
    string(SUBSTRING "${body}" 0 ${end} body)
    string(SUBSTRING "${isa}" ${desc} 4000 desc)

    set(want global_store_dwordx4 v_readfirstlane_b32)
    if(kernel MATCHES "^mpsc_")
        list(APPEND want global_atomic_add_x2 s_sleep)
    endif()
    set(avoid flat_store flat_atomic scratch_ buffer_store buffer_load ds_bpermute)
    if(NOT kernel MATCHES "_invoke_(context|loop)$")
        list(APPEND avoid flat_load)
    endif()
    set(bad "")
    string(REGEX MATCH "v_cmp_[a-z]+_u64[^\n]*v\\[[0-9]+:[0-9]+\\], v\\[[0-9]+:[0-9]+\\]"
           divergent_cmp "${body}")
    if(divergent_cmp)
        list(APPEND bad "divergent 64-bit compare (${divergent_cmp})")
    endif()
    if(kernel STREQUAL "spsc_invoke_loop")
        string(REGEX MATCH "v_cmpx?_[a-z]+_[ui]64[^\n]*" valu_cmp "${body}")
        if(valu_cmp)
            list(APPEND bad "64-bit VALU credit compare (${valu_cmp})")
        endif()
    endif()
    if(NOT desc MATCHES "amdhsa_private_segment_fixed_size 0\n")
        list(APPEND bad "uses scratch")
    endif()
    foreach(pattern IN LISTS want)
        string(FIND "${body}" "${pattern}" pos)
        if(pos EQUAL -1)
            list(APPEND bad "missing ${pattern}")
        endif()
    endforeach()
    foreach(pattern IN LISTS avoid)
        string(FIND "${body}" "${pattern}" pos)
        if(NOT pos EQUAL -1)
            list(APPEND bad "has ${pattern}")
        endif()
    endforeach()
    if(bad)
        message(SEND_ERROR "${kernel}: ${bad}")
        math(EXPR failures "${failures} + 1")
    else()
        message(STATUS "${kernel}: ok")
    endif()
endforeach()
if(failures GREATER 0)
    message(FATAL_ERROR "${failures} push kernel(s) failed the ISA check")
endif()
