# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

find_package(MPI REQUIRED)

# Tests for kiwi-p2p

add_executable(test_hello test_hello.hip.cpp)
set_source_files_properties(test_hello.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(test_hello PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(test_hello PRIVATE kiwi_p2p)

add_test(NAME test_hello COMMAND test_hello)

add_executable(test_context test_context.hip.cpp)
set_source_files_properties(test_context.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(test_context PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(test_context PRIVATE kiwi_p2p)

add_test(NAME test_context COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_context>)

add_executable(test_comp_counter test_comp_counter.hip.cpp)
set_source_files_properties(test_comp_counter.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(test_comp_counter PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(test_comp_counter PRIVATE kiwi_p2p)

add_test(NAME test_comp_counter COMMAND test_comp_counter)

add_executable(test_memory test_memory.hip.cpp)
set_source_files_properties(test_memory.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(test_memory PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(test_memory PRIVATE kiwi_p2p)

add_test(NAME test_memory COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_memory>)

add_executable(test_pingpong test_pingpong.hip.cpp)
set_source_files_properties(test_pingpong.hip.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(test_pingpong PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(test_pingpong PRIVATE kiwi_p2p)

add_test(NAME test_pingpong COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_pingpong>)

# Hardware-dependent manual test. Do not register with CTest: it must run
# across two nodes with the intended network provider to exercise NIC writes.
add_executable(test_lci_put_atomicity test_lci_put_atomicity.hip.cpp)
set_source_files_properties(test_lci_put_atomicity.hip.cpp
    PROPERTIES LANGUAGE HIP
)
set_target_properties(test_lci_put_atomicity PROPERTIES
    HIP_ARCHITECTURES "${HIP_ARCHITECTURES}"
)
target_link_libraries(test_lci_put_atomicity PRIVATE kiwi_p2p CLI11::CLI11)
