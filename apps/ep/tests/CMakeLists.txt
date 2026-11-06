# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

find_package(MPI REQUIRED)

function(add_ep_test name source)
    add_executable(${name} ${source})
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    target_link_libraries(${name} PRIVATE kiwi_ep MPI::MPI_CXX)
    add_test(NAME ${name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:${name}>)
endfunction()

add_ep_test(test_dispatch test_dispatch.hip.cpp)
add_ep_test(test_combine  test_combine.hip.cpp)
add_ep_test(test_ep_runtime_guard test_ep_runtime_guard.hip.cpp)
add_executable(test_ep_block_count test_ep_block_count.hip.cpp)
set_source_files_properties(test_ep_block_count.hip.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(test_ep_block_count PRIVATE kiwi_ep)
add_test(NAME test_ep_block_count COMMAND $<TARGET_FILE:test_ep_block_count>)
add_test(NAME test_combine_ws4
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:test_combine>)
add_ep_test(test_combine_graph_replay test_combine_graph_replay.hip.cpp)
add_ep_test(test_ep_collective_primitives test_ep_collective_primitives.hip.cpp)

add_executable(test_dispatch_combine_harness test_dispatch_combine_harness.hip.cpp)
set_source_files_properties(test_dispatch_combine_harness.hip.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(test_dispatch_combine_harness PRIVATE kiwi_ep CLI11::CLI11 MPI::MPI_CXX)
add_test(NAME test_dispatch_combine_harness
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_dispatch_combine_harness>
                 --tokens 4 --hidden 128 --experts 8 --topk 2 --iters 2 --warmup 1)
add_test(NAME test_dispatch_combine_harness_combine
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_dispatch_combine_harness>
                 --tokens 4 --hidden 128 --experts 8 --topk 2 --iters 2 --warmup 1
                 --combine)

add_executable(test_rcomp_readiness_layout
    test_rcomp_readiness_layout.hip.cpp)
set_source_files_properties(test_rcomp_readiness_layout.hip.cpp
                            PROPERTIES LANGUAGE HIP)
target_link_libraries(test_rcomp_readiness_layout PRIVATE kiwi_ep)
add_test(NAME test_rcomp_readiness_layout
         COMMAND $<TARGET_FILE:test_rcomp_readiness_layout>)

# These tests intentionally require two ranks on one hostname. Hybrid setup is
# strict: unavailable peer access, native atomics, or HIP IPC mappings fail
# rather than silently falling back to RDMA.
add_ep_test(test_rcomp_hybrid_ipc test_rcomp_hybrid_ipc.hip.cpp)
add_test(NAME test_combine_rcomp_hybrid
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 $<TARGET_FILE:test_combine>
                 --rcomp-backend hybrid)
add_test(NAME test_combine_graph_replay_rcomp_hybrid
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 $<TARGET_FILE:test_combine_graph_replay>
                 --rcomp-backend hybrid)
add_test(NAME test_combine_graph_replay_auto_policy
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 $<TARGET_FILE:test_combine_graph_replay>
                 --rcomp-backend auto)
add_test(NAME test_dispatch_combine_harness_rcomp_hybrid_route
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                 $<TARGET_FILE:test_dispatch_combine_harness>
                 --tokens 4 --hidden 128 --experts 8 --topk 2
                 --iters 1 --warmup 0 --combine
                 --routing all-ranks --verify-results
                 --rcomp-backend hybrid --assert-ipc-peer-count 1
                 --assert-rdma-peer-count 0)
