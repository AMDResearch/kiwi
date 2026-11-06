# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# invoke tests

add_invoke_test(test_invoke_basic test_invoke_basic.hip.cpp)
add_invoke_test(test_invoke_concurrent test_invoke_concurrent.hip.cpp)
add_invoke_test(test_invoke_debug test_invoke_debug.hip.cpp)
add_invoke_test(test_invoke_shared test_invoke_shared.hip.cpp)
add_invoke_test(test_invoke_chunk_queue test_invoke_chunk_queue.hip.cpp)
