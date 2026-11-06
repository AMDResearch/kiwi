# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Python helper modules shared by the frontends of the cross-frontend harnesses
# in every area (benchmarks/*/frontends/, apps/p2p/benchmarks/frontends/),
# copied into the build tree's python/.
if(KIWI_BUILD_FLYDSL)
    configure_file(flydsl_bench_ops.py "${KIWI_PYTHON_BINARY_DIR}/flydsl_bench_ops.py" COPYONLY)
endif()
if(KIWI_BUILD_TRITON)
    configure_file(triton_bench_ops.py "${KIWI_PYTHON_BINARY_DIR}/triton_bench_ops.py" COPYONLY)
endif()
