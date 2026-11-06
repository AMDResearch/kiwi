# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# LCI is an application dependency, not part of the KIWI core.
set(LCI_WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LCI_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(LCI_WITH_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LCI_WITH_DOC OFF CACHE BOOL "" FORCE)
set(LCI_USE_HIP ON CACHE BOOL "" FORCE)
set(LCI_HIP_ARCH "${HIP_ARCHITECTURES}" CACHE STRING "" FORCE)
set(LCI_BUILD_WARN_AS_ERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    lci
    GIT_REPOSITORY https://github.com/uiuc-hpc/lci.git
    GIT_TAG d366a4053c204182e5129eefd4c9d71d63a06d68
)
FetchContent_MakeAvailable(lci)

add_subdirectory(common/topology)
if(KIWI_BUILD_P2P)
    add_subdirectory(p2p)
endif()
if(KIWI_BUILD_EP)
    add_subdirectory(ep)
endif()
