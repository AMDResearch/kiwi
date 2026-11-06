# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Byte layout of ``kiwi::invoke::DeviceContext<SPSCD2HQueue>`` and its handles.

The FlyDSL device code reads the device ``DeviceContext`` returned by C++
``Context::device_context()`` and loads and saves the same per-endpoint
handles as C++ ``get_device_handle()``/``save_device_handle()``. Messages use
the 16-byte LL8 chunk format of ``kiwi/queue/detail/chunk.hip.hpp``.
``test_flydsl_invoke`` checks these values against the C++ headers.
"""

CONTEXT_HANDLES_OFFSET = 0
CONTEXT_NUM_ENDPOINTS_OFFSET = 8

HANDLE_SIZE = 40
CHUNKS_OFFSET = 0
CONSUMER_INDEX_OFFSET = 8
PRODUCER_INDEX_OFFSET = 16
CONSUMER_INDEX_CACHED_OFFSET = 24
LOG2_CAPACITY_OFFSET = 32  # uint32_t

CHUNK_BYTES = 16
CHUNK_PAYLOAD_BYTES = 8
CHUNKS_PER_LINE = 4
MAX_ARG_BYTES = 512  # kiwi::queue::kMaxMessageBytes
# kiwi::queue::kMaxCapacityLines: the host rejects larger rings, so lines in
# flight fit in 32 bits and the push checks credit in 32 bits.
MAX_CAPACITY_LINES = 1 << 31
MAX_CALLBACKS = 256
