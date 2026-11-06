# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Host-side reference to C++ KIWI invoke endpoints for FlyDSL kernels."""

import ctypes
import hashlib
from pathlib import Path

from flydsl._mlir import ir

# FlyDSL's disk cache does not hash library code that kernels call, so key
# cached kernels on this package's sources to drop them when it changes.
_SOURCE_DIGEST = hashlib.sha256(
    b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
).hexdigest()[:16]


class Endpoints:
    """The device context of a C++ ``kiwi::invoke::Context<SPSCD2HQueue>``.

    ``device_context`` is the integer address returned by the C++
    ``Context::device_context()``, the same pointer C++ kernels receive.
    Callbacks are registered, and ``progress()`` is called, on the C++
    context. Pass this object to a ``@flyc.jit`` launcher and on to a
    ``@flyc.kernel``; inside the kernel it is a
    :class:`kiwi_flydsl.DeviceContext`.

    This object does not own the endpoints: the C++ context must outlive every
    kernel launched with it.
    """

    def __init__(self, device_context):
        if not device_context:
            raise ValueError("device_context must be a non-null device address")
        self.device_context = int(device_context)

    # FlyDSL JitArgument protocol: lowered to the DeviceContext address.

    def __get_ir_types__(self):
        return [ir.IntegerType.get_signless(64)]

    def __cache_signature__(self):
        return (Endpoints, _SOURCE_DIGEST)

    def __c_abi_spec__(self):
        def fill(arg, slot):
            slot.value = arg.device_context

        return [(ctypes.c_uint64, fill)]
