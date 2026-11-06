# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""KIWI invoke for FlyDSL kernels.

Callbacks are registered and progressed on a C++ ``kiwi::invoke::Context``.
Host Python wraps that context's ``device_context()`` pointer in
:class:`Endpoints` and passes it to a ``@flyc.jit`` launcher. Inside a ``@flyc.kernel`` it is a
:class:`DeviceContext`, whose handles invoke the C++ callbacks through the
same device-to-host queues as C++ kernels::

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import kiwi_flydsl as kiwi

    @flyc.kernel
    def producer(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32):
        h = ctx.get_device_handle(0)
        for i in range(n):
            h = h.invoke(cb, i, fx.Float32(fx.Int32(i)) * 0.5)

    @flyc.jit
    def launch(ctx: kiwi.DeviceContext, cb: fx.Int32, n: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        producer(ctx, cb, n).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    # device_context and cb come from the application's C++ binding:
    # Context::device_context() and the id returned by
    # Context::register_callback for a std::function<void(int32_t, float)>.
    launch(kiwi.Endpoints(device_context), cb, 10)
"""

import os
import warnings

import flydsl

# kiwi_flydsl uses FlyDSL internals (flydsl._mlir, JitArgumentRegistry,
# Numeric.from_ir_type) that can change between releases.
TESTED_FLYDSL_VERSIONS = ("0.2.4",)

if flydsl.__version__ not in TESTED_FLYDSL_VERSIONS:
    _message = (
        f"kiwi_flydsl is tested with FlyDSL {', '.join(TESTED_FLYDSL_VERSIONS)}, "
        f"found {flydsl.__version__}"
    )
    if os.environ.get("KIWI_FLYDSL_ALLOW_UNTESTED") != "1":
        raise ImportError(f"{_message}; set KIWI_FLYDSL_ALLOW_UNTESTED=1 to try it anyway")
    warnings.warn(_message, RuntimeWarning, stacklevel=2)

from flydsl.compiler.jit_argument import JitArgumentRegistry  # noqa: E402

from ._layout import MAX_ARG_BYTES, MAX_CALLBACKS  # noqa: E402
from .device import DeviceContext, DeviceHandle  # noqa: E402
from .host import Endpoints  # noqa: E402

JitArgumentRegistry.register_jit_arg(Endpoints, DeviceContext)

__all__ = [
    "DeviceContext",
    "DeviceHandle",
    "Endpoints",
    "MAX_ARG_BYTES",
    "MAX_CALLBACKS",
]
