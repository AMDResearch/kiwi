# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Invoke argument types and their serialized sizes.

Arguments are packed back to back with no padding, little-endian, exactly like
``kiwi::invoke::internal::ArgumentSerializer``. ``Boolean`` occupies one byte,
matching C++ ``bool``.
"""

import flydsl.expr as fx

# FlyDSL numeric type -> size in bytes
ARG_TYPES = {
    fx.Boolean: 1,
    fx.Int8: 1,
    fx.Uint8: 1,
    fx.Int16: 2,
    fx.Uint16: 2,
    fx.Int32: 4,
    fx.Uint32: 4,
    fx.Int64: 8,
    fx.Uint64: 8,
    fx.Float16: 2,
    fx.Float32: 4,
    fx.Float64: 8,
}


def arg_type_size(arg_type):
    """Return the serialized size of a supported FlyDSL numeric type."""
    try:
        return ARG_TYPES[arg_type]
    except (KeyError, TypeError):
        supported = ", ".join(t.__name__ for t in ARG_TYPES)
        raise TypeError(
            f"unsupported KIWI invoke argument type {arg_type!r}; supported: {supported}"
        ) from None
