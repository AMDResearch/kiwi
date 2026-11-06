# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Kernel-side KIWI invoke for FlyDSL.

Mirrors ``kiwi::invoke::DeviceContext<SPSCD2HQueue>`` and its
``DeviceHandle::invoke()``: each invoke is one message in the 16-byte chunk
format of ``kiwi/queue/detail/chunk.hip.hpp``, with the argument bytes as
payload and the callback id as metadata, so the C++ dispatcher of the context
behind :class:`kiwi_flydsl.Endpoints` executes it unchanged.

Invokes are wavefront-cooperative: all 64 lanes of one wavefront must call
with identical arguments, and only one wavefront may produce into an endpoint
at a time.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm, rocdl, scf, vector
import flydsl.expr as fx
from flydsl.expr.numeric import Numeric

from . import _layout
from ._types import arg_type_size

def _i1():
    return ir.IntegerType.get_signless(1)


def _i32():
    return ir.IntegerType.get_signless(32)


def _i64():
    return ir.IntegerType.get_signless(64)


def _state_type():
    return ir.VectorType.get([2], _i64())


def _const(value, ty=None):
    ty = _i64() if ty is None else ty
    return arith.ConstantOp(ty, value).result


def _global_ptr(addr):
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), addr).result


def _load_i64(addr, volatile=False):
    return llvm.LoadOp(_i64(), _global_ptr(addr), alignment=8,
                       volatile_=volatile or None).result


def _load_u32_as_i64(addr):
    value = llvm.LoadOp(_i32(), _global_ptr(addr), alignment=4).result
    return arith.ExtUIOp(_i64(), value).result


def _store(value, addr, alignment=8):
    llvm.StoreOp(value, _global_ptr(addr), alignment=alignment)


def _add(a, b):
    return arith.AddIOp(a, b).result


def _sub(a, b):
    return arith.SubIOp(a, b).result


def _shl(a, b):
    return arith.ShLIOp(a, b).result


def _lshr(a, b):
    return arith.ShRUIOp(a, b).result


def _and(a, b):
    return arith.AndIOp(a, b).result


def _or(a, b):
    return arith.OrIOp(a, b).result


def _cmp(pred, a, b):
    return arith.CmpIOp(getattr(arith.CmpIPredicate, pred), a, b).result


def _select(cond, a, b):
    return arith.SelectOp(cond, a, b).result


def _to_i64(value, what):
    """Zero-extend or truncate an integer-like DSL value to i64."""
    if isinstance(value, bool):
        raise TypeError(f"{what} must be an integer, got bool")
    if isinstance(value, int):
        return _const(value)
    if isinstance(value, Numeric):
        value = value.ir_value()
    if not isinstance(value, ir.Value):
        raise TypeError(f"{what} must be an int or FlyDSL integer, got {type(value).__name__}")
    if isinstance(value.type, ir.IndexType):
        return arith.IndexCastUIOp(_i64(), value).result
    if not isinstance(value.type, ir.IntegerType):
        raise TypeError(f"{what} must be an integer, got {value.type}")
    width = ir.IntegerType(value.type).width
    if width < 64:
        return arith.ExtUIOp(_i64(), value).result
    if width > 64:
        return arith.TruncIOp(_i64(), value).result
    return value


def _wave_uniform(value):
    """Lane 0's ``value`` in every lane (C++ ``detail::wave_uniform``)."""
    return rocdl.readfirstlane(value.type, value)


def _lane_id():
    lo = rocdl.mbcnt_lo(_i32(), _const(-1, _i32()), _const(0, _i32()))
    return rocdl.mbcnt_hi(_i32(), _const(-1, _i32()), lo)


def _value_size(value):
    """Byte size of an integer or float MLIR value, or None if unsupported."""
    ty = value.type
    if isinstance(ty, ir.IntegerType):
        width = ir.IntegerType(ty).width
        if width == 1:
            return 1
        return width // 8 if width in (8, 16, 32, 64) else None
    for float_ty, nbytes in ((ir.F16Type, 2), (ir.F32Type, 4), (ir.F64Type, 8)):
        if isinstance(ty, float_ty):
            return nbytes
    return None


def _arg_bits(arg):
    """Return ``(bits zero-extended to i64, size in bytes)`` for one argument."""
    if isinstance(arg, Numeric):
        nbytes = arg_type_size(type(arg))
        value = arg.ir_value()
    elif isinstance(arg, ir.Value) and _value_size(arg) is not None:
        nbytes, value = _value_size(arg), arg
    else:
        raise TypeError(
            "KIWI invoke arguments must be FlyDSL numeric values such as "
            f"fx.Int32(...) or fx.Float32(...), got {type(arg).__name__}; wrap "
            "Python constants so their size matches the registered callback"
        )
    int_ty = ir.IntegerType.get_signless(8 * nbytes)
    if isinstance(value.type, ir.IntegerType):
        if ir.IntegerType(value.type).width < 8 * nbytes:
            value = arith.ExtUIOp(int_ty, value).result
    else:
        value = arith.BitcastOp(int_ty, value).result
    if nbytes < 8:
        value = arith.ExtUIOp(_i64(), value).result
    return value, nbytes


def _cast_args(args, arg_types):
    if arg_types is None:
        return args
    if len(args) != len(arg_types):
        raise TypeError(f"invoke got {len(args)} argument(s) for {len(arg_types)} arg_types")
    cast = []
    for arg, arg_type in zip(args, arg_types):
        arg_type_size(arg_type)
        if isinstance(arg, ir.Value) and not isinstance(arg, Numeric):
            arg = Numeric.from_ir_type(arg.type)(arg)
        cast.append(arg_type(arg))
    return cast


def _serialize(args):
    """Pack arguments into 64-bit payload words; returns ``(words, bytes)``.

    Word ``k`` holds message bytes ``[8k, 8k + 8)`` (the payload of chunk
    ``k``), in ``ArgumentSerializer`` order with no padding.
    """
    words = {}
    offset = 0
    for arg in args:
        bits, nbytes = _arg_bits(arg)
        word, shift = divmod(offset * 8, 64)
        part = bits if shift == 0 else _shl(bits, _const(shift))
        words[word] = part if word not in words else _or(words[word], part)
        if shift + nbytes * 8 > 64:
            spill = _lshr(bits, _const(64 - shift))
            words[word + 1] = spill if word + 1 not in words else _or(words[word + 1], spill)
        offset += nbytes

    if offset > _layout.MAX_ARG_BYTES:
        raise ValueError(
            f"KIWI invoke arguments total {offset} bytes; the limit is "
            f"{_layout.MAX_ARG_BYTES} bytes"
        )
    return words, offset


class DeviceHandle:
    """Wavefront-level producer handle for one KIWI invoke endpoint.

    Obtain one with :meth:`DeviceContext.get_device_handle`. The producer
    state is a single SSA value, so handles can be reassigned inside dynamic
    ``for``/``if`` bodies: always use the handle returned by
    :meth:`invoke`/:meth:`try_invoke`, as with the by-value C++ handle.
    """

    def __init__(self, chunks, consumer_index, log2_capacity, state):
        self._chunks = chunks
        self._consumer_index = consumer_index
        self._log2_capacity = log2_capacity
        self._state = state

    def __extract_to_ir_values__(self):
        return [self._state]

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar=None):
        if exemplar is None:
            raise TypeError("DeviceHandle can only be rebuilt from an existing handle")
        return cls(exemplar._chunks, exemplar._consumer_index, exemplar._log2_capacity, values[0])

    def _with_state(self, state):
        return DeviceHandle(self._chunks, self._consumer_index, self._log2_capacity, state)

    @property
    def producer_index(self):
        return fx.Uint64(vector.ExtractOp(self._state, [], [0]).result)

    @property
    def consumer_index_cached(self):
        return fx.Uint64(vector.ExtractOp(self._state, [], [1]).result)

    def try_invoke(self, callback_id, *args, arg_types=None):
        """Invoke once, as C++ ``invoke()``; returns ``(handle, ok)`` where ``ok`` is
        false, and nothing was sent, if the queue is full.

        See :meth:`invoke` for the arguments.
        """
        words, nbytes = _serialize(_cast_args(args, arg_types))
        meta = self._metadata(callback_id)
        state, ok = self._push(self._state, words, nbytes, meta)
        return self._with_state(state), fx.Boolean(ok)

    def invoke(self, callback_id, *args, arg_types=None):
        """Invoke, retrying while the queue is full, as C++
        ``while (!h.invoke(...)) {}``; returns the updated handle.

        ``callback_id`` is a Python int or a dynamic integer. Each argument is
        serialized with the size of its FlyDSL type, so it must match the type
        registered for the callback. FlyDSL arithmetic can widen types (for
        example ``fx.Int16(x) - 300`` is an ``Int32``); pass the registered
        types as ``arg_types`` to cast the arguments before serialization.
        """
        words, nbytes = _serialize(_cast_args(args, arg_types))
        meta = self._metadata(callback_id)
        state_ty = _state_type()
        loop = scf.WhileOp([state_ty], [self._state])
        before = loop.before.blocks.append(state_ty)
        with ir.InsertionPoint(before):
            state, ok = self._push(before.arguments[0], words, nbytes, meta)
            retry = arith.XOrIOp(ok, _const(1, _i1())).result
            scf.ConditionOp(retry, [state])
        after = loop.after.blocks.append(state_ty)
        with ir.InsertionPoint(after):
            scf.YieldOp([after.arguments[0]])
        return self._with_state(loop.results[0])

    @staticmethod
    def _metadata(callback_id):
        if isinstance(callback_id, int) and not isinstance(callback_id, bool):
            if not 0 <= callback_id < _layout.MAX_CALLBACKS:
                raise ValueError(
                    f"callback_id {callback_id} is outside 0..{_layout.MAX_CALLBACKS - 1}")
            return _const(callback_id << 8)
        # Metadata bits 0-7 sit in bits 8-15 of sig 0 (chunk.hip.hpp
        # make_signal()); callback ids fit in them, so _push leaves the
        # metadata bits of sig 1 zero.
        return _shl(_and(_to_i64(callback_id, "callback_id"), _const(0xFF)), _const(8))

    def _push(self, state, words, nbytes, meta):
        """Emit SPSCD2HQueue::DeviceHandle::push_lanes() for an ``nbytes`` message;
        returns ``(state, ok)``."""
        prod = vector.ExtractOp(state, [], [0]).result
        cached = vector.ExtractOp(state, [], [1]).result
        cap = _shl(_const(1), self._log2_capacity)
        num_chunks = max(1, -(-nbytes // _layout.CHUNK_PAYLOAD_BYTES))
        num_lines = -(-num_chunks // _layout.CHUNKS_PER_LINE)
        lines = _const(num_lines)

        # Credit check in 32 bits, as C++: lines in flight never exceed the
        # capacity, which the host keeps at most kMaxCapacityLines (2^31), so
        # used + lines cannot wrap.
        cap32 = arith.TruncIOp(_i32(), cap).result

        def in_flight(credit):
            used = arith.TruncIOp(_i32(), _sub(prod, credit)).result
            return arith.AddIOp(used, _const(num_lines, _i32())).result

        # Rare path: refresh the consumer index from host memory when the cached
        # value says the queue is full. readfirstlane keeps it wave-uniform
        # (C++ broadcasts lane 0's load), so every lane agrees on ok and state.
        # A message longer than the ring could never fit, so it traps as in C++.
        full = llvm.intr_expect(_cmp("ugt", in_flight(cached), cap32), _const(0, _i1()))
        refresh = scf.IfOp(full, [_i64(), _i1()], has_else=True)
        with ir.InsertionPoint(refresh.regions[0].blocks[0]):
            too_long = scf.IfOp(_cmp("ugt", lines, cap), [], has_else=False)
            with ir.InsertionPoint(too_long.regions[0].blocks[0]):
                llvm.Trap()
                scf.YieldOp([])
            fresh = rocdl.readfirstlane(_i64(), _load_i64(self._consumer_index, volatile=True))
            scf.YieldOp([fresh, _cmp("ule", in_flight(fresh), cap32)])
        with ir.InsertionPoint(refresh.regions[1].blocks[0]):
            scf.YieldOp([cached, _const(1, _i1())])
        cached, ok = refresh.results[0], refresh.results[1]

        # write_message(): lane c stores chunk c = {data lo, sig 0, data hi,
        # sig 1} for payload word c and signal pair sig 0 | sig 1 << 32,
        # only when ok (C++ writes under ok instead of returning early).
        lane = _lane_id()
        num_lanes = num_lines * _layout.CHUNKS_PER_LINE
        active = _and(ok, _cmp("ult", lane, _const(num_lanes, _i32())))
        store = scf.IfOp(active, [], has_else=False)
        with ir.InsertionPoint(store.regions[0].blocks[0]):
            lane64 = arith.ExtUIOp(_i64(), lane).result
            line = _add(prod, _lshr(lane64, _const(2)))

            data = _const(0)
            for k, word in sorted(words.items()):
                data = _select(_cmp("eq", lane64, _const(k)), word, data)

            # Built from 32-bit parts, the chunk's four dwords as C++
            # make_chunk(). Lane 0's header carries no epoch, so it is
            # loop-invariant; every lane ORs in its line's epoch (C++
            # write_message()). Callback ids are below 256, so the metadata
            # sits entirely in sig 0 (bits 8-15) and sig 1 is just the epoch.
            epoch = arith.TruncIOp(
                _i32(), _and(_lshr(line, self._log2_capacity), _const(0xFF))).result
            header = _select(_cmp("eq", lane64, _const(0)),
                             arith.TruncIOp(_i32(), _or(_const(nbytes << 16), meta)).result,
                             _const(0, _i32()))
            sig0 = arith.OrIOp(header, epoch).result
            data_lo = arith.TruncIOp(_i32(), data).result
            data_hi = arith.TruncIOp(_i32(), _lshr(data, _const(32))).result
            chunk = vector.FromElementsOp(ir.VectorType.get([4], _i32()),
                                          [data_lo, sig0, data_hi, epoch]).result

            ring_line = _and(line, _sub(cap, _const(1)))
            addr = _add(self._chunks, _add(_shl(ring_line, _const(6)),
                                           _shl(_and(lane64, _const(3)), _const(4))))
            _store(chunk, addr, alignment=_layout.CHUNK_BYTES)
            scf.YieldOp([])

        new_prod = _select(ok, _add(prod, lines), prod)
        new_state = vector.FromElementsOp(_state_type(), [new_prod, cached]).result
        return new_state, ok


class DeviceContext:
    """Kernel-side view of :class:`kiwi_flydsl.Endpoints`.

    Holds the address of the device ``kiwi::invoke::DeviceContext``, which
    points to the per-endpoint ``DeviceHandle`` array. Annotate ``@flyc.jit``
    and ``@flyc.kernel`` parameters with this type.
    """

    def __init__(self, context):
        self._context = context

    def __extract_to_ir_values__(self):
        return [self._context]

    @classmethod
    def __construct_from_ir_values__(cls, values, exemplar=None):
        return cls(values[0])

    @property
    def num_endpoints(self):
        return fx.Uint64(_load_i64(
            _add(self._context, _const(_layout.CONTEXT_NUM_ENDPOINTS_OFFSET))))

    def _handle_addr(self, endpoint_id):
        if endpoint_id is None:
            endpoint_id = fx.block_idx.x
        ep = _to_i64(endpoint_id, "endpoint_id")
        handles = _load_i64(_add(self._context, _const(_layout.CONTEXT_HANDLES_OFFSET)))
        return _add(handles, arith.MulIOp(ep, _const(_layout.HANDLE_SIZE)).result)

    def get_device_handle(self, endpoint_id=None):
        """Load the handle for ``endpoint_id`` (default: ``block_idx.x``).

        As in C++, the handle starts from the producer progress last written
        by :meth:`save_device_handle` (or the queue's state when the context
        was created). Call this once per endpoint and keep using the returned
        handle.

        The handle is wave-uniform, as C++ ``get_device_handle()`` returns it:
        every field is lane 0's value (``readfirstlane``), so the queue state
        stays in SGPRs even when ``endpoint_id`` is computed per lane. One
        wavefront owns an endpoint, and the whole wavefront must be active.
        """
        base = self._handle_addr(endpoint_id)

        def field(offset):
            return _wave_uniform(_load_i64(_add(base, _const(offset))))

        state = vector.FromElementsOp(
            _state_type(),
            [field(_layout.PRODUCER_INDEX_OFFSET), field(_layout.CONSUMER_INDEX_CACHED_OFFSET)],
        ).result
        return DeviceHandle(
            field(_layout.CHUNKS_OFFSET),
            field(_layout.CONSUMER_INDEX_OFFSET),
            _wave_uniform(_load_u32_as_i64(_add(base, _const(_layout.LOG2_CAPACITY_OFFSET)))),
            state,
        )

    def save_device_handle(self, endpoint_id, handle):
        """Persist ``handle``'s producer progress for later kernels (lane 0 stores)."""
        base = self._handle_addr(endpoint_id)
        lane0 = _cmp("eq", _lane_id(), _const(0, _i32()))
        store = scf.IfOp(lane0, [], has_else=False)
        with ir.InsertionPoint(store.regions[0].blocks[0]):
            _store(vector.ExtractOp(handle._state, [], [0]).result,
                   _add(base, _const(_layout.PRODUCER_INDEX_OFFSET)))
            _store(vector.ExtractOp(handle._state, [], [1]).result,
                   _add(base, _const(_layout.CONSUMER_INDEX_CACHED_OFFSET)))
            scf.YieldOp([])
