# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for ``AsyncStreamingEncoder`` — asyncio streaming encoder.

The async sibling of :class:`StreamingEncoder`, wrapping the Rust core
``AsyncStreamingEncoder`` over an in-memory buffer.  Every write returns
a coroutine; :meth:`finish` yields the complete wire-format bytes.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import tensogram

_DESCRIPTOR = {
    "type": "ntensor",
    "shape": [4],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}
_PAYLOAD = np.arange(4, dtype=np.float32)
_GLOBAL_META = {"base": [{"name": "test"}]}


def _preamble_total_length(message: bytes) -> int:
    return struct.unpack(">Q", message[16:24])[0]


def _postamble_total_length(message: bytes) -> int:
    return struct.unpack(">Q", message[-16:-8])[0]


@pytest.mark.asyncio
async def test_create_write_finish_round_trip() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = await enc.finish()

    assert isinstance(msg, bytes)
    decoded = tensogram.decode(msg)
    assert len(decoded.objects) == 1
    descriptor, data = decoded.objects[0]
    assert descriptor.dtype == "float32"
    assert list(descriptor.shape) == [4]
    np.testing.assert_array_equal(data, _PAYLOAD)


@pytest.mark.asyncio
async def test_finish_writes_streaming_mode_zero_lengths() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = await enc.finish()
    assert _preamble_total_length(msg) == 0
    assert _postamble_total_length(msg) == 0


@pytest.mark.asyncio
async def test_finish_backfilled_is_backward_locatable() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = await enc.finish_backfilled()

    assert _preamble_total_length(msg) == len(msg)
    assert _postamble_total_length(msg) == len(msg)
    # Backfilled messages are forward-scannable in a concatenated stream.
    assert list(tensogram.scan(msg)) == [(0, len(msg))]


@pytest.mark.asyncio
async def test_structurally_matches_sync_encoder() -> None:
    # Same logical writes → same object count, same payloads, same length
    # (the wire bytes differ only in the random per-message provenance UUID).
    sync = tensogram.StreamingEncoder(_GLOBAL_META)
    sync.write_object(_DESCRIPTOR, _PAYLOAD)
    sync_msg = sync.finish()

    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    async_msg = await enc.finish()

    assert len(async_msg) == len(sync_msg)
    sync_dec = tensogram.decode(sync_msg)
    async_dec = tensogram.decode(async_msg)
    assert len(sync_dec.objects) == len(async_dec.objects) == 1
    np.testing.assert_array_equal(async_dec.objects[0][1], sync_dec.objects[0][1])


@pytest.mark.asyncio
async def test_write_preceder_metadata_lands_in_base() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_preceder({"step": 7, "param": "2t"})
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = await enc.finish()

    decoded = tensogram.decode(msg)
    assert decoded.metadata.base[0].get("step") == 7
    assert decoded.metadata.base[0].get("param") == "2t"


@pytest.mark.asyncio
async def test_write_object_pre_encoded() -> None:
    # `encoding="none"` + `compression="none"` → the encoded payload is just
    # the native-endian bytes, so we can feed them straight to pre_encoded.
    pre = _PAYLOAD.tobytes()
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object_pre_encoded(_DESCRIPTOR, pre)
    msg = await enc.finish()

    decoded = tensogram.decode(msg)
    np.testing.assert_array_equal(decoded.objects[0][1], _PAYLOAD)


@pytest.mark.asyncio
async def test_write_object_pre_encoded_rejects_numpy() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    with pytest.raises(ValueError, match=r"(?i)bytes"):
        await enc.write_object_pre_encoded(_DESCRIPTOR, _PAYLOAD)


@pytest.mark.asyncio
async def test_introspection() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    assert enc.object_count() == 0
    after_header = enc.bytes_written()
    assert after_header > 0

    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    assert enc.object_count() == 1
    assert enc.bytes_written() > after_header

    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    assert enc.object_count() == 2


@pytest.mark.asyncio
async def test_multi_object_message() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    for i in range(3):
        await enc.write_preceder({"idx": i})
        await enc.write_object(_DESCRIPTOR, _PAYLOAD + float(i))
    assert enc.object_count() == 3
    msg = await enc.finish()

    decoded = tensogram.decode(msg)
    assert len(decoded.objects) == 3
    for i, (_desc, data) in enumerate(decoded.objects):
        np.testing.assert_array_equal(data, _PAYLOAD + float(i))


@pytest.mark.asyncio
async def test_double_finish_raises() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    await enc.finish()
    with pytest.raises(RuntimeError, match="finished"):
        await enc.finish()


@pytest.mark.asyncio
async def test_write_after_finish_raises() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    await enc.finish()
    with pytest.raises(RuntimeError, match="finished"):
        await enc.write_object(_DESCRIPTOR, _PAYLOAD)


@pytest.mark.asyncio
async def test_introspection_after_finish_raises() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    await enc.finish()
    with pytest.raises(RuntimeError, match="finished"):
        enc.object_count()


@pytest.mark.asyncio
async def test_hashing_produces_inline_hashes() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = await enc.finish()

    hashes = tensogram.object_inline_hashes(msg)
    assert len(hashes) == 1
    assert isinstance(hashes[0], str)
    assert len(hashes[0]) == 16
    # verify_hash decode succeeds → the inline slots are the real digests.
    tensogram.decode(msg, verify_hash=True)


@pytest.mark.asyncio
async def test_hashing_disabled() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META, hash=None)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = await enc.finish()
    assert tensogram.object_inline_hashes(msg) == [None]


@pytest.mark.asyncio
async def test_header_aggregate_hash_rejected_in_streaming_mode() -> None:
    # Streaming mode cannot honour a header-placed aggregate hash (the
    # header is written before any object), so `create` rejects it.
    with pytest.raises(ValueError, match=r"(?i)streaming|header"):
        await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META, aggregate_hash="header")


@pytest.mark.asyncio
async def test_repr() -> None:
    enc = await tensogram.AsyncStreamingEncoder.create(_GLOBAL_META)
    assert "AsyncStreamingEncoder" in repr(enc)
    await enc.write_object(_DESCRIPTOR, _PAYLOAD)
    assert "objects=1" in repr(enc)
