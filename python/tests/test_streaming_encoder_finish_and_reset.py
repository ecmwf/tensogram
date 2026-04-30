# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for ``StreamingEncoder.finish_and_reset()`` and
``StreamingEncoder.finish_and_reset_backfilled()``.

These methods finalise the current message and immediately reset the encoder
so that it can accumulate the next message without re-construction.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import tensogram

_DESCRIPTOR = {
    "type": "ntensor",
    "ndim": 1,
    "shape": [4],
    "dtype": "float32",
    "encoding": "none",
    "filter": "none",
    "compression": "none",
}
_PAYLOAD = np.arange(4, dtype=np.float32)
_GLOBAL_META = {"base": [{"name": "test"}]}


def _preamble_total_length(message: bytes) -> int:
    """Bytes 16..24 of the preamble carry total_length as big-endian u64."""
    return struct.unpack(">Q", message[16:24])[0]


def _postamble_total_length(message: bytes) -> int:
    """The second u64 field of the postamble is the mirrored total_length."""
    return struct.unpack(">Q", message[-16:-8])[0]


# ── finish_and_reset (streaming mode, total_length=0) ────────────────────────


def test_finish_and_reset_returns_decodable_message() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish_and_reset()

    decoded = tensogram.decode(msg)
    assert len(decoded.objects) == 1
    descriptor, data = decoded.objects[0]
    assert descriptor.dtype == "float32"
    assert list(descriptor.shape) == [4]
    np.testing.assert_array_equal(data, _PAYLOAD)


def test_finish_and_reset_streaming_mode_zero_lengths() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish_and_reset()

    assert _preamble_total_length(msg) == 0
    assert _postamble_total_length(msg) == 0


def test_finish_and_reset_encoder_still_usable() -> None:
    """After finish_and_reset the encoder must accept new write_object calls."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg1 = enc.finish_and_reset()

    payload2 = _PAYLOAD * 2
    enc.write_object(_DESCRIPTOR, payload2)
    msg2 = enc.finish_and_reset()

    decoded1 = tensogram.decode(msg1)
    np.testing.assert_array_equal(decoded1.objects[0][1], _PAYLOAD)

    decoded2 = tensogram.decode(msg2)
    np.testing.assert_array_equal(decoded2.objects[0][1], payload2)


def test_finish_and_reset_multiple_resets() -> None:
    """A single encoder can produce many messages via repeated resets."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    messages: list[tuple[bytes, np.ndarray]] = []
    for i in range(5):
        payload = _PAYLOAD + float(i)
        enc.write_object(_DESCRIPTOR, payload)
        messages.append((enc.finish_and_reset(), payload))

    for msg, expected in messages:
        decoded = tensogram.decode(msg)
        assert len(decoded.objects) == 1
        np.testing.assert_array_equal(decoded.objects[0][1], expected)


def test_finish_and_reset_then_finish() -> None:
    """Encoder remains valid for a final finish() after finish_and_reset()."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    _msg1 = enc.finish_and_reset()

    payload2 = _PAYLOAD + 10.0
    enc.write_object(_DESCRIPTOR, payload2)
    msg2 = enc.finish()

    decoded2 = tensogram.decode(msg2)
    np.testing.assert_array_equal(decoded2.objects[0][1], payload2)


def test_finish_and_reset_after_finish_raises() -> None:
    """Calling finish_and_reset after finish() must raise RuntimeError."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.finish()
    with pytest.raises(RuntimeError):
        enc.finish_and_reset()


def test_finish_and_reset_empty_message_is_decodable() -> None:
    """finish_and_reset with no objects must produce a valid empty message."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    msg = enc.finish_and_reset()

    decoded = tensogram.decode(msg)
    assert len(decoded.objects) == 0


def test_finish_and_reset_multi_object_message() -> None:
    """Messages with multiple objects round-trip correctly after reset."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.write_object(_DESCRIPTOR, _PAYLOAD * 2)
    msg = enc.finish_and_reset()

    decoded = tensogram.decode(msg)
    assert len(decoded.objects) == 2
    np.testing.assert_array_equal(decoded.objects[0][1], _PAYLOAD)
    np.testing.assert_array_equal(decoded.objects[1][1], _PAYLOAD * 2)


# ── finish_and_reset_backfilled (total_length back-filled) ───────────────────


def test_finish_and_reset_backfilled_writes_real_length_in_both_slots() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish_and_reset_backfilled()

    assert _preamble_total_length(msg) == len(msg)
    assert _postamble_total_length(msg) == len(msg)


def test_finish_and_reset_backfilled_decoder_round_trip() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish_and_reset_backfilled()

    decoded = tensogram.decode(msg)
    assert len(decoded.objects) == 1
    np.testing.assert_array_equal(decoded.objects[0][1], _PAYLOAD)


def test_finish_and_reset_backfilled_concatenated_messages_scannable() -> None:
    """Concatenation of backfilled messages must be scannable as separate messages."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    parts: list[bytes] = []
    for i in range(3):
        enc.write_object(_DESCRIPTOR, _PAYLOAD + float(i))
        parts.append(enc.finish_and_reset_backfilled())

    combined = b"".join(parts)
    layouts = list(tensogram.scan(combined))

    assert len(layouts) == 3

    offset = 0
    for i, part in enumerate(parts):
        assert layouts[i] == (offset, len(part))
        assert _postamble_total_length(combined[offset : offset + len(part)]) == len(part)
        offset += len(part)


def test_finish_and_reset_backfilled_then_finish_backfilled() -> None:
    """Encoder works correctly when mixing finish_and_reset_backfilled and finish_backfilled."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg1 = enc.finish_and_reset_backfilled()

    enc.write_object(_DESCRIPTOR, _PAYLOAD * 3)
    msg2 = enc.finish_backfilled()

    assert _preamble_total_length(msg1) == len(msg1)
    assert _preamble_total_length(msg2) == len(msg2)

    decoded1 = tensogram.decode(msg1)
    decoded2 = tensogram.decode(msg2)
    np.testing.assert_array_equal(decoded1.objects[0][1], _PAYLOAD)
    np.testing.assert_array_equal(decoded2.objects[0][1], _PAYLOAD * 3)


def test_finish_and_reset_backfilled_after_finish_backfilled_raises() -> None:
    """finish_and_reset_backfilled after finish_backfilled must raise RuntimeError."""
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.finish_backfilled()
    with pytest.raises(RuntimeError):
        enc.finish_and_reset_backfilled()


def test_finish_and_reset_global_meta_preserved_across_resets() -> None:
    """The global metadata supplied at construction must appear in every message."""
    meta = {"base": [{"source": "test-suite"}]}
    enc = tensogram.StreamingEncoder(meta)
    messages: list[bytes] = []
    for _ in range(3):
        enc.write_object(_DESCRIPTOR, _PAYLOAD)
        messages.append(enc.finish_and_reset_backfilled())

    for msg in messages:
        decoded = tensogram.decode(msg)
        # The "source" key must be present in base[0] of every message.
        assert decoded.metadata.base[0].get("source") == "test-suite"
