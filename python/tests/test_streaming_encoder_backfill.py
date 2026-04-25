# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for ``StreamingEncoder.finish_backfilled()``.

The default ``finish()`` writes ``total_length = 0`` in both the preamble
and postamble — the streaming-mode contract from wire-format §9.2.
``finish_backfilled()`` seeks back into the in-memory cursor and patches
both length slots with the real message length, satisfying the
backward-locatability invariant in §7 that the bidirectional remote
walker relies on.
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
    # Preamble v3 layout: bytes 16..24 carry total_length as big-endian uint64.
    return struct.unpack(">Q", message[16:24])[0]


def _postamble_total_length(message: bytes) -> int:
    # Postamble v3 layout (last 24 bytes of message): the second u64
    # at offset [end-16, end-8) is the mirrored total_length.
    return struct.unpack(">Q", message[-16:-8])[0]


def test_finish_writes_streaming_mode_zero_lengths() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish()

    assert _preamble_total_length(msg) == 0
    assert _postamble_total_length(msg) == 0


def test_finish_backfilled_writes_real_length_in_both_slots() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish_backfilled()

    assert _preamble_total_length(msg) == len(msg)
    assert _postamble_total_length(msg) == len(msg)


def test_finish_backfilled_round_trips_through_decode() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish_backfilled()

    decoded = tensogram.decode(msg)
    assert len(decoded.objects) == 1
    descriptor, data = decoded.objects[0]
    assert descriptor.dtype == "float32"
    assert list(descriptor.shape) == [4]
    np.testing.assert_array_equal(data, _PAYLOAD)


def test_finish_backfilled_then_finish_raises() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.finish_backfilled()
    with pytest.raises(RuntimeError):
        enc.finish()


def test_finish_then_finish_backfilled_raises() -> None:
    enc = tensogram.StreamingEncoder(_GLOBAL_META)
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    enc.finish()
    with pytest.raises(RuntimeError):
        enc.finish_backfilled()


def test_finish_backfilled_concatenated_yields_two_locatable_messages() -> None:
    # Two messages produced by separate StreamingEncoder + finish_backfilled
    # calls and concatenated should both be backward-locatable; the harness's
    # bidirectional walker relies on this fixture pattern.
    parts: list[bytes] = []
    for i in range(2):
        enc = tensogram.StreamingEncoder({"base": [{"msg_index": i}]})
        enc.write_object(_DESCRIPTOR, _PAYLOAD + float(i))
        parts.append(enc.finish_backfilled())

    combined = b"".join(parts)
    layouts = list(tensogram.scan(combined))

    assert len(layouts) == 2
    assert layouts[0] == (0, len(parts[0]))
    assert layouts[1] == (len(parts[0]), len(parts[1]))
    assert _postamble_total_length(combined[: len(parts[0])]) == len(parts[0])
    assert _postamble_total_length(combined[len(parts[0]) :]) == len(parts[1])
