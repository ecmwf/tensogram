# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for ``tensogram.object_inline_hashes()`` (BUG-PY fix).

In v3 the per-object integrity digest moved from the CBOR descriptor to
the data-object frame's inline footer slot (``plans/WIRE_FORMAT.md``
§2.4).  ``DataObjectDescriptor.hash`` therefore always returns ``None``
and its old docstring pointed at ``Message.object_inline_hashes()`` /
``Message.object_hash(i)`` — methods that never existed (``Message`` is a
bare namedtuple).  Reading inline hashes from Python was impossible.

``tensogram.object_inline_hashes(buf)`` is the real, working accessor —
it mirrors the Rust core ``data_object_inline_hashes`` walker and returns
one entry per data object.
"""

from __future__ import annotations

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


def _encode(payload: np.ndarray, *, hash: str | None = "xxh3") -> bytes:
    return tensogram.encode({"base": [{}]}, [(_DESCRIPTOR, payload)], hash=hash)


def test_reads_a_real_inline_hash() -> None:
    msg = _encode(_PAYLOAD)
    hashes = tensogram.object_inline_hashes(msg)

    assert isinstance(hashes, list)
    assert len(hashes) == 1
    digest = hashes[0]
    # Canonical xxh3-64: 16-char lowercase hex, identical form to
    # `compute_hash` and the C FFI `tgm_object_hash_value`.
    assert isinstance(digest, str)
    assert len(digest) == 16
    int(digest, 16)  # parses as hex — raises if malformed
    assert digest == digest.lower()


def test_inline_hash_is_deterministic() -> None:
    # Same descriptor + payload → identical inline digest across encodes
    # (the random per-message provenance UUID does not touch the frame body).
    first = tensogram.object_inline_hashes(_encode(_PAYLOAD))
    second = tensogram.object_inline_hashes(_encode(_PAYLOAD))
    assert first == second
    assert first[0] is not None


def test_different_payload_yields_different_hash() -> None:
    a = tensogram.object_inline_hashes(_encode(_PAYLOAD))
    b = tensogram.object_inline_hashes(_encode(_PAYLOAD + 1.0))
    assert a[0] != b[0]


def test_hashing_disabled_yields_none() -> None:
    msg = _encode(_PAYLOAD, hash=None)
    hashes = tensogram.object_inline_hashes(msg)
    assert hashes == [None]


def test_multi_object_message_one_entry_per_object() -> None:
    payloads = [
        np.arange(4, dtype=np.float32),
        np.arange(8, dtype=np.float32),
        np.arange(2, dtype=np.float32),
    ]
    descriptors = [{**_DESCRIPTOR, "shape": [len(p)]} for p in payloads]
    msg = tensogram.encode(
        {"base": [{} for _ in payloads]},
        list(zip(descriptors, payloads)),
    )
    hashes = tensogram.object_inline_hashes(msg)
    assert len(hashes) == len(payloads)
    assert all(isinstance(h, str) and len(h) == 16 for h in hashes)
    # The three objects differ, so their digests differ.
    assert len(set(hashes)) == 3


def test_inline_hashes_match_verify_hash_contract() -> None:
    # If `object_inline_hashes` reports non-None digests, they are the
    # very slots the decoder recomputes and checks under verify_hash — so
    # a clean verify_hash decode is a round-trip proof the digests are real.
    msg = _encode(_PAYLOAD)
    assert all(h is not None for h in tensogram.object_inline_hashes(msg))
    decoded = tensogram.decode(msg, verify_hash=True)
    np.testing.assert_array_equal(decoded.objects[0][1], _PAYLOAD)


def test_streaming_encoder_message_exposes_inline_hashes() -> None:
    enc = tensogram.StreamingEncoder({"base": [{}]})
    enc.write_object(_DESCRIPTOR, _PAYLOAD)
    msg = enc.finish()
    hashes = tensogram.object_inline_hashes(msg)
    assert len(hashes) == 1
    assert isinstance(hashes[0], str)
    assert len(hashes[0]) == 16


def test_descriptor_hash_getter_still_none() -> None:
    # The deprecated `.hash` getter stays a documented `None` stub (the
    # descriptor cannot reach the footer slot); the docstring now points
    # at `object_inline_hashes`.
    msg = _encode(_PAYLOAD)
    descriptor, _ = tensogram.decode(msg).objects[0]
    assert descriptor.hash is None


def test_invalid_buffer_raises() -> None:
    with pytest.raises(ValueError, match=r"(?i)framing|too short|frame"):
        tensogram.object_inline_hashes(b"not a tensogram message")
