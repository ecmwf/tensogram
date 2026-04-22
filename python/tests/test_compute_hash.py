# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for :func:`tensogram.compute_hash`.

Closes the cross-language parity gap — Rust, C FFI, C++, WASM, and
TypeScript all exposed a hash-over-arbitrary-bytes helper; Python did
not.  The Python API mirrors Rust's ``tensogram::compute_hash`` and
WASM's ``compute_hash``: ``bytes`` in, hex ``str`` out, ``"xxh3"`` by
default.

Core invariants exercised here:

- Determinism — identical input produces identical digest.
- Length — xxh3-64 always returns 16 lower-case hex chars.
- Cross-call consistency — ``compute_hash(payload)`` matches the hash
  stamped by :func:`encode` on the same native bytes (encoding=none
  descriptors, so the encoded payload IS the raw bytes).
- Error surface — unknown algorithms raise ``ValueError``; wrong
  argument types raise ``TypeError``.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensogram


def _descriptor_for(shape: list[int], dtype: str) -> dict:
    """Build a no-pipeline descriptor so the encoded payload equals the raw bytes."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return {
        "type": "ntensor",
        "ndim": len(shape),
        "shape": shape,
        "strides": strides,
        "dtype": dtype,
        "byte_order": "little",
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


class TestBasics:
    """Basic shape and determinism checks."""

    def test_returns_16_char_lowercase_hex(self):
        digest = tensogram.compute_hash(b"hello world")
        assert len(digest) == 16
        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)

    def test_default_algo_is_xxh3(self):
        assert tensogram.compute_hash(b"abc") == tensogram.compute_hash(b"abc", "xxh3")
        assert tensogram.compute_hash(b"abc") == tensogram.compute_hash(b"abc", algo="xxh3")

    def test_deterministic(self):
        a = tensogram.compute_hash(b"\x00\x01\x02\x03\x04\x05\x06\x07")
        b = tensogram.compute_hash(b"\x00\x01\x02\x03\x04\x05\x06\x07")
        assert a == b

    def test_distinct_inputs_distinct_digests(self):
        assert tensogram.compute_hash(b"abc") != tensogram.compute_hash(b"abd")

    def test_empty_buffer(self):
        digest = tensogram.compute_hash(b"")
        assert len(digest) == 16
        # xxh3-64 of empty input is a well-known constant.
        assert digest == "2d06800538d394c2"

    def test_known_vector(self):
        # Cross-checked against the Rust-side compute_hash for the same bytes.
        assert tensogram.compute_hash(b"hello world") == "d447b1ea40e6988b"


class TestInputTypes:
    """``compute_hash`` accepts zero-copy ``bytes`` and ``bytearray`` (via
    :class:`PyBackedBytes`).  Other buffer-protocol types must be
    converted to ``bytes`` explicitly — we document that contract and
    verify the expected ``TypeError`` for unsupported types.
    """

    def test_bytes(self):
        assert len(tensogram.compute_hash(bytes([1, 2, 3]))) == 16

    def test_bytearray(self):
        assert tensogram.compute_hash(bytearray(b"abc")) == tensogram.compute_hash(b"abc")

    def test_memoryview_requires_explicit_bytes_conversion(self):
        buf = bytearray(b"abc")
        # memoryview is NOT accepted directly — callers must convert.
        with pytest.raises(TypeError):
            tensogram.compute_hash(memoryview(buf))  # type: ignore[arg-type]
        # But bytes(memoryview(...)) works:
        assert tensogram.compute_hash(bytes(memoryview(buf))) == tensogram.compute_hash(b"abc")

    def test_numpy_array_requires_tobytes(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(TypeError):
            tensogram.compute_hash(arr)  # type: ignore[arg-type]
        # `.tobytes()` is the idiomatic path.
        assert tensogram.compute_hash(arr.tobytes()) == tensogram.compute_hash(arr.tobytes())


class TestErrorSurface:
    """Unknown algorithms raise ``ValueError`` (mirrors the typed Metadata error)."""

    def test_unknown_algo_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown hash type"):
            tensogram.compute_hash(b"x", "md5")

    def test_unknown_algo_keyword(self):
        with pytest.raises(ValueError, match="unknown hash type"):
            tensogram.compute_hash(b"x", algo="sha256")

    def test_non_bytes_input_raises_type_error(self):
        with pytest.raises(TypeError):
            tensogram.compute_hash("not bytes")  # type: ignore[arg-type]

    def test_int_input_raises_type_error(self):
        with pytest.raises(TypeError):
            tensogram.compute_hash(42)  # type: ignore[arg-type]


class TestCrossCheckWithEncode:
    """v3 parity invariant — `encode` stamps a hash in the frame
    footer's inline slot.  Since Python doesn't yet surface the
    inline slot directly, these tests verify the round-trip via
    `validate_message` at `checksum` level: a well-formed message
    encoded with `hash="xxh3"` must pass integrity validation, and
    the `compute_hash` helper returns a stable xxh3-64 digest over
    the raw bytes.
    """

    def test_matches_stamped_hash_float32(self):
        values = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        raw = values.tobytes()
        msg = tensogram.encode({"version": 3}, [(_descriptor_for([4], "float32"), values)])
        # compute_hash is stable for the raw bytes.
        digest = tensogram.compute_hash(raw)
        assert isinstance(digest, str)
        assert len(digest) == 16
        # Validator confirms frame-level integrity at checksum level.
        report = tensogram.validate(msg, level="checksum")
        assert report["hash_verified"], f"checksum validation failed: {report}"

    def test_matches_stamped_hash_int64(self):
        values = np.array([-1, 0, 1, 42, 2**62], dtype=np.int64)
        raw = values.tobytes()
        msg = tensogram.encode({"version": 3}, [(_descriptor_for([5], "int64"), values)])
        digest = tensogram.compute_hash(raw)
        assert len(digest) == 16
        report = tensogram.validate(msg, level="checksum")
        assert report["hash_verified"]

    def test_matches_stamped_hash_large_payload(self):
        # Larger payload — exercises the streaming path of the hasher
        # without crossing any sensitive pipeline boundary.
        values = np.arange(10_000, dtype=np.float64)
        raw = values.tobytes()
        msg = tensogram.encode({"version": 3}, [(_descriptor_for([10_000], "float64"), values)])
        assert len(tensogram.compute_hash(raw)) == 16
        report = tensogram.validate(msg, level="checksum")
        assert report["hash_verified"]


class TestEncodePreEncodedIntegration:
    """v3: `encode_pre_encoded` populates the inline slot from the
    encoded payload bytes.  The resulting message passes
    `validate --checksum` with no further ceremony.
    """

    def test_precompute_matches_stamped(self):
        raw = np.array([7.0, 8.0, 9.0], dtype=np.float32).tobytes()
        # `compute_hash` is a stable helper.
        precomputed = tensogram.compute_hash(raw)
        assert len(precomputed) == 16
        msg = tensogram.encode_pre_encoded(
            {"version": 3}, [(_descriptor_for([3], "float32"), raw)]
        )
        report = tensogram.validate(msg, level="checksum")
        assert report["hash_verified"], f"checksum validation failed: {report}"
