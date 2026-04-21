# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for encode_pre_encoded() and StreamingEncoder.write_object_pre_encoded().

These tests exercise the pre-encoded encode path: callers hand already-encoded
payload bytes (plus a matching descriptor) and the library wraps them into the
wire format without re-running the encoding pipeline.

Key invariants:
  - The library always overwrites the descriptor's ``hash`` field with
    ``xxh3(payload)`` — callers cannot inject a hash.
  - Wire bytes are NOT compared directly because provenance fields
    (``_reserved_.uuid``, ``_reserved_.time``) are non-deterministic.
    Instead, decoded payloads are compared via ``hashlib.sha256``.
  - ``szip_block_offsets`` are **BIT** offsets, not byte offsets.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_global_meta(version: int = 2, **extra: Any) -> dict[str, Any]:
    """Build a global metadata dict."""
    return {"version": version, **extra}


def make_descriptor(
    shape: list[int],
    dtype: str = "float32",
    byte_order: str = "little",
    encoding: str = "none",
    compression: str = "none",
    filter_: str = "none",
    **extra: Any,
) -> dict[str, Any]:
    """Build a DataObjectDescriptor dict."""
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "byte_order": byte_order,
        "encoding": encoding,
        "filter": filter_,
        "compression": compression,
        **extra,
    }


def decoded_sha256(desc: dict, payload: np.ndarray) -> str:
    """SHA-256 over (descriptor repr + raw payload bytes).

    Excludes hash and _reserved_ from the descriptor for stability.
    """
    h = hashlib.sha256()
    # Sort keys for determinism; exclude volatile fields
    stable = {k: v for k, v in sorted(desc.items()) if k not in ("hash", "_reserved_")}
    h.update(repr(stable).encode())
    h.update(payload.tobytes())
    return h.hexdigest()


def simple_pack_python(
    values: np.ndarray,
    ref_val: float,
    bsf: int,
    dsf: int,
    bpv: int,
) -> bytes:
    """Manually quantise float64 values and bit-pack into bytes.

    This reproduces the simple_packing encoding in pure Python so we can
    build pre-encoded payloads without calling tensogram.encode().

    Formula: packed_int = round((value - ref_val) * 10^dsf / 2^bsf)
    """
    scale = (10.0**dsf) / (2.0**bsf)
    packed_ints = np.round((values - ref_val) * scale).astype(np.uint64)

    # Bit-pack into a byte stream (big-endian bit order, MSB first)
    total_bits = len(packed_ints) * bpv
    total_bytes = (total_bits + 7) // 8
    buf = bytearray(total_bytes)

    bit_pos = 0
    for val in packed_ints:
        # Write bpv bits of val into buf starting at bit_pos
        remaining = bpv
        v = int(val)
        while remaining > 0:
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            space = 8 - bit_offset
            write_bits = min(remaining, space)

            # Extract the top `write_bits` bits from the remaining value
            shift = remaining - write_bits
            bits = (v >> shift) & ((1 << write_bits) - 1)

            buf[byte_idx] |= bits << (space - write_bits)
            bit_pos += write_bits
            remaining -= write_bits

    return bytes(buf)


# ---------------------------------------------------------------------------
# Test 1: simple_packing round-trip
# ---------------------------------------------------------------------------


class TestEncodePreEncodedSimplePacking:
    """Pre-encoded simple_packing payloads decode correctly."""

    def test_simple_packing_roundtrip(self):
        """Manually pack float64 values, pass to encode_pre_encoded, decode."""
        n = 1000
        temps = np.linspace(249.15, 349.05, n, dtype=np.float64)

        # Compute packing parameters via the library helper
        params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)
        ref_val = params["reference_value"]
        bsf = params["binary_scale_factor"]
        dsf = params["decimal_scale_factor"]
        bpv = params["bits_per_value"]

        # Manually pack in Python
        packed_bytes = simple_pack_python(temps, ref_val, bsf, dsf, bpv)

        # Build descriptor for pre-encoded path
        # Use byte_order="little" (native on ARM Mac) because simple_packing
        # bit-packing is byte-order-independent — only the reference value
        # and scale factors matter for decoding.
        desc = make_descriptor(
            shape=[n],
            dtype="float64",
            byte_order="little",
            encoding="simple_packing",
            **params,
        )
        meta = make_global_meta(2)

        # Encode via pre-encoded path
        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, packed_bytes)]))

        # Decode and verify lossy round-trip
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        max_err = np.abs(temps - decoded).max()
        assert max_err < 0.01, f"Max error {max_err} exceeds tolerance"
        assert decoded.dtype == np.float64
        assert decoded.shape == (n,)

    def test_simple_packing_matches_encode(self):
        """Pre-encoded simple_packing produces same decoded payload as encode()."""
        n = 500
        temps = np.linspace(200.0, 300.0, n, dtype=np.float64)
        params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)

        desc = make_descriptor(
            shape=[n],
            dtype="float64",
            byte_order="little",
            encoding="simple_packing",
            **params,
        )
        meta = make_global_meta(2)

        # Path A: normal encode
        msg_a = bytes(tensogram.encode(meta, [(desc, temps)]))
        _, objs_a = tensogram.decode(msg_a)
        _, decoded_a = objs_a[0]

        # Path B: manually pack then encode_pre_encoded
        packed = simple_pack_python(
            temps,
            params["reference_value"],
            params["binary_scale_factor"],
            params["decimal_scale_factor"],
            params["bits_per_value"],
        )
        msg_b = bytes(tensogram.encode_pre_encoded(meta, [(desc, packed)]))
        _, objs_b = tensogram.decode(msg_b)
        _, decoded_b = objs_b[0]

        # Both paths should produce identical decoded payloads
        np.testing.assert_array_equal(decoded_a, decoded_b)


# ---------------------------------------------------------------------------
# Test 2: szip with block offsets
# ---------------------------------------------------------------------------


class TestEncodePreEncodedSzip:
    """Pre-encoded szip payloads with block offsets."""

    def test_szip_with_block_offsets_via_pre_encoded(self):
        """Encode with szip via encode(), extract descriptor, re-encode via encode_pre_encoded.

        Since Python has no API to extract raw on-wire payload bytes directly,
        we use encode() to get a valid szip message, then re-use the descriptor
        (which includes szip_block_offsets) with manually packed bytes via
        encode_pre_encoded().  The descriptor's block offsets won't match the
        new payload, but the library must accept them structurally (validation
        checks monotonicity and range, not payload correspondence).
        """
        n = 10000
        temps = np.linspace(200.0, 300.0, n, dtype=np.float64)
        params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)

        desc_szip = make_descriptor(
            shape=[n],
            dtype="float64",
            byte_order="little",
            encoding="simple_packing",
            compression="szip",
            szip_rsi=128,
            szip_block_size=16,
            szip_flags=8,
            **params,
        )
        meta = make_global_meta(2)

        # Step 1: Encode normally with szip compression to get valid block offsets
        msg_orig = bytes(tensogram.encode(meta, [(desc_szip, temps)]))
        _, objs_orig = tensogram.decode(msg_orig)
        desc_orig, decoded_orig = objs_orig[0]
        max_err = np.abs(temps - decoded_orig).max()
        assert max_err < 0.01

        # Verify szip_block_offsets are present
        p = desc_orig.params
        assert "szip_block_offsets" in p, "szip should produce block offsets"
        offsets = p["szip_block_offsets"]
        assert len(offsets) > 0, "should have at least one block offset"

        # Step 2: Build a manually packed payload (simple_packing, no szip)
        packed_bytes = simple_pack_python(
            temps,
            params["reference_value"],
            params["binary_scale_factor"],
            params["decimal_scale_factor"],
            params["bits_per_value"],
        )

        # Step 3: Build a pre-encoded descriptor with valid szip_block_offsets
        # Use the block offsets from the real szip encode — they're structurally
        # valid (monotonically increasing, within range).
        pre_desc = make_descriptor(
            shape=[n],
            dtype="float64",
            byte_order="little",
            encoding="simple_packing",
            compression="szip",
            szip_rsi=int(p.get("szip_rsi", 128)),
            szip_block_size=int(p.get("szip_block_size", 16)),
            szip_flags=int(p.get("szip_flags", 8)),
            szip_block_offsets=[int(o) for o in offsets],
            **params,
        )

        # Step 4: encode_pre_encoded must accept the descriptor with szip_block_offsets
        msg_pre = bytes(tensogram.encode_pre_encoded(meta, [(pre_desc, packed_bytes)]))
        assert len(msg_pre) > 0, "pre-encoded message should not be empty"

        # Step 5: decode_descriptors to verify the offsets survived round-trip
        _, descs = tensogram.decode_descriptors(msg_pre)
        d = descs[0]
        assert "szip_block_offsets" in d.params
        assert d.params["szip_block_offsets"] == [int(o) for o in offsets]

    def test_szip_block_offsets_non_monotonic_rejected(self):
        """Non-monotonic szip_block_offsets must be rejected."""
        n = 100
        temps = np.linspace(200.0, 300.0, n, dtype=np.float64)
        params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)
        packed = simple_pack_python(
            temps,
            params["reference_value"],
            params["binary_scale_factor"],
            params["decimal_scale_factor"],
            params["bits_per_value"],
        )
        desc = make_descriptor(
            shape=[n],
            dtype="float64",
            byte_order="little",
            encoding="simple_packing",
            compression="szip",
            szip_rsi=128,
            szip_block_size=16,
            szip_flags=8,
            szip_block_offsets=[0, 200, 100],  # not monotonic
            **params,
        )
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="szip_block_offsets must be strictly increasing"):
            tensogram.encode_pre_encoded(meta, [(desc, packed)])

    def test_szip_block_offsets_with_non_szip_rejected(self):
        """szip_block_offsets with non-szip compression must be rejected."""
        n = 100
        temps = np.linspace(200.0, 300.0, n, dtype=np.float64)
        params = tensogram.compute_packing_params(temps, bits_per_value=16, decimal_scale_factor=0)
        packed = simple_pack_python(
            temps,
            params["reference_value"],
            params["binary_scale_factor"],
            params["decimal_scale_factor"],
            params["bits_per_value"],
        )
        desc = make_descriptor(
            shape=[n],
            dtype="float64",
            byte_order="little",
            encoding="simple_packing",
            compression="zstd",
            szip_block_offsets=[0, 100, 200],
            **params,
        )
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="szip_block_offsets provided but compression"):
            tensogram.encode_pre_encoded(meta, [(desc, packed)])


# ---------------------------------------------------------------------------
# Test 3: hash overwriting
# ---------------------------------------------------------------------------


class TestEncodePreEncodedHashOverwrite:
    """v3: `DataObjectDescriptor.hash` is gone from the CBOR
    descriptor; the per-object hash lives in the frame footer's
    inline slot.  The "caller injects garbage hash on descriptor,
    library overwrites" scenario is structurally impossible —
    the descriptor dict can't carry a `hash` key because the Rust
    struct no longer has one.  This test is retained as a v3
    integrity smoke-test: pre-encoded round-trips through encode/
    decode/validate cleanly with `hash="xxh3"` on the
    message-level options.
    """

    def test_overwrites_caller_hash(self):
        """v3 integrity smoke-test — see class docstring above."""
        data = np.arange(20, dtype=np.float32)
        raw_bytes = data.tobytes()

        # No `hash` kwarg in v3 — the library populates the inline
        # slot from the payload regardless.
        desc = make_descriptor(
            shape=[20],
            dtype="float32",
            byte_order="little",
        )
        meta = make_global_meta(3)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw_bytes)]))

        # Decode the payload and confirm round-trip equality.
        _, objects = tensogram.decode(msg, verify_hash=True)
        _desc_out, arr = objects[0]
        np.testing.assert_array_equal(arr, data)

        # Validate at checksum level: inline slot must verify
        # against the recomputed frame-body hash.
        report = tensogram.validate(msg, level="checksum")
        assert report["hash_verified"], (
            f"encode_pre_encoded inline slot must verify at checksum level, got: {report}"
        )


# ---------------------------------------------------------------------------
# Test 4: invalid descriptor rejection
# ---------------------------------------------------------------------------


class TestEncodePreEncodedRejectsInvalid:
    """Invalid descriptors are rejected with an exception."""

    def test_rejects_unknown_encoding(self):
        """Unknown encoding string raises ValueError."""
        data = b"\x00" * 40
        desc = make_descriptor(
            shape=[10],
            dtype="float32",
            byte_order="little",
            encoding="bogus_encoding",
        )
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="encoding"):
            tensogram.encode_pre_encoded(meta, [(desc, data)])

    def test_rejects_missing_version(self):
        """Missing version in global metadata raises ValueError."""
        data = b"\x00" * 40
        desc = make_descriptor(shape=[10], dtype="float32")
        with pytest.raises(ValueError, match="version"):
            tensogram.encode_pre_encoded({}, [(desc, data)])

    def test_rejects_missing_shape(self):
        """Missing shape in descriptor raises ValueError."""
        data = b"\x00" * 40
        desc = {"type": "ntensor", "dtype": "float32"}
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="shape"):
            tensogram.encode_pre_encoded(meta, [(desc, data)])

    def test_rejects_unknown_dtype(self):
        """Unknown dtype string raises ValueError."""
        data = b"\x00" * 40
        desc = make_descriptor(shape=[10], dtype="complex256")
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="unknown dtype"):
            tensogram.encode_pre_encoded(meta, [(desc, data)])

    def test_rejects_numpy_array_data(self):
        """encode_pre_encoded must reject numpy arrays (requires bytes)."""
        arr = np.arange(10, dtype=np.float32)
        desc = make_descriptor(shape=[10], dtype="float32", byte_order="little")
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError)):
            tensogram.encode_pre_encoded(meta, [(desc, arr)])


# ---------------------------------------------------------------------------
# Test 5: zero objects
# ---------------------------------------------------------------------------


class TestEncodePreEncodedZeroObjects:
    """Encoding with zero data objects."""

    def test_zero_objects(self):
        """Pre-encoding an empty list of objects produces a valid message."""
        meta = make_global_meta(2, note="empty")
        msg = bytes(tensogram.encode_pre_encoded(meta, []))
        decoded_meta, objects = tensogram.decode(msg)
        assert len(objects) == 0
        assert decoded_meta["note"] == "empty"


# ---------------------------------------------------------------------------
# Test 6: StreamingEncoder mixed mode
# ---------------------------------------------------------------------------


class TestStreamingEncoderPreEncoded:
    """StreamingEncoder.write_object_pre_encoded interleaved with write_object."""

    def test_mixed_mode(self):
        """Alternate write_object (normal) and write_object_pre_encoded."""
        meta = make_global_meta(2)
        enc = tensogram.StreamingEncoder(meta)

        # Object 0: normal encode
        arr0 = np.arange(10, dtype=np.float32)
        desc0 = make_descriptor(shape=[10], dtype="float32", byte_order="little")
        enc.write_object(desc0, arr0)

        # Object 1: pre-encoded (raw bytes, encoding=none)
        arr1 = np.arange(5, dtype=np.int32)
        raw1 = arr1.tobytes()
        desc1 = make_descriptor(
            shape=[5],
            dtype="int32",
            byte_order="little",
        )
        enc.write_object_pre_encoded(desc1, raw1)

        # Object 2: normal encode again
        arr2 = np.full(3, 42, dtype=np.uint8)
        desc2 = make_descriptor(shape=[3], dtype="uint8")
        enc.write_object(desc2, arr2)

        msg = bytes(enc.finish())

        # Decode and verify all three objects
        _, objects = tensogram.decode(msg)
        assert len(objects) == 3

        _, d0 = objects[0]
        np.testing.assert_array_equal(d0, arr0)

        _, d1 = objects[1]
        np.testing.assert_array_equal(d1, arr1)

        _, d2 = objects[2]
        np.testing.assert_array_equal(d2, arr2)

    def test_streaming_pre_encoded_only(self):
        """StreamingEncoder with only pre-encoded writes."""
        meta = make_global_meta(2)
        enc = tensogram.StreamingEncoder(meta)

        arr = np.arange(20, dtype=np.float64)
        raw = arr.tobytes()
        desc = make_descriptor(
            shape=[20],
            dtype="float64",
            byte_order="little",
        )
        enc.write_object_pre_encoded(desc, raw)

        msg = bytes(enc.finish())
        _, objects = tensogram.decode(msg)
        assert len(objects) == 1
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded, arr)

    def test_streaming_finish_twice_raises(self):
        """Calling finish() twice on a StreamingEncoder raises RuntimeError."""
        meta = make_global_meta(2)
        enc = tensogram.StreamingEncoder(meta)
        enc.finish()
        with pytest.raises(RuntimeError, match="already finished"):
            enc.finish()


# ---------------------------------------------------------------------------
# Test 7: szip_block_offsets rejected for non-szip compression
# ---------------------------------------------------------------------------


class TestEncodePreEncodedRejectsSzipOffsetsForNonSzip:
    """szip_block_offsets in descriptor when compression != 'szip' → error."""

    def test_rejects_szip_offsets_for_non_szip(self):
        """Passing szip_block_offsets with compression='none' should raise."""
        data = b"\x00" * 40
        desc = make_descriptor(
            shape=[10],
            dtype="float32",
            byte_order="little",
            szip_block_offsets=[0, 80, 160],
        )
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="szip_block_offsets"):
            tensogram.encode_pre_encoded(meta, [(desc, data)])


# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestEncodePreEncodedEdgeCases:
    """Additional edge cases for pre-encoded encoding."""

    def test_encoding_none_roundtrip(self):
        """encoding=none with raw bytes round-trips correctly."""
        data = np.arange(100, dtype=np.float32)
        raw = data.tobytes()

        desc = make_descriptor(
            shape=[100],
            dtype="float32",
            byte_order="little",
        )
        meta = make_global_meta(2)
        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)]))

        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded, data)

    def test_encoding_none_multiple_objects(self):
        """Pre-encode multiple objects with encoding=none."""
        a = np.arange(10, dtype=np.float32)
        b = np.arange(5, dtype=np.int64)

        desc_a = make_descriptor(
            shape=[10],
            dtype="float32",
            byte_order="little",
        )
        desc_b = make_descriptor(
            shape=[5],
            dtype="int64",
            byte_order="little",
        )
        meta = make_global_meta(2)

        msg = bytes(
            tensogram.encode_pre_encoded(
                meta,
                [(desc_a, a.tobytes()), (desc_b, b.tobytes())],
            )
        )

        _, objects = tensogram.decode(msg)
        assert len(objects) == 2
        _, d_a = objects[0]
        _, d_b = objects[1]
        np.testing.assert_array_equal(d_a, a)
        np.testing.assert_array_equal(d_b, b)

    def test_no_hash(self):
        """Pre-encode with hash=None succeeds."""
        data = np.ones(10, dtype=np.float32)
        raw = data.tobytes()
        desc = make_descriptor(shape=[10], dtype="float32", byte_order="little")
        meta = make_global_meta(2)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)], hash=None))
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded, data)
        desc_out = objects[0][0]
        assert desc_out.hash is None

    def test_big_endian_encoding_none(self):
        """Big-endian encoding=none pre-encoded: decode converts to native.

        When ``encoding=none`` is used with ``byte_order=big``, the library
        stores the payload bytes verbatim.  On decode, with the default
        ``native_byte_order=True``, the library automatically converts
        from big-endian to native byte order so the caller gets correct
        values without manual byte-swapping.
        """
        data = np.arange(10, dtype=np.float32)
        # Big-endian bytes: swap byte order
        raw_be = data.byteswap().tobytes()

        desc = make_descriptor(
            shape=[10],
            dtype="float32",
            byte_order="big",
        )
        meta = make_global_meta(2)
        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw_be)]))

        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]

        # Library converts to native byte order — values match directly.
        np.testing.assert_array_equal(decoded, data)

    def test_big_endian_wire_byte_order_opt_out(self):
        """native_byte_order=False returns raw wire-order (big-endian) bytes."""
        data = np.arange(10, dtype=np.float32)
        raw_be = data.byteswap().tobytes()

        desc = make_descriptor(
            shape=[10],
            dtype="float32",
            byte_order="big",
        )
        meta = make_global_meta(2)
        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw_be)]))

        # native_byte_order=False: get raw wire bytes (big-endian)
        _, objects = tensogram.decode(msg, native_byte_order=False)
        _, decoded_wire = objects[0]

        # Wire bytes should be big-endian — manually swap to verify content
        swapped = decoded_wire.byteswap()
        np.testing.assert_array_equal(swapped, data)

    def test_native_byte_order_on_decode_object(self):
        """native_byte_order parameter works on decode_object too."""
        data = np.arange(5, dtype=np.float32)
        raw_be = data.byteswap().tobytes()

        desc = make_descriptor(shape=[5], dtype="float32", byte_order="big")
        meta = make_global_meta(2)
        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw_be)]))

        # native=True (default): values match
        _, _, decoded_native = tensogram.decode_object(msg, 0)
        np.testing.assert_array_equal(decoded_native, data)

        # native=False: raw wire bytes
        _, _, decoded_wire = tensogram.decode_object(msg, 0, native_byte_order=False)
        swapped = decoded_wire.byteswap()
        np.testing.assert_array_equal(swapped, data)

    def test_metadata_preserved(self):
        """Extra metadata keys survive encode_pre_encoded round-trip."""
        data = np.ones(5, dtype=np.float32)
        raw = data.tobytes()
        desc = make_descriptor(shape=[5], dtype="float32", byte_order="little")
        meta = make_global_meta(2, experiment="test-01", step=42)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)]))
        decoded_meta, _ = tensogram.decode(msg)
        assert decoded_meta["experiment"] == "test-01"
        assert decoded_meta["step"] == 42

    def test_descriptor_params_preserved(self):
        """Custom params in the descriptor survive round-trip."""
        data = np.ones(5, dtype=np.float32)
        raw = data.tobytes()
        desc = make_descriptor(
            shape=[5],
            dtype="float32",
            byte_order="little",
            my_custom_key="hello",
        )
        meta = make_global_meta(2)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)]))
        _, objects = tensogram.decode(msg)
        desc_out = objects[0][0]
        assert desc_out.params["my_custom_key"] == "hello"


# ---------------------------------------------------------------------------
# Test: Additional edge cases for coverage
# ---------------------------------------------------------------------------


class TestEncodePreEncodedAdditionalEdgeCases:
    """Additional edge cases for code coverage hardening."""

    def test_bytearray_accepted(self):
        """bytearray is accepted as data (PyO3 extracts Vec<u8> from it)."""
        data = np.arange(8, dtype=np.float32)
        raw = bytearray(data.tobytes())
        desc = make_descriptor(shape=[8], dtype="float32", byte_order="little")
        meta = make_global_meta(2)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)]))
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded, data)

    def test_memoryview_accepted(self):
        """memoryview is accepted as data (PyO3 extracts Vec<u8> from it)."""
        data = np.arange(8, dtype=np.float32)
        raw = memoryview(data.tobytes())
        desc = make_descriptor(shape=[8], dtype="float32", byte_order="little")
        meta = make_global_meta(2)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)]))
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded, data)

    def test_empty_bytes_zero_element_shape(self):
        """Empty bytes with shape=[0] succeeds."""
        desc = make_descriptor(shape=[0], dtype="float32", byte_order="little")
        meta = make_global_meta(2)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, b"")]))
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        assert decoded.shape == (0,)
        assert decoded.dtype == np.float32

    def test_non_tuple_in_list_raises(self):
        """Non-tuple element in the data list raises ValueError."""
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError)):
            tensogram.encode_pre_encoded(meta, ["not a tuple"])

    def test_non_dict_descriptor_raises(self):
        """Non-dict as descriptor raises ValueError."""
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError)):
            tensogram.encode_pre_encoded(meta, [("not_a_dict", b"\x00" * 4)])

    def test_tuple_wrong_length_raises(self):
        """Tuple with wrong length raises ValueError."""
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError)):
            tensogram.encode_pre_encoded(meta, [({}, b"\x00", "extra")])

    def test_rejects_numpy_with_type_in_error(self):
        """encode_pre_encoded error message includes the rejected type name."""
        arr = np.arange(10, dtype=np.float32)
        desc = make_descriptor(shape=[10], dtype="float32", byte_order="little")
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError), match=r"ndarray|numpy|got"):
            tensogram.encode_pre_encoded(meta, [(desc, arr)])

    def test_rejects_int_data_with_type_in_error(self):
        """Passing an int as data shows the type name in the error."""
        desc = make_descriptor(shape=[1], dtype="float32", byte_order="little")
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError), match=r"int|got"):
            tensogram.encode_pre_encoded(meta, [(desc, 42)])

    def test_rejects_none_data_with_type_in_error(self):
        """Passing None as data shows the type name in the error."""
        desc = make_descriptor(shape=[1], dtype="float32", byte_order="little")
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError), match=r"None|got"):
            tensogram.encode_pre_encoded(meta, [(desc, None)])

    def test_single_element_roundtrip(self):
        """Single-element array round-trips correctly."""
        data = np.array([42.0], dtype=np.float32)
        raw = data.tobytes()
        desc = make_descriptor(shape=[1], dtype="float32", byte_order="little")
        meta = make_global_meta(2)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)]))
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded, data)

    def test_2d_array_roundtrip(self):
        """2D array [3, 4] round-trips correctly."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        raw = data.tobytes()
        desc = make_descriptor(shape=[3, 4], dtype="float32", byte_order="little")
        # strides for row-major float32 2D
        desc["strides"] = [16, 4]
        meta = make_global_meta(2)

        msg = bytes(tensogram.encode_pre_encoded(meta, [(desc, raw)]))
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded.reshape(3, 4), data)

    def test_streaming_write_after_finish_raises(self):
        """write_object_pre_encoded after finish() raises RuntimeError."""
        meta = make_global_meta(2)
        enc = tensogram.StreamingEncoder(meta)
        enc.finish()
        desc = make_descriptor(shape=[4], dtype="float32", byte_order="little")
        with pytest.raises(RuntimeError, match="already finished"):
            enc.write_object_pre_encoded(desc, b"\x00" * 16)

    def test_streaming_pre_encoded_multiple_objects(self):
        """StreamingEncoder with multiple pre-encoded writes."""
        meta = make_global_meta(2)
        enc = tensogram.StreamingEncoder(meta)

        a = np.arange(5, dtype=np.float32)
        b = np.arange(3, dtype=np.int32)

        desc_a = make_descriptor(shape=[5], dtype="float32", byte_order="little")
        desc_b = make_descriptor(shape=[3], dtype="int32", byte_order="little")

        enc.write_object_pre_encoded(desc_a, a.tobytes())
        enc.write_object_pre_encoded(desc_b, b.tobytes())

        msg = bytes(enc.finish())
        _, objects = tensogram.decode(msg)
        assert len(objects) == 2
        _, d_a = objects[0]
        _, d_b = objects[1]
        np.testing.assert_array_equal(d_a, a)
        np.testing.assert_array_equal(d_b, b)

    def test_encoding_none_size_mismatch_too_short(self):
        """encoding=none with data too short raises ValueError."""
        desc = make_descriptor(shape=[10], dtype="float32", byte_order="little")
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="does not match expected"):
            tensogram.encode_pre_encoded(meta, [(desc, b"\x00" * 20)])

    def test_encoding_none_size_mismatch_too_long(self):
        """encoding=none with data too long raises ValueError."""
        desc = make_descriptor(shape=[4], dtype="float32", byte_order="little")
        meta = make_global_meta(2)
        with pytest.raises(ValueError, match="does not match expected"):
            tensogram.encode_pre_encoded(meta, [(desc, b"\x00" * 32)])
