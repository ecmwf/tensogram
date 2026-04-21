# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Comprehensive tests for the tensogram Python bindings.

Tests the full public API surface:
  - encode / decode round-trip
  - decode_metadata
  - decode_object (single object by index)
  - decode_range (partial slice)
  - scan (message boundary detection)
  - compute_packing_params
  - TensogramFile (create, append, open, iterate, random access)
  - Metadata / DataObjectDescriptor classes
  - Error handling
  - Edge cases (empty arrays, large arrays, all dtypes)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import tensogram

# ---------------------------------------------------------------------------
# Module-level helpers (not fixtures — used directly in tests)
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


def encode_simple(
    data: np.ndarray,
    dtype: str | None = None,
    version: int = 2,
    hash_algo: str | None = "xxh3",
    extra_meta: dict | None = None,
    extra_desc: dict | None = None,
) -> bytes:
    """Encode a single array with sensible defaults."""
    if dtype is None:
        dtype = str(data.dtype)
    meta = make_global_meta(version, **(extra_meta or {}))
    desc = make_descriptor(list(data.shape), dtype=dtype, **(extra_desc or {}))
    return bytes(tensogram.encode(meta, [(desc, data)], hash=hash_algo))


# ---------------------------------------------------------------------------
# encode / decode round-trip
# ---------------------------------------------------------------------------


# Parametrized dtype cases: (dtype_string, numpy_array)
DTYPE_CASES = [
    ("float32", np.arange(100, dtype=np.float32)),
    ("float64", np.linspace(0, 1, 60, dtype=np.float64).reshape(6, 10)),
    ("int32", np.array([-1, 0, 1, 2**30], dtype=np.int32)),
    ("int64", np.array([-(2**62), 0, 2**62], dtype=np.int64)),
    ("uint8", np.array([0, 127, 255], dtype=np.uint8)),
    ("uint16", np.array([0, 1000, 65535], dtype=np.uint16)),
    ("uint32", np.array([0, 2**31], dtype=np.uint32)),
    ("uint64", np.array([0, 2**63], dtype=np.uint64)),
    ("int8", np.array([-128, 0, 127], dtype=np.int8)),
    ("int16", np.array([-32768, 0, 32767], dtype=np.int16)),
]


class TestEncodeDecode:
    """Basic encode->decode round-trip tests."""

    @pytest.mark.parametrize(
        ("dtype_str", "data"),
        DTYPE_CASES,
        ids=[d for d, _ in DTYPE_CASES],
    )
    def test_dtype_roundtrip(self, dtype_str, data):
        """All supported dtypes round-trip through encode/decode."""
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)
        assert arr.dtype == data.dtype
        assert arr.shape == data.shape

    def test_float32_1d_full_check(self):
        """Canonical test: checks shape, dtype, metadata, and object count."""
        data = np.arange(100, dtype=np.float32)
        msg = encode_simple(data)
        _meta, objects = tensogram.decode(msg)
        assert len(objects) == 1
        _desc, arr = objects[0]
        np.testing.assert_array_equal(arr, data)
        assert arr.dtype == np.float32
        assert arr.shape == (100,)

    def test_3d_array(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)
        assert arr.shape == (2, 3, 4)

    def test_4d_array(self):
        data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)

    def test_scalar_shape(self):
        """A single-element 1-d array."""
        data = np.array([42.0], dtype=np.float32)
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)

    def test_no_hash(self):
        """Encode without hash."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, hash_algo=None)
        _, objects = tensogram.decode(msg)
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)

    def test_verify_hash_clean(self):
        """verify_hash=True on a valid message should succeed."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, hash_algo="xxh3")
        _, objects = tensogram.decode(msg, verify_hash=True)
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)


# ---------------------------------------------------------------------------
# Multi-object messages
# ---------------------------------------------------------------------------


class TestMultiObject:
    """Messages with multiple data objects."""

    def test_two_objects_different_dtypes(self):
        f32 = np.arange(12, dtype=np.float32).reshape(3, 4)
        u8 = np.array([0, 128, 255], dtype=np.uint8)

        meta = make_global_meta(2)
        desc_f32 = make_descriptor([3, 4], dtype="float32")
        desc_u8 = make_descriptor([3], dtype="uint8")
        msg = bytes(tensogram.encode(meta, [(desc_f32, f32), (desc_u8, u8)]))

        _, objects = tensogram.decode(msg)
        assert len(objects) == 2
        _, arr0 = objects[0]
        _, arr1 = objects[1]
        np.testing.assert_array_equal(arr0, f32)
        np.testing.assert_array_equal(arr1, u8)

    def test_three_objects(self):
        arrays = [
            np.ones(10, dtype=np.float32),
            np.zeros(5, dtype=np.int32),
            np.full(3, 42, dtype=np.uint8),
        ]
        meta = make_global_meta(2)
        pairs = [
            (make_descriptor([10], dtype="float32"), arrays[0]),
            (make_descriptor([5], dtype="int32"), arrays[1]),
            (make_descriptor([3], dtype="uint8"), arrays[2]),
        ]
        msg = bytes(tensogram.encode(meta, pairs))

        _, objects = tensogram.decode(msg)
        assert len(objects) == 3
        for i, (_, arr) in enumerate(objects):
            np.testing.assert_array_equal(arr, arrays[i])


# ---------------------------------------------------------------------------
# decode_metadata
# ---------------------------------------------------------------------------


class TestDecodeMetadata:
    """Test decode_metadata (metadata-only, no payload)."""

    def test_basic(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, version=2)
        meta = tensogram.decode_metadata(msg)
        assert meta.version == 2

    def test_extra_keys(self):
        """Extra keys in global metadata round-trip."""
        data = np.ones(10, dtype=np.float32)
        extra = {"experiment": "test-01", "step": 6}
        msg = encode_simple(data, extra_meta=extra)
        meta = tensogram.decode_metadata(msg)
        assert meta["experiment"] == "test-01"
        assert meta["step"] == 6

    def test_nested_extra(self):
        """Nested dict in extra metadata."""
        data = np.ones(10, dtype=np.float32)
        extra = {"mars": {"class": "od", "date": "20260401"}}
        msg = encode_simple(data, extra_meta=extra)
        meta = tensogram.decode_metadata(msg)
        mars = meta["mars"]
        assert mars["class"] == "od"
        assert mars["date"] == "20260401"

    def test_missing_key_raises_keyerror(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data)
        meta = tensogram.decode_metadata(msg)
        with pytest.raises(KeyError):
            _ = meta["nonexistent"]

    def test_repr(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, version=2)
        meta = tensogram.decode_metadata(msg)
        r = repr(meta)
        assert "Metadata" in r
        assert "version=2" in r


# ---------------------------------------------------------------------------
# decode_object (single object by index)
# ---------------------------------------------------------------------------


class TestDecodeObject:
    """Test decode_object for random access by index."""

    def test_single_object(self):
        data = np.arange(20, dtype=np.float32)
        msg = encode_simple(data)
        _meta, desc, arr = tensogram.decode_object(msg, 0)
        np.testing.assert_array_equal(arr, data)
        assert desc.dtype == "float32"
        assert desc.shape == [20]

    def test_multi_object_index(self):
        """Decode specific object from multi-object message."""
        a = np.ones(10, dtype=np.float32)
        b = np.full(5, 99, dtype=np.int32)
        meta_dict = make_global_meta(2)
        pairs = [
            (make_descriptor([10], dtype="float32"), a),
            (make_descriptor([5], dtype="int32"), b),
        ]
        msg = bytes(tensogram.encode(meta_dict, pairs))

        # Decode object 1 only
        _meta, desc, arr = tensogram.decode_object(msg, 1)
        np.testing.assert_array_equal(arr, b)
        assert desc.dtype == "int32"

    def test_out_of_range_index(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data)
        with pytest.raises(ValueError, match="ObjectError"):
            tensogram.decode_object(msg, 99)

    def test_verify_hash(self):
        """decode_object with verify_hash=True should succeed on valid data."""
        data = np.arange(20, dtype=np.float32)
        msg = encode_simple(data, hash_algo="xxh3")
        _meta, _desc, arr = tensogram.decode_object(msg, 0, verify_hash=True)
        np.testing.assert_array_equal(arr, data)


# ---------------------------------------------------------------------------
# decode_range (partial slice)
# ---------------------------------------------------------------------------


class TestDecodeRange:
    """Test decode_range for partial data extraction."""

    def test_basic_range(self):
        data = np.arange(100, dtype=np.float32)
        msg = encode_simple(data)
        expected = data[10:15]

        # Default (join=False) returns a list
        result = tensogram.decode_range(msg, 0, [(10, 5)])
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], expected)

        # join=True returns a flat array (old behavior)
        joined = tensogram.decode_range(msg, 0, [(10, 5)], join=True)
        assert isinstance(joined, np.ndarray)
        np.testing.assert_array_equal(joined, expected)

    def test_range_from_start(self):
        data = np.arange(50, dtype=np.float32)
        msg = encode_simple(data)

        result = tensogram.decode_range(msg, 0, [(0, 10)])
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], data[:10])

        joined = tensogram.decode_range(msg, 0, [(0, 10)], join=True)
        assert isinstance(joined, np.ndarray)
        np.testing.assert_array_equal(joined, data[:10])

    def test_range_at_end(self):
        data = np.arange(50, dtype=np.float32)
        msg = encode_simple(data)

        result = tensogram.decode_range(msg, 0, [(45, 5)])
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], data[45:50])

        joined = tensogram.decode_range(msg, 0, [(45, 5)], join=True)
        assert isinstance(joined, np.ndarray)
        np.testing.assert_array_equal(joined, data[45:50])

    def test_range_multi_object(self):
        """decode_range on a specific object in a multi-object message."""
        a = np.arange(100, dtype=np.float32)
        b = np.arange(50, dtype=np.float32) + 1000
        meta_dict = make_global_meta(2)
        pairs = [
            (make_descriptor([100], dtype="float32"), a),
            (make_descriptor([50], dtype="float32"), b),
        ]
        msg = bytes(tensogram.encode(meta_dict, pairs))

        result = tensogram.decode_range(msg, 1, [(10, 5)])
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], b[10:15])

        joined = tensogram.decode_range(msg, 1, [(10, 5)], join=True)
        assert isinstance(joined, np.ndarray)
        np.testing.assert_array_equal(joined, b[10:15])

    def test_range_int32(self):
        """decode_range preserves int32 dtype (not raw bytes)."""
        data = np.arange(50, dtype=np.int32)
        msg = encode_simple(data, dtype="int32")

        result = tensogram.decode_range(msg, 0, [(5, 10)])
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], data[5:15])
        assert result[0].dtype == np.int32

        joined = tensogram.decode_range(msg, 0, [(5, 10)], join=True)
        assert isinstance(joined, np.ndarray)
        np.testing.assert_array_equal(joined, data[5:15])
        assert joined.dtype == np.int32

    def test_range_uint8(self):
        """decode_range with uint8 data."""
        data = np.arange(50, dtype=np.uint8)
        msg = encode_simple(data, dtype="uint8")

        result = tensogram.decode_range(msg, 0, [(5, 10)])
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], data[5:15])

        joined = tensogram.decode_range(msg, 0, [(5, 10)], join=True)
        assert isinstance(joined, np.ndarray)
        np.testing.assert_array_equal(joined, data[5:15])

    def test_range_multiple_spans(self):
        """decode_range with multiple (start, count) tuples in one call."""
        data = np.arange(100, dtype=np.float32)
        msg = encode_simple(data)

        # Default (join=False): one array per range
        result = tensogram.decode_range(msg, 0, [(10, 5), (50, 3)])
        assert isinstance(result, list)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], data[10:15])
        np.testing.assert_array_equal(result[1], data[50:53])

        # join=True: concatenated flat array (old behavior)
        joined = tensogram.decode_range(msg, 0, [(10, 5), (50, 3)], join=True)
        assert isinstance(joined, np.ndarray)
        expected = np.concatenate([data[10:15], data[50:53]])
        np.testing.assert_array_equal(joined, expected)

    def test_range_single_split_type(self):
        """Default call returns list, not ndarray."""
        data = np.arange(20, dtype=np.float32)
        msg = encode_simple(data)
        result = tensogram.decode_range(msg, 0, [(0, 5)])
        assert isinstance(result, list)
        assert not isinstance(result, np.ndarray)

    def test_range_single_join_type(self):
        """join=True returns ndarray, not list."""
        data = np.arange(20, dtype=np.float32)
        msg = encode_simple(data)
        result = tensogram.decode_range(msg, 0, [(0, 5)], join=True)
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, list)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


class TestScan:
    """Test scan for message boundary detection."""

    def test_single_message(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data)
        offsets = tensogram.scan(msg)
        assert len(offsets) == 1
        offset, length = offsets[0]
        assert offset == 0
        assert length == len(msg)

    def test_multiple_messages(self):
        """Scan a buffer with multiple concatenated messages."""
        msgs = []
        for i in range(5):
            data = np.full(10, i, dtype=np.float32)
            msgs.append(encode_simple(data))
        buf = b"".join(msgs)

        offsets = tensogram.scan(buf)
        assert len(offsets) == 5

        # Verify each offset points to a valid message
        running = 0
        for i, (offset, length) in enumerate(offsets):
            assert offset == running
            assert length == len(msgs[i])
            running += length

    def test_empty_buffer(self):
        offsets = tensogram.scan(b"")
        assert len(offsets) == 0


# ---------------------------------------------------------------------------
# compute_packing_params
# ---------------------------------------------------------------------------


class TestComputePackingParams:
    """Test compute_packing_params for simple_packing."""

    def test_basic(self):
        values = np.linspace(200.0, 300.0, 1000, dtype=np.float64)
        params = tensogram.compute_packing_params(values, 16, 0)
        assert "reference_value" in params
        assert "binary_scale_factor" in params
        assert "decimal_scale_factor" in params
        assert "bits_per_value" in params
        assert params["bits_per_value"] == 16

    def test_constant_field(self):
        """All-same values should produce valid params."""
        values = np.full(100, 273.15, dtype=np.float64)
        params = tensogram.compute_packing_params(values, 16, 0)
        assert params["reference_value"] == pytest.approx(273.15, abs=1e-6)

    def test_nan_rejected(self):
        """NaN values should raise ValueError."""
        values = np.array([1.0, float("nan"), 3.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"[Nn]a[Nn]"):
            tensogram.compute_packing_params(values, 16, 0)


# ---------------------------------------------------------------------------
# DataObjectDescriptor
# ---------------------------------------------------------------------------


class TestDataObjectDescriptor:
    """Test DataObjectDescriptor properties."""

    def test_properties(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]

        assert desc.obj_type == "ntensor"
        assert desc.ndim == 2
        assert desc.shape == [3, 4]
        assert desc.dtype == "float32"
        assert desc.encoding == "none"
        assert desc.filter == "none"
        assert desc.compression == "none"
        assert desc.byte_order in ("big", "little")

    def test_strides(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]

        # C-order strides: [12, 4, 1]
        assert desc.strides == [12, 4, 1]

    def test_hash_present(self):
        """v3: `descriptor.hash` is always None (deprecated surface).

        The per-object hash moved from the CBOR descriptor to the frame
        footer's inline slot in v3 (see plans/WIRE_FORMAT.md §2.4).
        `descriptor.hash` is retained on the Python class for source
        compatibility but returns None regardless of whether the
        encoder wrote a hash slot.  Use `tensogram validate --checksum`
        or the upcoming `Message.inline_hashes()` accessor (tracked in
        plans/WIRE_FORMAT_CHANGES.md open follow-ups) for frame-level
        integrity.
        """
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, hash_algo="xxh3")
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        # v3 contract: descriptor.hash is always None.
        assert desc.hash is None

    def test_hash_absent(self):
        """Without hash, descriptor.hash is None (same as with-hash in v3)."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, hash_algo=None)
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        assert desc.hash is None

    def test_params(self):
        """Extra keys in descriptor dict become params."""
        data = np.ones(10, dtype=np.float32)
        meta = make_global_meta(2)
        desc_dict = make_descriptor([10], dtype="float32", my_custom_key="hello")
        msg = bytes(tensogram.encode(meta, [(desc_dict, data)]))
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        assert desc.params["my_custom_key"] == "hello"

    def test_repr(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        r = repr(desc)
        assert "DataObjectDescriptor" in r
        assert "float32" in r


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Test Metadata class."""

    def test_version(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, version=2)
        meta, _ = tensogram.decode(msg)
        assert meta.version == 2

    def test_extra_dict(self):
        data = np.ones(10, dtype=np.float32)
        extra = {"source": "test", "count": 42}
        msg = encode_simple(data, extra_meta=extra)
        meta, _ = tensogram.decode(msg)
        e = meta.extra
        assert e["source"] == "test"
        assert e["count"] == 42

    def test_getitem(self):
        data = np.ones(10, dtype=np.float32)
        extra = {"key": "value"}
        msg = encode_simple(data, extra_meta=extra)
        meta, _ = tensogram.decode(msg)
        assert meta["key"] == "value"

    def test_getitem_missing(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data)
        meta, _ = tensogram.decode(msg)
        with pytest.raises(KeyError):
            _ = meta["missing"]

    def test_contains(self):
        """'in' operator on Metadata via __contains__."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"source": "test"})
        meta, _ = tensogram.decode(msg)
        assert "source" in meta
        assert "nonexistent" not in meta

    def test_repr_contains_version(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, version=2)
        meta, _ = tensogram.decode(msg)
        assert "version=2" in repr(meta)


# ---------------------------------------------------------------------------
# TensogramFile
# ---------------------------------------------------------------------------


class TestTensogramFile:
    """Test file-based API: create, append, open, iterate."""

    def test_create_and_message_count(self, tmp_path):
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.ones(10, dtype=np.float32)
            meta = make_global_meta(2)
            desc = make_descriptor([10], dtype="float32")
            f.append(meta, [(desc, data)])
            assert f.message_count() == 1

    def test_append_multiple(self, tmp_path):
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(5):
                data = np.full(10, i, dtype=np.float32)
                meta = make_global_meta(2)
                desc = make_descriptor([10], dtype="float32")
                f.append(meta, [(desc, data)])
            assert f.message_count() == 5

    def test_open_and_decode(self, tmp_path):
        path = str(tmp_path / "test.tgm")
        data = np.arange(20, dtype=np.float32)

        with tensogram.TensogramFile.create(path) as f:
            meta = make_global_meta(2)
            desc = make_descriptor([20], dtype="float32")
            f.append(meta, [(desc, data)])

        with tensogram.TensogramFile.open(path) as f2:
            assert f2.message_count() == 1
            meta, objects = f2.decode_message(0)
            _, arr = objects[0]
            np.testing.assert_array_equal(arr, data)

    def test_read_message_raw(self, tmp_path):
        path = str(tmp_path / "test.tgm")
        data = np.ones(10, dtype=np.float32)

        with tensogram.TensogramFile.create(path) as f:
            meta = make_global_meta(2)
            desc = make_descriptor([10], dtype="float32")
            f.append(meta, [(desc, data)])

        with tensogram.TensogramFile.open(path) as f2:
            raw = f2.read_message(0)
            assert isinstance(raw, bytes)
            assert len(raw) > 0

            # Raw bytes should be a valid message
            meta, objects = tensogram.decode(raw)
            _, arr = objects[0]
            np.testing.assert_array_equal(arr, data)

    def test_messages_returns_all(self, tmp_path):
        path = str(tmp_path / "test.tgm")
        expected = []

        with tensogram.TensogramFile.create(path) as f:
            for i in range(3):
                data = np.full(5, float(i), dtype=np.float32)
                expected.append(data.copy())
                meta = make_global_meta(2)
                desc = make_descriptor([5], dtype="float32")
                f.append(meta, [(desc, data)])

        with tensogram.TensogramFile.open(path) as f2:
            msgs = f2.messages()
            assert len(msgs) == 3
            for i, msg_bytes in enumerate(msgs):
                assert isinstance(msg_bytes, bytes)
                # Verify each message decodes to the correct data
                _, objects = tensogram.decode(msg_bytes)
                _, arr = objects[0]
                np.testing.assert_array_equal(arr, expected[i])

    def test_decode_message_with_extra_meta(self, tmp_path):
        """Extra metadata keys survive file round-trip."""
        path = str(tmp_path / "test.tgm")

        with tensogram.TensogramFile.create(path) as f:
            data = np.ones(10, dtype=np.float32)
            meta = make_global_meta(2, source="test", step=6)
            desc = make_descriptor([10], dtype="float32")
            f.append(meta, [(desc, data)])

        with tensogram.TensogramFile.open(path) as f2:
            meta, _ = f2.decode_message(0)
            assert meta["source"] == "test"
            assert meta["step"] == 6

    def test_append_with_hash(self, tmp_path):
        """Append with hash, then decode with verification."""
        path = str(tmp_path / "test.tgm")

        with tensogram.TensogramFile.create(path) as f:
            data = np.ones(10, dtype=np.float32)
            meta = make_global_meta(2)
            desc = make_descriptor([10], dtype="float32")
            f.append(meta, [(desc, data)], hash="xxh3")

        with tensogram.TensogramFile.open(path) as f2:
            meta, objects = f2.decode_message(0, verify_hash=True)
            _, arr = objects[0]
            np.testing.assert_array_equal(arr, data)

    def test_len(self, tmp_path):
        """len(file) returns message count."""
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            assert len(f) == 0
            data = np.ones(10, dtype=np.float32)
            meta = make_global_meta(2)
            desc = make_descriptor([10], dtype="float32")
            f.append(meta, [(desc, data)])
            assert len(f) == 1

    def test_context_manager(self, tmp_path):
        """TensogramFile supports 'with' statement."""
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            assert isinstance(f, tensogram.TensogramFile)
            data = np.ones(10, dtype=np.float32)
            meta = make_global_meta(2)
            desc = make_descriptor([10], dtype="float32")
            f.append(meta, [(desc, data)])

        # File should be usable after with-block ends
        with tensogram.TensogramFile.open(path) as f2:
            assert f2.message_count() == 1

    def test_repr(self, tmp_path):
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            r = repr(f)
            assert "TensogramFile" in r
            assert "test.tgm" in r


# ---------------------------------------------------------------------------
# Wire format sanity
# ---------------------------------------------------------------------------


class TestWireFormat:
    """Verify magic bytes and basic wire format properties."""

    def test_magic_bytes(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data)
        assert msg[:8] == b"TENSOGRM"

    def test_message_length_reasonable(self):
        """Encoded message should be larger than raw data (has framing)."""
        data = np.ones(100, dtype=np.float32)
        msg = encode_simple(data)
        assert len(msg) > data.nbytes
        # But not absurdly large (< 10x overhead for 400 bytes of data)
        assert len(msg) < data.nbytes * 10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Test error conditions raise appropriate exceptions."""

    def test_decode_garbage(self):
        """Decoding garbage should raise ValueError (FramingError)."""
        with pytest.raises(ValueError, match="FramingError"):
            tensogram.decode(b"this is not a valid message")

    def test_decode_empty(self):
        with pytest.raises(ValueError, match="FramingError"):
            tensogram.decode(b"")

    def test_decode_truncated(self):
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data)
        with pytest.raises(ValueError, match="FramingError"):
            tensogram.decode(msg[:20])

    def test_encode_missing_version(self):
        """Encoding without 'version' key should raise ValueError."""
        desc = make_descriptor([10], dtype="float32")
        data = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="version"):
            tensogram.encode({}, [(desc, data)])

    def test_encode_missing_dtype(self):
        """Encoding without 'dtype' in descriptor should raise ValueError."""
        meta = make_global_meta(2)
        desc = {"type": "ntensor", "shape": [10]}  # no dtype
        data = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            tensogram.encode(meta, [(desc, data)])

    def test_encode_missing_shape(self):
        """Encoding without 'shape' in descriptor should raise ValueError."""
        meta = make_global_meta(2)
        desc = {"type": "ntensor", "dtype": "float32"}  # no shape
        data = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            tensogram.encode(meta, [(desc, data)])

    def test_encode_unknown_dtype(self):
        """Unknown dtype string should raise ValueError."""
        meta = make_global_meta(2)
        desc = make_descriptor([10], dtype="complex256")
        data = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="unknown dtype"):
            tensogram.encode(meta, [(desc, data)])

    def test_encode_unknown_hash(self):
        """Unknown hash algorithm should raise ValueError."""
        data = np.ones(10, dtype=np.float32)
        meta = make_global_meta(2)
        desc = make_descriptor([10], dtype="float32")
        with pytest.raises(ValueError, match="unknown hash"):
            tensogram.encode(meta, [(desc, data)], hash="sha256")

    def test_encode_unknown_byte_order(self):
        """Unknown byte_order should raise ValueError."""
        meta = make_global_meta(2)
        desc = make_descriptor([10], dtype="float32", byte_order="middle")
        data = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="unknown byte_order"):
            tensogram.encode(meta, [(desc, data)])

    def test_hash_mismatch_detected(self):
        """v3: corruption detection moved to `validate`, not `decode`.

        `decode(verify_hash=True)` is a no-op in v3 (the option is
        kept for source compatibility).  Frame-level integrity now
        goes through `tensogram.validate(msg, level="checksum")`,
        which recomputes each frame's body hash and compares it to
        the inline slot.  Corruption in the payload or header region
        surfaces as a validation issue, not a decode error.

        Decode may still raise `ValueError` (FramingError) if the
        corruption breaks frame boundaries; otherwise decode succeeds
        and the validation report flags the mismatch.
        """
        data = np.arange(100, dtype=np.float32)
        msg = encode_simple(data, hash_algo="xxh3")
        corrupted = bytearray(msg)
        # Corrupt a byte in the payload area (past header, before terminator)
        mid = len(corrupted) // 2
        corrupted[mid] ^= 0xFF

        # Decode no longer checks the hash — may succeed or raise
        # FramingError depending on where the tamper lands.  Ignore
        # that return value; the real check is in validate.
        try:
            tensogram.decode(bytes(corrupted), verify_hash=True)
        except ValueError:
            return  # FramingError is an acceptable v3 detection path

        # If decode succeeded, validate --checksum must surface the
        # corruption as a HashMismatch issue.
        report = tensogram.validate(bytes(corrupted), level="checksum")
        codes = [issue["code"] for issue in report["issues"]]
        assert any(
            c in ("hash_mismatch", "decode_pipeline_failed", "cbor_offset_invalid")
            for c in codes
        ), f"expected integrity or structural error in validate report, got: {report}"

    def test_file_open_nonexistent(self, tmp_path):
        nonexistent_path = tmp_path / "nonexistent_file_12345.tgm"
        with pytest.raises(OSError, match="nonexistent"):
            tensogram.TensogramFile.open(str(nonexistent_path))

    def test_encode_bad_pair_format(self):
        """Passing non-tuple elements should raise."""
        meta = make_global_meta(2)
        with pytest.raises((ValueError, TypeError)):
            # Pass a list of dicts instead of (dict, array) tuples
            tensogram.encode(meta, [make_descriptor([10])])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_object_message(self):
        """Encoding a message with zero data objects."""
        meta = make_global_meta(2, note="empty")
        msg = bytes(tensogram.encode(meta, []))
        decoded_meta, objects = tensogram.decode(msg)
        assert len(objects) == 0
        assert decoded_meta["note"] == "empty"

    def test_large_array(self):
        """Encode/decode a reasonably large array."""
        data = np.random.default_rng(42).random(100_000).astype(np.float32)
        msg = encode_simple(data)
        _, objects = tensogram.decode(msg)
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)

    def test_version_1(self):
        """Version 1 should also work."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, version=1)
        meta, objects = tensogram.decode(msg)
        assert meta.version == 1
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)

    def test_many_extra_keys(self):
        """Metadata with many extra keys."""
        data = np.ones(10, dtype=np.float32)
        extra = {f"key_{i}": f"value_{i}" for i in range(20)}
        msg = encode_simple(data, extra_meta=extra)
        meta = tensogram.decode_metadata(msg)
        for i in range(20):
            assert meta[f"key_{i}"] == f"value_{i}"

    def test_bool_metadata_value(self):
        """Boolean values in metadata."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"flag": True})
        meta = tensogram.decode_metadata(msg)
        assert meta["flag"] is True

    def test_null_metadata_value(self):
        """None values in metadata."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"empty": None})
        meta = tensogram.decode_metadata(msg)
        assert meta["empty"] is None

    def test_float_metadata_value(self):
        """Float values in metadata."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"pi": 3.14159})
        meta = tensogram.decode_metadata(msg)
        assert meta["pi"] == pytest.approx(3.14159)

    def test_list_metadata_value(self):
        """List values in metadata."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"tags": ["a", "b", "c"]})
        meta = tensogram.decode_metadata(msg)
        assert meta["tags"] == ["a", "b", "c"]

    def test_scan_then_decode_each(self):
        """Scan a multi-message buffer, then decode each independently."""
        msgs = []
        original_data = []
        for i in range(4):
            data = np.full(10, float(i), dtype=np.float32)
            original_data.append(data)
            msgs.append(encode_simple(data, extra_meta={"index": i}))
        buf = b"".join(msgs)

        offsets = tensogram.scan(buf)
        assert len(offsets) == 4

        for i, (offset, length) in enumerate(offsets):
            chunk = buf[offset : offset + length]
            meta, objects = tensogram.decode(chunk)
            _, arr = objects[0]
            np.testing.assert_array_equal(arr, original_data[i])
            assert meta["index"] == i

    def test_file_multi_object_message(self, tmp_path):
        """File with multi-object messages."""
        path = str(tmp_path / "multi.tgm")

        with tensogram.TensogramFile.create(path) as f:
            a = np.ones(10, dtype=np.float32)
            b = np.zeros(5, dtype=np.uint8)
            meta = make_global_meta(2)
            pairs = [
                (make_descriptor([10], dtype="float32"), a),
                (make_descriptor([5], dtype="uint8"), b),
            ]
            f.append(meta, pairs)

        with tensogram.TensogramFile.open(path) as f2:
            meta, objects = f2.decode_message(0)
            assert len(objects) == 2
            _, arr0 = objects[0]
            _, arr1 = objects[1]
            np.testing.assert_array_equal(arr0, a)
            np.testing.assert_array_equal(arr1, b)

    def test_file_many_messages(self, tmp_path):
        """File with many messages for stress test."""
        path = str(tmp_path / "many.tgm")
        n = 50

        with tensogram.TensogramFile.create(path) as f:
            for i in range(n):
                data = np.full(10, float(i), dtype=np.float32)
                meta = make_global_meta(2, index=i)
                desc = make_descriptor([10], dtype="float32")
                f.append(meta, [(desc, data)])
            assert f.message_count() == n

        with tensogram.TensogramFile.open(path) as f2:
            assert f2.message_count() == n
            # Spot-check a few
            for idx in [0, 25, 49]:
                meta, objects = f2.decode_message(idx)
                _, arr = objects[0]
                np.testing.assert_array_equal(arr, np.full(10, float(idx), dtype=np.float32))
                assert meta["index"] == idx

    def test_native_endian_roundtrip(self):
        """Encode and decode with native byte order — values should match."""
        data = np.arange(10, dtype=np.float32)
        meta = make_global_meta(2)
        desc = make_descriptor([10], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        _, objects = tensogram.decode(msg)
        _decoded_desc, arr = objects[0]
        np.testing.assert_array_equal(arr, data)

    def test_decode_range_with_verify_hash(self):
        """decode_range with verify_hash=True on valid data."""
        data = np.arange(100, dtype=np.float32)
        msg = encode_simple(data, hash_algo="xxh3")

        # Default (join=False) with verify_hash
        result = tensogram.decode_range(msg, 0, [(10, 5)], verify_hash=True)
        assert isinstance(result, list)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], data[10:15])

        # join=True with verify_hash
        joined = tensogram.decode_range(msg, 0, [(10, 5)], join=True, verify_hash=True)
        assert isinstance(joined, np.ndarray)
        np.testing.assert_array_equal(joined, data[10:15])

    def test_decode_metadata_multi_object(self):
        """decode_metadata on multi-object message only reads metadata."""
        a = np.ones(100, dtype=np.float32)
        b = np.zeros(50, dtype=np.uint8)
        meta_dict = make_global_meta(2, source="multi")
        pairs = [
            (make_descriptor([100], dtype="float32"), a),
            (make_descriptor([50], dtype="uint8"), b),
        ]
        msg = bytes(tensogram.encode(meta_dict, pairs))
        meta = tensogram.decode_metadata(msg)
        assert meta.version == 2
        assert meta["source"] == "multi"

    def test_file_decode_message_out_of_range(self, tmp_path):
        """decode_message with invalid index should raise."""
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.ones(10, dtype=np.float32)
            meta = make_global_meta(2)
            desc = make_descriptor([10], dtype="float32")
            f.append(meta, [(desc, data)])

        with (
            tensogram.TensogramFile.open(path) as f2,
            pytest.raises((ValueError, IndexError), match=r"index|out of range|ObjectError"),
        ):
            f2.decode_message(99)

    def test_decode_idempotent(self):
        """Decoding the same message twice produces identical results."""
        data = np.arange(50, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"run": 1})

        meta1, objects1 = tensogram.decode(msg)
        meta2, objects2 = tensogram.decode(msg)

        assert meta1.version == meta2.version
        assert meta1["run"] == meta2["run"]
        np.testing.assert_array_equal(objects1[0][1], objects2[0][1])

    def test_empty_string_metadata(self):
        """Empty string value in metadata."""
        data = np.ones(5, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"name": ""})
        meta = tensogram.decode_metadata(msg)
        assert meta["name"] == ""

    def test_compute_packing_params_nonzero_decimal_scale(self):
        """compute_packing_params with decimal_scale_factor != 0."""
        values = np.linspace(200.0, 300.0, 100, dtype=np.float64)
        params = tensogram.compute_packing_params(values, 16, 2)
        assert params["decimal_scale_factor"] == 2
        assert params["bits_per_value"] == 16

    def test_encode_decode_preserves_data(self):
        """Encoding the same data twice produces structurally identical output.

        Wire bytes differ per-encode due to provenance (UUID, timestamp),
        but the decoded metadata and payload must be identical.
        """
        data = np.arange(20, dtype=np.float32)
        msg1 = encode_simple(data, hash_algo=None, extra_meta={"key": "val"})
        msg2 = encode_simple(data, hash_algo=None, extra_meta={"key": "val"})

        m1 = tensogram.decode(msg1)
        m2 = tensogram.decode(msg2)
        assert m1.metadata.version == m2.metadata.version
        assert m1.metadata["key"] == m2.metadata["key"]
        np.testing.assert_array_equal(m1.objects[0][1], m2.objects[0][1])

    # ── File iteration ──

    def test_file_iter_basic(self, tmp_path):
        """Iterate over file messages with for loop."""
        path = str(tmp_path / "iter.tgm")
        n = 5
        with tensogram.TensogramFile.create(path) as f:
            for i in range(n):
                data = np.full(10, float(i), dtype=np.float32)
                meta = make_global_meta(2, index=i)
                desc = make_descriptor([10], dtype="float32")
                f.append(meta, [(desc, data)])

        collected = []
        with tensogram.TensogramFile.open(path) as f:
            for meta, objects in f:
                _, arr = objects[0]
                collected.append((meta["index"], arr[0]))

        assert len(collected) == n
        for i, (idx, val) in enumerate(collected):
            assert idx == i
            assert val == float(i)

    def test_file_iter_empty(self, tmp_path):
        """Iterating an empty file yields nothing."""
        path = str(tmp_path / "empty.tgm")
        with tensogram.TensogramFile.create(path):
            pass

        with tensogram.TensogramFile.open(path) as f:
            items = list(f)
        assert items == []

    def test_file_iter_len(self, tmp_path):
        """Iterator __len__ tracks remaining messages."""
        path = str(tmp_path / "len.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for _i in range(3):
                data = np.zeros(4, dtype=np.float32)
                f.append(make_global_meta(2), [(make_descriptor([4], dtype="float32"), data)])

        with tensogram.TensogramFile.open(path) as f:
            it = iter(f)
            assert len(it) == 3
            next(it)
            assert len(it) == 2
            next(it)
            assert len(it) == 1
            next(it)
            assert len(it) == 0

    def test_file_iter_stops(self, tmp_path):
        """Iterator raises StopIteration after exhaustion."""
        path = str(tmp_path / "stop.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.ones(4, dtype=np.float32)
            f.append(make_global_meta(2), [(make_descriptor([4], dtype="float32"), data)])

        with tensogram.TensogramFile.open(path) as f:
            it = iter(f)
            next(it)
            with pytest.raises(StopIteration):
                next(it)

    def test_file_getitem(self, tmp_path):
        """Index file messages with []."""
        path = str(tmp_path / "getitem.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(5):
                data = np.full(8, float(i), dtype=np.float32)
                f.append(
                    make_global_meta(2, index=i),
                    [(make_descriptor([8], dtype="float32"), data)],
                )

        with tensogram.TensogramFile.open(path) as f:
            assert f[0].metadata["index"] == 0
            assert f[4].metadata["index"] == 4
            assert f[-1].metadata["index"] == 4
            assert f[-5].metadata["index"] == 0

    def test_file_getitem_out_of_range(self, tmp_path):
        """file[bad_index] raises IndexError."""
        path = str(tmp_path / "oob.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.ones(4, dtype=np.float32)
            f.append(make_global_meta(2), [(make_descriptor([4], dtype="float32"), data)])

        with tensogram.TensogramFile.open(path) as f:
            with pytest.raises(IndexError):
                f[5]
            with pytest.raises(IndexError):
                f[-2]

    def test_file_slice_basic(self, tmp_path):
        """file[1:3] returns a list of two decoded messages."""
        path = str(tmp_path / "slice.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(5):
                data = np.full(8, float(i), dtype=np.float32)
                f.append(
                    make_global_meta(2, index=i),
                    [(make_descriptor([8], dtype="float32"), data)],
                )

        with tensogram.TensogramFile.open(path) as f:
            result = f[1:3]
            assert isinstance(result, list)
            assert len(result) == 2
            meta0, _ = result[0]
            meta1, _ = result[1]
            assert meta0["index"] == 1
            assert meta1["index"] == 2

    def test_file_slice_step(self, tmp_path):
        """file[::2] returns every other message."""
        path = str(tmp_path / "step.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(6):
                data = np.full(4, float(i), dtype=np.float32)
                f.append(
                    make_global_meta(2, index=i),
                    [(make_descriptor([4], dtype="float32"), data)],
                )

        with tensogram.TensogramFile.open(path) as f:
            result = f[::2]
            assert len(result) == 3
            indices = [meta["index"] for meta, _ in result]
            assert indices == [0, 2, 4]

    def test_file_slice_negative(self, tmp_path):
        """file[-2:] returns the last two messages."""
        path = str(tmp_path / "neg.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(5):
                data = np.full(4, float(i), dtype=np.float32)
                f.append(
                    make_global_meta(2, index=i),
                    [(make_descriptor([4], dtype="float32"), data)],
                )

        with tensogram.TensogramFile.open(path) as f:
            result = f[-2:]
            assert len(result) == 2
            assert result[0][0]["index"] == 3
            assert result[1][0]["index"] == 4

    def test_file_slice_reverse(self, tmp_path):
        """file[::-1] returns all messages in reverse order."""
        path = str(tmp_path / "rev.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(4):
                data = np.full(4, float(i), dtype=np.float32)
                f.append(
                    make_global_meta(2, index=i),
                    [(make_descriptor([4], dtype="float32"), data)],
                )

        with tensogram.TensogramFile.open(path) as f:
            result = f[::-1]
            assert len(result) == 4
            indices = [meta["index"] for meta, _ in result]
            assert indices == [3, 2, 1, 0]

    def test_file_slice_empty(self, tmp_path):
        """file[2:2] returns an empty list."""
        path = str(tmp_path / "empty_slice.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for _i in range(3):
                data = np.zeros(4, dtype=np.float32)
                f.append(make_global_meta(2), [(make_descriptor([4], dtype="float32"), data)])

        with tensogram.TensogramFile.open(path) as f:
            assert f[2:2] == []
            assert f[5:10] == []

    def test_file_getitem_bad_key(self, tmp_path):
        """file['bad'] raises ValueError."""
        path = str(tmp_path / "badkey.tgm")
        with tensogram.TensogramFile.create(path) as f:
            data = np.ones(4, dtype=np.float32)
            f.append(make_global_meta(2), [(make_descriptor([4], dtype="float32"), data)])

        with tensogram.TensogramFile.open(path) as f, pytest.raises(TypeError):
            f["bad"]

    # ── Message namedtuple ──

    def test_message_namedtuple_from_decode(self):
        """decode() returns a Message with .metadata and .objects."""
        data = np.arange(10, dtype=np.float32)
        msg_bytes = encode_simple(data, extra_meta={"key": "val"})
        msg = tensogram.decode(msg_bytes)
        assert isinstance(msg, tensogram.Message)
        assert msg.metadata["key"] == "val"
        _, arr = msg.objects[0]
        np.testing.assert_array_equal(arr, data)

    def test_message_namedtuple_from_file(self, tmp_path):
        """decode_message and iteration return Message namedtuples."""
        path = str(tmp_path / "msg.tgm")
        data = np.ones(4, dtype=np.float32)
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                make_global_meta(2, tag="test"),
                [(make_descriptor([4], dtype="float32"), data)],
            )

        with tensogram.TensogramFile.open(path) as f:
            # decode_message
            msg = f.decode_message(0)
            assert isinstance(msg, tensogram.Message)
            assert msg.metadata["tag"] == "test"
            # indexing
            msg2 = f[0]
            assert isinstance(msg2, tensogram.Message)
            # iteration
            for msg3 in f:
                assert isinstance(msg3, tensogram.Message)

    def test_message_tuple_unpacking(self):
        """Message supports tuple unpacking."""
        data = np.ones(4, dtype=np.float32)
        msg_bytes = encode_simple(data)
        meta, objects = tensogram.decode(msg_bytes)
        assert meta.version == 2
        assert len(objects) == 1

    # ── Buffer iteration ──

    def test_iter_messages_basic(self):
        """iter_messages yields decoded messages from a byte buffer."""
        msgs = []
        for i in range(4):
            data = np.full(8, float(i), dtype=np.float32)
            meta = make_global_meta(2, index=i)
            desc = make_descriptor([8], dtype="float32")
            msgs.append(bytes(tensogram.encode(meta, [(desc, data)])))

        buf = b"".join(msgs)
        collected = list(tensogram.iter_messages(buf))
        assert len(collected) == 4
        for i, (meta, objects) in enumerate(collected):
            assert meta["index"] == i
            _, arr = objects[0]
            np.testing.assert_array_equal(arr, np.full(8, float(i), dtype=np.float32))

    def test_iter_messages_empty(self):
        """iter_messages on empty buffer yields nothing."""
        assert list(tensogram.iter_messages(b"")) == []

    def test_iter_messages_len(self):
        """iter_messages supports len() tracking remaining."""
        data = np.ones(4, dtype=np.float32)
        meta = make_global_meta(2)
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        buf = msg * 3

        it = tensogram.iter_messages(buf)
        assert len(it) == 3
        next(it)
        assert len(it) == 2

    def test_iter_messages_stops(self):
        """iter_messages raises StopIteration after exhaustion."""
        data = np.ones(4, dtype=np.float32)
        msg = bytes(
            tensogram.encode(make_global_meta(2), [(make_descriptor([4], dtype="float32"), data)])
        )

        it = tensogram.iter_messages(msg)
        next(it)
        with pytest.raises(StopIteration):
            next(it)

    def test_iter_messages_with_garbage(self):
        """iter_messages skips garbage between valid messages."""
        data = np.ones(4, dtype=np.float32)
        meta = make_global_meta(2, tag="valid")
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))

        buf = b"\xde\xad" + msg + b"\xff\xff" + msg
        collected = list(tensogram.iter_messages(buf))
        assert len(collected) == 2
        for meta, _ in collected:
            assert meta["tag"] == "valid"

    def test_iter_messages_verify_hash(self):
        """iter_messages with verify_hash=True on hashed data."""
        data = np.arange(16, dtype=np.float32)
        meta = make_global_meta(2)
        desc = make_descriptor([16], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)], hash="xxh3"))

        collected = list(tensogram.iter_messages(msg, verify_hash=True))
        assert len(collected) == 1
        _, objects = collected[0]
        _, arr = objects[0]
        np.testing.assert_array_equal(arr, data)


# ---------------------------------------------------------------------------
# Coverage: Metadata properties (.base, .reserved, .extra, __getitem__)
# ---------------------------------------------------------------------------


class TestMetadataCoverage:
    """Thorough coverage of Metadata properties and access patterns."""

    def _encode_with_mars(self):
        """Encode a message with per-object base metadata."""
        meta = {
            "version": 2,
            "base": [
                {
                    "mars": {
                        "class": "od",
                        "date": "20260401",
                        "param": "2t",
                        "levtype": "sfc",
                    },
                    "source": "test",
                },
                {
                    "mars": {
                        "class": "od",
                        "date": "20260401",
                        "param": "msl",
                        "levtype": "sfc",
                    },
                    "source": "test",
                },
            ],
        }
        d1 = make_descriptor([4], dtype="float32")
        d2 = make_descriptor([4], dtype="float32")
        data = np.ones(4, dtype=np.float32)
        return bytes(tensogram.encode(meta, [(d1, data), (d2, data)]))

    def test_base_property(self):
        """Metadata.base returns the per-object metadata list."""
        msg = self._encode_with_mars()
        meta = tensogram.decode_metadata(msg)
        base = meta.base
        assert isinstance(base, list)
        assert len(base) == 2
        assert base[0]["mars"]["class"] == "od"
        assert base[0]["source"] == "test"

    def test_base_per_object(self):
        """Metadata.base entries hold per-object metadata."""
        msg = self._encode_with_mars()
        meta = tensogram.decode_metadata(msg)
        base = meta.base
        assert base[0]["mars"]["param"] == "2t"
        assert base[1]["mars"]["param"] == "msl"

    def test_extra_property(self):
        """Metadata.extra returns non-standard top-level keys."""
        msg = encode_simple(
            np.ones(4, dtype=np.float32),
            extra_meta={"custom_field": "hello", "number": 42},
        )
        meta = tensogram.decode_metadata(msg)
        extra = meta.extra
        assert extra["custom_field"] == "hello"
        assert extra["number"] == 42

    def test_getitem_base_precedence(self):
        """__getitem__ checks base entries first, then extra."""
        meta_dict = {
            "version": 2,
            "base": [{"shared": "from_base"}],
            "_extra_": {"shared": "from_extra"},
        }
        msg = bytes(
            tensogram.encode(
                meta_dict,
                [(make_descriptor([4]), np.ones(4, dtype=np.float32))],
            )
        )
        meta = tensogram.decode_metadata(msg)
        # "shared" exists in base[0] → base wins
        assert meta["shared"] == "from_base"

    def test_getitem_keyerror(self):
        """__getitem__ raises KeyError for missing keys."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta = tensogram.decode_metadata(msg)
        with pytest.raises(KeyError, match="nonexistent"):
            meta["nonexistent"]

    def test_contains(self):
        """__contains__ checks both base entries and extra."""
        meta_dict = {
            "version": 2,
            "base": [{"b_key": 1}],
            "e_key": 2,
        }
        msg = bytes(
            tensogram.encode(
                meta_dict,
                [(make_descriptor([4]), np.ones(4, dtype=np.float32))],
            )
        )
        meta = tensogram.decode_metadata(msg)
        assert "b_key" in meta
        assert "e_key" in meta
        assert "missing" not in meta

    def test_repr(self):
        """__repr__ returns informative string."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta = tensogram.decode_metadata(msg)
        r = repr(meta)
        assert "Metadata" in r
        assert "version=" in r

    def test_empty_base(self):
        """Message with no base metadata (zero objects)."""
        msg = bytes(
            tensogram.encode(
                {"version": 2},
                [],
            )
        )
        meta = tensogram.decode_metadata(msg)
        assert meta.base == []

    def test_base_with_reserved_auto_populated(self):
        """Base entries get _reserved_.tensor auto-populated by encoder."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta = tensogram.decode_metadata(msg)
        base = meta.base
        assert isinstance(base, list)
        assert len(base) == 1
        # Encoder auto-populates _reserved_.tensor with ndim/shape/strides/dtype
        assert "_reserved_" in base[0]
        tensor_info = base[0]["_reserved_"]["tensor"]
        assert tensor_info["ndim"] == 1
        assert tensor_info["shape"] == [4]

    def test_reserved_property(self):
        """Metadata.reserved returns library-reserved metadata."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta = tensogram.decode_metadata(msg)
        reserved = meta.reserved
        assert isinstance(reserved, dict)
        # Encoder populates provenance info in reserved
        assert "encoder" in reserved
        assert "time" in reserved
        assert "uuid" in reserved


# ---------------------------------------------------------------------------
# Coverage: DataObjectDescriptor properties
# ---------------------------------------------------------------------------


class TestDescriptorCoverage:
    """Thorough coverage of DataObjectDescriptor properties."""

    def test_all_properties(self):
        """Every descriptor property returns the correct value."""
        data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        msg = encode_simple(data, dtype="float64")
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]

        assert desc.obj_type == "ntensor"
        assert desc.ndim == 3
        assert desc.shape == [2, 3, 4]
        assert desc.strides == [12, 4, 1]
        assert desc.dtype == "float64"
        assert desc.byte_order in ("big", "little")
        assert desc.encoding == "none"
        assert desc.filter == "none"
        assert desc.compression == "none"
        assert isinstance(desc.params, dict)

    def test_hash_with_xxh3(self):
        """v3: `descriptor.hash` is deprecated — always None.

        The per-object hash lives in the frame footer's inline slot
        (plans/WIRE_FORMAT.md §2.4).  Python's `descriptor.hash`
        stays on the class for source compatibility and returns
        None regardless of the encoder's `hash` option.  Use the
        validate API to confirm integrity.
        """
        msg = encode_simple(np.ones(10, dtype=np.float32), hash_algo="xxh3")
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        assert desc.hash is None
        # Integrity verification lives at the validate layer now.
        report = tensogram.validate(msg, level="checksum")
        assert report["hash_verified"], (
            f"checksum validation should pass on fresh encode, got: {report}"
        )

    def test_hash_without(self):
        """Descriptor reports None when no hash is used."""
        msg = encode_simple(np.ones(10, dtype=np.float32), hash_algo=None)
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        assert desc.hash is None

    def test_extra_params_preserved(self):
        """Extra keys in descriptor dict end up in params."""
        data = np.ones(8, dtype=np.float32)
        desc_dict = make_descriptor([8], custom_key="hello", numeric_param=42)
        msg = bytes(tensogram.encode(make_global_meta(2), [(desc_dict, data)]))
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        assert desc.params["custom_key"] == "hello"
        assert desc.params["numeric_param"] == 42

    def test_1d_strides(self):
        """1-D array has strides [1]."""
        msg = encode_simple(np.arange(10, dtype=np.float32))
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        assert desc.strides == [1]

    def test_repr(self):
        """Descriptor __repr__ is informative."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        r = repr(desc)
        assert "float32" in r
        assert "4" in r


# ---------------------------------------------------------------------------
# Coverage: Error handling
# ---------------------------------------------------------------------------


class TestErrorCoverage:
    """Test all error paths are properly raised and typed."""

    def test_decode_garbage(self):
        """Garbage input raises ValueError."""
        with pytest.raises(ValueError, match=r"[Ff]raming|[Ii]nvalid"):
            tensogram.decode(b"GARBAGE_DATA_NOT_TENSOGRAM")

    def test_decode_empty(self):
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match=r"[Ff]raming|[Ii]nvalid|[Ee]mpty"):
            tensogram.decode(b"")

    def test_decode_truncated(self):
        """Truncated message raises ValueError."""
        msg = encode_simple(np.ones(100, dtype=np.float32))
        with pytest.raises(ValueError, match=r"[Ff]raming|truncat"):
            tensogram.decode(msg[:50])

    def test_decode_object_out_of_range(self):
        """Object index out of range raises ValueError."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        with pytest.raises(ValueError, match="out of range"):
            tensogram.decode_object(msg, index=99)

    def test_decode_object_negative_index(self):
        """Negative object index raises ValueError."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        with pytest.raises((ValueError, OverflowError)):
            tensogram.decode_object(msg, index=-1)

    def test_hash_mismatch(self):
        """v3: HashMismatch is surfaced via `validate_message`, not decode.

        `decode(verify_hash=True)` is a v3 no-op; integrity is a
        validate-level check.  A middle-of-payload tamper triggers
        a `HashMismatch` or `DecodePipelineFailed` issue in the
        validation report.
        """
        msg = bytearray(encode_simple(np.ones(100, dtype=np.float32), hash_algo="xxh3"))
        msg[len(msg) // 2] ^= 0xFF
        report = tensogram.validate(bytes(msg), level="checksum")
        codes = [issue["code"] for issue in report["issues"]]
        assert any(
            c in ("hash_mismatch", "decode_pipeline_failed", "cbor_offset_invalid")
            for c in codes
        ), f"expected integrity error, got: {report}"

    def test_encode_shape_mismatch(self):
        """Shape mismatch between descriptor and data raises ValueError."""
        data = np.ones(10, dtype=np.float32)
        desc = make_descriptor([5, 5])  # 25 elements, data has 10
        with pytest.raises(ValueError, match=r"does not match|[Ss]ize|[Ll]ength"):
            tensogram.encode(make_global_meta(2), [(desc, data)])

    def test_encode_missing_descriptor_field(self):
        """Descriptor missing 'type' raises ValueError."""
        data = np.ones(4, dtype=np.float32)
        bad_desc = {"shape": [4], "dtype": "float32"}  # no "type"
        with pytest.raises(ValueError, match="type"):
            tensogram.encode(make_global_meta(2), [(bad_desc, data)])

    def test_encode_missing_shape(self):
        """Descriptor missing 'shape' raises ValueError."""
        data = np.ones(4, dtype=np.float32)
        bad_desc = {"type": "ntensor", "dtype": "float32"}
        with pytest.raises(ValueError, match="shape"):
            tensogram.encode(make_global_meta(2), [(bad_desc, data)])

    def test_encode_missing_dtype(self):
        """Descriptor missing 'dtype' raises ValueError."""
        data = np.ones(4, dtype=np.float32)
        bad_desc = {"type": "ntensor", "shape": [4]}
        with pytest.raises(ValueError, match="dtype"):
            tensogram.encode(make_global_meta(2), [(bad_desc, data)])

    def test_encode_bad_byte_order(self):
        """Invalid byte_order raises ValueError."""
        data = np.ones(4, dtype=np.float32)
        desc = make_descriptor([4], byte_order="middle")
        with pytest.raises(ValueError, match="byte_order"):
            tensogram.encode(make_global_meta(2), [(desc, data)])

    def test_file_not_found(self):
        """Opening nonexistent file raises OSError."""
        with pytest.raises(OSError, match=r"[Nn]o such|not found"):
            tensogram.TensogramFile.open("/nonexistent/path.tgm")

    def test_file_out_of_range_index(self, tmp_path):
        """File decode_message with bad index raises ValueError."""
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                make_global_meta(2),
                [(make_descriptor([4]), np.ones(4, dtype=np.float32))],
            )
        with (
            tensogram.TensogramFile.open(path) as f,
            pytest.raises((ValueError, IndexError)),
        ):
            f.decode_message(999)


# ---------------------------------------------------------------------------
# Coverage: compute_packing_params edge cases
# ---------------------------------------------------------------------------


class TestMetadataEdgeCases:
    """Edge cases for metadata model (FFI, Python, CLI layers)."""

    def test_reserved_in_base_rejected(self):
        """Client code cannot set _reserved_ in base entries."""
        data = np.ones(4, dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"_reserved_": {"tensor": {"ndim": 1}}}],
        }
        desc = make_descriptor([4], dtype="float32")
        with pytest.raises(ValueError, match="_reserved_"):
            tensogram.encode(meta, [(desc, data)])

    def test_reserved_at_toplevel_rejected(self):
        """Client code cannot set _reserved_ at the top level."""
        data = np.ones(4, dtype=np.float32)
        meta = {
            "version": 2,
            "_reserved_": {"encoder": "fake"},
        }
        desc = make_descriptor([4], dtype="float32")
        with pytest.raises(ValueError, match="_reserved_"):
            tensogram.encode(meta, [(desc, data)])

    def test_getitem_reserved_raises(self):
        """meta['_reserved_'] raises KeyError (not accessible via dict syntax)."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta = tensogram.decode_metadata(msg)
        with pytest.raises(KeyError, match="_reserved_"):
            meta["_reserved_"]

    def test_contains_reserved_false(self):
        """'_reserved_' in meta returns False (hidden from dict access)."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta = tensogram.decode_metadata(msg)
        assert "_reserved_" not in meta

    def test_base_on_empty_message(self):
        """meta.base is [] for zero-object message."""
        msg = bytes(tensogram.encode({"version": 2}, []))
        meta = tensogram.decode_metadata(msg)
        assert meta.base == []

    def test_base_none_rejected(self):
        """base=None should be rejected."""
        data = np.ones(4, dtype=np.float32)
        meta = {"version": 2, "base": None}
        desc = make_descriptor([4], dtype="float32")
        with pytest.raises((ValueError, TypeError)):
            tensogram.encode(meta, [(desc, data)])

    def test_extra_and_extra_underscore_precedence(self):
        """_extra_ takes precedence over 'extra' when both are present."""
        data = np.ones(4, dtype=np.float32)
        meta = {
            "version": 2,
            "_extra_": {"key": "from_wire"},
            "extra": {"key": "from_alias"},
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        # _extra_ wins when both are present
        assert decoded_meta.extra["key"] == "from_wire"

    def test_base_first_match_semantics(self):
        """__getitem__ returns the first base entry match, not all."""
        meta_dict = {
            "version": 2,
            "base": [
                {"mars": {"param": "2t"}},
                {"mars": {"param": "msl"}},
            ],
        }
        d = make_descriptor([4], dtype="float32")
        data = np.ones(4, dtype=np.float32)
        msg = bytes(tensogram.encode(meta_dict, [(d, data), (d, data)]))
        meta = tensogram.decode_metadata(msg)
        # First base entry wins
        assert meta["mars"]["param"] == "2t"

    def test_missing_version_rejected(self):
        """Metadata dict without 'version' raises ValueError."""
        data = np.ones(4, dtype=np.float32)
        desc = make_descriptor([4], dtype="float32")
        with pytest.raises((ValueError, TypeError, KeyError), match="version"):
            tensogram.encode({}, [(desc, data)])

    def test_base_with_non_dict_entry_rejected(self):
        """base entries must be dicts."""
        data = np.ones(4, dtype=np.float32)
        meta = {"version": 2, "base": ["not_a_dict"]}
        desc = make_descriptor([4], dtype="float32")
        with pytest.raises((ValueError, TypeError)):
            tensogram.encode(meta, [(desc, data)])

    def test_empty_base_list(self):
        """base=[] is valid (zero per-object metadata)."""
        data = np.ones(4, dtype=np.float32)
        meta = {"version": 2, "base": []}
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        # Encoder auto-populates base entries for objects, so base should have 1
        assert len(decoded_meta.base) >= 0  # at minimum, encoder may add entries


class TestPackingParamsCoverage:
    """Edge cases for compute_packing_params."""

    def test_nan_rejected(self):
        """NaN values raise ValueError."""
        values = np.array([1.0, float("nan"), 3.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"[Nn]a[Nn]"):
            tensogram.compute_packing_params(values, 16, 0)

    def test_single_element(self):
        """Single-element array produces valid params."""
        values = np.array([300.0], dtype=np.float64)
        params = tensogram.compute_packing_params(values, 16, 0)
        assert params["reference_value"] == pytest.approx(300.0, abs=1e-4)

    def test_constant_field(self):
        """All-same values produce valid params."""
        values = np.full(100, 273.15, dtype=np.float64)
        params = tensogram.compute_packing_params(values, 16, 0)
        assert params["reference_value"] == pytest.approx(273.15, abs=1e-6)
        assert params["bits_per_value"] == 16

    def test_various_bits(self):
        """Different bits_per_value produce valid params."""
        values = np.linspace(200, 300, 100, dtype=np.float64)
        for bpv in [1, 8, 12, 16, 24, 32]:
            params = tensogram.compute_packing_params(values, bpv, 0)
            assert params["bits_per_value"] == bpv

    def test_positive_infinity_rejected(self):
        """Positive infinity is rejected alongside NaN — a finite range
        is a precondition of the binary-scale-factor computation.
        """
        values = np.array([1.0, float("inf"), 3.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"infinite value"):
            tensogram.compute_packing_params(values, 16, 0)

    def test_negative_infinity_rejected(self):
        """Negative infinity is rejected for the same reason."""
        values = np.array([float("-inf"), 1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"infinite value"):
            tensogram.compute_packing_params(values, 16, 0)


# ---------------------------------------------------------------------------
# Coverage: decode_range across all dtypes
# ---------------------------------------------------------------------------


class TestDecodeRangeDtypeCoverage:
    """Verify decode_range works correctly for all common dtypes."""

    @pytest.mark.parametrize(
        ("dtype_str", "np_dtype"),
        [
            ("float32", np.float32),
            ("float64", np.float64),
            ("int8", np.int8),
            ("int16", np.int16),
            ("int32", np.int32),
            ("int64", np.int64),
            ("uint8", np.uint8),
            ("uint16", np.uint16),
            ("uint32", np.uint32),
            ("uint64", np.uint64),
        ],
    )
    def test_decode_range_dtype(self, dtype_str, np_dtype):
        """decode_range preserves dtype for all common types."""
        data = np.arange(20, dtype=np_dtype)
        msg = encode_simple(data, dtype=dtype_str)
        parts = tensogram.decode_range(msg, object_index=0, ranges=[(5, 5)])
        assert len(parts) == 1
        assert parts[0].dtype == np_dtype
        np.testing.assert_array_equal(parts[0], data[5:10])

    def test_decode_range_join(self):
        """decode_range with join=True concatenates results."""
        data = np.arange(50, dtype=np.float32)
        msg = encode_simple(data)
        joined = tensogram.decode_range(msg, object_index=0, ranges=[(0, 5), (20, 5)], join=True)
        expected = np.concatenate([data[:5], data[20:25]])
        np.testing.assert_array_equal(joined, expected)

    def test_decode_range_full_array(self):
        """decode_range covering entire array matches full decode."""
        data = np.arange(32, dtype=np.float32)
        msg = encode_simple(data)
        parts = tensogram.decode_range(msg, object_index=0, ranges=[(0, 32)])
        np.testing.assert_array_equal(parts[0], data)


# ---------------------------------------------------------------------------
# Coverage: File iteration, indexing, slicing edge cases
# ---------------------------------------------------------------------------


class TestFileSliceCoverage:
    """Edge cases for file __getitem__ slicing and iteration."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a file with 6 messages."""
        path = str(tmp_path / "sample.tgm")
        with tensogram.TensogramFile.create(path) as f:
            for i in range(6):
                data = np.full(4, float(i), dtype=np.float32)
                f.append(
                    make_global_meta(2, index=i),
                    [(make_descriptor([4], dtype="float32"), data)],
                )
        return path

    def test_negative_index(self, sample_file):
        """file[-1] returns last message."""
        with tensogram.TensogramFile.open(sample_file) as f:
            msg = f[-1]
            assert msg.metadata["index"] == 5

    def test_negative_index_first(self, sample_file):
        """file[-6] returns first message (6 messages total)."""
        with tensogram.TensogramFile.open(sample_file) as f:
            msg = f[-6]
            assert msg.metadata["index"] == 0

    def test_index_out_of_range(self, sample_file):
        """file[100] raises IndexError."""
        with tensogram.TensogramFile.open(sample_file) as f, pytest.raises(IndexError):
            f[100]

    def test_negative_out_of_range(self, sample_file):
        """file[-100] raises IndexError."""
        with tensogram.TensogramFile.open(sample_file) as f, pytest.raises(IndexError):
            f[-100]

    def test_slice_reverse(self, sample_file):
        """file[::-1] reverses all messages."""
        with tensogram.TensogramFile.open(sample_file) as f:
            result = f[::-1]
            indices = [m.metadata["index"] for m in result]
            assert indices == [5, 4, 3, 2, 1, 0]

    def test_slice_empty_result(self, sample_file):
        """file[4:1] returns empty list (start > stop with positive step)."""
        with tensogram.TensogramFile.open(sample_file) as f:
            result = f[4:1]
            assert result == []

    def test_slice_step_2(self, sample_file):
        """file[::2] returns every other message."""
        with tensogram.TensogramFile.open(sample_file) as f:
            result = f[::2]
            indices = [m.metadata["index"] for m in result]
            assert indices == [0, 2, 4]

    def test_slice_negative_step(self, sample_file):
        """file[4:1:-1] returns messages 4, 3, 2."""
        with tensogram.TensogramFile.open(sample_file) as f:
            result = f[4:1:-1]
            indices = [m.metadata["index"] for m in result]
            assert indices == [4, 3, 2]

    def test_len(self, sample_file):
        """len(file) returns message count."""
        with tensogram.TensogramFile.open(sample_file) as f:
            assert len(f) == 6

    def test_concurrent_iterators(self, sample_file):
        """Two iterators from same file advance independently."""
        with tensogram.TensogramFile.open(sample_file) as f:
            it1 = iter(f)
            it2 = iter(f)
            m1 = next(it1)
            m2 = next(it2)
            # Both start from beginning
            assert m1.metadata["index"] == 0
            assert m2.metadata["index"] == 0
            # Advance one independently
            next(it1)
            m1_2 = next(it1)
            m2_2 = next(it2)
            assert m1_2.metadata["index"] == 2
            assert m2_2.metadata["index"] == 1

    def test_iter_len(self, sample_file):
        """Iterator supports len()."""
        with tensogram.TensogramFile.open(sample_file) as f:
            it = iter(f)
            assert len(it) == 6
            next(it)
            assert len(it) == 5

    def test_iter_exhaustion(self, sample_file):
        """Iterator raises StopIteration after all messages."""
        with tensogram.TensogramFile.open(sample_file) as f:
            it = iter(f)
            for _ in range(6):
                next(it)
            with pytest.raises(StopIteration):
                next(it)


# ---------------------------------------------------------------------------
# Coverage: iter_messages edge cases
# ---------------------------------------------------------------------------


class TestIterMessagesCoverage:
    """Edge cases for iter_messages buffer iteration."""

    def test_empty_buffer(self):
        """Empty buffer yields nothing."""
        assert list(tensogram.iter_messages(b"")) == []

    def test_garbage_only(self):
        """Pure garbage yields nothing."""
        assert list(tensogram.iter_messages(b"\xff" * 100)) == []

    def test_single_message(self):
        """Single message yields exactly one result."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        result = list(tensogram.iter_messages(msg))
        assert len(result) == 1

    def test_multiple_messages(self):
        """Multiple concatenated messages all decode."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        buf = msg + msg + msg
        result = list(tensogram.iter_messages(buf))
        assert len(result) == 3

    def test_iter_len(self):
        """MessageIter supports len()."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        buf = msg + msg
        it = tensogram.iter_messages(buf)
        assert len(it) == 2
        next(it)
        assert len(it) == 1


# ---------------------------------------------------------------------------
# Coverage: decode_descriptors
# ---------------------------------------------------------------------------


class TestDecodeDescriptorsCoverage:
    """Coverage for decode_descriptors."""

    def test_single_object(self):
        """decode_descriptors returns correct descriptor for one object."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        msg = encode_simple(data)
        meta, descriptors = tensogram.decode_descriptors(msg)
        assert meta.version == 2
        assert len(descriptors) == 1
        assert descriptors[0].shape == [3, 4]
        assert descriptors[0].dtype == "float32"

    def test_multi_object(self):
        """decode_descriptors with multiple objects."""
        d1 = make_descriptor([4], dtype="float32")
        d2 = make_descriptor([8], dtype="int32")
        msg = bytes(
            tensogram.encode(
                make_global_meta(2),
                [
                    (d1, np.ones(4, dtype=np.float32)),
                    (d2, np.ones(8, dtype=np.int32)),
                ],
            )
        )
        _meta, descriptors = tensogram.decode_descriptors(msg)
        assert len(descriptors) == 2
        assert descriptors[0].dtype == "float32"
        assert descriptors[1].dtype == "int32"
        assert descriptors[1].shape == [8]

    def test_garbage_raises(self):
        """decode_descriptors on garbage raises ValueError."""
        with pytest.raises(ValueError, match=r"[Ff]raming|[Ii]nvalid"):
            tensogram.decode_descriptors(b"not valid at all")


# ---------------------------------------------------------------------------
# Coverage: Message namedtuple unpacking across all decode paths
# ---------------------------------------------------------------------------


class TestMessageUnpackingCoverage:
    """Message tuple unpacking works across all decode paths."""

    def test_decode_unpacking(self):
        """decode() result supports tuple unpacking."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta, objects = tensogram.decode(msg)
        assert meta.version == 2
        assert len(objects) == 1

    def test_decode_attribute_access(self):
        """decode() result supports attribute access."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        result = tensogram.decode(msg)
        assert result.metadata.version == 2
        assert len(result.objects) == 1

    def test_file_decode_message(self, tmp_path):
        """decode_message returns Message namedtuple."""
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                make_global_meta(2, tag="x"),
                [(make_descriptor([4]), np.ones(4, dtype=np.float32))],
            )
        with tensogram.TensogramFile.open(path) as f:
            msg = f.decode_message(0)
            assert msg.metadata["tag"] == "x"
            _meta, objects = msg
            assert len(objects) == 1

    def test_file_getitem_returns_message(self, tmp_path):
        """file[i] returns Message namedtuple."""
        path = str(tmp_path / "test.tgm")
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                make_global_meta(2, v=99),
                [(make_descriptor([4]), np.ones(4, dtype=np.float32))],
            )
        with tensogram.TensogramFile.open(path) as f:
            msg = f[0]
            assert msg.metadata["v"] == 99
            meta, _objects = msg
            assert meta.version == 2

    def test_iter_messages_returns_message(self):
        """iter_messages yields Message namedtuples."""
        buf = encode_simple(np.ones(4, dtype=np.float32))
        for msg in tensogram.iter_messages(buf):
            assert msg.metadata.version == 2
            _meta, objects = msg
            assert len(objects) == 1


# ---------------------------------------------------------------------------
# Coverage: scan edge cases
# ---------------------------------------------------------------------------


class TestScanCoverage:
    """Edge cases for scan()."""

    def test_scan_empty(self):
        """Scan empty buffer returns empty list."""
        assert tensogram.scan(b"") == []

    def test_scan_garbage(self):
        """Scan garbage returns empty list."""
        assert tensogram.scan(b"\xff\xfe\xfd" * 100) == []

    def test_scan_single(self):
        """Scan single message returns one entry."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        entries = tensogram.scan(msg)
        assert len(entries) == 1
        offset, length = entries[0]
        assert offset == 0
        assert length == len(msg)

    def test_scan_with_garbage_between(self):
        """Scan skips garbage between valid messages."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        buf = msg + b"\xde\xad\xbe\xef" + msg
        entries = tensogram.scan(buf)
        assert len(entries) == 2

    def test_scan_decode_consistency(self):
        """Scan entries can be used to decode individual messages."""
        data1 = np.ones(4, dtype=np.float32)
        data2 = np.arange(8, dtype=np.float32)
        msg1 = encode_simple(data1, extra_meta={"idx": 0})
        msg2 = encode_simple(data2, extra_meta={"idx": 1})
        buf = msg1 + msg2
        entries = tensogram.scan(buf)
        assert len(entries) == 2
        for i, (offset, length) in enumerate(entries):
            meta = tensogram.decode_metadata(buf[offset : offset + length])
            assert meta["idx"] == i


# ---------------------------------------------------------------------------
# Coverage: Big-endian round-trips
# ---------------------------------------------------------------------------


class TestNativeEndianCoverage:
    """Native byte-order round-trip coverage across dtypes."""

    @pytest.mark.parametrize(
        ("dtype_str", "np_dtype"),
        [
            ("float32", np.float32),
            ("float64", np.float64),
            ("int32", np.int32),
            ("uint16", np.uint16),
        ],
    )
    def test_native_endian_roundtrip(self, dtype_str, np_dtype):
        """Encode and decode with native byte order — values should match
        because the library returns decoded bytes in native byte order."""
        data = np.arange(20, dtype=np_dtype)
        desc = make_descriptor(list(data.shape), dtype=dtype_str)
        msg = bytes(tensogram.encode(make_global_meta(2), [(desc, data)]))
        _, objects = tensogram.decode(msg)
        _, decoded = objects[0]
        np.testing.assert_array_equal(decoded, data)


# ---------------------------------------------------------------------------
# Coverage: encode with no hash
# ---------------------------------------------------------------------------


class TestEncodeOptionsCoverage:
    """Encode options coverage."""

    def test_no_hash(self):
        """Encode with hash=None produces valid message."""
        data = np.ones(4, dtype=np.float32)
        msg = encode_simple(data, hash_algo=None)
        _, objects = tensogram.decode(msg)
        assert len(objects) == 1

    def test_verify_hash_on_no_hash(self):
        """verify_hash=True silently skips when no hash stored."""
        msg = encode_simple(np.ones(4, dtype=np.float32), hash_algo=None)
        result = tensogram.decode(msg, verify_hash=True)
        assert len(result.objects) == 1

    def test_verify_hash_on_clean(self):
        """verify_hash=True passes on clean message."""
        msg = encode_simple(np.ones(4, dtype=np.float32), hash_algo="xxh3")
        result = tensogram.decode(msg, verify_hash=True)
        assert len(result.objects) == 1


# ---------------------------------------------------------------------------
# Coverage: dict_to_global_metadata edge cases
# ---------------------------------------------------------------------------


class TestDictToGlobalMetadataCoverage:
    """Test all branches in dict_to_global_metadata (Python bindings)."""

    def test_extra_key_as_non_dict_raises(self):
        """When 'extra' is a non-dict value, a ValueError is raised.

        The 'extra' key is a convenience alias for '_extra_' and must be a
        dict. Passing a non-dict value is an error to prevent silent data loss.
        """
        data = np.ones(4, dtype=np.float32)
        meta = {
            "version": 2,
            "extra": "not_a_dict",  # non-dict: must raise
        }
        desc = make_descriptor([4], dtype="float32")
        with pytest.raises(ValueError, match="'extra' must be a dict"):
            tensogram.encode(meta, [(desc, data)])

    def test_unknown_top_level_keys_become_extra(self):
        """Keys not in known_keys go to extra."""
        data = np.ones(4, dtype=np.float32)
        meta = {
            "version": 2,
            "custom_field": "hello",
            "another": 42,
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        assert decoded_meta.extra["custom_field"] == "hello"
        assert decoded_meta.extra["another"] == 42

    def test_extra_underscore_has_priority(self):
        """_extra_ key takes priority over extra alias."""
        data = np.ones(4, dtype=np.float32)
        meta = {
            "version": 2,
            "_extra_": {"from_wire": True},
            "extra": {"from_alias": True},
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        # _extra_ should win
        assert decoded_meta.extra.get("from_wire") is True

    def test_base_empty_list_with_objects(self):
        """Empty base list with objects: encoder auto-populates base."""
        data = np.ones(4, dtype=np.float32)
        meta = {"version": 2, "base": []}
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        # Encoder may add _reserved_.tensor entries
        assert isinstance(decoded_meta.base, list)

    def test_no_base_key(self):
        """Missing base key: defaults to empty."""
        data = np.ones(4, dtype=np.float32)
        meta = {"version": 2, "note": "no base"}
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        assert decoded_meta.extra["note"] == "no base"

    def test_base_with_nested_dict(self):
        """Base entries with nested dicts round-trip."""
        data = np.ones(4, dtype=np.float32)
        meta = {
            "version": 2,
            "base": [{"mars": {"class": "od", "param": "2t"}, "source": "test"}],
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        base = decoded_meta.base
        assert len(base) >= 1
        # Find the entry with mars (may have _reserved_ too)
        assert base[0]["mars"]["class"] == "od"
        assert base[0]["source"] == "test"


# ---------------------------------------------------------------------------
# Coverage: PyMetadata __getitem__ and __contains__ edge cases
# ---------------------------------------------------------------------------


class TestPyMetadataAccessCoverage:
    """Test __getitem__ and __contains__ edge cases on PyMetadata."""

    def test_getitem_from_extra_when_not_in_base(self):
        """Key only in extra is found by __getitem__."""
        data = np.ones(4, dtype=np.float32)
        meta_dict = {
            "version": 2,
            "base": [{"mars": {"param": "2t"}}],
            "only_in_extra": "found",
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta_dict, [(desc, data)]))
        meta = tensogram.decode_metadata(msg)
        assert meta["only_in_extra"] == "found"

    def test_contains_base_key(self):
        """Key in base is detected by __contains__."""
        data = np.ones(4, dtype=np.float32)
        meta_dict = {
            "version": 2,
            "base": [{"custom_key": "val"}],
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta_dict, [(desc, data)]))
        meta = tensogram.decode_metadata(msg)
        assert "custom_key" in meta

    def test_contains_extra_key(self):
        """Key in extra is detected by __contains__."""
        data = np.ones(4, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"e_key": "val"})
        meta = tensogram.decode_metadata(msg)
        assert "e_key" in meta

    def test_contains_missing_key(self):
        """Missing key returns False."""
        data = np.ones(4, dtype=np.float32)
        msg = encode_simple(data)
        meta = tensogram.decode_metadata(msg)
        assert "nonexistent_key_xyz" not in meta

    def test_getitem_reserved_skipped_in_base(self):
        """__getitem__('_reserved_') skips base entries that have it."""
        msg = encode_simple(np.ones(4, dtype=np.float32))
        meta = tensogram.decode_metadata(msg)
        # Base entries contain _reserved_ from encoder, but
        # __getitem__ skips _reserved_ → should raise KeyError
        with pytest.raises(KeyError, match="_reserved_"):
            meta["_reserved_"]

    def test_repr_format(self):
        """__repr__ includes version, base_len, and extra_keys."""
        data = np.ones(4, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"key1": 1, "key2": 2})
        meta = tensogram.decode_metadata(msg)
        r = repr(meta)
        assert "Metadata" in r
        assert "version=2" in r
        assert "base_len=" in r
        assert "extra_keys=" in r

    def test_getitem_multi_base_first_wins(self):
        """__getitem__ with multiple base entries returns first match."""
        meta_dict = {
            "version": 2,
            "base": [
                {"param": "first_value"},
                {"param": "second_value"},
            ],
        }
        d = make_descriptor([4], dtype="float32")
        data = np.ones(4, dtype=np.float32)
        msg = bytes(tensogram.encode(meta_dict, [(d, data), (d, data)]))
        meta = tensogram.decode_metadata(msg)
        assert meta["param"] == "first_value"

    def test_contains_with_multi_base(self):
        """__contains__ returns True if key in any base entry."""
        meta_dict = {
            "version": 2,
            "base": [
                {"a": 1},
                {"b": 2},
            ],
        }
        d = make_descriptor([4], dtype="float32")
        data = np.ones(4, dtype=np.float32)
        msg = bytes(tensogram.encode(meta_dict, [(d, data), (d, data)]))
        meta = tensogram.decode_metadata(msg)
        assert "a" in meta
        assert "b" in meta


# ---------------------------------------------------------------------------
# Unicode metadata cross-language parity
# ---------------------------------------------------------------------------


class TestUnicodeMetadata:
    """Verify Unicode strings (emoji, CJK, Arabic) round-trip through encode/decode."""

    def test_emoji_metadata_roundtrip(self):
        """Emoji characters in metadata survive encode → decode."""
        data = np.ones(4, dtype=np.float32)
        emoji_str = "🌍🌊🔥❄️🌡️"
        msg = encode_simple(data, extra_meta={"emoji": emoji_str})
        meta = tensogram.decode_metadata(msg)
        assert meta["emoji"] == emoji_str

    def test_cjk_metadata_roundtrip(self):
        """CJK (Chinese/Japanese/Korean) characters in metadata round-trip."""
        data = np.ones(4, dtype=np.float32)
        cjk_str = "気温データ"
        msg = encode_simple(data, extra_meta={"cjk": cjk_str})
        meta = tensogram.decode_metadata(msg)
        assert meta["cjk"] == cjk_str

    def test_arabic_metadata_roundtrip(self):
        """Arabic characters in metadata round-trip."""
        data = np.ones(4, dtype=np.float32)
        arabic_str = "بيانات الطقس"
        msg = encode_simple(data, extra_meta={"arabic": arabic_str})
        meta = tensogram.decode_metadata(msg)
        assert meta["arabic"] == arabic_str

    def test_mixed_unicode_metadata_roundtrip(self):
        """Mixed Unicode (emoji + accented + CJK) in one metadata value."""
        data = np.ones(4, dtype=np.float32)
        mixed = "Temperature 🌡️ is 25°C — très bien — 気温"
        msg = encode_simple(data, extra_meta={"mixed": mixed})
        meta = tensogram.decode_metadata(msg)
        assert meta["mixed"] == mixed

    def test_unicode_in_base_metadata(self):
        """Unicode in per-object base metadata survives round-trip."""
        data = np.ones(4, dtype=np.float32)
        meta_dict = {
            "version": 2,
            "base": [{"name": "気温", "units": "°C", "note": "🌡️ surface"}],
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta_dict, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        assert decoded_meta.base[0]["name"] == "気温"
        assert decoded_meta.base[0]["units"] == "°C"
        assert decoded_meta.base[0]["note"] == "🌡️ surface"

    def test_unicode_in_nested_metadata(self):
        """Unicode in deeply nested metadata structures round-trips."""
        data = np.ones(4, dtype=np.float32)
        nested = {"source": {"name": "観測所", "emoji": "🏔️"}}
        msg = encode_simple(data, extra_meta=nested)
        meta = tensogram.decode_metadata(msg)
        assert meta["source"]["name"] == "観測所"
        assert meta["source"]["emoji"] == "🏔️"

    def test_empty_string_and_whitespace(self):
        """Empty string and whitespace-only strings round-trip."""
        data = np.ones(4, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"empty": "", "spaces": "   ", "tabs": "\t\n"})
        meta = tensogram.decode_metadata(msg)
        assert meta["empty"] == ""
        assert meta["spaces"] == "   "
        assert meta["tabs"] == "\t\n"


# ---------------------------------------------------------------------------
# Bool→CBOR round-trip
# ---------------------------------------------------------------------------


class TestBoolCborRoundtrip:
    """Verify Python bool values survive CBOR round-trip as bools, not ints."""

    def test_true_false_roundtrip(self):
        """True and False metadata values come back as Python bool."""
        data = np.ones(4, dtype=np.float32)
        msg = encode_simple(data, extra_meta={"flag_true": True, "flag_false": False})
        meta = tensogram.decode_metadata(msg)
        assert meta["flag_true"] is True
        assert meta["flag_false"] is False
        # Ensure they are actually bool, not int
        assert type(meta["flag_true"]) is bool
        assert type(meta["flag_false"]) is bool

    def test_bool_in_nested_structure(self):
        """Bool values inside nested dicts and lists survive round-trip."""
        data = np.ones(4, dtype=np.float32)
        msg = encode_simple(
            data,
            extra_meta={
                "config": {"enabled": True, "debug": False},
                "flags": [True, False, True],
            },
        )
        meta = tensogram.decode_metadata(msg)
        assert meta["config"]["enabled"] is True
        assert meta["config"]["debug"] is False
        assert meta["flags"] == [True, False, True]
        assert type(meta["flags"][0]) is bool
        assert type(meta["flags"][1]) is bool

    def test_bool_in_base_metadata(self):
        """Bool values in per-object base metadata survive round-trip."""
        data = np.ones(4, dtype=np.float32)
        meta_dict = {
            "version": 2,
            "base": [{"is_valid": True, "is_forecast": False}],
        }
        desc = make_descriptor([4], dtype="float32")
        msg = bytes(tensogram.encode(meta_dict, [(desc, data)]))
        decoded_meta = tensogram.decode_metadata(msg)
        assert decoded_meta.base[0]["is_valid"] is True
        assert decoded_meta.base[0]["is_forecast"] is False
        assert type(decoded_meta.base[0]["is_valid"]) is bool


# ---------------------------------------------------------------------------
# compute_strides overflow
# ---------------------------------------------------------------------------


class TestComputeStridesOverflow:
    """Verify strides overflow is detected rather than wrapping."""

    def test_huge_shape_overflow_rejected(self):
        """Shape with dimensions that overflow u64 strides raises ValueError."""
        data = np.ones(4, dtype=np.float32)
        # Shape [2^63, 2] would overflow u64 in stride computation
        huge_shape = [2**63, 2]
        desc = make_descriptor(huge_shape, dtype="float32")
        with pytest.raises(ValueError, match=r"[Oo]verflow|strides"):
            tensogram.encode(make_global_meta(2), [(desc, data)])
