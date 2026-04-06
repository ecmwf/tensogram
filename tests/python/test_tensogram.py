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
        """When hash is used, descriptor should expose it."""
        data = np.ones(10, dtype=np.float32)
        msg = encode_simple(data, hash_algo="xxh3")
        _, objects = tensogram.decode(msg)
        desc, _ = objects[0]
        h = desc.hash
        assert h is not None
        assert h["type"] == "xxh3"
        assert isinstance(h["value"], str)
        assert len(h["value"]) > 0

    def test_hash_absent(self):
        """Without hash, descriptor.hash should be None."""
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
        """Corrupted payload should fail verify_hash=True.

        Corruption may manifest as FramingError (ValueError) if the
        corruption breaks the frame structure, or RuntimeError if only
        the hash mismatches. Both are acceptable detection.
        """
        data = np.arange(100, dtype=np.float32)
        msg = encode_simple(data, hash_algo="xxh3")
        corrupted = bytearray(msg)
        # Corrupt a byte in the payload area (past header, before terminator)
        mid = len(corrupted) // 2
        corrupted[mid] ^= 0xFF
        with pytest.raises((ValueError, RuntimeError)):
            tensogram.decode(bytes(corrupted), verify_hash=True)

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

    def test_big_endian_roundtrip(self):
        """Encode with byte_order='big' and verify round-trip."""
        data = np.arange(10, dtype=np.float32)
        meta = make_global_meta(2)
        desc = make_descriptor([10], dtype="float32", byte_order="big")
        msg = bytes(tensogram.encode(meta, [(desc, data)]))
        _, objects = tensogram.decode(msg)
        decoded_desc, arr = objects[0]
        np.testing.assert_array_equal(arr, data)
        assert decoded_desc.byte_order == "big"

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

    def test_encode_decode_preserves_wire_bytes(self):
        """Encoding the same data twice produces identical wire bytes."""
        data = np.arange(20, dtype=np.float32)
        msg1 = encode_simple(data, hash_algo=None, extra_meta={"key": "val"})
        msg2 = encode_simple(data, hash_algo=None, extra_meta={"key": "val"})
        assert msg1 == msg2

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
                    make_global_meta(2, index=i), [(make_descriptor([8], dtype="float32"), data)]
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
                    make_global_meta(2, index=i), [(make_descriptor([8], dtype="float32"), data)]
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
                    make_global_meta(2, index=i), [(make_descriptor([4], dtype="float32"), data)]
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
                    make_global_meta(2, index=i), [(make_descriptor([4], dtype="float32"), data)]
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
                    make_global_meta(2, index=i), [(make_descriptor([4], dtype="float32"), data)]
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
                make_global_meta(2, tag="test"), [(make_descriptor([4], dtype="float32"), data)]
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
