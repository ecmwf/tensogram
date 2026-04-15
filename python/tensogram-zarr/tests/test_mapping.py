# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for tensogram_zarr.mapping — dtype/metadata conversion."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from tensogram_zarr.mapping import (
    _json_safe_metadata,
    build_array_zarr_json,
    build_group_zarr_json,
    deserialize_zarr_json,
    numpy_dtype_to_tgm,
    parse_array_zarr_json,
    resolve_variable_name,
    serialize_zarr_json,
    tgm_dtype_to_numpy,
    tgm_dtype_to_zarr,
    zarr_dtype_to_tgm,
)

# ---------------------------------------------------------------------------
# Dtype mapping tests
# ---------------------------------------------------------------------------


class TestDtypeMapping:
    """Test bidirectional dtype conversions."""

    @pytest.mark.parametrize(
        ("tgm", "zarr_expected"),
        [
            ("float32", "float32"),
            ("float64", "float64"),
            ("int8", "int8"),
            ("int32", "int32"),
            ("uint8", "uint8"),
            ("uint64", "uint64"),
            ("complex128", "complex128"),
            ("bitmask", "uint8"),
        ],
    )
    def test_tgm_to_zarr(self, tgm: str, zarr_expected: str):
        assert tgm_dtype_to_zarr(tgm) == zarr_expected

    @pytest.mark.parametrize(
        ("zarr_dt", "tgm_expected"),
        [
            ("float32", "float32"),
            ("float64", "float64"),
            ("int16", "int16"),
            ("uint32", "uint32"),
        ],
    )
    def test_zarr_to_tgm(self, zarr_dt: str, tgm_expected: str):
        assert zarr_dtype_to_tgm(zarr_dt) == tgm_expected

    @pytest.mark.parametrize(
        ("tgm", "np_expected"),
        [
            ("float32", np.dtype("<f4")),
            ("float64", np.dtype("<f8")),
            ("int32", np.dtype("<i4")),
            ("uint8", np.dtype("|u1")),
        ],
    )
    def test_tgm_to_numpy(self, tgm: str, np_expected: np.dtype):
        assert tgm_dtype_to_numpy(tgm) == np_expected

    @pytest.mark.parametrize(
        ("np_dtype", "tgm_expected"),
        [
            (np.dtype("float32"), "float32"),
            (np.dtype("int64"), "int64"),
            (np.dtype("uint16"), "uint16"),
        ],
    )
    def test_numpy_to_tgm(self, np_dtype: np.dtype, tgm_expected: str):
        assert numpy_dtype_to_tgm(np_dtype) == tgm_expected

    def test_unknown_tgm_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported TGM dtype"):
            tgm_dtype_to_zarr("imaginary42")

    def test_unknown_zarr_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported Zarr dtype"):
            zarr_dtype_to_tgm("bfloat256")

    def test_unknown_numpy_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported numpy dtype"):
            numpy_dtype_to_tgm(np.dtype("U10"))


# ---------------------------------------------------------------------------
# Variable naming
# ---------------------------------------------------------------------------


class TestVariableNaming:
    """Test resolve_variable_name logic."""

    def test_explicit_key(self):
        meta = {"mars": {"param": "2t"}}
        assert resolve_variable_name(0, meta, variable_key="mars.param") == "2t"

    def test_auto_name_from_name_key(self):
        meta = {"name": "temperature"}
        assert resolve_variable_name(0, meta) == "temperature"

    def test_auto_name_from_mars_param(self):
        meta = {"mars": {"param": "sp"}}
        assert resolve_variable_name(0, meta) == "sp"

    def test_fallback_to_index(self):
        assert resolve_variable_name(3, {}) == "object_3"

    def test_common_meta_not_searched(self):
        """common_meta (extra) is NOT consulted for variable naming.

        Variable names come exclusively from per-object metadata (base[i])
        to avoid all objects in a message sharing the same name.
        """
        assert resolve_variable_name(0, {}, {"mars": {"param": "global_var"}}) == "object_0"

    def test_per_object_only(self):
        per = {"mars": {"param": "local"}}
        com = {"mars": {"param": "global"}}
        assert resolve_variable_name(0, per, com) == "local"


# ---------------------------------------------------------------------------
# Zarr JSON synthesis (read path)
# ---------------------------------------------------------------------------


class TestGroupZarrJson:
    """Test group metadata synthesis."""

    def test_basic_structure(self):
        class FakeMeta:
            version: ClassVar[int] = 2
            extra: ClassVar[dict[str, object]] = {"mars": {"class": "od"}}

        result = build_group_zarr_json(FakeMeta(), ["temp", "pressure"])
        assert result["zarr_format"] == 3
        assert result["node_type"] == "group"
        assert result["attributes"]["mars"] == {"class": "od"}
        assert result["attributes"]["_tensogram_version"] == 2
        assert result["attributes"]["_tensogram_variables"] == ["temp", "pressure"]


class TestArrayZarrJson:
    """Test array metadata synthesis from descriptors."""

    def test_basic_float32(self):
        class FakeDesc:
            shape: ClassVar[list[object]] = [6, 10]
            dtype: ClassVar[str] = "float32"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[None] = None
            params: ClassVar[dict[str, object]] = {}

        result = build_array_zarr_json(FakeDesc())
        assert result["zarr_format"] == 3
        assert result["node_type"] == "array"
        assert result["shape"] == [6, 10]
        assert result["data_type"] == "float32"
        assert result["chunk_grid"]["configuration"]["chunk_shape"] == [6, 10]

    def test_per_object_metadata_in_attrs(self):
        class FakeDesc:
            shape: ClassVar[list[object]] = [3]
            dtype: ClassVar[str] = "int32"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[dict[str, object]] = {"type": "xxh3", "value": "abc"}
            params: ClassVar[dict[str, object]] = {}

        per_obj = {"mars": {"param": "2t"}}
        result = build_array_zarr_json(FakeDesc(), per_obj)
        assert result["attributes"]["mars"] == {"param": "2t"}
        assert result["attributes"]["_tensogram_hash"] == {"type": "xxh3", "value": "abc"}


# ---------------------------------------------------------------------------
# Zarr JSON parsing (write path)
# ---------------------------------------------------------------------------


class TestParseArrayZarrJson:
    """Test parsing zarr.json back to TGM-relevant fields."""

    def test_round_trip(self):
        class FakeDesc:
            shape: ClassVar[list[object]] = [4, 8]
            dtype: ClassVar[str] = "float64"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[None] = None
            params: ClassVar[dict[str, object]] = {}

        zarr_meta = build_array_zarr_json(FakeDesc())
        parsed = parse_array_zarr_json(zarr_meta)
        assert parsed["shape"] == [4, 8]
        assert parsed["dtype"] == "float64"
        assert parsed["byte_order"] == "little"
        assert parsed["encoding"] == "none"


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test serialize/deserialize round-trip."""

    def test_round_trip(self):
        original = {"zarr_format": 3, "node_type": "group", "attributes": {"key": "val"}}
        data = serialize_zarr_json(original)
        assert isinstance(data, bytes)
        restored = deserialize_zarr_json(data)
        assert restored == original


# ---------------------------------------------------------------------------
# JSON RFC 8259 compliance — NaN / Infinity handling
# ---------------------------------------------------------------------------


class TestJsonSafeMetadata:
    """Test that non-finite floats are converted to Zarr v3 string sentinels."""

    def test_nan_becomes_string(self):
        result = _json_safe_metadata(float("nan"))
        assert result == "NaN"

    def test_positive_inf_becomes_string(self):
        result = _json_safe_metadata(float("inf"))
        assert result == "Infinity"

    def test_negative_inf_becomes_string(self):
        result = _json_safe_metadata(float("-inf"))
        assert result == "-Infinity"

    def test_normal_float_unchanged(self):
        assert _json_safe_metadata(3.14) == 3.14

    def test_nested_dict_nan_converted(self):
        obj = {"fill_value": float("nan"), "other": 42}
        result = _json_safe_metadata(obj)
        assert result == {"fill_value": "NaN", "other": 42}

    def test_nested_list_nan_converted(self):
        obj = [float("nan"), 1.0, float("inf")]
        result = _json_safe_metadata(obj)
        assert result == ["NaN", 1.0, "Infinity"]

    def test_deeply_nested(self):
        obj = {"a": {"b": [float("-inf"), {"c": float("nan")}]}}
        result = _json_safe_metadata(obj)
        assert result == {"a": {"b": ["-Infinity", {"c": "NaN"}]}}

    def test_non_float_passthrough(self):
        assert _json_safe_metadata("hello") == "hello"
        assert _json_safe_metadata(42) == 42
        assert _json_safe_metadata(None) is None
        assert _json_safe_metadata(True) is True


class TestSerializeNanCompliance:
    """Verify serialize_zarr_json produces valid RFC 8259 JSON with NaN."""

    def test_nan_fill_value_valid_json(self):
        """Float NaN fill_value should not produce bare NaN token."""
        import json

        meta = {"fill_value": float("nan"), "zarr_format": 3}
        data = serialize_zarr_json(meta)
        # Must be valid JSON (json.loads with strict mode rejects bare NaN)
        parsed = json.loads(data)
        assert parsed["fill_value"] == "NaN"

    def test_array_zarr_json_float_fill_value(self):
        """build_array_zarr_json produces NaN fill → serialize should be valid."""
        import json

        class Desc:
            shape: ClassVar[list[object]] = [3]
            dtype: ClassVar[str] = "float32"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[None] = None
            params: ClassVar[dict[str, object]] = {}

        zarr_meta = build_array_zarr_json(Desc())
        data = serialize_zarr_json(zarr_meta)
        # Must be valid JSON
        parsed = json.loads(data)
        assert parsed["fill_value"] == "NaN"
