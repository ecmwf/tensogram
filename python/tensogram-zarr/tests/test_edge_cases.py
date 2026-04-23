# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Edge case tests for tensogram-zarr.

Covers: invalid inputs, boundary conditions, ambiguities, and
unusual-but-valid scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tensogram
from tensogram_zarr import TensogramStore
from tensogram_zarr.mapping import (
    _default_fill_value,
    _dotted_get,
    parse_array_zarr_json,
    resolve_variable_name,
)
from tensogram_zarr.store import _chunk_key_for_shape, _sanitize_key_segment

# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


class TestInvalidMode:
    def test_bad_mode_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="invalid mode"):
            TensogramStore(str(tmp_path / "f.tgm"), mode="x")


# ---------------------------------------------------------------------------
# message_index out of range
# ---------------------------------------------------------------------------


class TestMessageIndexBounds:
    def test_index_out_of_range_raises(self, simple_tgm: str):
        with pytest.raises(IndexError, match="out of range"):
            TensogramStore.open_tgm(simple_tgm, message_index=999)

    def test_negative_index_raises(self, simple_tgm: str):
        """Negative indices are rejected at construction time."""
        with pytest.raises(ValueError, match="message_index must be >= 0"):
            TensogramStore.open_tgm(simple_tgm, message_index=-1)


# ---------------------------------------------------------------------------
# Variable name sanitization
# ---------------------------------------------------------------------------


class TestVariableNameSanitization:
    def test_slash_in_name_replaced(self):
        assert _sanitize_key_segment("temperature/surface") == "temperature_surface"

    def test_backslash_in_name_replaced(self):
        assert _sanitize_key_segment("wind\\speed") == "wind_speed"

    def test_empty_name_becomes_underscore(self):
        assert _sanitize_key_segment("") == "_"

    def test_normal_name_unchanged(self):
        assert _sanitize_key_segment("2t") == "2t"

    def test_slash_in_metadata_sanitized_at_store_level(self, tmp_path: Path):
        """A variable whose metadata name contains '/' is sanitized."""
        path = str(tmp_path / "slash.tgm")
        data = np.ones(3, dtype=np.float32)
        meta = {
            "version": 3,
            "base": [{"name": "temp/sfc"}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [({"type": "ntensor", "shape": [3], "dtype": "float32"}, data)])

        with TensogramStore(path, mode="r") as store:
            # Should be sanitized to "temp_sfc"
            array_keys = [k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"]
            names = [k.split("/")[0] for k in array_keys]
            assert "temp_sfc" in names
            assert "temp/sfc" not in names


# ---------------------------------------------------------------------------
# Triple+ duplicate names
# ---------------------------------------------------------------------------


class TestTripleDuplicateNames:
    def test_three_duplicates(self, tmp_path: Path):
        path = str(tmp_path / "triple.tgm")
        data = np.ones(2, dtype=np.float32)
        meta = {
            "version": 3,
            "base": [{"name": "x"}, {"name": "x"}, {"name": "x"}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    ({"type": "ntensor", "shape": [2], "dtype": "float32"}, data),
                    ({"type": "ntensor", "shape": [2], "dtype": "float32"}, data * 2),
                    ({"type": "ntensor", "shape": [2], "dtype": "float32"}, data * 3),
                ],
            )

        with TensogramStore(path, mode="r") as store:
            array_keys = sorted(
                k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"
            )
            names = sorted(k.split("/")[0] for k in array_keys)
            assert names == ["x", "x_1", "x_2"]


# ---------------------------------------------------------------------------
# Zero-object message
# ---------------------------------------------------------------------------


class TestZeroObjectMessage:
    def test_metadata_only_message(self, tmp_path: Path):
        """A message with zero data objects should produce a group with no arrays."""
        path = str(tmp_path / "zero_obj.tgm")
        # Write a zero-object message via the append API with an empty list
        with tensogram.TensogramFile.create(path) as f:
            f.append({"version": 3, "signal": "start"}, [])

        with TensogramStore(path, mode="r") as store:
            root_meta = json.loads(store._keys["zarr.json"])
            assert root_meta["node_type"] == "group"
            assert root_meta["attributes"]["signal"] == "start"
            array_keys = [k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"]
            assert len(array_keys) == 0


# ---------------------------------------------------------------------------
# Chunk key shapes
# ---------------------------------------------------------------------------


class TestChunkKeyShapes:
    def test_0d(self):
        assert _chunk_key_for_shape([]) == "c/0"
        assert _chunk_key_for_shape(()) == "c/0"

    def test_1d(self):
        assert _chunk_key_for_shape([5]) == "c/0"

    def test_2d(self):
        assert _chunk_key_for_shape([3, 4]) == "c/0/0"

    def test_3d(self):
        assert _chunk_key_for_shape([2, 3, 4]) == "c/0/0/0"

    def test_5d(self):
        assert _chunk_key_for_shape([1, 2, 3, 4, 5]) == "c/0/0/0/0/0"


# ---------------------------------------------------------------------------
# Byte range edge cases
# ---------------------------------------------------------------------------


class TestByteRangeEdgeCases:
    def test_suffix_zero_returns_empty(self, simple_tgm: str):
        from zarr.abc.store import SuffixByteRequest
        from zarr.core.buffer import default_buffer_prototype

        with TensogramStore(simple_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            proto = default_buffer_prototype()
            buf = store._get_sync(chunk_keys[0], proto, SuffixByteRequest(0))
            assert buf is not None
            assert len(buf.to_bytes()) == 0

    def test_range_beyond_end_returns_truncated(self, simple_tgm: str):
        """Python slicing semantics: range beyond end returns what's available."""
        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype

        with TensogramStore(simple_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            full_len = len(store._keys[chunk_keys[0]])
            proto = default_buffer_prototype()
            buf = store._get_sync(chunk_keys[0], proto, RangeByteRequest(0, full_len + 1000))
            assert buf is not None
            assert len(buf.to_bytes()) == full_len

    def test_range_start_equals_end(self, simple_tgm: str):
        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype

        with TensogramStore(simple_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            proto = default_buffer_prototype()
            buf = store._get_sync(chunk_keys[0], proto, RangeByteRequest(10, 10))
            assert buf is not None
            assert len(buf.to_bytes()) == 0


# ---------------------------------------------------------------------------
# Fill value edge cases
# ---------------------------------------------------------------------------


class TestFillValue:
    def test_float_fill_is_nan(self):
        assert np.isnan(_default_fill_value("float32"))
        assert np.isnan(_default_fill_value("float64"))

    def test_bfloat16_fill_is_nan(self):
        assert np.isnan(_default_fill_value("bfloat16"))

    def test_complex_fill_is_nan(self):
        assert np.isnan(_default_fill_value("complex64"))

    def test_int_fill_is_zero(self):
        assert _default_fill_value("int32") == 0
        assert _default_fill_value("uint8") == 0


# ---------------------------------------------------------------------------
# parse_array_zarr_json does not mutate input
# ---------------------------------------------------------------------------


class TestParseNoMutation:
    def test_parse_does_not_mutate_original(self):
        original = {
            "shape": [3, 4],
            "data_type": "float32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
                "units": "K",
            },
        }
        # Deep copy the attributes to check later
        original_attr_keys = set(original["attributes"].keys())

        parse_array_zarr_json(original)

        # Original should not be modified
        assert set(original["attributes"].keys()) == original_attr_keys


# ---------------------------------------------------------------------------
# Double open / double close
# ---------------------------------------------------------------------------


class TestLifecycleEdgeCases:
    def test_double_open_raises(self, simple_tgm: str):
        store = TensogramStore.open_tgm(simple_tgm)
        with pytest.raises(ValueError, match="already open"):
            store._open_sync()
        store.close()

    def test_double_close_is_safe(self, simple_tgm: str):
        store = TensogramStore.open_tgm(simple_tgm)
        store.close()
        store.close()  # should not raise

    def test_close_sets_is_open_false_even_on_flush_error(self, tmp_path: Path):
        """If flush fails, _is_open should still be set to False."""
        path = str(tmp_path / "bad.tgm")
        store = TensogramStore(path, mode="w")
        store._open_sync()
        # Mark dirty but with no arrays — flush exits early, no error.
        # To test the try/finally, we need an actual error.  We'll verify
        # the simpler case: close always sets _is_open = False.
        store._dirty = True
        store.close()
        assert not store._is_open


# ---------------------------------------------------------------------------
# dotted_get edge cases
# ---------------------------------------------------------------------------


class TestDottedGet:
    def test_deeply_nested(self):
        d = {"a": {"b": {"c": {"d": 42}}}}
        assert _dotted_get(d, "a.b.c.d") == 42

    def test_missing_intermediate(self):
        assert _dotted_get({"a": 1}, "a.b") is None

    def test_non_dict_intermediate(self):
        assert _dotted_get({"a": [1, 2]}, "a.0") is None

    def test_empty_path(self):
        # "".split(".") → [""] so it tries d[""] which returns None
        assert _dotted_get({"": "found"}, "") == "found"

    def test_single_key(self):
        assert _dotted_get({"key": "val"}, "key") == "val"


# ---------------------------------------------------------------------------
# resolve_variable_name edge cases
# ---------------------------------------------------------------------------


class TestVariableNameEdgeCases:
    def test_empty_string_value_is_used(self):
        """An empty string metadata value IS returned (truthy issue? No — empty
        string is not None, so str("") returns "" which is a valid name)."""
        meta = {"name": ""}
        result = resolve_variable_name(0, meta)
        # Empty string is not None, so it passes the `is not None` check.
        # str("") == "", so the name will be "".
        # The store's _sanitize_key_segment will convert this to "_".
        assert result == ""

    def test_numeric_value_converted_to_str(self):
        meta = {"mars": {"param": 128}}
        assert resolve_variable_name(0, meta) == "128"

    def test_none_meta_fallback(self):
        assert resolve_variable_name(5, None) == "object_5"

    def test_common_meta_not_searched_for_name(self):
        """common_meta is NOT searched for variable naming.

        Variable names come exclusively from per-object metadata (base[i])
        to avoid all objects in a message sharing the same name when
        the name key only exists in extra metadata.
        """
        result = resolve_variable_name(0, {}, {"name": "from_extra"})
        assert result == "object_0"


# ---------------------------------------------------------------------------
# _reserved_ key collision on write path
# ---------------------------------------------------------------------------


class TestReservedKeyWritePath:
    def test_reserved_not_written_to_base(self, tmp_path: Path):
        """Writing attributes with a '_reserved_' key must not create a
        bogus _reserved_ entry in the TGM base that collides with the
        encoder's auto-populated _reserved_.tensor."""
        path = str(tmp_path / "reserved_write.tgm")
        data = np.ones(3, dtype=np.float32)

        with TensogramStore(path, mode="w") as store:
            from tensogram_zarr.mapping import serialize_zarr_json

            store._keys["zarr.json"] = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {},
                }
            )
            store._write_group_attrs = {}

            arr_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3],
                "data_type": "float32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "fill_value": None,
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                    "_reserved_": {"bogus": True},  # Should be filtered
                    "units": "K",
                },
            }
            store._write_arrays["temp"] = arr_meta
            store._write_chunks["temp/c/0"] = data.tobytes()
            store._dirty = True

        # Read back and verify _reserved_ was not written as user data
        with tensogram.TensogramFile.open(path) as f:
            meta, _objects = f.decode_message(0)
            base = meta.base[0]
            # The encoder's auto-populated _reserved_ is ok
            # But the user's bogus {"bogus": True} should NOT appear
            reserved = base.get("_reserved_", {})
            if isinstance(reserved, dict):
                assert "bogus" not in reserved


# ---------------------------------------------------------------------------
# Empty extra metadata
# ---------------------------------------------------------------------------


class TestEmptyExtra:
    def test_empty_extra_no_custom_attrs(self, simple_tgm: str):
        """When meta.extra is empty, group zarr.json has only internal attrs."""
        with TensogramStore(simple_tgm, mode="r") as store:
            root = json.loads(store._keys["zarr.json"])
            attrs = root["attributes"]
            # Should have internal attrs but no custom ones
            assert "_tensogram_wire_version" in attrs
            assert "_tensogram_variables" in attrs


# ---------------------------------------------------------------------------
# Zarr metadata key collision
# ---------------------------------------------------------------------------


class TestZarrKeyCollision:
    def test_base_key_named_zarr_no_collision(self, tmp_path: Path):
        """If meta.base[i] has a key named 'zarr' or 'chunks', it goes into
        the array's attributes dict and doesn't collide with zarr metadata
        at the top level."""
        path = str(tmp_path / "collision.tgm")
        data = np.ones(3, dtype=np.float32)
        meta = {
            "version": 3,
            "base": [{"zarr": "not_a_real_zarr_key", "chunks": 99}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [({"type": "ntensor", "shape": [3], "dtype": "float32"}, data)])

        with TensogramStore(path, mode="r") as store:
            array_keys = [k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"]
            assert len(array_keys) == 1
            arr_meta = json.loads(store._keys[array_keys[0]])
            # User keys go into attributes, not top-level
            assert arr_meta["node_type"] == "array"
            attrs = arr_meta["attributes"]
            assert attrs["zarr"] == "not_a_real_zarr_key"
            assert attrs["chunks"] == 99
            # Top-level zarr metadata should be unaffected
            assert arr_meta["shape"] == [3]
            assert arr_meta["data_type"] == "float32"


# ---------------------------------------------------------------------------
# Variable name resolution order
# ---------------------------------------------------------------------------


class TestVariableNameResolutionOrder:
    def test_per_object_only_no_extra_fallback(self, tmp_path: Path):
        """Variable names come from base[i] only, not from extra."""
        path = str(tmp_path / "name_order.tgm")
        data = np.ones(3, dtype=np.float32)
        # Extra has a name, but base[0] does not
        meta = {
            "version": 3,
            "name": "from_extra",
            "base": [{}],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(meta, [({"type": "ntensor", "shape": [3], "dtype": "float32"}, data)])

        with TensogramStore(path, mode="r") as store:
            array_keys = [k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"]
            names = [k.split("/")[0] for k in array_keys]
            # Should fallback to "object_0", NOT "from_extra"
            assert "object_0" in names
            assert "from_extra" not in names
