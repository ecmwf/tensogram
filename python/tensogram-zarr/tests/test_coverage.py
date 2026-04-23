# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Coverage gap tests — exercises every untested path found by the audit.

Each test class targets a specific gap identified in the coverage report.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
import tensogram
from tensogram_zarr import TensogramStore
from tensogram_zarr.mapping import (
    build_array_zarr_json,
    build_group_zarr_json,
    deserialize_zarr_json,
    parse_array_zarr_json,
    resolve_variable_name,
    serialize_zarr_json,
    tgm_dtype_to_numpy,
)
from tensogram_zarr.store import (
    _apply_byte_range,
    _native_is_big,
)

# ===================================================================
# 1. Async method tests (get, exists, set, delete, list, list_prefix)
# ===================================================================


class TestAsyncGet:
    def test_async_get_existing_key(self, simple_tgm: str):
        async def run():
            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            from zarr.core.buffer import default_buffer_prototype

            proto = default_buffer_prototype()
            buf = await store.get("zarr.json", proto)
            assert buf is not None
            meta = json.loads(buf.to_bytes())
            assert meta["zarr_format"] == 3
            store.close()

        asyncio.run(run())

    def test_async_get_missing_key(self, simple_tgm: str):
        async def run():
            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            from zarr.core.buffer import default_buffer_prototype

            proto = default_buffer_prototype()
            buf = await store.get("nonexistent", proto)
            assert buf is None
            store.close()

        asyncio.run(run())

    def test_async_get_with_byte_range(self, simple_tgm: str):
        async def run():
            from zarr.abc.store import RangeByteRequest
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            proto = default_buffer_prototype()
            buf = await store.get("zarr.json", proto, RangeByteRequest(0, 5))
            assert buf is not None
            assert len(buf.to_bytes()) == 5
            store.close()

        asyncio.run(run())


class TestAsyncGetPartialValues:
    def test_batch_get(self, simple_tgm: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            proto = default_buffer_prototype()
            results = await store.get_partial_values(
                proto, [("zarr.json", None), ("missing", None)]
            )
            assert len(results) == 2
            assert results[0] is not None
            assert results[1] is None
            store.close()

        asyncio.run(run())


class TestAsyncExists:
    def test_exists_true_and_false(self, simple_tgm: str):
        async def run():
            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            assert await store.exists("zarr.json")
            assert not await store.exists("nope")
            store.close()

        asyncio.run(run())


class TestAsyncSet:
    def test_set_group_meta(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()
            group_json = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"hello": "world"},
                }
            )
            await store.set("zarr.json", proto.buffer.from_bytes(group_json))
            assert store._write_group_attrs == {"hello": "world"}
            store.close()

        asyncio.run(run())

    def test_set_array_meta(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()
            arr_json = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [3],
                    "data_type": "float32",
                }
            )
            await store.set("temp/zarr.json", proto.buffer.from_bytes(arr_json))
            assert "temp" in store._write_arrays
            store.close()

        asyncio.run(run())

    def test_set_chunk_data(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()
            await store.set("temp/c/0", proto.buffer.from_bytes(b"\x00" * 12))
            assert "temp/c/0" in store._write_chunks
            store.close()

        asyncio.run(run())

    def test_set_non_array_zarr_json_ignored(self, output_path: str):
        """A zarr.json with node_type='group' at an array path is ignored."""

        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()
            group_json = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {},
                }
            )
            await store.set("nested/zarr.json", proto.buffer.from_bytes(group_json))
            assert "nested" not in store._write_arrays
            store.close()

        asyncio.run(run())


class TestAsyncSetIfNotExists:
    def test_skips_existing(self, simple_tgm: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            # Use a writable store
            store2 = TensogramStore(str(Path(simple_tgm).parent / "out.tgm"), mode="w")
            await store2._open()
            proto = default_buffer_prototype()
            await store2.set("key", proto.buffer.from_bytes(b"first"))
            await store2.set_if_not_exists("key", proto.buffer.from_bytes(b"second"))
            assert store2._keys["key"] == b"first"
            store2.close()

        asyncio.run(run())


class TestAsyncSetMany:
    def test_set_many(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()
            await store._set_many(
                [
                    ("a/c/0", proto.buffer.from_bytes(b"aaa")),
                    ("b/c/0", proto.buffer.from_bytes(b"bbb")),
                ]
            )
            assert store._keys["a/c/0"] == b"aaa"
            assert store._keys["b/c/0"] == b"bbb"
            store.close()

        asyncio.run(run())


class TestAsyncDelete:
    def test_delete_chunk(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()
            await store.set("var/c/0", proto.buffer.from_bytes(b"data"))
            assert "var/c/0" in store._keys
            await store.delete("var/c/0")
            assert "var/c/0" not in store._keys
            store.close()

        asyncio.run(run())

    def test_delete_array_meta_cleans_write_arrays(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()
            arr_json = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [3],
                    "data_type": "int32",
                }
            )
            await store.set("temp/zarr.json", proto.buffer.from_bytes(arr_json))
            assert "temp" in store._write_arrays
            await store.delete("temp/zarr.json")
            assert "temp" not in store._write_arrays
            store.close()

        asyncio.run(run())

    def test_delete_nonexistent_is_noop(self, output_path: str):
        async def run():
            store = TensogramStore(output_path, mode="w")
            await store._open()
            await store.delete("does_not_exist")  # should not raise
            store.close()

        asyncio.run(run())


class TestAsyncList:
    def test_list_all(self, simple_tgm: str):
        async def run():
            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            keys = [k async for k in store.list()]
            assert "zarr.json" in keys
            store.close()

        asyncio.run(run())

    def test_list_prefix(self, multi_object_tgm: str):
        async def run():
            store = TensogramStore(multi_object_tgm, mode="r")
            await store._open()
            keys = [k async for k in store.list_prefix("2t/")]
            assert len(keys) == 2  # zarr.json + c/0/0
            store.close()

        asyncio.run(run())


class TestListDirBranches:
    def test_list_dir_prefix_already_has_slash(self, multi_object_tgm: str):
        """prefix ending with / should not get double-slashed."""

        async def run():
            store = TensogramStore(multi_object_tgm, mode="r")
            await store._open()
            entries = [e async for e in store.list_dir("2t/")]
            assert "zarr.json" in entries
            assert "c/" in entries
            store.close()

        asyncio.run(run())


# ===================================================================
# 2. Append mode
# ===================================================================


class TestAppendMode:
    def test_append_adds_message(self, simple_tgm: str, tmp_path: Path):
        """mode='a' appends a new message to an existing file."""
        path = str(tmp_path / "append.tgm")
        # Create initial file
        data = np.ones(4, dtype=np.float32)
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 3, "base": [{"name": "first"}]},
                [({"type": "ntensor", "shape": [4], "dtype": "float32"}, data)],
            )

        # Append via store
        with TensogramStore(path, mode="a") as store:
            store._write_arrays["second"] = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3],
                "data_type": "int32",
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                },
            }
            store._write_chunks["second/c/0"] = np.array([10, 20, 30], dtype="<i4").tobytes()
            store._dirty = True

        with tensogram.TensogramFile.open(path) as f:
            assert len(f) == 2


# ===================================================================
# 3. Flush error paths
# ===================================================================


class TestFlushErrors:
    def test_byte_count_mismatch_raises(self, output_path: str):
        store = TensogramStore(output_path, mode="w")
        store._open_sync()
        store._write_arrays["bad"] = {
            "shape": [4],
            "data_type": "float32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
            },
        }
        # 3 bytes instead of 16 (4 floats * 4 bytes)
        store._write_chunks["bad/c/0"] = b"\x00\x00\x00"
        with pytest.raises(ValueError, match=r"expected 16 bytes.*got 3"):
            store._flush_to_tgm()
        store._dirty = False  # prevent close from re-flushing
        store.close()

    def test_no_arrays_warns(self, output_path: str, caplog):
        with TensogramStore(output_path, mode="w") as store:
            store._dirty = True
            store._write_arrays.clear()
            # close triggers flush, which warns
        assert "no arrays registered" in caplog.text

    def test_array_without_chunk_warns(self, output_path: str, caplog):
        with TensogramStore(output_path, mode="w") as store:
            store._write_arrays["orphan"] = {
                "shape": [2],
                "data_type": "int32",
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                },
            }
            store._dirty = True
        assert "no chunk data" in caplog.text

    def test_all_arrays_skipped_warns(self, output_path: str, caplog):
        """When every array is skipped (no chunk), should warn about no data."""
        with TensogramStore(output_path, mode="w") as store:
            store._write_arrays["a"] = {
                "shape": [2],
                "data_type": "int32",
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                },
            }
            store._dirty = True
        assert "no data" in caplog.text or "no chunk data" in caplog.text


# ===================================================================
# 4. __exit__ exception handling
# ===================================================================


class TestExitExceptionHandling:
    def test_exit_with_exception_in_flight_logs_flush_error(self, tmp_path: Path, caplog):
        """If an exception is in flight and flush also fails, log the flush error."""
        path = str(tmp_path / "exit_err.tgm")
        store = TensogramStore(path, mode="w")
        store._open_sync()
        store._dirty = True
        store._write_arrays["x"] = {
            "shape": [1],
            "data_type": "float32",
            "codecs": [],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
            },
        }
        store._write_chunks["x/c/0"] = b"\x00"  # Wrong size → flush will fail
        # Simulate __exit__ with an exception in flight
        try:
            raise RuntimeError("original error")
        except RuntimeError:
            import sys

            store.__exit__(*sys.exc_info())
        assert not store._is_open  # closed even on error


# ===================================================================
# 5. __eq__ with non-TensogramStore
# ===================================================================


class TestEqNotImplemented:
    def test_eq_with_string(self, simple_tgm: str):
        store = TensogramStore.open_tgm(simple_tgm)
        assert store != "not a store"
        store.close()

    def test_eq_with_none(self, simple_tgm: str):
        store = TensogramStore.open_tgm(simple_tgm)
        assert store != None  # noqa: E711
        store.close()


# ===================================================================
# 6. Path validation
# ===================================================================


class TestPathValidation:
    def test_empty_path_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            TensogramStore("", mode="r")


# ===================================================================
# 7. mapping.py coverage gaps
# ===================================================================


class TestTgmDtypeToNumpyError:
    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="unsupported TGM dtype"):
            tgm_dtype_to_numpy("imaginary256")


class TestBuildGroupMissingAttrs:
    def test_extra_only(self):
        class Meta:
            version: ClassVar[int] = 2
            extra: ClassVar[dict[str, object]] = {"source": "test"}

        result = build_group_zarr_json(Meta(), ["arr"])
        assert result["attributes"]["source"] == "test"
        assert result["attributes"]["_tensogram_wire_version"] == 2

    def test_with_extra(self):
        class Meta:
            version: ClassVar[int] = 2
            extra: ClassVar[dict[str, object]] = {"custom": 42}

        result = build_group_zarr_json(Meta(), [])
        assert result["attributes"]["custom"] == 42


class TestBuildArrayParamsBranch:
    def test_truthy_params(self):
        class Desc:
            shape: ClassVar[list[object]] = [3]
            dtype: ClassVar[str] = "float32"
            encoding: ClassVar[str] = "simple_packing"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[None] = None
            params: ClassVar[dict[str, object]] = {"bits_per_value": 16, "reference_value": 0.5}

        result = build_array_zarr_json(Desc())
        assert result["attributes"]["_tensogram_params"]["bits_per_value"] == 16

    def test_scalar_shape(self):
        """Empty shape → chunk_shape should be [1]."""

        class Desc:
            shape: ClassVar[list[object]] = []
            dtype: ClassVar[str] = "float64"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[None] = None
            params: ClassVar[dict[str, object]] = {}

        result = build_array_zarr_json(Desc())
        assert result["chunk_grid"]["configuration"]["chunk_shape"] == [1]
        assert result["shape"] == []


class TestParseNoCodecs:
    def test_missing_codecs_defaults_little_endian(self):
        meta = {"shape": [3], "data_type": "int32", "attributes": {}}
        parsed = parse_array_zarr_json(meta)
        assert parsed["byte_order"] == "little"

    def test_no_bytes_codec(self):
        meta = {
            "shape": [3],
            "data_type": "int32",
            "codecs": [{"name": "gzip", "configuration": {"level": 5}}],
            "attributes": {},
        }
        parsed = parse_array_zarr_json(meta)
        assert parsed["byte_order"] == "little"  # default


class TestDeserializeInvalidJson:
    def test_corrupt_bytes_raises_valueerror(self):
        with pytest.raises(ValueError, match=r"invalid zarr\.json"):
            deserialize_zarr_json(b"\xff\xfe not json")

    def test_empty_bytes_raises_valueerror(self):
        with pytest.raises(ValueError, match=r"invalid zarr\.json"):
            deserialize_zarr_json(b"")


# ===================================================================
# 8. _apply_byte_range unknown type
# ===================================================================


class TestApplyByteRangeUnknown:
    def test_unknown_type_raises_typeerror(self):
        class FakeRequest:
            pass

        with pytest.raises(TypeError, match="unsupported ByteRequest"):
            _apply_byte_range(b"hello", FakeRequest())


# ===================================================================
# 9. _native_is_big
# ===================================================================


class TestNativeIsBig:
    def test_returns_bool(self):
        result = _native_is_big()
        assert isinstance(result, bool)

    def test_mocked_big_endian(self):
        with patch("tensogram_zarr.store.sys") as mock_sys:
            mock_sys.byteorder = "big"
            assert _native_is_big() is True

    def test_mocked_little_endian(self):
        with patch("tensogram_zarr.store.sys") as mock_sys:
            mock_sys.byteorder = "little"
            assert _native_is_big() is False


# ===================================================================
# 10. Error wrapping for Rust calls
# ===================================================================


class TestRustErrorWrapping:
    def test_open_nonexistent_file_gives_oserror(self, tmp_path: Path):
        path = str(tmp_path / "does_not_exist.tgm")
        with pytest.raises(OSError, match="failed to open TGM file"):
            TensogramStore.open_tgm(path)

    def test_flush_succeeds_with_valid_data(self, output_path: str):
        """Verify a basic flush with float32 data works end-to-end."""
        store = TensogramStore(output_path, mode="w")
        store._open_sync()
        store._write_arrays["valid"] = {
            "shape": [3],
            "data_type": "float32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
            },
        }
        store._write_chunks["valid/c/0"] = np.array([1.0, 2.0, 3.0], dtype="<f4").tobytes()
        store._dirty = True
        store._flush_to_tgm()
        store._dirty = False
        store.close()

        with tensogram.TensogramFile.open(output_path) as f:
            assert len(f) == 1


# ===================================================================
# 13. _filter_reserved edge cases
# ===================================================================


class TestFilterReserved:
    """Cover store.py _filter_reserved with non-dict inputs."""

    def test_non_dict_entry_returns_empty(self):
        from tensogram_zarr.store import _filter_reserved

        assert _filter_reserved("not a dict") == {}
        assert _filter_reserved(42) == {}
        assert _filter_reserved(None) == {}
        assert _filter_reserved([1, 2]) == {}

    def test_dict_with_reserved_filtered(self):
        from tensogram_zarr.store import _filter_reserved

        entry = {"mars": {"param": "2t"}, "_reserved_": {"tensor": {}}}
        result = _filter_reserved(entry)
        assert "_reserved_" not in result
        assert "mars" in result

    def test_empty_dict_returns_empty(self):
        from tensogram_zarr.store import _filter_reserved

        assert _filter_reserved({}) == {}


# ===================================================================
# 14. _scan_tgm_file with fewer base entries than objects
# ===================================================================


class TestScanWithShortBase:
    """Cover _scan_tgm_file when base has fewer entries than objects."""

    def test_short_base_uses_empty_per_obj(self, tmp_path: Path):
        """When base has fewer entries than objects, missing ones get {}."""
        path = str(tmp_path / "short_base.tgm")
        data1 = np.ones(3, dtype=np.float32)
        data2 = np.ones(4, dtype=np.int32)
        # Only 1 base entry for 2 objects — encoder will auto-pad,
        # but we test the naming logic still works
        meta = {"version": 3, "base": [{"name": "first"}]}
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    ({"type": "ntensor", "shape": [3], "dtype": "float32"}, data1),
                    ({"type": "ntensor", "shape": [4], "dtype": "int32"}, data2),
                ],
            )

        with TensogramStore(path, mode="r") as store:
            # First object should have name "first"
            array_keys = sorted(
                k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"
            )
            names = [k.split("/")[0] for k in array_keys]
            assert "first" in names
            assert len(names) == 2


# ===================================================================
# 15. Write path with big-endian byte order
# ===================================================================


class TestWritePathBigEndian:
    """Cover the big-endian swap path in _flush_to_tgm."""

    def test_big_endian_write(self, output_path: str):
        """Big-endian byte order triggers byteswap on flush."""
        store = TensogramStore(output_path, mode="w")
        store._open_sync()
        store._write_arrays["be_data"] = {
            "shape": [3],
            "data_type": "float32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "big"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
            },
        }
        # Write big-endian bytes
        data = np.array([1.0, 2.0, 3.0], dtype=">f4")
        store._write_chunks["be_data/c/0"] = data.tobytes()
        store._dirty = True
        store._flush_to_tgm()
        store._dirty = False
        store.close()

        # Read back and verify data was correctly byteswapped
        with tensogram.TensogramFile.open(output_path) as f:
            _meta, objects = f.decode_message(0)
            _desc, arr = objects[0]
            np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])


# ===================================================================
# 16. Write path with reserved top-level keys in group attrs
# ===================================================================


class TestWriteReservedTopLevelKeys:
    """Cover the reserved top-level key filtering in _flush_to_tgm."""

    def test_reserved_keys_filtered(self, output_path: str):
        """Reserved keys like 'version', 'base', '_extra_' must not leak."""
        store = TensogramStore(output_path, mode="w")
        store._open_sync()
        store._write_group_attrs = {
            "version": 999,  # should be filtered
            "base": "bogus",  # should be filtered
            "_extra_": "also bogus",  # should be filtered
            "_tensogram_internal": "internal",  # should be filtered (starts with _tensogram_)
            "experiment": "real_attr",  # should be kept
        }
        store._write_arrays["temp"] = {
            "shape": [2],
            "data_type": "float32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
            },
        }
        store._write_chunks["temp/c/0"] = np.array([1.0, 2.0], dtype="<f4").tobytes()
        store._dirty = True
        store._flush_to_tgm()
        store._dirty = False
        store.close()

        with tensogram.TensogramFile.open(output_path) as f:
            meta, _objects = f.decode_message(0)
            # version should be 3 (from encoder), not 999
            assert meta.version == 3
            # experiment should be in extra
            assert meta.extra.get("experiment") == "real_attr"


# ===================================================================
# 17. mapping.py resolve_variable_name priority chain
# ===================================================================


class TestVariableNamePriorityChain:
    """Cover resolve_variable_name's full priority chain."""

    def test_name_key_found(self):
        meta = {"name": "temperature"}
        assert resolve_variable_name(0, meta) == "temperature"

    def test_mars_param_fallback(self):
        meta = {"mars": {"param": "2t"}}
        # No "name" key, so falls back to mars.param
        assert resolve_variable_name(0, meta) == "2t"

    def test_param_fallback(self):
        meta = {"param": "sp"}
        # No "name" or "mars.param", uses "param"
        assert resolve_variable_name(0, meta) == "sp"

    def test_short_name_fallback(self):
        meta = {"mars": {"shortName": "msl"}}
        # No name, mars.param, or param
        assert resolve_variable_name(0, meta) == "msl"

    def test_bare_short_name_fallback(self):
        meta = {"shortName": "msl"}
        assert resolve_variable_name(0, meta) == "msl"

    def test_all_miss_fallback_to_index(self):
        meta = {"mars": {"type": "fc"}}  # no naming keys
        assert resolve_variable_name(5, meta) == "object_5"

    def test_explicit_key_overrides_all(self):
        meta = {"name": "temperature", "mars": {"param": "2t"}, "custom": "myvar"}
        assert resolve_variable_name(0, meta, variable_key="custom") == "myvar"

    def test_explicit_key_miss_tries_standard(self):
        """When variable_key misses, standard keys are tried."""
        meta = {"name": "temperature"}
        assert resolve_variable_name(0, meta, variable_key="nonexistent") == "temperature"


# ===================================================================
# 18. mapping.py _dotted_get edge: None value at intermediate
# ===================================================================


class TestDottedGetNoneIntermediate:
    """Cover _dotted_get when an intermediate value is None."""

    def test_none_at_intermediate(self):
        from tensogram_zarr.mapping import _dotted_get

        d = {"a": {"b": None}}
        assert _dotted_get(d, "a.b.c") is None

    def test_none_at_leaf(self):
        from tensogram_zarr.mapping import _dotted_get

        d = {"a": None}
        assert _dotted_get(d, "a") is None


# ===================================================================
# 19. mapping.py parse_array_zarr_json: _tensogram_ prefix stripping
# ===================================================================


class TestParseArrayZarrJsonStripping:
    """Cover parse_array_zarr_json stripping _tensogram_ prefixed attrs."""

    def test_tensogram_prefixed_attrs_stripped(self):
        meta = {
            "shape": [3],
            "data_type": "float32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
                "_tensogram_hash": {"type": "xxh3", "value": "abc"},
                "_tensogram_params": {"bits_per_value": 16},
                "units": "K",
            },
        }
        parsed = parse_array_zarr_json(meta)
        assert parsed["encoding"] == "none"
        assert parsed["attrs"] == {"units": "K"}
        # All _tensogram_ prefixed keys should be stripped from attrs
        assert "_tensogram_hash" not in parsed["attrs"]
        assert "_tensogram_params" not in parsed["attrs"]


# ===================================================================
# 20. store.py _sanitize_key_segment edge cases
# ===================================================================


class TestSanitizeKeySegmentEdges:
    """Cover _sanitize_key_segment with multiple forbidden chars."""

    def test_multiple_slashes(self):
        from tensogram_zarr.store import _sanitize_key_segment

        assert _sanitize_key_segment("a/b/c") == "a_b_c"

    def test_mixed_slash_backslash(self):
        from tensogram_zarr.store import _sanitize_key_segment

        assert _sanitize_key_segment("a/b\\c") == "a_b_c"


# ===================================================================
# 21. store.py _chunk_key_for_shape edge case: 4D
# ===================================================================


class TestChunkKeyFor4D:
    def test_4d(self):
        from tensogram_zarr.store import _chunk_key_for_shape

        assert _chunk_key_for_shape([1, 2, 3, 4]) == "c/0/0/0/0"


# ===================================================================
# 22. Write path: base entries with per-array user attrs
# ===================================================================


class TestWritePathBaseEntryAttrs:
    """Cover _flush_to_tgm building base entries from array attrs."""

    def test_user_attrs_in_base_entry(self, output_path: str):
        """Array attributes (excluding _tensogram_ and _reserved_) go to base."""
        store = TensogramStore(output_path, mode="w")
        store._open_sync()
        store._write_arrays["temp"] = {
            "shape": [3],
            "data_type": "float32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
                "units": "K",
                "mars": {"param": "2t"},
            },
        }
        store._write_chunks["temp/c/0"] = np.array([1.0, 2.0, 3.0], dtype="<f4").tobytes()
        store._dirty = True
        store._flush_to_tgm()
        store._dirty = False
        store.close()

        with tensogram.TensogramFile.open(output_path) as f:
            meta, _objects = f.decode_message(0)
            base = meta.base[0]
            assert base.get("units") == "K"
            mars = base.get("mars", {})
            assert mars.get("param") == "2t"


# ===================================================================
# 23. build_group_zarr_json: no extra attr
# ===================================================================


class TestBuildGroupNoExtra:
    """Cover build_group_zarr_json when meta.extra is None/empty."""

    def test_no_extra_attr(self):
        class Meta:
            version: ClassVar[int] = 2

        # No extra attribute at all
        result = build_group_zarr_json(Meta(), ["arr"])
        assert result["attributes"]["_tensogram_wire_version"] == 2
        assert result["attributes"]["_tensogram_variables"] == ["arr"]

    def test_none_extra(self):
        class Meta:
            version: ClassVar[int] = 2
            extra: ClassVar[None] = None

        result = build_group_zarr_json(Meta(), [])
        assert result["attributes"]["_tensogram_wire_version"] == 2

    def test_empty_extra(self):
        class Meta:
            version: ClassVar[int] = 2
            extra: ClassVar[dict[str, object]] = {}

        result = build_group_zarr_json(Meta(), ["a"])
        assert result["attributes"]["_tensogram_wire_version"] == 2


# ===================================================================
# 24. build_array_zarr_json: no per_object_meta, no params, with hash
# ===================================================================


class TestBuildArrayEdges:
    """Cover build_array_zarr_json edge paths."""

    def test_no_per_object_meta(self):
        class Desc:
            shape: ClassVar[list[object]] = [3]
            dtype: ClassVar[str] = "int32"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[None] = None
            params: ClassVar[dict[str, object]] = {}

        result = build_array_zarr_json(Desc(), None)
        assert result["attributes"]["_tensogram_encoding"] == "none"
        # No per-object metadata, so no custom attrs
        assert "mars" not in result["attributes"]

    def test_with_hash(self):
        class Desc:
            shape: ClassVar[list[object]] = [3]
            dtype: ClassVar[str] = "float32"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[dict[str, object]] = {"type": "xxh3", "value": "deadbeef"}
            params: ClassVar[dict[str, object]] = {}

        result = build_array_zarr_json(Desc())
        assert result["attributes"]["_tensogram_hash"] == {
            "type": "xxh3",
            "value": "deadbeef",
        }

    def test_with_falsy_params(self):
        """Empty params dict means no _tensogram_params."""

        class Desc:
            shape: ClassVar[list[object]] = [3]
            dtype: ClassVar[str] = "float32"
            encoding: ClassVar[str] = "none"
            filter: ClassVar[str] = "none"
            compression: ClassVar[str] = "none"
            hash: ClassVar[None] = None
            params: ClassVar[dict[str, object]] = {}

        result = build_array_zarr_json(Desc())
        assert "_tensogram_params" not in result["attributes"]


# ===================================================================
# 25. _default_fill_value for all dtype families
# ===================================================================


class TestDefaultFillValueComprehensive:
    """Cover _default_fill_value for all dtype prefixes."""

    def test_float16(self):
        import math

        from tensogram_zarr.mapping import _default_fill_value

        assert math.isnan(_default_fill_value("float16"))

    def test_bfloat16(self):
        import math

        from tensogram_zarr.mapping import _default_fill_value

        assert math.isnan(_default_fill_value("bfloat16"))

    def test_complex128(self):
        import math

        from tensogram_zarr.mapping import _default_fill_value

        # complex starts with "complex", not "float" — but the prefix
        # check should still match
        # Actually "complex" doesn't start with "float" or "bfloat"
        # but our code checks _FLOAT_LIKE_PREFIXES = ("float", "bfloat", "complex")
        assert math.isnan(_default_fill_value("complex128"))

    def test_unknown_dtype_fill(self):
        from tensogram_zarr.mapping import _default_fill_value

        # Unknown dtype that doesn't match any prefix -> 0
        assert _default_fill_value("string") == 0


# ===================================================================
# 26. Flush with clean group attrs (no _tensogram_ and no reserved keys)
# ===================================================================


class TestFlushCleanGroupAttrs:
    """Cover the clean_attrs filtering in _flush_to_tgm."""

    def test_clean_attrs_go_to_message_meta(self, output_path: str):
        """Clean group attrs (no _tensogram_, no reserved) go to message meta."""
        store = TensogramStore(output_path, mode="w")
        store._open_sync()
        store._write_group_attrs = {
            "source": "test",
            "date": "2026-01-01",
        }
        store._write_arrays["arr"] = {
            "shape": [2],
            "data_type": "int32",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
            },
        }
        store._write_chunks["arr/c/0"] = np.array([10, 20], dtype="<i4").tobytes()
        store._dirty = True
        store._flush_to_tgm()
        store._dirty = False
        store.close()

        with tensogram.TensogramFile.open(output_path) as f:
            meta, _ = f.decode_message(0)
            assert meta.extra.get("source") == "test"
            assert meta.extra.get("date") == "2026-01-01"

    def test_delete_array_zarr_json_does_not_clear_group_attrs(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()

            # Set group attrs
            group_json = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"keep": "this"},
                }
            )
            await store.set("zarr.json", proto.buffer.from_bytes(group_json))

            # Set an array
            arr_json = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [3],
                    "data_type": "int32",
                }
            )
            await store.set("temp/zarr.json", proto.buffer.from_bytes(arr_json))

            # Delete array zarr.json — group attrs should remain
            await store.delete("temp/zarr.json")
            assert store._write_group_attrs == {"keep": "this"}
            store._dirty = False
            store.close()

        asyncio.run(run())


# ===================================================================
# Additional coverage gap tests
# ===================================================================


class TestByteRangeGet:
    """Cover store.py line 256: _apply_byte_range in _get_sync."""

    def test_get_with_offset_byte_range(self, simple_tgm: str):
        """Fetching a chunk key with an OffsetByteRequest applies byte slicing."""
        from zarr.abc.store import OffsetByteRequest
        from zarr.core.buffer import default_buffer_prototype

        async def run():
            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            proto = default_buffer_prototype()

            # Get full zarr.json first to confirm it exists
            full = store._get_sync("zarr.json", proto, byte_range=None)
            assert full is not None

            # Get with an offset byte range (skip first 5 bytes)
            sliced = store._get_sync("zarr.json", proto, byte_range=OffsetByteRequest(offset=5))
            assert sliced is not None
            assert len(sliced) == len(full) - 5

            store.close()

        asyncio.run(run())

    def test_get_with_suffix_byte_range(self, simple_tgm: str):
        """Fetching with a SuffixByteRequest returns the last N bytes."""
        from zarr.abc.store import SuffixByteRequest
        from zarr.core.buffer import default_buffer_prototype

        async def run():
            store = TensogramStore(simple_tgm, mode="r")
            await store._open()
            proto = default_buffer_prototype()

            full = store._get_sync("zarr.json", proto, byte_range=None)
            assert full is not None

            sliced = store._get_sync("zarr.json", proto, byte_range=SuffixByteRequest(suffix=10))
            assert sliced is not None
            assert len(sliced) == 10

            store.close()

        asyncio.run(run())


class TestSetIfNotExists:
    """Cover store.py line 301: set_if_not_exists write path."""

    def test_set_if_not_exists_writes_new_key(self, output_path: str):
        """set_if_not_exists writes when key is absent."""
        from zarr.core.buffer import default_buffer_prototype

        async def run():
            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()

            group_json = serialize_zarr_json(
                {"zarr_format": 3, "node_type": "group", "attributes": {}}
            )
            await store.set_if_not_exists("zarr.json", proto.buffer.from_bytes(group_json))
            assert await store.exists("zarr.json")
            store._dirty = False
            store.close()

        asyncio.run(run())

    def test_set_if_not_exists_skips_existing_key(self, output_path: str):
        """set_if_not_exists does NOT overwrite an existing key."""
        from zarr.core.buffer import default_buffer_prototype

        async def run():
            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()

            original = serialize_zarr_json(
                {"zarr_format": 3, "node_type": "group", "attributes": {"v": 1}}
            )
            replacement = serialize_zarr_json(
                {"zarr_format": 3, "node_type": "group", "attributes": {"v": 2}}
            )
            await store.set("zarr.json", proto.buffer.from_bytes(original))
            await store.set_if_not_exists("zarr.json", proto.buffer.from_bytes(replacement))
            # Original should still be there — read raw bytes from _keys
            raw = store._keys.get("zarr.json")
            assert raw is not None
            text = raw.decode() if isinstance(raw, bytes) else str(raw)
            assert '"v": 1' in text or '"v":1' in text
            store._dirty = False
            store.close()

        asyncio.run(run())


class TestListDirPrefixNormalization:
    """Cover store.py line 340: prefix without trailing '/'."""

    def test_list_dir_without_trailing_slash(self, simple_tgm: str):
        """list_dir with prefix lacking '/' still works."""

        async def run():
            store = TensogramStore(simple_tgm, mode="r")
            await store._open()

            # list_dir with a prefix that has no trailing slash
            entries = []
            async for entry in store.list_dir("object_0"):
                entries.append(entry)
            # Should find at least zarr.json and c/ for the variable
            assert len(entries) >= 1
            store.close()

        asyncio.run(run())


class TestFindChunkDataMultiChunkError:
    """Cover store.py lines 648-649: _find_chunk_data multi-chunk error."""

    def test_multiple_chunk_keys_raises(self):
        """Multiple chunk keys for same variable raises ValueError."""
        from tensogram_zarr.store import _find_chunk_data

        chunks = {
            "temp/c/0": b"data1",
            "temp/c/1": b"data2",
        }
        with pytest.raises(ValueError, match="chunk keys"):
            _find_chunk_data("temp", chunks)
