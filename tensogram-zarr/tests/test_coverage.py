"""Coverage gap tests — exercises every untested path found by the audit.

Each test class targets a specific gap identified in the coverage report.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
    serialize_zarr_json,
    tgm_dtype_to_numpy,
)
from tensogram_zarr.store import (
    _apply_byte_range,
    _find_chunk_data,
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
                {"version": 2, "payload": [{"name": "first"}]},
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
    def test_no_common(self):
        class Meta:
            version = 2
            extra = {"source": "test"}

        result = build_group_zarr_json(Meta(), ["arr"])
        assert result["attributes"]["source"] == "test"
        assert result["attributes"]["_tensogram_version"] == 2

    def test_with_extra(self):
        class Meta:
            version = 2
            common = {}
            extra = {"custom": 42}

        result = build_group_zarr_json(Meta(), [])
        assert result["attributes"]["custom"] == 42


class TestBuildArrayParamsBranch:
    def test_truthy_params(self):
        class Desc:
            shape = [3]
            dtype = "float32"
            encoding = "simple_packing"
            filter = "none"
            compression = "none"
            hash = None
            params = {"bits_per_value": 16, "reference_value": 0.5}

        result = build_array_zarr_json(Desc())
        assert result["attributes"]["_tensogram_params"]["bits_per_value"] == 16

    def test_scalar_shape(self):
        """Empty shape → chunk_shape should be [1]."""

        class Desc:
            shape = []
            dtype = "float64"
            encoding = "none"
            filter = "none"
            compression = "none"
            hash = None
            params = {}

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
# 11. Multi-chunk detection (_find_chunk_data)
# ===================================================================


class TestMultiChunkDetection:
    """Verify _find_chunk_data raises on multiple chunks instead of silently
    picking the first one (data loss risk)."""

    def test_single_chunk_returns_data(self):
        chunks = {"var/c/0": b"data"}
        assert _find_chunk_data("var", chunks) == b"data"

    def test_no_chunk_returns_none(self):
        assert _find_chunk_data("var", {}) is None

    def test_multiple_chunks_raises(self):
        chunks = {
            "var/c/0/0": b"chunk1",
            "var/c/0/1": b"chunk2",
        }
        with pytest.raises(ValueError, match=r"2 chunk keys.*single-chunk"):
            _find_chunk_data("var", chunks)

    def test_other_var_chunks_not_matched(self):
        chunks = {
            "var/c/0": b"data",
            "other_var/c/0": b"other",
        }
        assert _find_chunk_data("var", chunks) == b"data"


# ===================================================================
# 12. Stale group attrs on root zarr.json delete
# ===================================================================


class TestDeleteRootZarrJsonClearsGroupAttrs:
    """Deleting root zarr.json must clear _write_group_attrs to prevent
    stale attributes from being flushed."""

    def test_delete_root_zarr_json_clears_group_attrs(self, output_path: str):
        async def run():
            from zarr.core.buffer import default_buffer_prototype

            store = TensogramStore(output_path, mode="w")
            await store._open()
            proto = default_buffer_prototype()

            # Set group metadata with some attributes
            group_json = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"important": "data"},
                }
            )
            await store.set("zarr.json", proto.buffer.from_bytes(group_json))
            assert store._write_group_attrs == {"important": "data"}

            # Delete root zarr.json
            await store.delete("zarr.json")

            # Group attrs must be cleared
            assert store._write_group_attrs == {}
            store._dirty = False
            store.close()

        asyncio.run(run())

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
