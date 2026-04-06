"""Tests for TensogramStore read operations."""

from __future__ import annotations

import json

import numpy as np
from tensogram_zarr import TensogramStore


class TestOpenTgm:
    """Test the open_tgm factory and basic read properties."""

    def test_open_simple(self, simple_tgm: str):
        store = TensogramStore.open_tgm(simple_tgm)
        assert store._is_open
        assert store.read_only
        assert not store.supports_writes
        assert store.supports_listing
        store.close()

    def test_context_manager(self, simple_tgm: str):
        with TensogramStore(simple_tgm, mode="r") as store:
            assert store._is_open
        assert not store._is_open

    def test_repr(self, simple_tgm: str):
        store = TensogramStore.open_tgm(simple_tgm)
        assert "TensogramStore" in repr(store)
        assert simple_tgm in repr(store)
        store.close()

    def test_equality(self, simple_tgm: str):
        s1 = TensogramStore.open_tgm(simple_tgm)
        s2 = TensogramStore.open_tgm(simple_tgm)
        assert s1 == s2
        s1.close()
        s2.close()


class TestReadRootGroup:
    """Test reading root group zarr.json."""

    def test_root_zarr_json_exists(self, simple_tgm: str):
        with TensogramStore(simple_tgm, mode="r") as store:
            data = store._keys.get("zarr.json")
            assert data is not None
            meta = json.loads(data)
            assert meta["zarr_format"] == 3
            assert meta["node_type"] == "group"

    def test_root_has_tensogram_version(self, simple_tgm: str):
        with TensogramStore(simple_tgm, mode="r") as store:
            meta = json.loads(store._keys["zarr.json"])
            assert meta["attributes"]["_tensogram_version"] == 2

    def test_root_has_variable_list(self, multi_object_tgm: str):
        with TensogramStore(multi_object_tgm, mode="r") as store:
            meta = json.loads(store._keys["zarr.json"])
            var_names = meta["attributes"]["_tensogram_variables"]
            assert len(var_names) == 3

    def test_mars_metadata_in_root(self, mars_metadata_tgm: str):
        with TensogramStore(mars_metadata_tgm, mode="r") as store:
            meta = json.loads(store._keys["zarr.json"])
            mars = meta["attributes"]["mars"]
            assert mars["class"] == "od"
            assert mars["type"] == "fc"


class TestReadArrayMetadata:
    """Test reading per-array zarr.json."""

    def test_single_array_metadata(self, simple_tgm: str):
        with TensogramStore(simple_tgm, mode="r") as store:
            # Find the array key — name is object_0 (no metadata key)
            array_keys = [k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"]
            assert len(array_keys) == 1

            meta = json.loads(store._keys[array_keys[0]])
            assert meta["node_type"] == "array"
            assert meta["shape"] == [6, 10]
            assert meta["data_type"] == "float32"
            assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [6, 10]

    def test_multi_object_names(self, multi_object_tgm: str):
        with TensogramStore(multi_object_tgm, mode="r") as store:
            array_keys = sorted(
                k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"
            )
            # Names from mars.param: 2t, sp, q
            names = [k.split("/")[0] for k in array_keys]
            assert "2t" in names
            assert "sp" in names
            assert "q" in names

    def test_custom_variable_key(self, multi_object_tgm: str):
        with TensogramStore(multi_object_tgm, mode="r", variable_key="mars.param") as store:
            array_keys = sorted(
                k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"
            )
            names = [k.split("/")[0] for k in array_keys]
            assert "2t" in names


class TestReadChunkData:
    """Test reading array chunk data."""

    def test_simple_chunk_data(self, simple_tgm: str):
        with TensogramStore(simple_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            assert len(chunk_keys) == 1
            # 2D array → chunk key is "object_0/c/0/0"
            assert chunk_keys[0].endswith("/c/0/0")

            chunk_bytes = store._keys[chunk_keys[0]]
            arr = np.frombuffer(chunk_bytes, dtype=np.float32).reshape(6, 10)
            expected = np.arange(60, dtype=np.float32).reshape(6, 10)
            np.testing.assert_array_equal(arr, expected)

    def test_multi_object_chunks(self, multi_object_tgm: str):
        with TensogramStore(multi_object_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            assert len(chunk_keys) == 3

    def test_int_types(self, int_types_tgm: str):
        with TensogramStore(int_types_tgm, mode="r") as store:
            # "counts" array (int32, shape [4] → chunk key c/0)
            counts_data = store._keys.get("counts/c/0")
            assert counts_data is not None
            arr = np.frombuffer(counts_data, dtype=np.int32)
            np.testing.assert_array_equal(arr, [1, 2, 3, 4])

            # "flags" array (uint16, shape [3] → chunk key c/0)
            flags_data = store._keys.get("flags/c/0")
            assert flags_data is not None
            arr = np.frombuffer(flags_data, dtype=np.uint16)
            np.testing.assert_array_equal(arr, [10, 20, 30])


class TestReadEmptyFile:
    """Test reading an empty .tgm file."""

    def test_empty_file_has_root_group(self, empty_tgm: str):
        with TensogramStore(empty_tgm, mode="r") as store:
            assert "zarr.json" in store._keys
            meta = json.loads(store._keys["zarr.json"])
            assert meta["node_type"] == "group"

    def test_empty_file_has_no_arrays(self, empty_tgm: str):
        with TensogramStore(empty_tgm, mode="r") as store:
            array_keys = [k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"]
            assert len(array_keys) == 0


class TestByteRange:
    """Test partial reads with byte range requests."""

    def test_range_request(self, simple_tgm: str):
        from zarr.abc.store import RangeByteRequest
        from zarr.core.buffer import default_buffer_prototype

        with TensogramStore(simple_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            key = chunk_keys[0]
            full_data = store._keys[key]

            # Request first 16 bytes
            proto = default_buffer_prototype()
            buf = store._get_sync(key, proto, RangeByteRequest(0, 16))
            assert buf is not None
            assert len(buf.to_bytes()) == 16
            assert buf.to_bytes() == full_data[:16]

    def test_offset_request(self, simple_tgm: str):
        from zarr.abc.store import OffsetByteRequest
        from zarr.core.buffer import default_buffer_prototype

        with TensogramStore(simple_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            key = chunk_keys[0]
            full_data = store._keys[key]

            proto = default_buffer_prototype()
            buf = store._get_sync(key, proto, OffsetByteRequest(100))
            assert buf is not None
            assert buf.to_bytes() == full_data[100:]

    def test_suffix_request(self, simple_tgm: str):
        from zarr.abc.store import SuffixByteRequest
        from zarr.core.buffer import default_buffer_prototype

        with TensogramStore(simple_tgm, mode="r") as store:
            chunk_keys = [k for k in store._keys if "/c/" in k]
            key = chunk_keys[0]
            full_data = store._keys[key]

            proto = default_buffer_prototype()
            buf = store._get_sync(key, proto, SuffixByteRequest(20))
            assert buf is not None
            assert buf.to_bytes() == full_data[-20:]

    def test_missing_key_returns_none(self, simple_tgm: str):
        from zarr.core.buffer import default_buffer_prototype

        with TensogramStore(simple_tgm, mode="r") as store:
            proto = default_buffer_prototype()
            buf = store._get_sync("nonexistent/c/0", proto)
            assert buf is None
