# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""End-to-end round-trip tests: write through Zarr → read through Zarr and tensogram."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensogram
from tensogram_zarr import TensogramStore
from tensogram_zarr.mapping import serialize_zarr_json


class TestTgmToZarrRoundTrip:
    """Write TGM with tensogram API, read back through TensogramStore."""

    def test_simple_round_trip(self, tmp_path: Path):
        path = str(tmp_path / "rt.tgm")
        original = np.arange(20, dtype=np.float32).reshape(4, 5)

        # Write with tensogram
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {"version": 2, "source": "test"},
                [
                    ({"type": "ntensor", "shape": [4, 5], "dtype": "float32"}, original),
                ],
            )

        # Read through TensogramStore
        with TensogramStore(path, mode="r") as store:
            # Check root
            root_meta = json.loads(store._keys["zarr.json"])
            assert root_meta["attributes"]["source"] == "test"

            # Find and read array
            chunk_keys = [k for k in store._keys if "/c/" in k]
            assert len(chunk_keys) == 1
            arr_bytes = store._keys[chunk_keys[0]]
            arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(4, 5)
            np.testing.assert_array_equal(arr, original)

    def test_multi_dtype_round_trip(self, tmp_path: Path):
        path = str(tmp_path / "multi_dt.tgm")
        f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        i64 = np.array([100, 200], dtype=np.int64)

        with tensogram.TensogramFile.create(path) as f:
            f.append(
                {
                    "version": 2,
                    "base": [{"name": "floats"}, {"name": "ints"}],
                },
                [
                    ({"type": "ntensor", "shape": [3], "dtype": "float32"}, f32),
                    ({"type": "ntensor", "shape": [2], "dtype": "int64"}, i64),
                ],
            )

        with TensogramStore(path, mode="r") as store:
            f_data = store._keys.get("floats/c/0")
            assert f_data is not None
            np.testing.assert_array_equal(np.frombuffer(f_data, dtype=np.float32), f32)

            i_data = store._keys.get("ints/c/0")
            assert i_data is not None
            np.testing.assert_array_equal(np.frombuffer(i_data, dtype=np.int64), i64)


class TestStoreWriteReadRoundTrip:
    """Write through TensogramStore, read back through TensogramStore."""

    def test_write_then_read(self, tmp_path: Path):
        path = str(tmp_path / "store_rt.tgm")
        data = np.random.rand(8, 6).astype(np.float64)

        # Write
        with TensogramStore(path, mode="w") as store:
            store._keys["zarr.json"] = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"experiment": "rt"},
                }
            )
            store._write_group_attrs = {"experiment": "rt"}

            arr_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [8, 6],
                "data_type": "float64",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [8, 6]}},
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "fill_value": None,
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                },
            }
            store._keys["field/zarr.json"] = serialize_zarr_json(arr_meta)
            store._write_arrays["field"] = arr_meta
            chunk_bytes = data.astype("<f8").tobytes()
            store._keys["field/c/0"] = chunk_bytes
            store._write_chunks["field/c/0"] = chunk_bytes
            store._dirty = True

        # Read back — written data goes through TGM encode/decode,
        # object name falls back to "object_0", 2D chunk key is c/0/0
        with TensogramStore(path, mode="r") as store:
            chunk_data = store._keys.get("object_0/c/0/0")
            assert chunk_data is not None
            arr = np.frombuffer(chunk_data, dtype=np.float64).reshape(8, 6)
            np.testing.assert_array_almost_equal(arr, data)

    def test_write_read_preserves_metadata(self, tmp_path: Path):
        path = str(tmp_path / "meta_rt.tgm")

        with TensogramStore(path, mode="w") as store:
            store._write_group_attrs = {"source": "unittest", "version": "1.0"}

            arr_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3],
                "data_type": "int32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "fill_value": 0,
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                    "units": "K",
                },
            }
            store._write_arrays["temp"] = arr_meta
            chunk = np.array([273, 280, 290], dtype=np.int32).tobytes()
            store._write_chunks["temp/c/0"] = chunk
            store._dirty = True

        # Read via raw tensogram and check metadata
        with tensogram.TensogramFile.open(path) as f:
            meta, objects = f.decode_message(0)
            assert meta.extra.get("source") == "unittest"
            _desc, arr = objects[0]
            np.testing.assert_array_equal(arr, [273, 280, 290])


class TestStoreListOperations:
    """Test listing operations on the store."""

    def test_list_all_keys(self, multi_object_tgm: str):
        with TensogramStore(multi_object_tgm, mode="r") as store:
            keys = list(store._keys.keys())
            # Root zarr.json + 3 arrays * (zarr.json + c/0) = 1 + 6 = 7
            assert len(keys) == 7

    def test_list_prefix(self, multi_object_tgm: str):
        with TensogramStore(multi_object_tgm, mode="r") as store:
            # Get all keys starting with "2t/"
            matching = [k for k in store._keys if k.startswith("2t/")]
            assert len(matching) == 2  # zarr.json + c/0

    def test_list_dir_root(self, multi_object_tgm: str):
        """list_dir at root should show zarr.json + 3 variable directories."""
        import asyncio

        async def run():
            entries = []
            with TensogramStore(multi_object_tgm, mode="r") as store:
                async for entry in store.list_dir(""):
                    entries.append(entry)
            return entries

        entries = asyncio.run(run())
        # Should have zarr.json and 3 directory entries (2t/, sp/, q/)
        assert "zarr.json" in entries
        dir_entries = [e for e in entries if e.endswith("/")]
        assert len(dir_entries) == 3


class TestDuplicateNames:
    """Test variable name deduplication."""

    def test_duplicate_names_get_suffix(self, tmp_path: Path):
        path = str(tmp_path / "dup.tgm")
        data = np.ones(4, dtype=np.float32)

        # Two objects with the same name
        meta = {
            "version": 2,
            "base": [
                {"name": "field"},
                {"name": "field"},
            ],
        }
        with tensogram.TensogramFile.create(path) as f:
            f.append(
                meta,
                [
                    ({"type": "ntensor", "shape": [4], "dtype": "float32"}, data),
                    ({"type": "ntensor", "shape": [4], "dtype": "float32"}, data * 2),
                ],
            )

        with TensogramStore(path, mode="r") as store:
            # Should have field and field_1
            array_keys = sorted(
                k for k in store._keys if k.endswith("/zarr.json") and k != "zarr.json"
            )
            names = [k.split("/")[0] for k in array_keys]
            assert "field" in names
            assert "field_1" in names
