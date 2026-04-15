# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Tests for TensogramStore write operations."""

from __future__ import annotations

import numpy as np
import pytest
import tensogram
from tensogram_zarr import TensogramStore
from tensogram_zarr.mapping import serialize_zarr_json


class TestWriteBasic:
    """Test basic write operations."""

    def test_write_mode_properties(self, output_path: str):
        store = TensogramStore(output_path, mode="w")
        store._open_sync()
        assert store.supports_writes
        assert store.supports_deletes
        assert not store.read_only
        store.close()

    def test_write_and_read_back(self, output_path: str):
        """Write a simple array through the store, then read back the TGM file."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)

        # Write via store
        with TensogramStore(output_path, mode="w") as store:
            # Set root group
            group_meta = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"experiment": "test"},
            }
            store._keys["zarr.json"] = serialize_zarr_json(group_meta)
            store._write_group_attrs = group_meta["attributes"]

            # Set array metadata
            array_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3, 4],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [3, 4]},
                },
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "fill_value": None,
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                },
            }
            store._keys["temperature/zarr.json"] = serialize_zarr_json(array_meta)
            store._write_arrays["temperature"] = array_meta

            # Set chunk data (2D → chunk key c/0/0)
            chunk_bytes = data.astype("<f4").tobytes()
            store._keys["temperature/c/0/0"] = chunk_bytes
            store._write_chunks["temperature/c/0/0"] = chunk_bytes
            store._dirty = True

        # Verify the TGM file was written correctly
        with tensogram.TensogramFile.open(output_path) as f:
            assert len(f) == 1
            _meta, objects = f.decode_message(0)
            assert len(objects) == 1
            desc, arr = objects[0]
            assert list(desc.shape) == [3, 4]
            assert desc.dtype == "float32"
            np.testing.assert_array_almost_equal(arr, data)


class TestWriteMultipleArrays:
    """Test writing multiple arrays."""

    def test_multi_array_write(self, output_path: str):
        temp = np.random.rand(5, 5).astype(np.float64)
        pressure = np.array([1000.0, 900.0, 800.0], dtype=np.float64)

        with TensogramStore(output_path, mode="w") as store:
            # Group
            store._keys["zarr.json"] = serialize_zarr_json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {},
                }
            )
            store._write_group_attrs = {}

            # Temperature
            temp_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [5, 5],
                "data_type": "float64",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5, 5]}},
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "fill_value": None,
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                },
            }
            store._keys["pressure_levels/zarr.json"] = serialize_zarr_json(temp_meta)
            store._write_arrays["pressure_levels"] = temp_meta

            press_meta = {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3],
                "data_type": "float64",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "fill_value": None,
                "attributes": {
                    "_tensogram_encoding": "none",
                    "_tensogram_filter": "none",
                    "_tensogram_compression": "none",
                },
            }
            store._keys["temperature/zarr.json"] = serialize_zarr_json(press_meta)
            store._write_arrays["temperature"] = press_meta

            # 2D chunk key: c/0/0; 1D chunk key: c/0
            store._keys["pressure_levels/c/0/0"] = temp.astype("<f8").tobytes()
            store._write_chunks["pressure_levels/c/0/0"] = temp.astype("<f8").tobytes()
            store._keys["temperature/c/0"] = pressure.astype("<f8").tobytes()
            store._write_chunks["temperature/c/0"] = pressure.astype("<f8").tobytes()
            store._dirty = True

        # Verify
        with tensogram.TensogramFile.open(output_path) as f:
            assert len(f) == 1
            _meta, objects = f.decode_message(0)
            assert len(objects) == 2


class TestWriteReadOnly:
    """Test that read-only stores reject writes."""

    def test_read_only_rejects_writes(self, simple_tgm: str):
        with (
            TensogramStore(simple_tgm, mode="r") as store,
            pytest.raises(ValueError, match="read-only"),
        ):
            store._check_writable()
