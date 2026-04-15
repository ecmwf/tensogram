#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 08: Zarr v3 backend — read and write .tgm files through Zarr.

Demonstrates using TensogramStore as a Zarr v3 Store backend, allowing
.tgm files to be accessed through the standard Zarr Python API.

Requirements:
    pip install tensogram tensogram-zarr zarr numpy
"""

import tempfile
from pathlib import Path

import numpy as np
import tensogram
import zarr
from tensogram_zarr import TensogramStore


def create_sample_tgm(path: str) -> None:
    """Create a sample .tgm file with multiple arrays."""
    temperature = np.random.rand(10, 20).astype(np.float32) * 50 + 230  # ~230-280 K
    pressure = np.array([1000, 925, 850, 700, 500, 300], dtype=np.float64)
    humidity = np.random.rand(10, 20).astype(np.float32)

    meta = {
        "version": 2,
        "source": "ifs-cycle49r1",
        "base": [
            {
                "mars": {
                    "class": "od",
                    "type": "fc",
                    "stream": "oper",
                    "date": "20260401",
                    "time": "1200",
                    "param": "2t",
                    "levtype": "sfc",
                },
                "units": "K",
            },
            {
                "mars": {
                    "class": "od",
                    "type": "fc",
                    "stream": "oper",
                    "date": "20260401",
                    "time": "1200",
                    "param": "sp",
                    "levtype": "pl",
                },
                "units": "hPa",
            },
            {
                "mars": {
                    "class": "od",
                    "type": "fc",
                    "stream": "oper",
                    "date": "20260401",
                    "time": "1200",
                    "param": "q",
                    "levtype": "sfc",
                },
                "units": "kg/kg",
            },
        ],
    }
    descs_and_data = [
        ({"type": "ntensor", "shape": [10, 20], "dtype": "float32"}, temperature),
        ({"type": "ntensor", "shape": [6], "dtype": "float64"}, pressure),
        ({"type": "ntensor", "shape": [10, 20], "dtype": "float32"}, humidity),
    ]

    with tensogram.TensogramFile.create(path) as f:
        f.append(meta, descs_and_data)
    print(f"Created sample .tgm file: {path}")


def read_with_zarr(path: str) -> None:
    """Read a .tgm file through the Zarr API."""
    print("\n--- Reading .tgm through Zarr ---")

    store = TensogramStore.open_tgm(path)
    root = zarr.open_group(store=store, mode="r")

    # List all arrays
    print("\nArrays in the store:")
    for name, arr in root.members():
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    # Access group attributes (from GlobalMetadata)
    print(f"\nMARS metadata: {dict(root.attrs).get('mars', {})}")
    print(f"Source: {root.attrs.get('source', 'unknown')}")

    # Read array data
    temp = root["2t"][:]
    print(
        f"\nTemperature (2t): min={temp.min():.1f}, max={temp.max():.1f}, mean={temp.mean():.1f}"
    )

    # Slicing works
    subset = root["2t"][2:5, 10:15]
    print(f"Slice [2:5, 10:15]: shape={subset.shape}")

    # Scalar indexing
    val = root["2t"][0, 0]
    print(f"Single value [0,0]: {val:.2f}")

    # Pressure levels
    levels = root["sp"][:]
    print(f"\nPressure levels: {levels}")

    store.close()


def write_with_zarr(path: str) -> None:
    """Write a .tgm file through the Zarr API (low-level store operations)."""
    print("\n--- Writing .tgm through Zarr ---")

    # The TensogramStore write path buffers data in memory
    # and flushes to .tgm on close
    from tensogram_zarr.mapping import serialize_zarr_json

    data = np.arange(30, dtype=np.float32).reshape(5, 6)

    with TensogramStore(path, mode="w") as store:
        # Set root group
        store._keys["zarr.json"] = serialize_zarr_json(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"experiment": "zarr-write-demo"},
            }
        )
        store._write_group_attrs = {"experiment": "zarr-write-demo"}

        # Set array metadata
        arr_meta = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [5, 6],
            "data_type": "float32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5, 6]}},
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "fill_value": None,
            "attributes": {
                "_tensogram_encoding": "none",
                "_tensogram_filter": "none",
                "_tensogram_compression": "none",
                "units": "demo-units",
            },
        }
        store._keys["output/zarr.json"] = serialize_zarr_json(arr_meta)
        store._write_arrays["output"] = arr_meta

        # Set chunk data
        chunk_bytes = data.astype("<f4").tobytes()
        store._keys["output/c/0/0"] = chunk_bytes
        store._write_chunks["output/c/0/0"] = chunk_bytes
        store._dirty = True

    print(f"Written .tgm file: {path}")

    # Verify by reading back
    with tensogram.TensogramFile.open(path) as f:
        meta, objects = f.decode_message(0)
        _desc, arr = objects[0]
        print(f"Verified: shape={arr.shape}, matches={np.array_equal(arr, data)}")
        print(f"Metadata extra: {meta.extra}")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = str(Path(tmpdir) / "sample.tgm")
        output_path = str(Path(tmpdir) / "output.tgm")

        # Create and read
        create_sample_tgm(sample_path)
        read_with_zarr(sample_path)

        # Write and verify
        write_with_zarr(output_path)

        print("\nDone!")


if __name__ == "__main__":
    main()
