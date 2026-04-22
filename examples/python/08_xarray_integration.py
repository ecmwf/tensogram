# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Example 08 — xarray integration

Demonstrates how to open tensogram .tgm files as xarray Datasets using
the ``tensogram-xarray`` backend (``engine="tensogram"``).

Covers:
  - Basic open with auto-generated dimension names
  - Coordinate auto-detection from named data objects
  - Variable naming via ``variable_key``
  - User-specified ``dim_names``
  - Multi-message files with ``open_datasets()``
  - Lazy loading and the ``range_threshold`` heuristic

Application metadata (``name``, ``mars``, ``cf``, ``product``, …) lives
in ``metadata["base"][i]`` — one dict per data object, indexed
positionally against the ``descriptors_and_data`` list passed to
``encode`` / ``append``.  Placing these keys *inside* the descriptor
dict triggers a ``UserWarning`` from tensogram 0.17+ (see
``02b_generic_metadata.py`` for the same pattern on a single-object
message).

Prerequisites:
  pip install tensogram-xarray   # or: pip install -e python/tensogram-xarray/
"""

import tempfile
from pathlib import Path

import numpy as np
import tensogram
import tensogram_xarray
import xarray as xr


def _desc(shape, dtype="float32"):
    """Shorthand for a plain ntensor descriptor with no encoding pipeline.

    Application metadata (``name``, ``mars``, ...) is deliberately NOT
    accepted here — put those in ``metadata["base"][i]`` instead.

    ``byte_order`` is deliberately omitted: the Python encoder writes
    numpy data in native byte order, and the descriptor parser defaults
    the missing key to native so the descriptor always matches the
    payload on any host.
    """
    return {
        "type": "ntensor",
        "shape": shape,
        "dtype": dtype,
        "encoding": "none",
        "filter": "none",
        "compression": "none",
    }


def example_basic(tmp: Path):
    """1. Basic open — single object, generic dimension names."""
    data = np.arange(60, dtype=np.float32).reshape(6, 10)
    path = str(tmp / "basic.tgm")

    with tensogram.TensogramFile.create(path) as f:
        f.append({"version": 3, "base": [{}]}, [(_desc([6, 10]), data)])

    ds = xr.open_dataset(path, engine="tensogram")
    print("1. Basic open")
    print(f"   Variables: {list(ds.data_vars)}")
    print(f"   Dims:      {dict(ds.sizes)}")
    np.testing.assert_array_equal(ds["object_0"].values, data)
    print("   Round-trip OK")
    print()


def example_coords(tmp: Path):
    """2. Coordinate auto-detection from named data objects.

    The xarray backend treats a 1-D object whose ``name`` matches a
    dimension of another object as that dimension's coordinate.
    """
    lat = np.linspace(-90, 90, 5, dtype=np.float64)
    lon = np.linspace(0, 360, 8, endpoint=False, dtype=np.float64)
    temp = np.random.default_rng(42).random((5, 8)).astype(np.float32)

    path = str(tmp / "coords.tgm")
    with tensogram.TensogramFile.create(path) as f:
        f.append(
            {
                "version": 3,
                "base": [
                    {"name": "latitude"},
                    {"name": "longitude"},
                    {"name": "temperature"},
                ],
            },
            [
                (_desc([5], dtype="float64"), lat),
                (_desc([8], dtype="float64"), lon),
                (_desc([5, 8]), temp),
            ],
        )

    ds = xr.open_dataset(path, engine="tensogram")
    print("2. Coordinate auto-detection")
    print(f"   Coords:    {list(ds.coords)}")
    print(f"   Variables: {list(ds.data_vars)}")
    np.testing.assert_allclose(ds.coords["latitude"].values, lat)
    np.testing.assert_allclose(ds.coords["longitude"].values, lon)
    print("   Lat/lon coords match")
    print()


def example_variable_key(tmp: Path):
    """3. Variable naming from metadata via variable_key."""
    t2m = np.ones((3, 4), dtype=np.float32) * 273.15
    u10 = np.ones((3, 4), dtype=np.float32) * 5.0

    path = str(tmp / "mars.tgm")
    with tensogram.TensogramFile.create(path) as f:
        f.append(
            {
                "version": 3,
                "base": [
                    {"mars": {"param": "2t"}},
                    {"mars": {"param": "10u"}},
                ],
            },
            [(_desc([3, 4]), t2m), (_desc([3, 4]), u10)],
        )

    ds1 = xr.open_dataset(path, engine="tensogram")
    print("3. Variable naming")
    print(f"   Without variable_key: {list(ds1.data_vars)}")

    ds2 = xr.open_dataset(path, engine="tensogram", variable_key="mars.param")
    print(f"   With variable_key:    {list(ds2.data_vars)}")
    assert "2t" in ds2.data_vars
    assert "10u" in ds2.data_vars
    print()


def example_dim_names(tmp: Path):
    """4. User-specified dimension names."""
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    path = str(tmp / "dims.tgm")

    with tensogram.TensogramFile.create(path) as f:
        f.append({"version": 3, "base": [{}]}, [(_desc([4, 5]), data)])

    ds = xr.open_dataset(path, engine="tensogram", dim_names=["latitude", "longitude"])
    print("4. User-specified dim_names")
    print(f"   Dims: {ds['object_0'].dims}")
    assert ds["object_0"].dims == ("latitude", "longitude")
    print()


def example_multi_message(tmp: Path):
    """5. Multi-message file with open_datasets()."""
    rng = np.random.default_rng(99)
    path = str(tmp / "multi.tgm")

    with tensogram.TensogramFile.create(path) as f:
        for param in ["2t", "10u"]:
            for date in ["20260401", "20260402"]:
                data = rng.random((3, 4), dtype=np.float32).astype(np.float32)
                f.append(
                    {"version": 3, "base": [{"mars": {"param": param, "date": date}}]},
                    [(_desc([3, 4]), data)],
                )

    datasets = tensogram_xarray.open_datasets(path, variable_key="mars.param")
    print("5. Multi-message open_datasets()")
    print(f"   Number of datasets: {len(datasets)}")
    for i, ds in enumerate(datasets):
        print(f"   Dataset {i}: vars={list(ds.data_vars)}")
    print()


def example_lazy_and_threshold(tmp: Path):
    """6. Lazy loading and range_threshold control."""
    data = np.arange(1000, dtype=np.float32).reshape(10, 100)
    path = str(tmp / "lazy.tgm")

    with tensogram.TensogramFile.create(path) as f:
        f.append({"version": 3, "base": [{}]}, [(_desc([10, 100]), data)])

    ds = xr.open_dataset(path, engine="tensogram")
    small_slice = ds["object_0"][2:4, 10:20].values
    np.testing.assert_array_equal(small_slice, data[2:4, 10:20])
    print("6. Lazy loading + range_threshold")
    print(f"   Small slice (20%): shape={small_slice.shape}, values match")

    ds2 = xr.open_dataset(path, engine="tensogram", range_threshold=0.1)
    small_slice2 = ds2["object_0"][2:4, 10:20].values
    np.testing.assert_array_equal(small_slice2, data[2:4, 10:20])
    print("   Custom threshold=0.1: values still correct (full decode fallback)")
    print()


def main():
    print("tensogram-xarray examples\n" + "=" * 40 + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        example_basic(tmp_path)
        example_coords(tmp_path)
        example_variable_key(tmp_path)
        example_dim_names(tmp_path)
        example_multi_message(tmp_path)
        example_lazy_and_threshold(tmp_path)

    print("All examples passed.")


if __name__ == "__main__":
    main()
