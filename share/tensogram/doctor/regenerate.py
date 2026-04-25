#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Generate deterministic binary fixtures for the tensogram doctor self-test.

Produces three files in the same directory as this script:

  sanity.grib2          GRIB2, 4×4 regular_ll grid, MARS param 2t,
                        simple_packing 16-bit, ~500 B.

  sanity-classic.nc     NetCDF-3 classic, temperature(lat=2, lon=2) f32, ~1 KB.

  sanity-hdf5.nc        NetCDF-4/HDF5, temperature(lat=2, lon=2) f32, ~4 KB.

Run this script whenever the fixture format needs to change.  The output is
byte-deterministic: running it twice produces identical files.

Requirements:
  pip install eccodes netCDF4 numpy
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ── GRIB2 fixture ─────────────────────────────────────────────────────────────


def write_grib2() -> None:
    """Write sanity.grib2 using the eccodes Python bindings."""
    try:
        import eccodes
    except ImportError:
        raise SystemExit(
            "eccodes Python package not found.  Install with: pip install eccodes"
        )

    path = os.path.join(HERE, "sanity.grib2")

    # 4×4 grid of 2-metre temperature values (deterministic, no randomness)
    values = np.array(
        [
            273.15,
            274.15,
            275.15,
            276.15,
            277.15,
            278.15,
            279.15,
            280.15,
            281.15,
            282.15,
            283.15,
            284.15,
            285.15,
            286.15,
            287.15,
            288.15,
        ],
        dtype=np.float64,
    )

    sample_id = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
    try:
        # Grid geometry: 4×4 regular lat/lon
        eccodes.codes_set(sample_id, "Ni", 4)
        eccodes.codes_set(sample_id, "Nj", 4)
        eccodes.codes_set(sample_id, "latitudeOfFirstGridPointInDegrees", 0.0)
        eccodes.codes_set(sample_id, "longitudeOfFirstGridPointInDegrees", 0.0)
        eccodes.codes_set(sample_id, "latitudeOfLastGridPointInDegrees", 3.0)
        eccodes.codes_set(sample_id, "longitudeOfLastGridPointInDegrees", 3.0)
        eccodes.codes_set(sample_id, "iDirectionIncrementInDegrees", 1.0)
        eccodes.codes_set(sample_id, "jDirectionIncrementInDegrees", 1.0)

        # MARS keys for 2-metre temperature
        eccodes.codes_set(sample_id, "shortName", "2t")
        eccodes.codes_set(sample_id, "typeOfLevel", "heightAboveGround")
        eccodes.codes_set(sample_id, "level", 2)

        # Packing: simple_packing, 16 bits per value
        eccodes.codes_set(sample_id, "packingType", "grid_simple")
        eccodes.codes_set(sample_id, "bitsPerValue", 16)

        # Deterministic date/time (fixed, not wall-clock)
        eccodes.codes_set(sample_id, "dataDate", 20260101)
        eccodes.codes_set(sample_id, "dataTime", 0)
        eccodes.codes_set(sample_id, "stepRange", "0")

        eccodes.codes_set_values(sample_id, values)

        with open(path, "wb") as fh:
            eccodes.codes_write(sample_id, fh)
    finally:
        eccodes.codes_release(sample_id)

    # Verify magic bytes
    with open(path, "rb") as fh:
        magic = fh.read(4)
    assert magic == b"GRIB", f"unexpected magic bytes: {magic!r}"
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


# ── NetCDF fixtures ───────────────────────────────────────────────────────────


def _write_netcdf(path: str, fmt: str) -> None:
    """Write a minimal NetCDF file with a 2×2 f32 temperature variable."""
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise SystemExit(
            "netCDF4 Python package not found.  Install with: pip install netCDF4"
        )

    lat = np.array([0.0, 1.0], dtype=np.float32)
    lon = np.array([0.0, 1.0], dtype=np.float32)
    temperature = np.array([[273.15, 274.15], [275.15, 276.15]], dtype=np.float32)

    with Dataset(path, "w", format=fmt) as ds:
        ds.Conventions = "CF-1.8"
        ds.history = "Created by tensogram doctor regenerate.py"

        ds.createDimension("lat", 2)
        ds.createDimension("lon", 2)

        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lat_var.units = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var[:] = lat

        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lon_var.units = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var[:] = lon

        temp_var = ds.createVariable("temperature", "f4", ("lat", "lon"))
        temp_var.units = "K"
        temp_var.standard_name = "air_temperature"
        temp_var.long_name = "2-metre temperature"
        temp_var[:] = temperature

    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def write_netcdf_classic() -> None:
    path = os.path.join(HERE, "sanity-classic.nc")
    _write_netcdf(path, "NETCDF3_CLASSIC")
    with open(path, "rb") as fh:
        magic = fh.read(4)
    assert magic == b"CDF\x01", f"unexpected magic bytes: {magic!r}"


def write_netcdf_hdf5() -> None:
    path = os.path.join(HERE, "sanity-hdf5.nc")
    _write_netcdf(path, "NETCDF4")
    with open(path, "rb") as fh:
        magic = fh.read(8)
    assert magic[:4] == b"\x89HDF", f"unexpected magic bytes: {magic!r}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    write_grib2()
    write_netcdf_classic()
    write_netcdf_hdf5()
    print("all fixtures written successfully")
