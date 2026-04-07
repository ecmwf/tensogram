#!/usr/bin/env python3
"""
Generate NetCDF test fixtures for tensogram-netcdf integration tests.

Generated with: netCDF4==1.7.1, numpy==1.26.4, cftime==1.6.4

Run this script to regenerate all fixtures:
    cd crates/tensogram-netcdf/testdata
    python generate.py

Fixtures are committed as binary artifacts. This script exists for reproducibility only.
CI does NOT run this script.
"""

import argparse
import os

import numpy as np
import netCDF4 as nc


def output_dir(args):
    if args.output_dir:
        return args.output_dir
    return os.path.dirname(os.path.abspath(__file__))


def make_simple_2d(outdir):
    """NETCDF3_CLASSIC, 1 var data(y,x) float64, no special attrs."""
    path = os.path.join(outdir, "simple_2d.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("y", 5)
        ds.createDimension("x", 4)
        v = ds.createVariable("data", "f8", ("y", "x"))
        v.set_auto_mask(False)
        v[:] = rng.standard_normal((5, 4))


def make_cf_temperature(outdir):
    """NETCDF4, CF-compliant temperature variable, packed int16 with CF attrs."""
    path = os.path.join(outdir, "cf_temperature.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.8"
        ds.title = "CF temperature fixture"
        ds.institution = "Tensogram tests"

        ds.createDimension("time", 3)
        ds.createDimension("lat", 5)
        ds.createDimension("lon", 4)

        lat = ds.createVariable("lat", "f4", ("lat",))
        lat.set_auto_mask(False)
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat.axis = "Y"
        lat[:] = np.linspace(-90, 90, 5, dtype=np.float32)

        lon = ds.createVariable("lon", "f4", ("lon",))
        lon.set_auto_mask(False)
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon.axis = "X"
        lon[:] = np.linspace(-180, 180, 4, dtype=np.float32)

        time = ds.createVariable("time", "f8", ("time",))
        time.set_auto_mask(False)
        time.units = "days since 1970-01-01"
        time.calendar = "gregorian"
        time.standard_name = "time"
        time.axis = "T"
        time[:] = [18262.0, 18263.0, 18264.0]

        temp = ds.createVariable(
            "temperature", "i2", ("time", "lat", "lon"), fill_value=-32768
        )
        temp.set_auto_mask(False)
        temp.standard_name = "air_temperature"
        temp.long_name = "Air Temperature"
        temp.units = "K"
        temp.scale_factor = np.float64(0.01)
        temp.add_offset = np.float64(273.15)
        raw = rng.integers(-10000, 10000, (3, 5, 4), dtype=np.int16)
        raw[0, 0, 0] = -32768
        temp[:] = raw


def make_multi_var(outdir):
    """NETCDF4, multiple vars sharing dims including a char/string var."""
    path = os.path.join(outdir, "multi_var.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("y", 5)
        ds.createDimension("x", 4)
        ds.createDimension("name_len", 8)

        for name in ("temperature", "humidity", "pressure"):
            v = ds.createVariable(name, "f4", ("y", "x"))
            v.set_auto_mask(False)
            v[:] = rng.standard_normal((5, 4)).astype(np.float32)

        desc = ds.createVariable("description", "S1", ("name_len",))
        desc.set_auto_mask(False)
        desc[:] = np.array([c.encode("ascii") for c in "fixture1"], dtype="S1")


def make_multi_dtype(outdir):
    """NETCDF4, one variable per supported dtype, a scalar, and a NaN-containing var."""
    path = os.path.join(outdir, "multi_dtype.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("n", 6)

        dtype_map = [
            ("i8", "i1"),
            ("i16", "i2"),
            ("i32", "i4"),
            ("i64", "i8"),
            ("u8", "u1"),
            ("u16", "u2"),
            ("u32", "u4"),
            ("u64", "u8"),
            ("f32", "f4"),
            ("f64", "f8"),
        ]
        for name, nc_type in dtype_map:
            v = ds.createVariable(name, nc_type, ("n",))
            v.set_auto_mask(False)
            if nc_type.startswith("u"):
                v[:] = rng.integers(0, 100, 6)
            elif nc_type in ("f4", "f8"):
                v[:] = rng.standard_normal(6)
            else:
                v[:] = rng.integers(-50, 50, 6)

        pi = ds.createVariable("pi", "f8")
        pi.set_auto_mask(False)
        pi[:] = np.float64(3.141592653589793)

        f64_nan = ds.createVariable("f64_with_nan", "f8", ("n",))
        f64_nan.set_auto_mask(False)
        data = rng.standard_normal(6)
        data[2] = np.nan
        f64_nan[:] = data


def make_unlimited_time(outdir):
    """NETCDF4, unlimited time dim, temp(time,y,x) + static mask(y,x)."""
    path = os.path.join(outdir, "unlimited_time.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("y", 4)
        ds.createDimension("x", 3)

        time = ds.createVariable("time", "f8", ("time",))
        time.set_auto_mask(False)
        time.units = "days since 1970-01-01"
        time[:] = [18260.0, 18261.0, 18262.0, 18263.0, 18264.0]

        temp = ds.createVariable("temp", "f4", ("time", "y", "x"))
        temp.set_auto_mask(False)
        temp[:] = rng.standard_normal((5, 4, 3)).astype(np.float32)

        mask = ds.createVariable("mask", "i1", ("y", "x"))
        mask.set_auto_mask(False)
        mask[:] = rng.integers(0, 2, (4, 3), dtype=np.int8)


def make_nc4_groups(outdir):
    """NETCDF4, root group var + sub-group var (for sub-group warning test)."""
    path = os.path.join(outdir, "nc4_groups.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("n", 5)

        root_var = ds.createVariable("root_var", "f8", ("n",))
        root_var.set_auto_mask(False)
        root_var[:] = rng.standard_normal(5)

        grp = ds.createGroup("forecast")
        grp.createDimension("n", 5)
        predicted = grp.createVariable("predicted", "f8", ("n",))
        predicted.set_auto_mask(False)
        predicted[:] = rng.standard_normal(5)


def make_nc3_classic(outdir):
    """NETCDF3_CLASSIC, float32 temperature variable."""
    path = os.path.join(outdir, "nc3_classic.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("y", 5)
        ds.createDimension("x", 4)
        temp = ds.createVariable("temperature", "f4", ("y", "x"))
        temp.set_auto_mask(False)
        temp[:] = rng.standard_normal((5, 4)).astype(np.float32)


def make_empty_file(outdir):
    """NETCDF3_CLASSIC, only global attributes, zero variables."""
    path = os.path.join(outdir, "empty_file.nc")
    with nc.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.title = "Empty fixture"
        ds.institution = "Tensogram tests"
        ds.source = "generate.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write fixtures (default: script directory)",
    )
    args = parser.parse_args()
    outdir = output_dir(args)
    os.makedirs(outdir, exist_ok=True)

    fixtures = [
        ("simple_2d.nc", make_simple_2d),
        ("cf_temperature.nc", make_cf_temperature),
        ("multi_var.nc", make_multi_var),
        ("multi_dtype.nc", make_multi_dtype),
        ("unlimited_time.nc", make_unlimited_time),
        ("nc4_groups.nc", make_nc4_groups),
        ("nc3_classic.nc", make_nc3_classic),
        ("empty_file.nc", make_empty_file),
    ]

    for name, fn in fixtures:
        fn(outdir)
        size = os.path.getsize(os.path.join(outdir, name))
        print(f"  {name}: {size:,} bytes")

    print(f"\nGenerated {len(fixtures)} fixtures in {outdir}")


if __name__ == "__main__":
    main()
