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


def make_attr_type_variants(outdir):
    """NETCDF4, several variables each using a different numeric type
    for the scale_factor / add_offset / _FillValue attributes so the
    converter's get_f64_attr helper hits every numeric arm."""
    path = os.path.join(outdir, "attr_type_variants.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("n", 4)

        # Float (f32) scale_factor → hits AttributeValue::Float arm.
        v_float = ds.createVariable(
            "scaled_float", "i2", ("n",), fill_value=np.int16(-9999)
        )
        v_float.set_auto_mask(False)
        v_float.setncattr("scale_factor", np.float32(0.5))
        v_float.setncattr("add_offset", np.float32(10.0))
        v_float[:] = rng.integers(-100, 100, 4, dtype=np.int16)

        # Int (i32) scale_factor → hits AttributeValue::Int arm.
        v_int = ds.createVariable(
            "scaled_int", "i2", ("n",), fill_value=np.int16(-9999)
        )
        v_int.set_auto_mask(False)
        v_int.setncattr("scale_factor", np.int32(2))
        v_int.setncattr("add_offset", np.int32(100))
        v_int[:] = rng.integers(-100, 100, 4, dtype=np.int16)

        # Short (i16) scale_factor → hits AttributeValue::Short arm.
        v_short = ds.createVariable(
            "scaled_short", "i2", ("n",), fill_value=np.int16(-9999)
        )
        v_short.set_auto_mask(False)
        v_short.setncattr("scale_factor", np.int16(3))
        v_short.setncattr("add_offset", np.int16(50))
        v_short[:] = rng.integers(-100, 100, 4, dtype=np.int16)

        # Longlong (i64) scale_factor → hits AttributeValue::Longlong arm.
        v_longlong = ds.createVariable(
            "scaled_longlong", "i2", ("n",), fill_value=np.int16(-9999)
        )
        v_longlong.set_auto_mask(False)
        v_longlong.setncattr("scale_factor", np.int64(4))
        v_longlong.setncattr("add_offset", np.int64(200))
        v_longlong[:] = rng.integers(-100, 100, 4, dtype=np.int16)

        # A variable with an explicit `missing_value` attribute of a
        # different type than _FillValue — exercises the
        # `missing_value` → `_FillValue` fallback chain in
        # read_and_unpack + covers the "at least one NaN" case.
        v_missing = ds.createVariable("with_missing", "i2", ("n",))
        v_missing.set_auto_mask(False)
        v_missing.setncattr("scale_factor", np.float64(1.0))
        v_missing.setncattr("missing_value", np.int16(-1))
        data = np.array([5, -1, 10, -1], dtype=np.int16)
        v_missing[:] = data

        # Variable with a non-numeric scale_factor attribute
        # (a Str value). get_f64_attr should return None and
        # read_and_unpack should still succeed as a raw read.
        # This covers the `_ => None` fallback arm in get_f64_attr.
        # We disable auto-scale so netCDF4 doesn't try to divide
        # the int values by the string attribute at write time.
        v_strscale = ds.createVariable("string_scale", "i2", ("n",))
        v_strscale.set_auto_mask(False)
        v_strscale.set_auto_scale(False)
        v_strscale[:] = np.array([1, 2, 3, 4], dtype=np.int16)
        v_strscale.setncattr("scale_factor", "non_numeric")


def make_empty_unlimited(outdir):
    """NETCDF4, an unlimited dim with zero records. Exercises the
    `record_count == 0` early-return in encode_by_record."""
    path = os.path.join(outdir, "empty_unlimited.nc")
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("n", 3)
        v = ds.createVariable("temp", "f4", ("time", "n"))
        v.set_auto_mask(False)
        # Don't write any values — size stays at 0.


def make_complex_types(outdir):
    """NETCDF4, one variable of each complex user-defined type
    (Enum/Compound/Vlen) alongside a normal numeric variable.
    Exercises the complex-type rejection arm in extract_variable."""
    path = os.path.join(outdir, "complex_types.nc")
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("n", 4)

        # A normal variable so the conversion overall succeeds
        # (otherwise we'd hit NoVariables after skipping everything).
        normal = ds.createVariable("value", "f4", ("n",))
        normal.set_auto_mask(False)
        normal[:] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        # Enum type variable — triggers NcVariableType::Enum(_) rejection.
        enum_t = ds.createEnumType(
            np.int8, "status_enum", {"ok": 0, "warn": 1, "fail": 2}
        )
        enum_var = ds.createVariable("status", enum_t, ("n",))
        enum_var.set_auto_mask(False)
        enum_var[:] = np.array([0, 1, 2, 0], dtype=np.int8)


def make_complex_types_unlimited(outdir):
    """NETCDF4, unlimited dim with both a regular numeric variable
    and an enum-typed variable along the same dim. Record-split
    converts the numeric variable and skips the enum variable via
    the complex-type rejection inside extract_variable_record.

    Also carries global attributes so the `_global` insertion in
    extract_variable_record is exercised."""
    path = os.path.join(outdir, "complex_types_unlimited.nc")
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "Complex-type unlimited fixture"
        ds.institution = "Tensogram tests"

        ds.createDimension("time", None)
        ds.createDimension("n", 3)

        v = ds.createVariable("value", "f4", ("time", "n"))
        v.set_auto_mask(False)
        v[:] = np.arange(6, dtype=np.float32).reshape(2, 3)

        enum_t = ds.createEnumType(np.int8, "state_enum", {"on": 0, "off": 1})
        e = ds.createVariable("state", enum_t, ("time", "n"))
        e.set_auto_mask(False)
        e[:] = np.zeros((2, 3), dtype=np.int8)


def make_record_with_char(outdir):
    """NETCDF4, unlimited dim + a char variable sharing the dim.
    Converting with --split-by=record exercises the char-rejection
    branch inside extract_variable_record."""
    path = os.path.join(outdir, "record_with_char.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("strlen", 4)
        ds.createDimension("n", 3)

        t = ds.createVariable("time", "f8", ("time",))
        t.set_auto_mask(False)
        t[:] = [0.0, 1.0]

        v = ds.createVariable("values", "f4", ("time", "n"))
        v.set_auto_mask(False)
        v[:] = rng.standard_normal((2, 3)).astype(np.float32)

        # Char variable along the unlimited dimension — triggers the
        # Char / String rejection path inside extract_variable_record.
        labels = ds.createVariable("labels", "S1", ("time", "strlen"))
        labels.set_auto_mask(False)
        labels[:] = (
            np.array([list("a"), list("b")], dtype="S1")
            .reshape((2, 1))
            .repeat(4, axis=1)
        )


def make_record_multi_dtype(outdir):
    """NETCDF4, unlimited dimension with one variable per supported
    numeric dtype — exercises every read_native_extents branch when
    converting with --split-by=record."""
    path = os.path.join(outdir, "record_multi_dtype.nc")
    rng = np.random.default_rng(42)
    with nc.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("n", 3)

        # Time coordinate.
        time = ds.createVariable("time", "f8", ("time",))
        time.set_auto_mask(False)
        time.units = "days since 2020-01-01"
        time[:] = [0.0, 1.0, 2.0]

        # One variable per numeric dtype along the unlimited dim.
        # Attribute types are also varied so get_f64_attr hits more arms.
        def mk(name, nc_dtype, np_dtype, scale_attr_type=None):
            v = ds.createVariable(name, nc_dtype, ("time", "n"))
            v.set_auto_mask(False)
            v[:] = rng.integers(0, 10, (3, 3)).astype(np_dtype)
            if scale_attr_type is not None:
                # Attach a numeric attribute in a specific type so
                # the converter's get_f64_attr hits non-Double arms.
                v.setncattr("scale_hint", scale_attr_type(1))
            return v

        mk("v_i8", "i1", np.int8, scale_attr_type=np.int16)
        mk("v_u8", "u1", np.uint8)
        mk("v_i16", "i2", np.int16, scale_attr_type=np.int32)
        mk("v_u16", "u2", np.uint16)
        mk("v_i32", "i4", np.int32, scale_attr_type=np.int64)
        mk("v_u32", "u4", np.uint32)
        mk("v_i64", "i8", np.int64)
        mk("v_u64", "u8", np.uint64)
        mk("v_f32", "f4", np.float32, scale_attr_type=np.float32)
        mk("v_f64", "f8", np.float64)


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
        ("record_multi_dtype.nc", make_record_multi_dtype),
        ("attr_type_variants.nc", make_attr_type_variants),
        ("empty_unlimited.nc", make_empty_unlimited),
        ("complex_types.nc", make_complex_types),
        ("complex_types_unlimited.nc", make_complex_types_unlimited),
        ("record_with_char.nc", make_record_with_char),
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
