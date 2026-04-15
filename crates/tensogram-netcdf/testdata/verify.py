#!/usr/bin/env python3
# (C) Copyright 2024- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Verify all test fixtures are present and have expected structure.

8 baseline fixtures (Tasks 6-12) + 6 coverage fixtures (v0.7.0).
"""

import os
import sys
import netCDF4 as nc


def verify(outdir: str) -> bool:
    ok = True

    def check(name: str, fn) -> None:
        nonlocal ok
        path = os.path.join(outdir, name)
        if not os.path.exists(path):
            print(f"MISSING: {name}")
            ok = False
            return
        try:
            fn(path)
            size = os.path.getsize(path)
            print(f"OK: {name} ({size:,} bytes)")
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            ok = False

    def check_simple_2d(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert "data" in ds.variables
            assert ds["data"].shape == (5, 4)

    def check_cf_temperature(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert "temperature" in ds.variables
            assert ds["temperature"].shape == (3, 5, 4)
            assert hasattr(ds["temperature"], "scale_factor")
            assert hasattr(ds["temperature"], "add_offset")
            assert ds.Conventions == "CF-1.8"

    def check_multi_var(path: str) -> None:
        with nc.Dataset(path) as ds:
            for name in ("temperature", "humidity", "pressure", "description"):
                assert name in ds.variables

    def check_multi_dtype(path: str) -> None:
        with nc.Dataset(path) as ds:
            for name in (
                "i8",
                "i16",
                "i32",
                "i64",
                "u8",
                "u16",
                "u32",
                "u64",
                "f32",
                "f64",
                "pi",
                "f64_with_nan",
            ):
                assert name in ds.variables

    def check_unlimited_time(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert "temp" in ds.variables
            assert "mask" in ds.variables
            assert ds["temp"].shape == (5, 4, 3)
            assert ds.dimensions["time"].isunlimited()

    def check_nc4_groups(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert "root_var" in ds.variables
            assert "forecast" in ds.groups

    def check_nc3_classic(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert "temperature" in ds.variables

    def check_empty_file(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert len(ds.variables) == 0
            assert ds.title == "Empty fixture"

    # ── Coverage fixtures (v0.7.0) ───────────────────────────────────

    def check_record_multi_dtype(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert ds.dimensions["time"].isunlimited()
            for name in (
                "v_i8",
                "v_u8",
                "v_i16",
                "v_u16",
                "v_i32",
                "v_u32",
                "v_i64",
                "v_u64",
                "v_f32",
                "v_f64",
            ):
                assert name in ds.variables, f"missing {name}"

    def check_attr_type_variants(path: str) -> None:
        with nc.Dataset(path) as ds:
            # Disable auto-scale so we inspect the raw attribute types.
            for v in ds.variables.values():
                v.set_auto_scale(False)
                v.set_auto_mask(False)
            for name in (
                "scaled_float",
                "scaled_int",
                "scaled_short",
                "scaled_longlong",
                "with_missing",
                "string_scale",
            ):
                assert name in ds.variables

    def check_empty_unlimited(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert ds.dimensions["time"].isunlimited()
            assert ds.dimensions["time"].size == 0

    def check_complex_types(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert "value" in ds.variables
            assert "status" in ds.variables

    def check_complex_types_unlimited(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert ds.dimensions["time"].isunlimited()
            assert "value" in ds.variables
            assert "state" in ds.variables
            assert ds.title == "Complex-type unlimited fixture"

    def check_record_with_char(path: str) -> None:
        with nc.Dataset(path) as ds:
            assert ds.dimensions["time"].isunlimited()
            assert "values" in ds.variables
            assert "labels" in ds.variables

    check("simple_2d.nc", check_simple_2d)
    check("cf_temperature.nc", check_cf_temperature)
    check("multi_var.nc", check_multi_var)
    check("multi_dtype.nc", check_multi_dtype)
    check("unlimited_time.nc", check_unlimited_time)
    check("nc4_groups.nc", check_nc4_groups)
    check("nc3_classic.nc", check_nc3_classic)
    check("empty_file.nc", check_empty_file)
    check("record_multi_dtype.nc", check_record_multi_dtype)
    check("attr_type_variants.nc", check_attr_type_variants)
    check("empty_unlimited.nc", check_empty_unlimited)
    check("complex_types.nc", check_complex_types)
    check("complex_types_unlimited.nc", check_complex_types_unlimited)
    check("record_with_char.nc", check_record_with_char)

    return ok


if __name__ == "__main__":
    outdir = os.path.dirname(os.path.abspath(__file__))
    if verify(outdir):
        print("\nAll fixtures verified.")
        sys.exit(0)
    else:
        print("\nSome fixtures failed verification.", file=sys.stderr)
        sys.exit(1)
