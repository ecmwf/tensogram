#!/usr/bin/env python3
# (C) Copyright 2026- ECMWF and individual contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""
12 — convert-netcdf via the CLI

End-to-end example: build a small NetCDF file with netCDF4, run
`tensogram convert-netcdf` on it through subprocess, then read the
resulting .tgm file back with the tensogram Python bindings.

The Python bindings do NOT expose `convert_netcdf_file()` directly in
v1 — the intended pattern is to shell out to the CLI binary, which
does support all the conversion flags. This example demonstrates that
exact pattern.

Run:
    cargo build -p tensogram-cli --features netcdf
    python examples/python/12_convert_netcdf.py

Requires:
    - netCDF4 (`uv pip install netCDF4`)
    - libnetcdf installed at the OS level (brew install netcdf / apt install libnetcdf-dev)
    - The `tensogram` CLI binary built with the `netcdf` feature
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import netCDF4
    import numpy as np
    import tensogram
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print(
        "Install with: uv pip install netCDF4 numpy && (cd crates/tensogram-python && maturin develop)",
        file=sys.stderr,
    )
    sys.exit(1)


def find_tensogram_binary() -> str:
    """Locate the tensogram binary, preferring the workspace target dir."""
    # Walk up from this script to find the workspace root.
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent  # examples/python/ -> repo root
    candidates = [
        repo_root / "target" / "debug" / "tensogram",
        repo_root / "target" / "release" / "tensogram",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    # Fall back to PATH lookup.
    found = shutil.which("tensogram")
    if found:
        return found
    print(
        "ERROR: tensogram binary not found. Run:\n"
        "    cargo build -p tensogram-cli --features netcdf\n",
        file=sys.stderr,
    )
    sys.exit(1)


def write_demo_netcdf(path: Path) -> None:
    """Create a small CF-compliant NetCDF file with one packed temperature
    variable, two coordinate variables, and a time axis."""
    rng = np.random.default_rng(seed=12345)
    nlat, nlon, ntime = 10, 16, 4

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.Conventions = "CF-1.10"
        ds.title = "Tensogram convert-netcdf example"
        ds.institution = "tensogram examples"

        ds.createDimension("time", ntime)
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)

        lat = ds.createVariable("lat", "f4", ("lat",))
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat.axis = "Y"
        lat[:] = np.linspace(-90.0, 90.0, nlat, dtype=np.float32)

        lon = ds.createVariable("lon", "f4", ("lon",))
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon.axis = "X"
        lon[:] = np.linspace(0.0, 359.0, nlon, dtype=np.float32)

        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01"
        time.calendar = "gregorian"
        time.standard_name = "time"
        time.axis = "T"
        time[:] = np.arange(ntime, dtype=np.float64)

        # Packed temperature: int16 + scale_factor / add_offset.
        temp = ds.createVariable(
            "temperature",
            "i2",
            ("time", "lat", "lon"),
            fill_value=-32768,
        )
        temp.units = "K"
        temp.standard_name = "air_temperature"
        temp.long_name = "2 metre temperature"
        temp.scale_factor = 0.01
        temp.add_offset = 273.15
        temp.cell_methods = "time: mean"
        # Random values in plausible Kelvin range, packed as int16.
        kelvin = rng.uniform(240.0, 310.0, size=(ntime, nlat, nlon))
        packed = ((kelvin - 273.15) / 0.01).astype(np.int16)
        temp[:] = packed


def main() -> int:
    binary = find_tensogram_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        nc_path = tmp / "demo.nc"
        tgm_path = tmp / "demo.tgm"

        print(f"1. Writing demo NetCDF to {nc_path}")
        write_demo_netcdf(nc_path)
        print(f"   ({nc_path.stat().st_size} bytes)")

        print()
        print("2. Running: tensogram convert-netcdf --cf --compression zstd")
        result = subprocess.run(
            [
                binary,
                "convert-netcdf",
                "--cf",
                "--compression",
                "zstd",
                str(nc_path),
                "-o",
                str(tgm_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print("   FAILED", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        print(f"   {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"   stderr: {result.stderr.strip()}")
        print(f"   ({tgm_path.stat().st_size} bytes)")

        print()
        print("3. Reading the .tgm with tensogram Python bindings")
        with tensogram.TensogramFile.open(str(tgm_path)) as f:
            count = len(f)
            print(f"   message_count = {count}")

            for idx, (meta, objects) in enumerate(f):
                print(f"   message[{idx}]: {len(objects)} object(s)")
                for obj_idx, (desc, array) in enumerate(objects):
                    name_value = (
                        meta.base[obj_idx].get("name", "?")
                        if obj_idx < len(meta.base)
                        else "?"
                    )
                    cf_keys = []
                    if obj_idx < len(meta.base):
                        cf = meta.base[obj_idx].get("cf")
                        if isinstance(cf, dict):
                            cf_keys = sorted(cf.keys())
                    print(
                        f"     object[{obj_idx}] name={name_value!r:20} "
                        f"shape={list(desc.shape)} dtype={desc.dtype} "
                        f"compression={desc.compression}"
                    )
                    if cf_keys:
                        print(f"       cf attrs: {cf_keys}")

    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
