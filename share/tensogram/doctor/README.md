# tensogram doctor — sanity fixtures

This directory contains three small binary fixtures used by `tensogram doctor`
to self-test the GRIB and NetCDF converter pipelines.

| File | Format | Content |
|---|---|---|
| `sanity.grib2` | GRIB2 | 4×4 regular_ll grid, MARS param `2t`, simple_packing 16-bit |
| `sanity-classic.nc` | NetCDF-3 classic | `temperature(lat=2, lon=2)` f32 |
| `sanity-hdf5.nc` | NetCDF-4/HDF5 | `temperature(lat=2, lon=2)` f32 |

## Why committed binary blobs?

The fixtures are embedded in the `tensogram` binary via `include_bytes!` so
that `tensogram doctor` works without any external files.  Committing them as
binary blobs keeps the self-test hermetic and avoids a runtime dependency on
eccodes or netCDF4 Python packages.

## When to regenerate

Regeneration is rarely needed — only when:

- The fixture format changes (e.g. different grid size or variable name).
- A new converter is added that needs a new fixture type.
- The existing fixtures are found to be corrupt.

## How to regenerate

```bash
pip install eccodes netCDF4 numpy
python3 share/tensogram/doctor/regenerate.py
```

The script is deterministic: running it twice produces byte-identical files.
After regenerating, verify the magic bytes and commit the updated binaries.
