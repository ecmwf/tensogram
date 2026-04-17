# tensogram-cli

CLI for Tensogram .tgm files.

Inspect, convert, and manipulate Tensogram files. Supports info, ls, dump, get/set, copy, merge, split, reshuffle, and GRIB/NetCDF conversion.

## Installation

```bash
cargo install tensogram-cli
```

Binary is installed as `tensogram`.

## Usage

```bash
# Inspect
tensogram info data.tgm

# Convert GRIB
tensogram convert-grib forecast.grib -o forecast.tgm

# Convert NetCDF
tensogram convert-netcdf --cf data.nc -o data.tgm

# Validate
tensogram validate data.tgm --full
```

## Documentation

- Full docs: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under Apache-2.0. See LICENSE for details.
