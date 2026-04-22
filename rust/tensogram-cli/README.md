# tensogram-cli

Command-line tool for Tensogram `.tgm` files.

Inspect, validate, manipulate, and convert Tensogram files from the
command line. Subcommands include `info`, `ls`, `dump`, `get`, `set`,
`copy`, `merge`, `split`, `reshuffle`, `validate`, and (feature-gated)
`convert-grib` / `convert-netcdf`.

## Installation

```bash
cargo install tensogram-cli
```

The binary is installed as `tensogram`.

To enable GRIB and NetCDF conversion (requires ecCodes / libnetcdf at
the OS level):

```bash
cargo install tensogram-cli --features grib,netcdf
```

## Usage

```bash
# Inspect a file
tensogram info data.tgm
tensogram ls data.tgm -w "mars.param=2t"
tensogram dump data.tgm -j

# Convert GRIB / NetCDF
tensogram convert-grib forecast.grib -o forecast.tgm
tensogram convert-netcdf --cf data.nc -o data.tgm

# Validate
tensogram validate data.tgm --full
```

Global flags such as `--threads`, `--allow-nan`, and `--allow-inf`
apply to every subcommand that encodes data.

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.
