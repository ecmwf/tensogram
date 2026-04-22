# tensogram-netcdf

NetCDF importer for Tensogram.

Imports NetCDF-3 and NetCDF-4 files into Tensogram messages, with
optional CF-convention metadata lifting. Conversion is one-way
(NetCDF → Tensogram). NetCDF is widely used in climate, ocean,
atmospheric, and Earth-observation science; this crate brings those
datasets into Tensogram pipelines.

## Requirements

Requires the netCDF C library (e.g. `apt install libnetcdf-dev` or
`brew install netcdf`).

## Usage

```rust,ignore
use std::path::Path;
use tensogram_netcdf::{convert_netcdf_file, ConvertOptions};

let messages = convert_netcdf_file(
    Path::new("data.nc"),
    &ConvertOptions::default(),
)?;
// messages: Vec<Vec<u8>> — one complete Tensogram message per split.
```

## Installation

```bash
cargo add tensogram-netcdf
```

## CLI

```bash
cargo install tensogram-cli --features netcdf
tensogram convert-netcdf --cf --compression zstd data.nc -o data.tgm
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.
