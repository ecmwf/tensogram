# tensogram-netcdf

NetCDF to Tensogram format converter.

Converts NetCDF-3/4 files to Tensogram with CF metadata lifting.

## Requirements

Requires netCDF C library (e.g. `apt install libnetcdf-dev` or `brew install netcdf`).

## Usage

```rust
use tensogram_netcdf::{convert_netcdf, NetcdfConvertOptions};

convert_netcdf("data.nc", "data.tgm", &NetcdfConvertOptions::default())?;
```

## Installation

```toml
[dependencies]
tensogram-netcdf = "0.14"
```

## CLI

```bash
cargo install tensogram-cli --features netcdf
tensogram convert-netcdf --cf --compression zstd data.nc -o data.tgm
```

## Documentation

- Full docs: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.
