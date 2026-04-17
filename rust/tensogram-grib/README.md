# tensogram-grib

GRIB to Tensogram format converter using ecCodes.

Converts GRIB files to Tensogram format, preserving MARS metadata.

## Requirements

Requires ecCodes C library (e.g. `apt install libeccodes-dev` or `brew install eccodes`).

## Usage

```rust
use tensogram_grib::{convert_grib, GribConvertOptions};

convert_grib("forecast.grib", "forecast.tgm", &GribConvertOptions::default())?;
```

## Installation

```toml
[dependencies]
tensogram-grib = "0.14"
```

## CLI

```bash
cargo install tensogram-cli --features grib
tensogram convert-grib forecast.grib -o forecast.tgm
```

## Documentation

- Full docs: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.
