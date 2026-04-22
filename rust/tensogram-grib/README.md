# tensogram-grib

GRIB importer for Tensogram, built on ecCodes.

Imports GRIB files into Tensogram messages while preserving MARS-namespace
metadata. Conversion is one-way (GRIB → Tensogram). GRIB is widely used in
operational weather forecasting; this crate lets you bring existing GRIB data
into Tensogram pipelines.

## Requirements

Requires the ecCodes C library (e.g. `apt install libeccodes-dev` or
`brew install eccodes`).

## Usage

```rust,ignore
use std::path::Path;
use tensogram_grib::{convert_grib_file, ConvertOptions};

let messages = convert_grib_file(
    Path::new("forecast.grib"),
    &ConvertOptions::default(),
)?;
// messages: Vec<Vec<u8>> — each element is a complete Tensogram wire-format message.
```

## Installation

```bash
cargo add tensogram-grib
```

## CLI

```bash
cargo install tensogram-cli --features grib
tensogram convert-grib forecast.grib -o forecast.tgm
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- ECMWF. Licensed under Apache-2.0. See LICENSE.
