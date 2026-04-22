# tensogram-sz3

High-level safe Rust API for SZ3 error-bounded lossy compression.

This crate is a clean-room reimplementation of the public surface of
the `sz3` crate (v0.4.3), backed by `tensogram-sz3-sys` instead of
`sz3-sys`. It is suitable for compressing floating-point scientific
arrays with tight error bounds.

## Usage

```rust
use tensogram_sz3::{compress, decompress, DimensionedData, ErrorBound};

let data: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
let dimensioned = DimensionedData::build(&data).dim(1000)?.remainder_dim()?;

let compressed = compress(&dimensioned, ErrorBound::Absolute(1e-4))?;
let (decoded, _config): (Vec<f32>, _) = decompress(&compressed)?;
# Ok::<(), tensogram_sz3::SZ3Error>(())
```

For full control, use `compress_with_config` with a `Config`:

```rust
use tensogram_sz3::{compress_with_config, Config, ErrorBound};

let config = Config::new(ErrorBound::Absolute(1e-4));
# let dimensioned: tensogram_sz3::DimensionedData<f32, &[f32]> =
#     tensogram_sz3::DimensionedData::build(&[][..]).dim(0)?.remainder_dim()?;
let compressed = compress_with_config(&dimensioned, &config)?;
# Ok::<(), tensogram_sz3::SZ3Error>(())
```

## Installation

```bash
cargo add tensogram-sz3
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.
