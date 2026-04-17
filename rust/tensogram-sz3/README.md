# tensogram-sz3

High-level safe Rust API for SZ3 lossy compression.

This crate provides a safe, idiomatic Rust interface to the SZ3 compressor, built on top of `tensogram-sz3-sys`. SZ3 is a high-ratio lossy compression method suitable for scientific floating-point data.

## Usage

```rust
use tensogram_sz3::{Sz3Compressor, Sz3Config};

let config = Sz3Config::default().with_error_bound(1e-4);
let compressed = Sz3Compressor::compress(&data, &config)?;
```

## Installation

```toml
[dependencies]
tensogram-sz3 = "0.14"
```

## Documentation

- Full documentation: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.
