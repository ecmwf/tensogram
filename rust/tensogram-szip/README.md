# tensogram-szip

Pure-Rust implementation of CCSDS 121.0-B-3 Adaptive Entropy Coding (AEC/SZIP).

This crate provides encode, decode, and range decode operations with no C dependencies. It implements the CCSDS Lossless Data Compression standard used in space missions and scientific data compression.

## Usage

```rust
use tensogram_szip::{AecEncoder, AecDecoder};

let compressed = AecEncoder::encode(&data, width, bits_per_pixel)?;
let decompressed = AecDecoder::decode(&compressed, width, height)?;
```

## Installation

```toml
[dependencies]
tensogram-szip = "0.14"
```

## Documentation

- Full documentation: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.
