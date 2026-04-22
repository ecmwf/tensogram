# tensogram-szip

Pure-Rust implementation of the CCSDS 121.0-B-3 Adaptive Entropy
Coding (AEC / SZIP) codec.

This crate provides `aec_compress`, `aec_decompress`, and
`aec_decompress_range` functions with the same `AecParams` interface
as the C libaec library. It is useful as a drop-in replacement for
`libaec-sys` in environments where C FFI is unavailable — notably
WebAssembly.

## Usage

```rust
use tensogram_szip::{aec_compress, aec_decompress, AecParams, AEC_DATA_PREPROCESS};

let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
let params = AecParams {
    bits_per_sample: 8,
    block_size: 16,
    rsi: 128,
    flags: AEC_DATA_PREPROCESS,
};

let (compressed, rsi_block_offsets) = aec_compress(&data, &params)?;
let decompressed = aec_decompress(&compressed, data.len(), &params)?;
assert_eq!(decompressed, data);
# Ok::<(), tensogram_szip::AecError>(())
```

`rsi_block_offsets` enables `aec_decompress_range` — partial decode
without decompressing the full stream.

## Installation

```bash
cargo add tensogram-szip
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.
