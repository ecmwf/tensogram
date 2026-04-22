# tensogram-encodings

Encoding pipeline and compression codec registry for the Tensogram
message format.

Most callers should depend on the top-level [`tensogram`](https://crates.io/crates/tensogram)
crate, which re-exports the high-level `encode` / `decode` API.
This crate is useful when you need to call the encoding primitives
directly (for example to integrate with a pipeline other than
Tensogram's framing), or to pick a specific codec backend via Cargo
features.

## Usage — simple_packing quantisation

```rust
use tensogram_encodings::simple_packing;

let values: Vec<f64> = (0..1000).map(|i| 273.15 + i as f64 / 100.0).collect();
let params = simple_packing::compute_params(&values, 16, 0)?;
let packed = simple_packing::encode(&values, &params)?;
let round_trip = simple_packing::decode(&packed, values.len(), &params)?;
# Ok::<(), tensogram_encodings::simple_packing::PackingError>(())
```

## Features

- `simple_packing`: GRIB-style lossy quantization (0–64 bits per value)
- `shuffle`: HDF5-style byte-level shuffle filter
- `szip`: CCSDS 121.0-B-3 compression (C FFI via libaec)
- `szip-pure`: pure-Rust szip (e.g. for WebAssembly)
- `zstd` / `zstd-pure`: Zstandard compression (C FFI or ruzstd)
- `lz4`: LZ4 fast compression (pure Rust)
- `blosc2`: Blosc2 multi-codec with chunk random access
- `zfp`: ZFP lossy floating-point compression (C FFI)
- `sz3`: SZ3 error-bounded lossy compression (clean-room shim)

All compression codecs default to on. Opt out selectively — for a
WebAssembly-friendly build use
`--no-default-features --features szip-pure,zstd-pure,lz4`.

## Installation

```bash
cargo add tensogram-encodings
```

## Documentation

- Full documentation: <https://sites.ecmwf.int/docs/tensogram/main/>
- Repository: <https://github.com/ecmwf/tensogram>

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.
