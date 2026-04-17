# tensogram-encodings

Encoding pipeline and compression codec registry for the Tensogram message format.

This crate provides the encoding pipeline infrastructure and supports multiple compression codecs through feature flags: szip, zstd, lz4, blosc2, zfp, and sz3. It serves as the encoding foundation for Tensogram messages.

## Usage

```toml
[dependencies]
tensogram-encodings = { version = "0.1", features = ["szip", "zstd"] }
```

```rust
use tensogram_encodings::{CodecRegistry, CompressionCodec};

let registry = CodecRegistry::default();
let compressed = registry.encode("szip", &data, config)?;
```

## Features

- `szip`: CCSDS Adaptive Entropy Coding (pure-Rust)
- `zstd`: Zstandard compression
- `lz4`: LZ4 fast compression
- `blosc2`: Blosc2 compressor
- `zfp`: ZFP floating-point compression
- `sz3`: SZ3 lossy compression

## Documentation

- Full documentation: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.
