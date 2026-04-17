# tensogram-core

The primary Tensogram library for fast binary N-tensor message format.

This crate provides encode, decode, file I/O, and streaming capabilities for scientific data. It supports self-describing messages with CBOR metadata, multiple tensors per message, partial decode, and remote access to cloud storage.

## Usage

```rust
use tensogram_core::{encode, decode, EncodeOptions, DecodeOptions};

let message = encode(&meta, &[(&desc, &data)], &EncodeOptions::default())?;
let (_, objects) = decode(&message, &DecodeOptions::default())?;
```

```rust
// File API with partial decode
use tensogram_core::{TensogramFile, decode_range};

let file = TensogramFile::open("data.tgm")?;
let slice = decode_range(&file, 0, &[0..100, 50..150])?;
```

## Installation

```toml
[dependencies]
tensogram-core = { version = "0.14", features = ["mmap", "async", "remote"] }
```

## Documentation

- Full documentation: https://sites.ecmwf.int/docs/tensogram/main/
- Repository: https://github.com/ecmwf/tensogram

## License

Copyright 2026- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the Apache License, Version 2.0. See LICENSE for details.
