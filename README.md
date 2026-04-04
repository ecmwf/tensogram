# Tensogram

[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#emerging)

> [!IMPORTANT]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

Tensogram is a fast, efficient *'telegram'* for multidimensional tensors. ⏩[NxM]⏩

A library to encode and decode binary N-Tensor scientific data with semantic metadata close to the data, in a serialisable format that can be sent over the network, encoded into in-memory buffers and decoded with zero-copy.

Tensogram defines a network-transmissible binary message format, not a file format. Multiple messages can be appended to a file, each carrying its own begin/terminator codes.

## Features

- **Self-describing messages** — CBOR-encoded metadata with structured `common`/`payload`/`reserved` sections
- **N-Tensor support** — multiple tensors of different dtypes per message
- **No panics** — all fallible operations return `Result<T, TensogramError>`
- **Streaming encoder** — progressive encode/transmit without buffering the full message
- **Compression** — szip, zstd, lz4, blosc2, zfp, sz3 per data object
- **Hash verification** — xxHash xxh3-64 integrity check per object
- **Multiple languages** — Rust, Python (NumPy), C/C++
- **GRIB conversion** — import GRIB data with MARS metadata preservation
- **CLI** — `tensogram info/ls/dump/get/set/copy/merge/split/reshuffle`
- **Optional features** — `mmap` (zero-copy file reads), `async` (tokio I/O)

## Quick Start

```rust
use std::collections::BTreeMap;
use tensogram_core::{
    encode, decode, ByteOrder, DataObjectDescriptor, DecodeOptions,
    Dtype, EncodeOptions, GlobalMetadata,
};

let desc = DataObjectDescriptor {
    obj_type: "ntensor".to_string(), ndim: 2,
    shape: vec![100, 200], strides: vec![200, 1],
    dtype: Dtype::Float32, byte_order: ByteOrder::Big,
    encoding: "none".to_string(), filter: "none".to_string(),
    compression: "none".to_string(), params: BTreeMap::new(), hash: None,
};

let meta = GlobalMetadata::default();
let raw: Vec<u8> = vec![0u8; 100 * 200 * 4];

let message = encode(&meta, &[(&desc, &raw)], &EncodeOptions::default())?;
let (_, objects) = decode(&message, &DecodeOptions::default())?;
assert_eq!(objects[0].1.len(), 100 * 200 * 4);
```

See `examples/rust/` for MARS metadata, streaming, compression, file API, and more.

## Build & Test

```bash
cargo build --workspace                                          # build
cargo test --workspace                                           # test
cargo clippy --workspace --all-targets --all-features -- -D warnings  # lint
```

**Optional features:**
```bash
cargo build -p tensogram-core --features mmap,async
```

**C++ wrapper** (`include/tensogram.hpp`):
```bash
cargo build --release                  # build Rust static library first
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure  # run C++ tests
```
See `examples/cpp/` for encode/decode, metadata, file API, and iterator examples.

**GRIB conversion** (requires [ecCodes](https://confluence.ecmwf.int/display/ECC)):
```bash
cargo build -p tensogram-cli --features grib
tensogram convert-grib forecast.grib -o forecast.tgm
```

## Documentation

- [mdbook docs](docs/) — full developer guide (`cd docs && mdbook build`)
- [Architecture](ARCHITECTURE.md) — crate structure and design decisions
- [Contributing](CONTRIBUTING.md) — setup and workflow
- [Changelog](CHANGELOG.md) — release history
- [Python API](PYTHON_API.md) — quick reference for Python interface

## Repository Layout

```
crates/
├── tensogram-core/       Core encode/decode library
├── tensogram-encodings/  Encoding pipeline + compression codecs
├── tensogram-cli/        CLI binary (tensogram command)
├── tensogram-ffi/        C FFI layer
├── tensogram-grib/       GRIB converter (ecCodes, excluded from default build)
└── tensogram-python/     Python bindings (PyO3, excluded from default build)
examples/{rust,cpp,python}/
docs/                     mdBook documentation
```

## Copyright and License

Copyright 2024- European Centre for Medium-Range Weather Forecasts (ECMWF).

This software is licensed under the terms of the [Apache License, Version 2.0](LICENSE) which can also be obtained at http://www.apache.org/licenses/LICENSE-2.0.

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.
