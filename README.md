# Tensogram

[![Static Badge](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity/emerging_badge.svg)](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity#emerging)

> [!IMPORTANT]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

A library to encode and decode binary N-Tensor scientific data with semantic metadata close to the data, in a serialisable format that can be sent over the network, encoded into in-memory buffers and decoded with zero-copy. It is geared to a lightweight implementation, self-description of data and high-performance with limited dependencies.

Tensogram defines a network-transmissible binary message format, not a file format.
Multiple messages can be appended to a file, and that remains valid since each
message carries its own begin/terminator codes.

## Features

- **Self-describing messages** — each message bundles CBOR-encoded key-value metadata with one or more typed data objects (tensors)
- **Support for n-Tensors** — each message can contain multiple n-Tensors of different element types
- **No panics** — all fallible operations return `Result<T, TensogramError>`
- **Stateless & thread-safe** — no global state
- **Compression** — optional szip (libaec) and Blosc2 encoding per data object
- **Hash verification** — xxHash xxh3-64 integrity check on every data object (can be skipped for trusted buffers)
- **Support for multiple languages** — Python NumPy-based API, C++ and Rust
- **File convenience API** — convinience API functions to handle files containing multiple messages
- **multiple data types** — float16/32/64, bfloat16, int8-64, uint8-64, complex64/128, bit, etc

## Quick Start

### Rust

Add `tensogram-core` to your `Cargo.toml`:

```toml
[dependencies]
tensogram-core = { path = "path/to/tensogram/crates/tensogram-core" }
tensogram-encodings = { path = "path/to/tensogram/crates/tensogram-encodings" }
```

Encode and decode a 2D float32 tensor:

```rust
use std::collections::BTreeMap;
use tensogram_core::{
    decode, encode, ByteOrder, DecodeOptions, Dtype, EncodeOptions,
    Metadata, ObjectDescriptor, PayloadDescriptor,
};

// Describe a 100×200 temperature grid
let metadata = Metadata {
    version: 1,
    objects: vec![ObjectDescriptor {
        obj_type: "ntensor".to_string(),
        ndim: 2,
        shape: vec![100, 200],
        strides: vec![200, 1],
        dtype: Dtype::Float32,
        extra: BTreeMap::new(),
    }],
    payload: vec![PayloadDescriptor {
        byte_order: ByteOrder::Big,
        encoding: "none".to_string(),
        filter: "none".to_string(),
        compression: "none".to_string(),
        params: BTreeMap::new(),
        hash: None,
    }],
    extra: BTreeMap::new(),
};

// Raw big-endian float32 bytes
let raw: Vec<u8> = (0u32..20_000)
    .flat_map(|i| (273.15f32 + i as f32 * 0.001).to_be_bytes())
    .collect();

// Encode to wire-format bytes
let message = encode(&metadata, &[raw.as_slice()], EncodeOptions::default())?;

// Decode back
let (meta, objects) = decode(&message, DecodeOptions::default())?;
assert_eq!(objects[0].data.len(), 20_000 * 4);
```

See `examples/rust/` for more: MARS metadata, simple packing, multi-object messages, file API.

### Python

Build the Python extension first:

```bash
cd crates/tensogram-python
maturin develop
```

Then:

```python
import numpy as np
import tensogram

temps = np.linspace(273.15, 283.15, 100 * 200, dtype=np.float32).reshape(100, 200)

metadata = tensogram.Metadata(
    version=1,
    objects=[tensogram.ObjectDescriptor(type="ntensor", shape=[100, 200], dtype="float32")],
)

message: bytes = tensogram.encode(metadata, temps)
meta, arrays = tensogram.decode(message)

assert arrays[0].shape == temps.shape
np.testing.assert_array_equal(arrays[0], temps)
```

See `examples/python/` for more examples.

### C++

Link against the FFI library and include the generated header:

```cpp
#include "tensogram.h"

const char *meta_json = R"({"version":1,"objects":[{
    "type":"ntensor","ndim":2,"shape":[100,200],"strides":[200,1],"dtype":"float32"
}],"payload":[{"byte_order":"big","encoding":"none","filter":"none","compression":"none"}]})";

tgm_message_t *msg = nullptr;
tgm_encode(meta_json, data_ptr, data_len, TGM_HASH_XXH3, &msg);
// ... use msg ...
tgm_message_free(msg);
```

See `examples/cpp/` for more examples.

### CLI

```bash
cargo install --path crates/tensogram-cli

tensogram info data.tgm
tensogram ls -p mars.param,mars.date data.tgm
tensogram dump -j data.tgm
tensogram get -p mars.param data.tgm
tensogram set -s mars.date=20260401 input.tgm output.tgm
tensogram copy -w mars.param=2t input.tgm output_[mars.param].tgm
```

## Building and Testing

### Prerequisites

- Rust 1.75+ (`rustup install stable`)
- For Python bindings: Python 3.9+, `maturin` (`pip install maturin`)
- For C/C++ bindings: a C++17 compiler

### Build

```bash
# Build all workspace crates (excludes tensogram-python)
cargo build --workspace

# Build in release mode
cargo build --workspace --release
```

### Format, Lint, Test

```bash
cargo fmt                                                        # format
cargo clippy --workspace --all-targets --all-features -- -D warnings  # lint
cargo test --workspace                                           # test
```

### Python bindings

```bash
cd crates/tensogram-python
maturin develop          # installs into the current venv
python -m pytest         # run Python tests
```

### C/C++ bindings

The `tensogram.h` header is auto-generated by `cbindgen` during the `tensogram-ffi` build:

```bash
cargo build -p tensogram-ffi
# Header: crates/tensogram-ffi/tensogram.h
# Static lib: target/debug/libtensogram_ffi.a
# Shared lib: target/debug/libtensogram_ffi.{so,dylib}
```

### Running examples

```bash
# Rust examples
cargo run --bin 01_encode_decode    # from examples/rust/
# or from workspace root:
cargo run -p examples --bin 01_encode_decode

# Python examples
python examples/python/01_encode_decode.py
```

## Dependencies

| Library         | Purpose                                          | Type           |
|-----------------|--------------------------------------------------|----------------|
| ciborium        | CBOR metadata encoding/decoding                  | Rust crate     |
| xxhash-rust     | Payload hashing (xxh3, 64-bit)                   | Rust crate     |
| thiserror       | Structured error types                           | Rust crate     |
| serde           | Serialization framework (used with ciborium)     | Rust crate     |
| sha1            | SHA-1 payload hashing (legacy support)           | Rust crate     |
| md5             | MD5 payload hashing (legacy support)             | Rust crate     |
| clap            | CLI argument parsing (`tensogram` binary)        | Rust crate     |
| serde_json      | JSON ↔ CBOR metadata bridge for C FFI            | Rust crate     |
| cbindgen        | C header generation from Rust FFI (build-time)   | Rust crate     |
| pyo3            | Python bindings                                  | Rust crate     |
| numpy (pyo3)    | NumPy array integration for Python bindings      | Rust crate     |
| maturin         | Python extension build tool                      | Python tool    |

**Optional system library:**

| Library | Purpose                              | Type           |
|---------|--------------------------------------|----------------|
| libaec  | szip lossless compression (CCSDS standard) | C library (FFI) |

## Repository Layout

```
tensogram
├── Cargo.toml                  # Workspace root (members + shared dependencies)
├── crates/
│   ├── tensogram-core/         # Core encode/decode library (Rust)
│   ├── tensogram-encodings/    # Encoding pipeline (Rust)
│   ├── tensogram-cli/          # CLI binary (`tensogram` command)
│   ├── tensogram-ffi/          # C FFI layer for C/C++ callers
│   └── tensogram-python/       # Python bindings (PyO3 / maturin)
├── examples/
│   ├── rust/                   # Rust usage examples (cargo run --bin <name>)
│   ├── cpp/                    # C++ usage examples (require tensogram-ffi)
│   └── python/                 # Python usage examples (require maturin develop)
├── docs/                       # mdBook documentation
└── plans/
    ├── DESIGN.md               # Architecture and wire format specification
    ├── DONE.md                 # Implementation status
    ├── IMPROVEMENTS.md         # Future improvements backlog
    └── TODO.md                 # Long-term feature ideas
```

## Copyright and License

Copyright 2024- European Centre for Medium-Range Weather Forecasts (ECMWF).

This software is licensed under the terms of the [Apache License, Version 2.0](LICENSE) which can also be obtained at http://www.apache.org/licenses/LICENSE-2.0.

In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of its status as an intergovernmental organisation nor does it submit to any jurisdiction.
