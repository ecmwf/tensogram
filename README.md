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


## Building


## Dependencies

| Library       | Purpose                           | Type       |
|---------------|-----------------------------------|------------|
| ciborium      | CBOR metadata encoding            | Rust crate |
| xxhash-rust   | Payload hashing (xxh3, 64-bit)    | Rust crate |

## Repository Layout

## License

Copyright 2024- European Centre for Medium-Range Weather Forecasts (ECMWF).

Licensed under the [Apache License, Version 2.0](LICENSE).
