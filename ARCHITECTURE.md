# Architecture

Tensogram is a binary message format library for N-dimensional scientific tensors. This document explains how the pieces fit together and why the design looks the way it does.

For the wire format byte layout, see [docs/src/format/wire-format.md](docs/src/format/wire-format.md). For the full design rationale, see [plans/DESIGN.md](plans/DESIGN.md).

## High-Level Structure

```
                    ┌─────────────────────────────────────┐
                    │         User Application            │
                    │  (Rust / Python / C++ / CLI)        │
                    └────────────┬────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                      │
    ┌──────▼──────┐    ┌────────▼────────┐    ┌───────▼───────┐
    │ tensogram-  │    │  tensogram-     │    │  tensogram-   │
    │   python    │    │     ffi         │    │     cli       │
    │  (PyO3)     │    │  (C header)     │    │  (clap)       │
    └──────┬──────┘    └────────┬────────┘    └───────┬───────┘
           │                    │                      │
           └────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   tensogram-core      │
                    │                       │
                    │  encode / decode      │
                    │  framing / wire       │
                    │  file / iter          │
                    │  metadata / hash      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ tensogram-encodings   │
                    │                       │
                    │  simple_packing       │
                    │  shuffle filter       │
                    │  compression pipeline │
                    │  (szip, zstd, lz4,   │
                    │   blosc2, zfp, sz3)   │
                    └──────────────────────┘
```

## Workspace Crates

| Crate | Purpose | Depends on |
|-------|---------|------------|
| `tensogram-core` | Wire format, framing, encode/decode, file API, iterators | `tensogram-encodings` |
| `tensogram-encodings` | Encoding pipeline: simple packing, shuffle, 6 compression codecs | (standalone) |
| `tensogram-cli` | Command-line tool (`tensogram info/ls/dump/get/set/copy/merge/split/reshuffle`) | `tensogram-core` |
| `tensogram-ffi` | C FFI with opaque handles and `tensogram.h` via cbindgen | `tensogram-core` |
| `tensogram-python` | Python bindings via PyO3, returns NumPy arrays | `tensogram-core` |
| `tensogram-grib` | GRIB→Tensogram converter via ecCodes (excluded from default build) | `tensogram-core` |

The dependency graph is a clean tree. `tensogram-encodings` has no internal dependencies. Everything else flows through `tensogram-core`. `tensogram-grib` is excluded from the default workspace build (requires the ecCodes C library).

## C++ Wrapper

The header-only C++17 wrapper lives at `include/tensogram.hpp`. It delegates all work to the C FFI (`tensogram-ffi`) and adds:

- **RAII classes** — `message`, `metadata`, `file`, `buffer_iterator`, `file_iterator`, `object_iterator`, `streaming_encoder`, each wrapping an opaque C handle via `std::unique_ptr` with a custom deleter.
- **Typed exceptions** — `tensogram::error` hierarchy maps every C error code to a specific subclass (`framing_error`, `io_error`, `hash_mismatch_error`, etc.).
- **C++17 idioms** — `[[nodiscard]]`, `std::string_view`, range-based for over decoded objects.

The CMake build system (`CMakeLists.txt`) invokes `cargo build --release` to produce the Rust static library, then exposes an INTERFACE header-only target that links the static lib and platform-specific system libraries.

## Core Modules (tensogram-core)

```
src/
├── wire.rs       Constants, Preamble, FrameHeader, Postamble, FrameType enum
├── framing.rs    Frame read/write, encode_message (5 helpers), decode_message, scan, scan_file
├── metadata.rs   CBOR serialization with canonical key ordering, verify_canonical_cbor
├── types.rs      GlobalMetadata, DataObjectDescriptor, IndexFrame, HashFrame
├── dtype.rs      15 data types (float16..complex128, bitmask)
├── encode.rs     Full encode pipeline: validate, build config, encode per object, hash, assemble
├── decode.rs     decode, decode_metadata, decode_object, decode_range
├── file.rs       TensogramFile: open, create, append, scan, mmap, async variants
├── iter.rs       MessageIter (zero-copy), ObjectIter (lazy decode), FileMessageIter (seek-based)
├── hash.rs       xxh3 hashing and verification
├── streaming.rs  StreamingEncoder<W: Write> for progressive encode to sinks
└── error.rs      TensogramError enum with 7 variants
```

## Wire Format (v2)

A message is a self-contained binary blob:

```
[Preamble 24B] [Header frames] [Data object frames] [Footer frames] [Postamble 16B]
```

Each frame starts with `FR` (2 bytes) and ends with `ENDF` (4 bytes). The preamble opens with `TENSOGRM` (8 bytes), the postamble closes with `39277777` (8 bytes).

Frame types: HeaderMetadata(1), HeaderIndex(2), HeaderHash(3), DataObject(4), FooterHash(5), FooterIndex(6), FooterMetadata(7), PrecederMetadata(8).

The decoder enforces ordering: header frames first, then data objects, then footer frames.

## Encoding Pipeline

Data flows through a per-object pipeline:

```
Raw bytes
  -> encoding (simple_packing: lossy quantization)
  -> filter (shuffle: byte-level transpose)
  -> compression (szip/zstd/lz4/blosc2/zfp/sz3)
  -> hashing (xxh3)
  -> data object frame
```

Each stage is optional and configured per object via `DataObjectDescriptor` fields (`encoding`, `filter`, `compression`). Decoding runs the same pipeline in reverse.

## Key Design Decisions

**CBOR for all metadata.** Messages are tens to hundreds of MiB, so CBOR parsing overhead is negligible. Fixed binary structs would create the same vocabulary rigidity Tensogram exists to escape.

**Frame-based, not monolithic.** Each data object carries its own CBOR descriptor. This supports streaming (write objects as they arrive, add index/hash in footer) and independent evolution of frame types.

**Vocabulary-agnostic.** The library never interprets metadata keys. ECMWF uses MARS namespace keys (`mars.class`, `mars.param`), but that is application-layer concern. The library just stores and retrieves `BTreeMap<String, ciborium::Value>`.

**Deterministic encoding.** All CBOR output is canonicalized (RFC 8949 section 4.2). Two messages with the same content produce identical bytes regardless of key insertion order. This enables byte-level comparison and deduplication.

**No panics.** Every fallible operation returns `Result<T, TensogramError>`. The library is designed for long-running operational pipelines where a panic would bring down the process.

## Feature Gates

### tensogram-core

| Feature | Dependency | What it enables | Default |
|---------|------------|-----------------|---------|
| `mmap` | `memmap2` | `TensogramFile::open_mmap()` for zero-copy file reads | off |
| `async` | `tokio` | `open_async()`, `read_message_async()`, `decode_message_async()` | off |
| `szip` | `libaec-sys` | CCSDS 121.0-B-3 szip compression with random access | **on** |
| `zstd` | `zstd` | Zstandard lossless compression | **on** |
| `lz4` | `lz4_flex` | LZ4 lossless compression (pure Rust) | **on** |
| `blosc2` | `blosc2` | Blosc2 multi-codec meta-compressor with chunk random access | **on** |
| `zfp` | `zfp-sys-cc` | ZFP lossy floating-point compression | **on** |
| `sz3` | `sz3` | SZ3 error-bounded lossy compression | **on** |

All compression features are on by default. For a lightweight build without C FFI dependencies:
```bash
cargo build -p tensogram-core --no-default-features --features mmap
```

### tensogram-cli

| Feature | Dependency | What it enables | Default |
|---------|------------|-----------------|---------|
| `grib` | `tensogram-grib` | `tensogram convert-grib` subcommand | off |

## Testing

1,262 tests across the workspace:

- Unit tests in `#[cfg(test)]` modules alongside the code
- Integration tests in `crates/tensogram-core/tests/` (round-trips, adversarial inputs, golden files)
- Python tests in `tests/python/` (226 pytest tests with parametrized dtypes)
- xarray backend tests in `tensogram-xarray/tests/` (179 tests, ~98% coverage)
- Zarr backend tests in `tensogram-zarr/tests/` (204 tests, ~95% coverage)
- C++ wrapper tests via GoogleTest (109 tests across 10 files)
- Golden binary files in `tests/golden/` for cross-language verification
- Feature-gated tests for `mmap` and `async` behind `#[cfg(feature = "...")]`

```bash
cargo test --workspace                                    # default features
cargo test -p tensogram-core --features mmap,async        # with optional features
```
