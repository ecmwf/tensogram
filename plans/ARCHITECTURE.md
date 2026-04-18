# Architecture

Tensogram is a binary message format library for N-dimensional scientific
tensors. This document explains how the pieces fit together and why the
design looks the way it does.

For the wire format byte layout, see
[../docs/src/format/wire-format.md](../docs/src/format/wire-format.md) and
[WIRE_FORMAT.md](WIRE_FORMAT.md) in this directory. For the full design
rationale, see [DESIGN.md](DESIGN.md). For the ongoing implementation path
followed, see [DONE.md](DONE.md).

## High-Level Structure

```
                   ┌────────────────────────────────────────┐
                   │            User Application            │
                   │   (Rust / Python / C++ / Web / CLI)    │
                   └───────────────────┬────────────────────┘
                                       │
        ┌─────────────┬─────────────┬──┴──────────┬──────────────┐
        │             │             │             │              │
 ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐ ┌────▼────┐ ┌───────▼────────┐
 │  tensogram  │ │tensogram│ │  tensogram  │ │tensogram│ │   tensogram-   │
 │   -python   │ │   -ffi  │ │    -wasm    │ │   -cli  │ │  xarray, -zarr │
 │    (PyO3)   │ │ (C hdr) │ │  (wasm-bg)  │ │  (clap) │ │  (pure Python) │
 └──────┬──────┘ └────┬────┘ └──────┬──────┘ └────┬────┘ └───────┬────────┘
        │             │             │             │              │
        └─────────────┴─────────────┼─────────────┴──────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │      tensogram        │
                        │                       │
                        │  encode / decode      │
                        │  framing / wire       │
                        │  file / iter          │
                        │  metadata / hash      │
                        │  remote / validate    │
                        └───────────┬───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │  tensogram-encodings  │
                        │                       │
                        │  simple_packing       │
                        │  shuffle filter       │
                        │  compression pipeline │
                        └───────────┬───────────┘
                                    │
        ┌────────────────┬──────────┴──────────┬────────────────┐
        │                │                     │                │
 ┌──────▼──────┐ ┌───────▼───────┐ ┌───────────▼─────┐ ┌────────▼─────────┐
 │  tensogram  │ │  tensogram    │ │  libaec /       │ │  lz4 / blosc2 /  │
 │    -szip    │ │    -sz3       │ │  zstd / zfp     │ │  ruzstd          │
 │ (pure Rust) │ │  (safe API)   │ │  (C libraries)  │ │  (Rust crates)   │
 └─────────────┘ └───────┬───────┘ └─────────────────┘ └──────────────────┘
                         │
                 ┌───────▼───────┐
                 │  tensogram    │
                 │   -sz3-sys    │
                 │  (vendored    │
                 │   C++ shim)   │
                 └───────────────┘
```

## Workspace Crates

The repository is a Rust workspace plus separate Python packages. The
default workspace build excludes crates that need Python linking,
WebAssembly tooling, or external system libraries.

### Default-workspace Rust crates

Public library crates (published to [crates.io](https://crates.io/crates/tensogram)):

| Crate | Purpose | Depends on |
|-------|---------|------------|
| `tensogram` | Wire format, framing, encode/decode, file API, iterators, validation, remote object store | `tensogram-encodings` |
| `tensogram-encodings` | Encoding pipeline: simple packing, shuffle, compression codecs (szip, zstd, lz4, blosc2, zfp, sz3 — all feature-gated) | — |
| `tensogram-cli` | Command-line tool (`info`, `ls`, `dump`, `get`, `set`, `copy`, `merge`, `split`, `reshuffle`, `validate`, plus feature-gated `convert-grib`, `convert-netcdf`) | `tensogram` |
| `tensogram-ffi` | C FFI surface with opaque handles and `tensogram.h` via cbindgen | `tensogram` |
| `tensogram-szip` | Pure-Rust CCSDS 121.0-B-3 szip codec (used via the `szip-pure` feature, e.g. for WebAssembly) | — |
| `tensogram-sz3` | Safe high-level SZ3 lossy-compression API | `tensogram-sz3-sys` |
| `tensogram-sz3-sys` | `-sys` crate vendoring the SZ3 C++ library (Apache-2.0 wrapper; Argonne BSD + Boost-1.0 for vendored source — see `LICENSES.md`) | (native C++ build) |

Internal crates (not published):

| Crate | Purpose | Depends on |
|-------|---------|------------|
| `tensogram-benchmarks` | Benchmark suite in `rust/benchmarks` (`codec-matrix`, `grib-comparison`, `threads-scaling`, `hash-overhead` binaries) | `tensogram`, `tensogram-encodings` |
| `tensogram-rust-examples` | Runnable Rust examples in `examples/rust` (numbered `NN_description.rs`) | `tensogram` |

### Excluded-from-default-workspace crates

These live in the repo but are opt-in because they require a Python
interpreter, WebAssembly tooling, or external C libraries.

| Cargo package | Directory | Excluded because | Build recipe |
|---------------|-----------|------------------|--------------|
| `tensogram-python` (PyPI package `tensogram`) | `python/bindings` | PyO3 extension requires the Python linker | `cd python/bindings && maturin develop` |
| `tensogram-grib` | `rust/tensogram-grib` | Needs `libeccodes` (ecCodes) | `cd rust/tensogram-grib && cargo build` |
| `tensogram-netcdf` | `rust/tensogram-netcdf` | Needs `libnetcdf` | `cd rust/tensogram-netcdf && cargo build` |
| `tensogram-wasm` | `rust/tensogram-wasm` | Needs `wasm-pack` and the `wasm32-unknown-unknown` target | `wasm-pack build rust/tensogram-wasm --target web` |

### Separate Python packages

Not part of the Cargo workspace; pure-Python packages distributed on PyPI.

| Package | Purpose |
|---------|---------|
| `python/tensogram-xarray` | xarray backend engine (`engine="tensogram"`) with lazy loading, coordinate auto-detection, and multi-message hypercube stacking |
| `python/tensogram-zarr` | Zarr v3 store backend (`TensogramStore`) for read/write access via the standard Zarr API |

The dependency graph is a clean tree. `tensogram-encodings` has no
internal dependencies. Everything else flows through `tensogram`.

## C++ Wrapper

The header-only C++17 wrapper lives at `cpp/include/tensogram.hpp`. It
delegates all work to the C FFI (`tensogram-ffi`) and adds:

- **RAII classes** — `message`, `metadata`, `file`, `buffer_iterator`,
  `file_iterator`, `object_iterator`, `streaming_encoder`, each wrapping
  an opaque C handle via `std::unique_ptr` with a custom deleter.
- **Typed exceptions** — `tensogram::error` hierarchy maps every C
  error code to a specific subclass (`framing_error`, `io_error`,
  `hash_mismatch_error`, etc.).
- **C++17 idioms** — `[[nodiscard]]`, `std::string_view`, range-based
  for over decoded objects.
- **`validate()` / `validate_file()`** — wrappers that return JSON
  strings with typed exception mapping.

The CMake build system (`cpp/CMakeLists.txt`) invokes `cargo build
--release` to produce the Rust static library, then exposes an
INTERFACE header-only target that links the static lib and
platform-specific system libraries.

## Core Modules (`tensogram`)

```
src/
├── wire.rs       Constants, Preamble, FrameHeader, Postamble, FrameType enum,
│                 MessageFlags, DataObjectFlags
├── framing.rs    Frame read/write, encode_message, decode_message,
│                 scan, scan_file
├── metadata.rs   CBOR serialization with canonical key ordering,
│                 verify_canonical_cbor
├── types.rs      GlobalMetadata, DataObjectDescriptor, IndexFrame, HashFrame
├── dtype.rs      15 data types (float16..complex128, bitmask) with
│                 swap-unit logic for native-endianness decode
├── encode.rs     Full encode pipeline plus encode_pre_encoded bypass
├── decode.rs     decode, decode_metadata, decode_descriptors, decode_object,
│                 decode_range (supports split/join, native byte order)
├── file.rs       TensogramFile: open, create, append, scan, mmap, async,
│                 remote-aware file handle
├── iter.rs       MessageIter, ObjectIter, FileMessageIter
├── pipeline.rs   Shared DataPipeline + apply_pipeline helper
│                 (used by tensogram-grib and tensogram-netcdf)
├── hash.rs       xxh3 hashing and verification
├── streaming.rs  StreamingEncoder<W: Write>, Preceder Metadata Frames
├── remote.rs     object_store-backed remote access (S3/GCS/Azure/HTTP),
│                 sync + async range reads, batched reads
├── error.rs      TensogramError enum
└── validate/     4-level validation (structure, metadata, integrity,
                  fidelity) with stable IssueCode enum
```

## Wire Format

A message is a self-contained binary blob:

```
[Preamble 24B] [Header frames] [Data object frames] [Footer frames] [Postamble 16B]
```

Each frame starts with `FR` (2 bytes) and ends with `ENDF` (4 bytes).
The preamble opens with `TENSOGRM` (8 bytes), the postamble closes with
`39277777` (8 bytes).

Frame types (8 total):
`HeaderMetadata(1)`, `HeaderIndex(2)`, `HeaderHash(3)`, `DataObject(4)`,
`FooterHash(5)`, `FooterIndex(6)`, `FooterMetadata(7)`,
`PrecederMetadata(8)`.

The decoder enforces ordering: header frames first, then data objects
(optionally each preceded by a Preceder Metadata Frame), then footer
frames.

## Encoding Pipeline

Data flows through a per-object pipeline:

```
Raw bytes
  → encoding     (simple_packing: lossy quantization, or none)
  → filter       (shuffle: byte-level transpose, or none)
  → compression  (szip / zstd / lz4 / blosc2 / zfp / sz3, or none)
  → hashing      (xxh3, optional)
  → data object frame
```

Each stage is optional and configured per object via
`DataObjectDescriptor` fields (`encoding`, `filter`, `compression`).
Decoding runs the same pipeline in reverse. Decoded payloads are
returned in the caller's native byte order by default; set
`DecodeOptions.native_byte_order = false` to preserve the wire-order
bytes.

## Key Design Decisions

**CBOR for all metadata.** Messages are tens to hundreds of MiB, so
CBOR parsing overhead is negligible. Fixed binary structs would create
the same vocabulary rigidity Tensogram exists to escape.

**Frame-based, not monolithic.** Each data object carries its own CBOR
descriptor. This supports streaming (write objects as they arrive, add
index/hash in footer) and independent evolution of frame types.

**Vocabulary-agnostic.** The library never interprets metadata keys.
ECMWF uses MARS namespace keys (`mars.class`, `mars.param`), but that
is an application-layer concern. The library just stores and retrieves
`BTreeMap<String, ciborium::Value>`.

**Deterministic encoding.** All CBOR output is canonicalized (RFC 8949
section 4.2). Two messages with the same content produce identical
bytes regardless of key insertion order. This enables byte-level
comparison and deduplication.

**No panics.** Every fallible operation returns `Result<T,
TensogramError>`. The library is designed for long-running operational
pipelines where a panic would bring down the process. `panic = "abort"`
is set on both release and dev profiles so panics cannot unwind across
the FFI boundary.

## Feature Gates

### `tensogram` and `tensogram-encodings`

| Feature | Dependency | What it enables | Default |
|---------|------------|-----------------|---------|
| `mmap` | `memmap2` | `TensogramFile::open_mmap()` for zero-copy file reads | off |
| `async` | `tokio` | `open_async()`, `read_message_async()`, `decode_message_async()`, etc. | off |
| `remote` | `object_store`, `tokio`, `bytes`, `url` | `open_remote()`, S3/GCS/Azure/HTTP access, async range reads | off |
| `szip` | `libaec-sys` | CCSDS 121.0-B-3 szip compression (C FFI) with random access | **on** |
| `szip-pure` | `tensogram-szip` | Pure-Rust szip (for WebAssembly / no-C-FFI builds) | off |
| `zstd` | `zstd` | Zstandard lossless compression (C FFI) | **on** |
| `zstd-pure` | `ruzstd` | Pure-Rust Zstandard decoder | off |
| `lz4` | `lz4_flex` | LZ4 lossless compression (pure Rust) | **on** |
| `blosc2` | `blosc2` | Blosc2 multi-codec meta-compressor with chunk random access | **on** |
| `zfp` | `zfp-sys-cc` | ZFP lossy floating-point compression | **on** |
| `sz3` | `tensogram-sz3` | SZ3 error-bounded lossy compression (via clean-room shim) | **on** |

All compression features default to on. For a lightweight build without
C FFI dependencies (e.g. WebAssembly):

```bash
cargo build -p tensogram --no-default-features \
    --features szip-pure,zstd-pure,lz4
```

### `tensogram-cli`

| Feature | Dependency | What it enables | Default |
|---------|------------|-----------------|---------|
| `grib` | `tensogram-grib` | `tensogram convert-grib` subcommand | off |
| `netcdf` | `tensogram-netcdf` | `tensogram convert-netcdf` subcommand | off |

## Testing

Tests are spread across the workspace, the optional crates, and the
Python packages. Rather than list frozen numbers here (they drift with
every release), see [TEST.md](TEST.md) for the current coverage shape.

To run them:

```bash
cargo test --workspace                                    # default features
cargo test -p tensogram --features mmap,async,remote # optional features
cd rust/tensogram-grib    && cargo test                   # requires libeccodes
cd rust/tensogram-netcdf  && cargo test                   # requires libnetcdf
wasm-pack test --node rust/tensogram-wasm                 # requires wasm-pack

python -m pytest python/tests/                            # core Python
python -m pytest python/tensogram-xarray/tests/           # xarray backend
python -m pytest python/tensogram-zarr/tests/             # zarr backend

cmake -S cpp -B build && cmake --build build -j
ctest --test-dir build --output-on-failure                # C++ wrapper
```

Golden binary `.tgm` fixtures in `rust/tensogram/tests/golden/`
provide byte-for-byte cross-language verification between Rust, Python,
and C++.
