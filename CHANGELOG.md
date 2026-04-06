# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] - 2026-04-05

### Added
- **tensogram-xarray** — xarray backend engine for `.tgm` files (`engine="tensogram"`)
  - Lazy loading via `BackendArray` with N-D slice-to-flat-range mapping
  - Coordinate auto-detection by name matching (13 known names)
  - User-specified dimension mapping (`dim_names`) and variable naming (`variable_key`)
  - Multi-message auto-merge via `open_datasets()` with hypercube stacking
  - `StackedBackendArray` for lazy hypercube composition
  - Ratio-based `range_threshold` heuristic (default 0.5) for partial vs full decode
  - 113 tests, 97% line coverage
- **`decode_descriptors(buf)`** — parse metadata + per-object descriptors without decoding payloads (Rust, Python, C, C++)
- **`meta.common`** and **`meta.payload`** getters in Python bindings
- **float16/bfloat16/complex** Python support — proper typed numpy arrays (`ml_dtypes.bfloat16` if installed)

### Changed
- **`decode_range` API (BREAKING)** — now returns split results by default (one buffer per range). `join` parameter opts into concatenated output. Affects Rust, Python, C, and C++ APIs.

### Stats
- 572 total tests (257 Rust + 97 Python + 113 xarray + 105 C++)

## [0.3.0] - 2026-04-04

### Added
- **Python bindings quality overhaul** — all 10 numeric numpy dtypes natively accepted/decoded
- **`TensogramFile` Python** — context manager (`with ... as f:`), `len(f)`, `"key" in meta`
- **95 Python tests** — parametrized dtype round-trips, multi-object, multi-range, big-endian, wire determinism
- **ruff** configured as Python linter/formatter (0 warnings)

### Fixed
- `decode_range` validates byte count vs expected elements (was silently truncating)
- Bitmask dtype fallback no longer produces empty array
- `byte_order` rejects unknown values (was silently defaulting)
- Safe i128-to-i64 bounds check for CBOR integers
- Flaky golden file test race condition fixed (tests now read-only)
- `cargo doc` and `cargo fmt` CI failures resolved

### Performance
- `from_slice` zero-copy for u8/i8 numpy decode (eliminates allocation)

### Stats
- 262 total tests (167 Rust + 95 Python), 0 clippy warnings

## [0.2.0] - 2026-04-04

### Added
- **Streaming API** — `StreamingEncoder<W: Write>` for progressive encode without buffering
- **Metadata structure** — `GlobalMetadata` now has `common`, `payload`, `reserved` CBOR sections (backwards-compatible)
- **CLI merge** — `tensogram merge` combines messages from multiple files
- **CLI split** — `tensogram split` separates multi-object messages into individual files
- **CLI reshuffle** — `tensogram reshuffle` converts streaming-mode to random-access-mode
- **GRIB converter** — `tensogram-grib` crate with ecCodes FFI for GRIB-to-Tensogram conversion
- **CLI convert-grib** — `tensogram convert-grib` subcommand (feature-gated behind `grib`)
- **Feature-gated compression** — all 6 codecs (szip, zstd, lz4, blosc2, zfp, sz3) are optional features (default on)
- **Streaming example** — `examples/rust/src/bin/11_streaming.rs`
- **GRIB docs** and **CLI docs** — mdbook pages for conversion and new commands

### Changed
- `README.md` shortened from 302 to 100 lines; detailed content moved to mdbook docs

### Removed
- **md5 and sha1 hash support** — only xxh3 is supported; unknown hash types return a clear error

### Stats
- 170 Rust tests, 0 clippy warnings

## [0.1.0] - 2026-04-04

Initial release of Tensogram, a binary N-Tensor message format library for scientific data.

### Core
- Encode and decode N-dimensional tensors with self-describing CBOR metadata
- Pack multiple tensors per message, each with own shape, dtype, and encoding pipeline
- 6 compression codecs per data object: szip (CCSDS), zstd, lz4, blosc2, zfp (lossy float), sz3 (error-bounded lossy)
- GRIB-style simple packing for lossy quantization (0-64 bits per value)
- xxh3 per-object payload hashing for integrity verification
- 15 data types: float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask

### File I/O
- `TensogramFile` with lazy scanning, O(1) random-access, and message append
- Memory-mapped I/O via `mmap` feature (zero-copy reads)
- Async file operations via `async` feature (tokio)

### Language Bindings
- Rust native API
- C FFI layer with auto-generated `tensogram.h` header (62 functions)
- Python bindings via PyO3 with NumPy integration

### CLI
- `tensogram info`, `ls`, `dump`, `get`, `set`, `copy` subcommands
- Where-clause filtering (`-w`), key selection (`-p`), JSON output (`-j`)

### Wire Format (v2)
- Frame-based message structure: Preamble + typed frames + Postamble
- Streaming support with `total_length=0` and footer-based index
- Deterministic CBOR encoding (RFC 8949 section 4.2 canonical key ordering)
- Corruption recovery via magic boundary detection

### Quality
- 157 tests across 5 workspace crates, 0 clippy warnings
- Golden binary test files for cross-language verification
