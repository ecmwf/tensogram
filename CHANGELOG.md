# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.7.0] - 2026-04-08

### Fixed
- **szip 24-bit data corruption** — `AEC_DATA_3BYTE` is now auto-set in `effective_flags()` for 17-24 bit samples, so libaec reads 3-byte-packed data correctly. Decoded values previously had ±60 max error; now match quantization step (~1.9×10⁻⁶ at 24 bits).
- **szip byte-order mismatch** — `AEC_DATA_MSB` is now set when the upstream encoding is `SimplePacking` (which produces MSB-first bytes). libaec's predictor now sees the correct byte significance order; compression ratio on 24-bit GRIB data now matches ecCodes (~27%).
- **Benchmark `AEC_DATA_PREPROCESS` constant** — was 1 (`AEC_DATA_SIGNED`), now correctly 8. Benchmarks were running without the preprocessor step.

### Added
- `simple_packing::encode_pipeline_f64()` — typed-input variant that avoids the bytes→f64 round-trip allocation
- Benchmark fidelity validation: lossless paths checked for exact round-trip; lossy paths report Linf, L1, and L2 (RMSE) norms
- Benchmark structured error handling: `BenchmarkError` enum, `BenchmarkRun` struct, non-zero exit on failures
- `--warmup` flag (default 3) and raised default iteration count from 5 to 10
- Throughput (MB/s) reporting and compressed-size variability detection in benchmarks
- Rewritten benchmark documentation with split tables (lossless / SimplePacking / lossy), human-readable method names, sizes in MiB, and fidelity norms explained

### Changed
- `simple_packing` encode is ~2.5× faster for typical SimplePacking cases:
  - Fused NaN + min + max into a single pass in `compute_params` (was 3 passes)
  - Precomputed `scale = 10^D × 2^(-E)` — eliminates per-value f64 division
  - Specialized `encode_aligned<N>` / `decode_aligned<N>` loops for 8/16/24/32-bit widths
  - Removed redundant NaN check from `encode()`
- Benchmark methodology page cleaned of internal jargon — no Rust API names, no C function signatures
- GRIB comparison timing is now symmetric end-to-end (both ecCodes and Tensogram include parameter setup)

## [0.6.0] - 2026-04-06

### Changed (BREAKING)
- **Metadata major refactor** — `GlobalMetadata` fields `common` and `payload` removed; replaced by `base` (per-object metadata array where each entry is independent and self-contained)
- **CBOR key renames** — `reserved` → `_reserved_`, `extra` → `_extra_` on the wire
- **Python API** — `meta.common` and `meta.payload` replaced by `meta.base` and `meta.reserved`; `meta.extra` now maps to `_extra_` in CBOR
- Auto-populated tensor metadata (ndim/shape/strides/dtype) now lives under `base[i]["_reserved_"]["tensor"]`

### Added
- `encode_pre_encoded()` API for advanced callers (e.g., GPU pipelines) across Rust, Python, C FFI, and C++ — bypasses the encoding pipeline for already-encoded payloads
- `StreamingEncoder::write_object_pre_encoded()` for streaming pre-encoded objects
- `compute_common()` utility for extracting shared keys from `base` entries in software
- Encoder validates that client code does not write to `_reserved_` at any level
- Preceder Metadata Frames (type 8) for streaming per-object metadata

### Stats
- 1008 total tests (283 Rust + 226 Python + 181 xarray + 204 Zarr + 105 C++ + 7 GRIB new + 2 streamer)

## [0.5.0] - 2026-04-06

### Added
- **Project logo** — centered at top of README with badge
- **82 new Python coverage tests** — metadata properties, error paths, descriptor coverage,
  decode_range across all 10 dtypes, file slice edge cases, concurrent iterators,
  iter_messages edges, Message unpacking across all 5 decode paths, scan edges,
  big-endian round-trips

### Fixed
- Replaced `unwrap()` in FFI decode macro with `Result` propagation (panic=abort safety)
- Fixed TOCTOU race condition in Python file iterator initialization
- Added buffer copy warning to `iter_messages()` docstring
- Python examples 01-03, 05-06 rewritten to use real API
- Removed binary `.so` build artifact, added `*.so` to `.gitignore`

### Stats
- 888 total tests (283 Rust + 200 Python + 124 xarray + 172 Zarr + 109 C++)

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
- **tensogram-zarr** — Zarr v3 Store backend for `.tgm` files
  - Read/write/append modes via `TensogramStore`
  - `zarr.open_group(store=TensogramStore.open_tgm("file.tgm"))` for standard Zarr API access
  - 14 numeric dtypes mapped bidirectionally (TGM <-> Zarr v3 <-> NumPy)
  - Variable naming from MARS metadata with dedup and sanitization
  - Byte-range support (RangeByteRequest, OffsetByteRequest, SuffixByteRequest)
  - 172 tests
- **Python iterator protocol** — `TensogramFile` now supports standard Python iteration
  - `for msg in file:` iterates all messages (owns independent file handle, free-threaded safe)
  - `file[i]`, `file[-1]` — index by position (negative indexing)
  - `file[1:10:2]` — slice returns list of Message namedtuples
  - `iter_messages(buf)` — iterate decoded messages from a byte buffer
  - `Message` namedtuple — `.metadata` and `.objects` fields, tuple unpacking
- **`decode_descriptors(buf)`** — parse metadata + per-object descriptors without decoding payloads (Rust, Python, C, C++)
- **`meta.base`**, **`meta.reserved`**, and **`meta.extra`** getters in Python bindings
- **float16/bfloat16/complex** Python support — proper typed numpy arrays (`ml_dtypes.bfloat16` if installed)

### Changed
- **`decode_range` API (BREAKING)** — now returns split results by default (one buffer per range). `join` parameter opts into concatenated output. Affects Rust, Python, C, and C++ APIs.
- **`decode()` and `decode_message()`** now return `Message` namedtuple (supports both attribute access and tuple unpacking — backward compatible)

### Fixed
- Removed binary `.so` build artifact from Python package
- Replaced `unwrap()` in FFI decode macro with proper error propagation
- Fixed TOCTOU race condition in Python file iterator initialization
- Python examples 01-03, 05-06 rewritten to use real API (were placeholder code)

### Stats
- 888 total tests (283 Rust + 200 Python + 124 xarray + 172 Zarr + 109 C++)

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
