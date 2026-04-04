# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-04-04

### Added
- **Streaming API** — `StreamingEncoder<W: Write>` for progressive encode/transmit without buffering
- **Metadata structure** — `GlobalMetadata` now has `common`, `payload`, `reserved` CBOR sections (backwards-compatible)
- **CLI merge** — `tensogram merge` combines messages from multiple files into one
- **CLI split** — `tensogram split` separates multi-object messages into individual files
- **CLI reshuffle** — `tensogram reshuffle` converts streaming-mode to random-access-mode messages
- **GRIB converter** — `tensogram-grib` crate with ecCodes FFI for GRIB→Tensogram conversion
- **CLI convert-grib** — `tensogram convert-grib` subcommand (feature-gated behind `grib`)
- **Feature-gated compression** — all 6 codecs (szip, zstd, lz4, blosc2, zfp, sz3) are optional features (default on)
- **Streaming example** — `examples/rust/src/bin/11_streaming.rs`
- **GRIB docs** — mdbook pages for GRIB conversion overview and MARS key mapping
- **CLI docs** — mdbook pages for merge, split, reshuffle commands

### Changed
- `README.md` shortened from 302 to 100 lines; detailed content moved to mdbook docs

### Removed
- **md5 and sha1 hash support** — only xxh3 is supported; unknown hash types return a clear error

## [0.1.0] - 2026-04-04

Initial release of Tensogram, a binary N-Tensor message format library for scientific data.

### What you can do

- **Encode and decode N-dimensional tensors** with self-describing CBOR metadata in a single binary message
- **Pack multiple tensors per message**, each with its own shape, dtype, and encoding pipeline
- **Choose from 6 compression codecs** per data object: szip (CCSDS), zstd, lz4, blosc2, zfp (lossy floating-point), and sz3 (error-bounded lossy)
- **Apply GRIB-style simple packing** for lossy quantization with configurable bit depth (0-64 bits)
- **Verify data integrity** with xxh3 per-object payload hashing
- **Work with files** containing multiple messages via `TensogramFile` with lazy scanning and O(1) random access
- **Use from Python** with NumPy integration (encode/decode returns `numpy.ndarray` directly)
- **Use from C/C++** via the FFI layer with auto-generated `tensogram.h` header
- **Inspect files from the command line** with `tensogram info`, `ls`, `dump`, `get`, `set`, and `copy`
- **Iterate lazily** over messages, objects, and file contents with zero-copy buffer iterators
- **Open files with memory-mapped I/O** (`mmap` feature) for zero-copy reads
- **Use async file operations** (`async` feature) with tokio for non-blocking pipelines

### Wire format (v2)

- Frame-based message structure: Preamble + typed frames (metadata, index, hash, data object) + Postamble
- Streaming support with `total_length=0` and footer-based index
- Deterministic CBOR encoding (RFC 8949 section 4.2 canonical key ordering)
- 8-byte frame alignment for memory-mapped access
- Corruption recovery via `TENSOGRM`/`39277777` magic boundary detection

### Data types

float16, float32, float64, bfloat16, complex64, complex128, int8-64, uint8-64, bitmask

### For contributors

- 157 tests across 5 workspace crates, 0 clippy warnings
- Golden binary test files in `tests/golden/` for cross-language verification
- Frame ordering validation in the decoder (headers, then data objects, then footers)
- Streaming file scanner (`scan_file`) that reads preamble chunks without loading entire files
- `verify_canonical_cbor()` utility for RFC 8949 compliance checking
