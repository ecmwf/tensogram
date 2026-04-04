# Tensogram Library — Implementation Status

Implemented: 2026-04-03

## Workspace: 5 crates, 137 tests, 0 clippy warnings

### tensogram-core (43 unit tests + 42 integration + 12 adversarial)
- `wire.rs` — v2 frame-based wire format: Preamble (24B), FrameHeader (16B), Postamble (16B), FrameType enum, MessageFlags, DataObjectFlags
- `framing.rs` — `encode_message()` with two-pass index construction, `decode_message()`, `scan()` for multi-message buffers
- `metadata.rs` — Deterministic CBOR encoding for GlobalMetadata, DataObjectDescriptor, IndexFrame, HashFrame (three-step: serialize → canonicalize → write)
- `types.rs` — `GlobalMetadata`, `DataObjectDescriptor` (merged object + payload), `IndexFrame`, `HashFrame`, `DecodedObject` type alias
- `dtype.rs` — All 15 dtypes (float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask)
- `hash.rs` — xxh3 hashing + verification (sha1/md5 removed)
- `encode.rs` — Full encode pipeline: validate → build pipeline config → encode per object → hash → assemble frames
- `decode.rs` — `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()`
- `file.rs` — `TensogramFile`: open, create, lazy scan, append, seek-based random access, `iter()` for lazy file iteration
- `iter.rs` — `MessageIter` (zero-copy buffer iteration), `ObjectIter` (lazy per-object decode), `FileMessageIter` (seek-based file iteration), `objects_metadata()` (descriptor-only)

### tensogram-encodings (47 tests)
- `simple_packing.rs` — GRIB-style lossy quantization, MSB-first bit packing, 0-64 bits, NaN rejection, `decode_range()` for arbitrary bit offsets
- `shuffle.rs` — Byte-level shuffle/unshuffle (HDF5-style)
- `libaec.rs` — Safe Rust wrapper around libaec (CCSDS 121.0-B-3): `aec_compress()` with RSI block offset tracking, `aec_decompress()`, `aec_decompress_range()` for partial range decode
- `compression/` — Module directory with `Compressor` trait, `NoopCompressor`, and 6 compressor implementations:
  - `szip.rs` — `SzipCompressor` (CCSDS 121.0-B-3 via libaec, RSI block random access)
  - `zstd.rs` — `ZstdCompressor` (Zstandard lossless, stream compressor)
  - `lz4.rs` — `Lz4Compressor` (LZ4 lossless via lz4_flex, fastest decompression)
  - `blosc2.rs` — `Blosc2Compressor` (multi-codec meta-compressor, chunk-based random access via SChunk)
  - `zfp.rs` — `ZfpCompressor` (lossy floating-point, fixed-rate/precision/accuracy modes, range decode support)
  - `sz3.rs` — `Sz3Compressor` (SZ3 error-bounded lossy, absolute/relative/PSNR modes)
- `zfp_ffi.rs` — Safe Rust wrapper around ZFP C library (compress/decompress/range for f64 arrays)
- `pipeline.rs` — Two-phase dispatch (build_compressor → compress/decompress), `decode_range_pipeline()` supports all compressors with random access capability

### tensogram-cli (5 tests)
- `tensogram info/ls/dump/get/set/copy` subcommands
- Where-clause filtering (`-w`), key selection (`-p`), JSON output (`-j`)
- Immutable key protection in `set`
- Object-level metadata mutations in `set` via `objects.<index>.<path>`
- Payload hash preservation in `set` when payload bytes are unchanged
- Filename placeholder expansion in `copy`
- First-match metadata lookup semantics in `get`, `copy`, and filters for multi-object messages

### tensogram-ffi (C FFI)
- Full C API with opaque handles (`TgmMessage`, `TgmMetadata`, `TgmFile`, `TgmScanResult`)
- `tgm_encode()` — JSON metadata + raw data slices → wire-format bytes
- `tgm_decode()`, `tgm_decode_metadata()`, `tgm_decode_object()`, `tgm_decode_range()`
- `tgm_scan()` — multi-message buffer scanning
- File API: `tgm_file_open/create/message_count/decode_message/read_message/append_raw/close`
- Typed accessors: `tgm_object_shape/strides/dtype/data`, `tgm_metadata_get_string/int/float`
- `tgm_simple_packing_compute_params()` — direct packing parameter computation
- Iterator API: `tgm_buffer_iter_*`, `tgm_file_iter_*`, `tgm_object_iter_*` with create/next/free pattern
- Thread-local error messages via `tgm_last_error()`
- Auto-generated `tensogram.h` via cbindgen
- Static library (`libtensogram_ffi.a`) + shared library (`libtensogram_ffi.dylib/.so`)

### tensogram-python (PyO3 bindings)
- Full Python API with numpy integration (returns `numpy.ndarray` directly)
- `tensogram.encode()` — dict metadata + list of `(descriptor_dict, ndarray)` pairs → bytes
- `tensogram.decode()` — bytes → `(Metadata, list[(DataObjectDescriptor, ndarray)])`
- `tensogram.decode_metadata()` — bytes → `Metadata` (no payload read)
- `tensogram.decode_object()` — bytes → `(Metadata, DataObjectDescriptor, ndarray)` by index
- `tensogram.decode_range()` — partial sub-tensor extraction → flat `ndarray`
- `tensogram.scan()` — `bytes → list[(offset, length)]`
- `tensogram.compute_packing_params()` — numpy array → packing parameters dict
- `TensogramFile` class: `open/create/append/message_count/decode_message/read_message/messages`
  - Context manager protocol (`with TensogramFile.create(...) as f:`)
  - `len(f)` returns message count
- `Metadata`: `meta.version`, `meta["key"]`, `"key" in meta`, `meta.extra`
- `DataObjectDescriptor`: all tensor + encoding fields as properties
- All 10 numeric numpy dtypes accepted natively by encoder (u8–u64, i8–i64, f32, f64)
- All 10 numeric dtypes decoded to correct numpy dtype (including `decode_range`)
- CBOR ↔ Python type conversion (str, int, float, bool, None, list, dict, bytes)
- Safe i128→i64 bounds check for CBOR integers
- Strict `byte_order` validation (rejects unknown values)
- DRY dtype handling via `decode_ne_vec!`/`numpy_from_ne!`/`numpy_flat_from_ne!` macros
- Overflow-safe dimension calculation for unsupported dtypes (checked_mul)
- `decode_range` validates byte count vs expected elements (no silent truncation)
- Bitmask/float16/complex fallback returns raw bytes (fixed: was producing empty array for bitmask)
- Performance: `from_slice` zero-copy for u8/i8 numpy conversion (avoids `.to_vec()` allocation)
- Built via `maturin develop` (excluded from default workspace build, inline deps)
- 95 Python tests (ruff-clean): parametrized dtype round-trip, multi-object, metadata, file API, multi-range decode, zero-object messages, big-endian, idempotency, wire determinism, errors, edge cases
- ruff configured: E/W/F/I/N/UP/B/SIM/PT/RUF rules, line-length=99, 0 warnings

## Wire format history

### v1 (initial implementation, 2026-04-02)
- Monolithic binary header: TENSOGRM magic + total_length + metadata_offset + metadata_length + num_objects + object_offsets[]
- Single CBOR metadata block with `objects[]` and `payload[]` arrays
- OBJS/OBJE markers per data object
- 39277777 terminator

### v2 (current, 2026-04-03)
- Frame-based format: Preamble + optional header frames + data object frames + optional footer frames + Postamble
- Each data object has its own CBOR descriptor embedded in its frame
- Merged `ObjectDescriptor` + `PayloadDescriptor` → `DataObjectDescriptor`
- `Metadata` → `GlobalMetadata` (version + free-form extra map)
- Streaming support: total_length=0, footer-based index
- Two-pass index construction for non-streaming mode
- All panic paths eliminated from library code

## Key design properties implemented
- Frame-based wire format with streaming support (v2)
- Deterministic CBOR (RFC 8949 §4.2 canonical key ordering)
- Per-object encoding pipelines with independent byte order
- Payload integrity hashing (xxh3 default)
- FR/ENDF frame markers for corruption detection
- Multi-message file scanning with corruption recovery
- Partial range decode (szip, blosc2, zfp fixed-rate; stream compressors zstd/lz4/sz3 return RangeNotSupported)
- No panics in library code — all fallible operations return Result

## Code quality & features (2026-04-04)

### Code quality improvements
- Decomposed `encode_message` (framing.rs) into 5 focused helpers: `build_hash_frame_cbor()`, `build_index_frame()`, `compute_object_offsets()`, `compute_message_flags()`, `assemble_message()` — orchestrator is now ~30 lines
- Added `DecodePhase` enum for frame ordering validation in `decode_message()` — rejects header frames after data objects, data objects after footers
- Replaced `fs::read()` in `ensure_scanned()` with streaming `scan_file()` that reads preamble-sized chunks + seeks, avoiding full-file memory load
- Added `Debug` derive on `DecodedMessage`

### Feature gates
- `mmap` feature: `memmap2` behind `#[cfg(feature = "mmap")]`, adds `TensogramFile::open_mmap()` with zero-copy scan + read
- `async` feature: `tokio` behind `#[cfg(feature = "async")]`, adds `open_async()`, `read_message_async()`, `decode_message_async()` — all FFI/CPU work runs via `spawn_blocking`

### Golden binary test files
- 5 canonical `.tgm` files in `tests/golden/`: simple_f32, multi_object (3 dtypes), mars_metadata, multi_message, hash_xxh3
- 6 integration tests verifying decode correctness, determinism, and hash verification
- Files are byte-for-byte deterministic for cross-language interoperability testing

### ciborium canonical encoding verification
- Added `verify_canonical_cbor()` utility in metadata.rs — checks RFC 8949 §4.2.1 canonical map key ordering
- 7 new tests: all CBOR outputs verified canonical, non-canonical CBOR rejected, nested maps sorted, insertion-order independence confirmed

### Test count: 157 tests (was 137), 0 clippy warnings

## TODO implementation progress (2026-04-04)

### Dependency cleanup: Remove md5/sha1 hash support
- Removed `sha1` 0.10 and `md5` 0.7 from workspace and tensogram-core dependencies
- Simplified `hash.rs` to xxh3-only: removed `Sha1`/`Md5` variants from `HashAlgorithm`
- Unknown hash types now return clear error: "unknown hash type: {name}"
- Updated FFI (`tensogram-ffi`), Python bindings (`tensogram-python`), examples, and docs
- Golden test files regenerated (byte-for-byte deterministic with new format)

### Metadata frame CBOR structure: common/payload/reserved
- Added `common`, `payload`, `reserved` fields to `GlobalMetadata` struct
- Fields use `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]` for backwards compatibility
- Old messages without these fields decode correctly (flat keys land in `extra`)
- Added `Default` impl for `GlobalMetadata` (version=2, all maps empty)
- 2 new integration tests: `test_metadata_common_payload_reserved_round_trip`, `test_metadata_empty_sections_not_serialized`

### Streaming API: `StreamingEncoder`
- New `streaming.rs` module in tensogram-core with `StreamingEncoder<W: Write>`
- API: `new()` → writes preamble (total_length=0) + header metadata frame
- `write_object()` → encodes via pipeline and writes data object frame immediately
- `finish()` → writes footer hash + footer index + postamble, returns writer
- Supports hash computation per-object, encoding pipeline, 8-byte alignment
- 6 tests: single/multi object round-trip, matches buffered encode, hash verification, zero objects, metadata preservation

### CLI tools: merge, split, reshuffle
- `tensogram merge` — merges messages from multiple files into one; metadata merged (first takes precedence)
- `tensogram split` — splits multi-object messages into separate single-object files with `[index]` template
- `tensogram reshuffle` — converts streaming-mode (footer) to random-access (header) via decode→re-encode
- 2 unit tests for split filename expansion

### tensogram-grib crate + convert-grib CLI
- New `crates/tensogram-grib/` crate (excluded from default workspace build, requires ecCodes C library)
- `convert_grib_file()` API: reads GRIB via ecCodes `CodesFile`, extracts ~40 MARS namespace keys
- Two grouping modes: `OneToOne` (1 GRIB → 1 TGM) and `MergeAll` (N GRIBs → 1 TGM with N objects)
- Key partitioning: identical keys → `GlobalMetadata.common`, varying keys → `DataObjectDescriptor.params`
- `tensogram convert-grib` CLI subcommand (feature-gated behind `grib` feature in tensogram-cli)
- 3 unit tests for key partitioning logic
- mdbook docs: GRIB overview + MARS key mapping with mermaid diagram

### Documentation & examples
- Shortened README.md from 302 lines to 100 lines; detailed examples moved to mdbook
- New streaming example: `examples/rust/src/bin/11_streaming.rs`
- New CLI docs: merge, split, reshuffle pages
- New GRIB docs: overview + metadata-mapping pages
- Updated SUMMARY.md with new CLI and GRIB sections

### Dependency audit: Feature-gated compression
- All 6 compression codecs (szip, zstd, lz4, blosc2, zfp, sz3) are now optional features in `tensogram-encodings`
- Features forwarded through `tensogram-core` via `szip = ["tensogram-encodings/szip"]` etc.
- All features are `default = [all 6]` so existing builds are unchanged
- Lightweight builds: `cargo build --no-default-features` skips all C FFI compression deps
- `CompressionType` enum variants, `build_compressor()` dispatch, and `build_pipeline_config()` match arms all gated with `#[cfg(feature = "...")]`
- Added `CompressionError::NotAvailable` variant for disabled features
- Updated ARCHITECTURE.md with feature gate table

### Test count: 167 Rust tests (was 165), 84 Python tests, 0 clippy warnings

## Examples

### examples/rust/ (10 runnable examples, workspace member)
- `01_encode_decode` — basic round-trip, all message fields
- `02_mars_metadata` — MARS namespace at message and object level
- `03_simple_packing` — lossy compression, precision measurement
- `04_shuffle_filter` — byte-level shuffle, direct API
- `05_multi_object` — multiple tensors per message, per-object metadata
- `06_hash_verification` — xxh3 hashing, corruption detection
- `07_scan_buffer` — multi-message buffer, injected corruption, recovery
- `08_decode_variants` — all four decode functions with edge cases
- `09_file_api` — full TensogramFile lifecycle
- `10_iterators` — buffer/object/file iteration patterns

### examples/cpp/ (intended C FFI API, 5 examples)
- `01_encode_decode.cpp`, `02_mars_metadata.cpp`, `03_simple_packing.cpp`, `04_file_api.cpp`, `05_iterators.cpp`
- `README.md` — planned header (`tensogram.h`) with full function signatures

### examples/python/ (intended PyO3 API, 7 examples)
- `01_encode_decode.py` through `07_iterators.py`
- `README.md` — planned module structure, NumPy dtype mapping, error hierarchy

## Documentation (mdbook)

- `docs/` — mdbook source (build with `PATH="$HOME/.cargo/bin:$PATH" mdbook build` from `docs/`)
- Introduction, Concepts (messages, metadata, objects, pipeline)
- Wire Format (message layout, CBOR schema, dtypes)
- Developer Guide (quickstart, encoding, decoding, file API, iterators)
- Encodings (simple_packing, shuffle, compression)
- CLI Reference (info, ls, dump, get, set, copy)
- Edge Cases and Internals reference pages
- Mermaid diagrams throughout

## Dependencies
- ciborium 0.2 — CBOR encode/decode
- serde 1 — serialization framework
- thiserror 2 — error derive macros
- libaec-sys 0.1 — CCSDS 121.0-B-3 (szip) compression via libaec 1.1.4
- zstd 0.13 — Zstandard compression
- lz4_flex 0.11 — LZ4 compression (pure Rust)
- blosc2 0.2 — Blosc2 meta-compressor with chunk random access
- zfp-sys-cc 0.2 — ZFP floating-point compression (FFI to C library)
- sz3 0.4 — SZ3 error-bounded lossy compression
- xxhash-rust 0.8 — xxh3 payload hashing
- ~~sha1 0.10, md5 0.7~~ — removed (xxh3-only since Sprint 1.1)
- clap 4 — CLI argument parsing
- serde_json 1 — JSON output in CLI
- tempfile 3 — dev dependency for file tests
