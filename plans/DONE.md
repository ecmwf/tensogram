# Tensogram Library — Implementation Status

Implemented: 2026-04-03

## Workspace: 5 crates, 137 tests, 0 clippy warnings

### tensogram-core (43 unit tests + 42 integration + 12 adversarial)
- `wire.rs` — v2 frame-based wire format: Preamble (24B), FrameHeader (16B), Postamble (16B), FrameType enum, MessageFlags, DataObjectFlags
- `framing.rs` — `encode_message()` with two-pass index construction, `decode_message()`, `scan()` for multi-message buffers
- `metadata.rs` — Deterministic CBOR encoding for GlobalMetadata, DataObjectDescriptor, IndexFrame, HashFrame (three-step: serialize → canonicalize → write)
- `types.rs` — `GlobalMetadata`, `DataObjectDescriptor` (merged object + payload), `IndexFrame`, `HashFrame`, `DecodedObject` type alias
- `dtype.rs` — All 15 dtypes (float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask)
- `hash.rs` — xxh3, sha1, md5 hashing + verification
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
- `tensogram.encode()` — dict metadata + list of byte arrays → bytes
- `tensogram.decode()` — bytes → `(Metadata, list[numpy.ndarray])`
- `tensogram.decode_metadata()` — bytes → `Metadata` (no payload read)
- `tensogram.decode_object()` — bytes → `(ObjectDescriptor, numpy.ndarray)` by index
- `tensogram.decode_range()` — partial sub-tensor extraction → `numpy.ndarray`
- `tensogram.scan()` — `bytes → list[(offset, length)]`
- `tensogram.compute_packing_params()` — numpy array → packing parameters dict
- `TensogramFile` class: `open/create/append/message_count/decode_message/read_message/messages`
- Dict-like access: `meta['mars']['class']`
- CBOR ↔ Python type conversion (str, int, float, bool, None, list, dict)
- Built via `maturin develop` (excluded from default workspace build)

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

## Examples

### examples/rust/ (10 runnable examples, workspace member)
- `01_encode_decode` — basic round-trip, all message fields
- `02_mars_metadata` — MARS namespace at message and object level
- `03_simple_packing` — lossy compression, precision measurement
- `04_shuffle_filter` — byte-level shuffle, direct API
- `05_multi_object` — multiple tensors per message, per-object metadata
- `06_hash_verification` — xxh3/sha1/md5, corruption detection
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
- sha1 0.10, md5 0.7 — legacy hash support
- clap 4 — CLI argument parsing
- serde_json 1 — JSON output in CLI
- tempfile 3 — dev dependency for file tests
