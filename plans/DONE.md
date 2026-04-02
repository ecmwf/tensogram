# Tensogram Library — Implementation Status

Implemented: 2026-04-02

## Workspace: 5 crates, 64 tests, 0 clippy warnings

### tensogram-core (23 unit tests + 14 integration tests)
- `wire.rs` — Binary header with TENSOGRM magic, terminator, object offsets
- `framing.rs` — `encode_frame()`, `decode_frame()`, `extract_object_payload()`, `scan()`
- `metadata.rs` — Deterministic CBOR encoding (two-step: serialize → canonicalize → write)
- `types.rs` — `Metadata`, `ObjectDescriptor`, `PayloadDescriptor`, `ByteOrder`, `HashDescriptor`
- `dtype.rs` — All 15 dtypes (float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask)
- `hash.rs` — xxh3, sha1, md5 hashing + verification
- `encode.rs` — Full encode pipeline: validate → encode per object → hash → CBOR → frame
- `decode.rs` — `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()`
- `file.rs` — `TensogramFile`: open, create, lazy scan, append, seek-based random access, deprecated `messages()` iterator

### tensogram-encodings (14 tests)
- `simple_packing.rs` — GRIB-style lossy quantization, MSB-first bit packing, 0-64 bits, NaN rejection
- `shuffle.rs` — Byte-level shuffle/unshuffle (HDF5-style)
- `compression.rs` — `Compressor` trait with `NoopCompressor` and `SzipCompressor` (stub)
- `pipeline.rs` — Encode → filter → compress dispatch

### tensogram-cli (3 tests)
- `tensogram info/ls/dump/get/set/copy` subcommands
- Where-clause filtering (`-w`), key selection (`-p`), JSON output (`-j`)
- Immutable key protection in `set`
- Filename placeholder expansion in `copy`

### tensogram-ffi (C FFI)
- Full C API with opaque handles (`TgmMessage`, `TgmMetadata`, `TgmFile`, `TgmScanResult`)
- `tgm_encode()` — JSON metadata + raw data slices → wire-format bytes
- `tgm_decode()`, `tgm_decode_metadata()`, `tgm_decode_object()`, `tgm_decode_range()`
- `tgm_scan()` — multi-message buffer scanning
- File API: `tgm_file_open/create/message_count/decode_message/read_message/append_raw/close`
- Typed accessors: `tgm_object_shape/strides/dtype/data`, `tgm_metadata_get_string/int/float`
- `tgm_simple_packing_compute_params()` — direct packing parameter computation
- Thread-local error messages via `tgm_last_error()`
- Auto-generated `tensogram.h` via cbindgen
- Static library (`libtenogram_ffi.a`) + shared library (`libtenogram_ffi.dylib/.so`)

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
- `Metadata`, `ObjectDescriptor`, `PayloadDescriptor` Python classes with property accessors
- Dict-like access: `meta['mars']['class']`
- CBOR ↔ Python type conversion (str, int, float, bool, None, list, dict)
- Built via `maturin develop` (excluded from default workspace build)

## Key design properties implemented
- Binary header index (deterministic size, O(1) object access)
- Deterministic CBOR (RFC 8949 §4.2 canonical key ordering)
- Per-object encoding pipelines with independent byte order
- Payload integrity hashing (xxh3 default)
- OBJS/OBJE corruption markers per object
- Multi-message file scanning with corruption recovery
- Partial range decode (uncompressed path)

## Not yet implemented
- szip/libaec compression (stub in place, returns error — needs libaec C bindings)
- Partial range decode for szip-compressed data (RSI block seeking, ~200-300 LOC)
- `async` feature gate (tokio + spawn_blocking for libaec FFI)
- `mmap` feature gate (memmap2 for memory-mapped file access)
- Streaming mode (total_length=0 path)
- Cross-language golden binary test files
- `tensogram filter` subcommand (v2 rules engine)
- ciborium canonical encoding verification (current two-step approach works but should be validated against a reference implementation)

## Examples

### examples/rust/ (9 runnable examples, workspace member)
- `01_encode_decode` — basic round-trip, all message fields
- `02_mars_metadata` — MARS namespace at message and object level
- `03_simple_packing` — lossy compression, precision measurement
- `04_shuffle_filter` — byte-level shuffle, direct API
- `05_multi_object` — multiple tensors per message, per-object metadata
- `06_hash_verification` — xxh3/sha1/md5, corruption detection
- `07_scan_buffer` — multi-message buffer, injected corruption, recovery
- `08_decode_variants` — all four decode functions with edge cases
- `09_file_api` — full TensogramFile lifecycle

### examples/cpp/ (intended C FFI API, 4 examples)
- `01_encode_decode.cpp`, `02_mars_metadata.cpp`, `03_simple_packing.cpp`, `04_file_api.cpp`
- `README.md` — planned header (`tensogram.h`) with full function signatures

### examples/python/ (intended PyO3 API, 6 examples)
- `01_encode_decode.py` through `06_hash_and_errors.py`
- `README.md` — planned module structure, NumPy dtype mapping, error hierarchy

## Documentation (mdbook)

- `docs/` — mdbook source (build with `PATH="$HOME/.cargo/bin:$PATH" mdbook build` from `docs/`)
- Introduction, Concepts (messages, metadata, objects, pipeline)
- Wire Format (message layout, CBOR schema, dtypes)
- Developer Guide (quickstart, encoding, decoding, file API)
- Encodings (simple_packing, shuffle, compression)
- CLI Reference (info, ls, dump, get, set, copy)
- Edge Cases and Internals reference pages
- Mermaid diagrams throughout

## Dependencies
- ciborium 0.2 — CBOR encode/decode
- serde 1 — serialization framework
- thiserror 2 — error derive macros
- xxhash-rust 0.8 — xxh3 payload hashing
- sha1 0.10, md5 0.7 — legacy hash support
- clap 4 — CLI argument parsing
- serde_json 1 — JSON output in CLI
- tempfile 3 — dev dependency for file tests
