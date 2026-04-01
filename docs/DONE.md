# Tensogram Library — Implementation Status

Implemented: 2026-04-02

## Workspace: 5 crates, 52 tests, 0 clippy warnings

### tensogram-core (21 unit tests + 14 integration tests)
- `wire.rs` — Binary header with TENSOGRM magic, terminator, object offsets
- `framing.rs` — `encode_frame()`, `decode_frame()`, `extract_object_payload()`, `scan()`
- `metadata.rs` — Deterministic CBOR encoding (two-step: serialize → canonicalize → write)
- `types.rs` — `Metadata`, `ObjectDescriptor`, `PayloadDescriptor`, `ByteOrder`, `HashDescriptor`
- `dtype.rs` — All 15 dtypes (float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask)
- `hash.rs` — xxh3, sha1, md5 hashing + verification
- `encode.rs` — Full encode pipeline: validate → encode per object → hash → CBOR → frame
- `decode.rs` — `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()`
- `file.rs` — `TensogramFile`: open, create, lazy scan, append, messages iterator, random access

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

### tensogram-ffi + tensogram-python
- Stubs ready for future implementation

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
- C FFI (opaque handles + typed getters via cbindgen)
- Python bindings (PyO3/maturin)
- Cross-language golden binary test files
- `tensogram filter` subcommand (v2 rules engine)
- ciborium canonical encoding verification (current two-step approach works but should be validated against a reference implementation)

## Dependencies
- ciborium 0.2 — CBOR encode/decode
- serde 1 — serialization framework
- thiserror 2 — error derive macros
- xxhash-rust 0.8 — xxh3 payload hashing
- sha1 0.10, md5 0.7 — legacy hash support
- clap 4 — CLI argument parsing
- serde_json 1 — JSON output in CLI
- tempfile 3 — dev dependency for file tests
