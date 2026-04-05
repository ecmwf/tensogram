# Tensogram Library — Implementation Status

Implemented: 2026-04-03

## Workspace: 5 crates, 253 Rust tests + 103 C++ tests, 0 clippy warnings, 90.5% line coverage

### tensogram-core (43 unit tests + 42 integration + 12 adversarial + 84 edge-case)
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

### tensogram-ffi (C FFI — redesigned 2026-04-04)
- Redesigned C API with clean `tgm_` naming: `tgm_error`, `tgm_message_t`, `tgm_bytes_t`, etc.
- Enum values use `TGM_ERROR_` prefix: `TGM_ERROR_OK`, `TGM_ERROR_FRAMING`, etc.
- cbindgen.toml: `[export.rename]` for type names, `prefix_with_name = true` for enum variants
- Opaque handles: `tgm_message_t`, `tgm_metadata_t`, `tgm_file_t`, `tgm_scan_result_t`, `tgm_streaming_encoder_t`
- Core API: `tgm_encode()`, `tgm_decode()`, `tgm_decode_metadata()`, `tgm_decode_object()`, `tgm_decode_range()`
- Scan: `tgm_scan()` — multi-message buffer scanning
- File API: `tgm_file_open/create/message_count/decode_message/read_message/append_raw/append/path/close`
- Message accessors: `tgm_message_version/num_objects/metadata`, `tgm_object_ndim/shape/strides/dtype/data/type/byte_order/filter/compression/hash_type/hash_value`, `tgm_payload_encoding/has_hash`
- Metadata accessors: `tgm_metadata_version/num_objects/get_string/get_int/get_float`
- Iterator API: `tgm_buffer_iter_*`, `tgm_file_iter_*`, `tgm_object_iter_*` with create/next/free pattern
- Streaming encoder: `tgm_streaming_encoder_create/write/count/finish/free`
- Hash utilities: `tgm_compute_hash()` for arbitrary data
- Error utilities: `tgm_error_string()` converts error code to static string, `tgm_last_error()` for thread-local messages
- Simple packing: `tgm_simple_packing_compute_params()` — direct packing parameter computation
- Centralized `build_message_caches()` helper: all TgmMessage construction sites share cache building for descriptor string accessors
- Auto-generated `tensogram.h` (484 lines) via cbindgen with `usize_is_size_t = true`
- Static library (`libtensogram_ffi.a`) + shared library (`libtensogram_ffi.dylib/.so`)

### C++ wrapper (header-only, 100 tests — new 2026-04-04)
- `include/tensogram.hpp` — single-header C++17 wrapper, ~870 lines
- RAII classes with `std::unique_ptr` + custom deleters: `message`, `metadata`, `file`, `buffer_iterator`, `file_iterator`, `object_iterator`, `streaming_encoder`
- Typed exception hierarchy: `error` → `framing_error`, `metadata_error`, `encoding_error`, `compression_error`, `object_error`, `io_error`, `hash_mismatch_error`, `invalid_arg_error`
- `decoded_object` non-owning view with `data_as<T>()`, `element_count<T>()`, full descriptor access
- `message::iterator` for range-based for loops over objects
- Free functions: `encode()`, `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()`, `scan()`, `compute_hash()`
- `file` class: `open()`, `create()`, `append()`, `append_raw()`, `decode_message()`, `read_message()`, `message_count()`, `path()`
- `streaming_encoder`: `write_object()`, `object_count()`, `finish()`
- Value types: `scan_entry`, `encode_options`, `decode_options`
- C++ Core Guidelines compliant: `[[nodiscard]]`, `noexcept` on deleters/moves, `const`-correct, `explicit` constructors, Rule of Five
- Doxygen-documented classes and methods, thread-safety notes, lifetime documentation
- CMake build system: `CMakeLists.txt` (root) + `tests/cpp/CMakeLists.txt` with GoogleTest v1.15.2
- 100 GoogleTest tests across 10 files: encode_decode (12), metadata (12), descriptor (11), file (12), iterators (9), error (8), edge_cases (16), simple_packing (5), streaming (6), multi_dtype (10)
- All 5 C++ examples rewritten to use the C++ wrapper API

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

### Test count:
- 181 Rust tests
- 95 Python tests
- 103 C++ tests
- 0 clippy warnings

## C++ wrapper & build system (2026-04-04)

### Header-only C++17 wrapper (`include/tensogram.hpp`)
- `tensogram` namespace wrapping entire C API
- RAII classes: `message`, `metadata`, `file`, `buffer_iterator`, `file_iterator`, `object_iterator`, `streaming_encoder`
- Each class holds `std::unique_ptr<T, custom_deleter>` for automatic cleanup
- Move-only semantics (copy suppressed, move defaulted)
- Typed exception hierarchy: `error` base → `framing_error`, `metadata_error`, `encoding_error`, `compression_error`, `object_error`, `io_error`, `hash_mismatch_error`, `invalid_arg_error`
- `detail::check()` maps all C error codes to typed exceptions
- `decoded_object` non-owning view with typed data access (`data_as<T>()`, `element_count<T>()`)
- `message::iterator` for range-based for over decoded objects
- Free functions: `encode()`, `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()`, `scan()`, `compute_hash()`
- Value types: `scan_entry`, `encode_options`, `decode_options`
- C++17 only (`std::string_view` for parameters, no C++20 features)

### CMake build system
- Root `CMakeLists.txt`: builds Rust static library via `cargo build --release`, imports as CMake target, INTERFACE header-only library, platform-specific system library linking (macOS frameworks, Linux dl/pthread)
- `tests/cpp/CMakeLists.txt`: GoogleTest v1.15.2 via FetchContent, `tensogram_tests` executable
- 12 passing C++ tests: basic round-trip, helper round-trip, metadata access, decode metadata only, decode single object, descriptor fields, scan buffer, compute hash, hash verification, message iterator, invalid buffer error, error code preservation

### Metadata improvements (2026-04-04)

#### Doc page: `docs/src/format/metadata-values.md`
- New documentation page on allowed/forbidden CBOR types for metadata values
- Covers: string, int, float, bool, null, array, map (string keys only)
- Forbidden: byte strings, CBOR tags, undefined, half-precision floats
- Documents `payload.objects` auto-populated summary structure
- Updated `docs/src/SUMMARY.md`, `docs/src/concepts/metadata.md`, `docs/src/format/cbor-metadata.md`

#### Auto-populate `payload["objects"]` in encoder
- `build_payload_objects_summary()` helper in `encode.rs` — builds CBOR array with `{ndim, shape, strides, dtype}` per object, plus `gridName` if present in params
- Buffered encoder (`encode()`) clones GlobalMetadata, inserts `payload["objects"]`, passes enriched metadata
- Streaming encoder (`StreamingEncoder`) accumulates `completed_objects`, writes `FooterMetadata` with payload-enriched metadata in `finish()`
- Regenerated all 5 golden test files

#### Dynamic MARS namespace iteration (no hardcoded keys)
- Deleted `crates/tensogram-grib/src/keys.rs` (was 49-line hardcoded key list)
- Rewrote `extract_mars_keys()` in `metadata.rs` to use `msg.new_keys_iterator("mars")` — discovers keys at runtime
- Two-phase approach: collect key names (holds `&mut msg`), drop iterator, then read values
- Library is now vocabulary-agnostic — only the CLI knows about specific MARS keys

#### Extract `gridType` into per-object params
- Read `msg.read_key::<String>("gridType")` for each GRIB message (not in MARS namespace)
- Stored as `"gridName"` in per-object `params` for `build_payload_objects_summary()` propagation
- Injected into params for both `OneToOne` and `MergeAll` grouping modes

#### ECMWF opendata GRIB test fixtures
- 4 real GRIB files in `crates/tensogram-grib/testdata/` downloaded via byte-range HTTP
- `lsm.grib2` (land-sea mask, sfc, 188KB), `2t.grib2` (2m temp, sfc, 661KB)
- `q_150.grib2` (specific humidity, 150 hPa, 477KB), `t_600.grib2` (temperature, 600 hPa, 511KB)
- Source: ECMWF IFS 0.25° operational forecast, 2026-04-04 00z step 0h
- `download.sh` script for reproducibility

#### Integration tests for tensogram-grib
- 7 integration tests in `crates/tensogram-grib/tests/integration.rs`
- `test_lsm_convert` — metadata verification (MARS keys, shape 721×1440)
- `test_2t_round_trip` — f64 data round-trip, finite values, temperature sanity
- `test_q_pl_convert` — pressure-level metadata (levelist/level key present)
- `test_t_pl_round_trip` — temperature range check (200–320 K)
- `test_multi_merge` — MergeAll: common keys shared, varying in per-object params
- `test_multi_split` — OneToOne: 2 GRIB → 2 Tensogram messages
- `test_payload_objects_metadata` — verifies ndim/shape/strides/dtype/gridName in payload.objects

### Metadata restructuring: payload as array + mars namespace (2026-04-04)

#### `GlobalMetadata.payload` type change
- Changed from `BTreeMap<String, CborValue>` to `Vec<BTreeMap<String, CborValue>>`
- Wire format: payload CBOR value changed from map `{"objects": [...]}` to array `[{...}, ...]`
- Each entry corresponds to one data object in the message
- Encoder auto-populates `ndim`, `shape`, `strides`, `dtype` into each entry
- Pre-existing keys (e.g. `"mars"`) preserved via merge

#### MARS keys under namespaced sub-object
- Common MARS keys → `common["mars"]` (shared across all objects)
- Per-object varying MARS keys → `payload[i]["mars"]`
- GRIB `gridType` stored as `"grid"` in the mars namespace
- `DataObjectDescriptor.params` no longer carries MARS keys — encoding params only
- All Rust examples migrated from `extra["mars"]` / `params["mars"]` to `common["mars"]` / `payload[i]["mars"]`

#### Encoder changes
- Renamed `build_payload_objects_summary()` → `populate_payload_entries()`
- New signature: mutates `&mut Vec<BTreeMap>` in-place (extend/truncate to object count)
- Streaming encoder (`finish()`) uses same pattern

#### CLI split correctness fix
- `split.rs` now extracts `payload[idx]` per-object when splitting multi-object messages
- Prevents loss of per-object metadata (mars keys) during split

#### Golden files regenerated
- All 5 `.tgm` files regenerated for new wire format
- `test_golden_mars_metadata` migrated from `extra["mars"]` to `common["mars"]`

#### `preserve_all_keys` option for GRIB converter
- New `preserve_all_keys: bool` field on `ConvertOptions` (default `false`)
- When enabled, extracts keys from 6 non-mars ecCodes namespaces: `ls`, `geography`, `time`, `vertical`, `parameter`, `statistics`
- Keys stored under `common["grib"]["<namespace>"]["<key>"]` (shared) and `payload[i]["grib"]["<namespace>"]["<key>"]` (varying)
- Same common/varying partitioning as mars keys, applied per-namespace independently
- Refactored `metadata.rs`: extracted `read_namespace_keys()` helper, `dynamic_to_cbor()`, `partition_flat_keys()`, `partition_grib_keys()`
- CLI: `--all-keys` flag on `convert-grib` subcommand
- 3 new integration tests + 2 new unit tests for `partition_grib_keys`

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

### examples/cpp/ (C++ wrapper API, 5 examples)
- `01_encode_decode.cpp`, `02_mars_metadata.cpp`, `03_simple_packing.cpp`, `04_file_api.cpp`, `05_iterators.cpp`
- All examples use the C++ wrapper (`tensogram.hpp`), not the raw C FFI
- `README.md` — API overview, error handling, CMake + manual build instructions

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

## Security/quality audit fixes (2026-04-04)

### Rust FFI (crates/tensogram-ffi/src/lib.rs)
- Fix 1: Vec capacity UB — added `shrink_to_fit()` before all 5 `std::mem::forget(bytes)` sites to ensure `capacity == len` for safe `tgm_bytes_free` reconstruction via `Vec::from_raw_parts`
- Fix 2: FFI panic safety — added `panic = "abort"` to both `[profile.release]` and `[profile.dev]` in workspace `Cargo.toml`, preventing UB from panic unwinding across FFI boundary
- Fix 3: `tgm_streaming_encoder_finish` double-free — removed `Box::from_raw(enc)` from success path; handle is now left valid-but-empty, caller must always call `tgm_streaming_encoder_free`
- Fix 4: Null data pointer UB — added per-element null check in both `tgm_encode` and `tgm_file_append` before constructing slices; returns `InvalidArg` with index in error message
- Fix 5: `tgm_error_string` non-exhaustive enum — changed to integer-based matching (`err as i32`) with wildcard `_ => "unknown error"` for safety against invalid discriminants from C callers

### C++ wrapper (include/tensogram.hpp)
- Fix 6: `streaming_encoder::finish()` — removed `handle_.release()` since C side no longer frees; destructor safely calls `tgm_streaming_encoder_free` on the empty shell
- Fix 7: Extracted `detail::scatter_gather` helper — deduplicates ptr/len vector building in `encode()` and `file::append()`
- Fix 8: Added `-Wall -Wextra -Wpedantic -Wno-unused-parameter` to `tests/cpp/CMakeLists.txt` via `target_compile_options`
- Fix 9: Fixed `examples/cpp/README.md` build command — changed `-I build/generated` to `-I crates/tensogram-ffi`

### Verification: 167 Rust tests + 103 C++ tests pass, 0 clippy warnings, 0 compiler warnings

## Third-pass audit fixes (2026-04-04)

### test_helpers.hpp
- Fix: `temp_path()` — `mkstemp()` creates a base file that was never deleted; added `std::remove(tmpl)` after `close(fd)` to release the base file before appending the suffix.

### include/tensogram.hpp
- Doc: `file::message_count()` — expanded Doxygen comment to explain non-const / lazy-scan semantics so callers are not surprised they cannot call it on `const file&`.
- Doc: `file::raw()` — added `@warning` documenting that the returned pointer is non-owning and must not be freed or stored beyond the `file` lifetime.

### tests/cpp/test_iterators.cpp
- Fix: `ObjectIteratorSingleObject` — removed two dead variables (`iter` and `msg`) that created unused iterator and message objects.

### tests/cpp/test_encode_decode.cpp
- New: `DecodeRangePartial` — exercises `tensogram::decode_range()` with a sub-range request `[2, 3)` on an 8-element float32 message; verifies partial element extraction.
- New: `DecodeRangeFull` — exercises `tensogram::decode_range()` with a full-range request; round-trips all 3 floats.

### tests/cpp/test_error.cpp
- New: `HashMismatchThrowsHashMismatchError` — encodes with xxh3, corrupts two bytes at the payload offset (53% into the wire buffer), decodes with `verify_hash=true`; asserts `tensogram::hash_mismatch_error` specifically (not just `tensogram::error`).

### examples/cpp/README.md
- Fix: manual build command was missing platform link flags; split into Linux (`-ldl -lpthread -lm`) and macOS (`-framework CoreFoundation -framework Security -framework SystemConfiguration -lc++ -lm`) sections.

## tensogram-xarray (separate Python package, 71 tests)
- xarray backend engine for `.tgm` files — `engine="tensogram"` via entry_points
- `TensogramBackendEntrypoint` — `open_dataset()`, `guess_can_open()` (`.tgm` extension)
- `TensogramBackendArray` — lazy loading via `BackendArray`, pickle-safe for dask
  - Full N-D random-access: maps N-D slices to flat byte ranges with adjacent-range merging, uses `decode_range()` with `none`/`szip`/`blosc2`/`zfp`(fixed_rate) compressors
  - Ratio-based heuristic (`range_threshold` parameter, default 0.5) — falls back to full decode when selected ranges exceed threshold fraction of total size
  - Falls back to `decode_object()` + in-memory slice for stream compressors or shuffle filter
- `decode_range` API changed: returns split results per range by default (`join=True` for concatenated flat array)
- `meta.common` and `meta.payload` getters added to PyMetadata
- Coordinate auto-detection by name matching (13 known names: lat/latitude, lon/longitude, time, level, etc.)
- User-specified dimension mapping (`dim_names`) and variable naming (`variable_key` with dotted paths)
- File scanner: metadata extraction from all messages/objects via `desc.params`
- Auto-merge: `open_datasets()` groups compatible objects (same shape/dtype) into hypercubes
- Auto-split: incompatible objects go to separate Datasets
- Documentation: `docs/src/guide/xarray-integration.md` — 7 worked examples covering the full conversion philosophy
- Per-object metadata stored in descriptor extra keys (accessible via `desc.params`)

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
