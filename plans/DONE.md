# Tensogram — Current Implementation Status (v0.6.0)

> For historical release notes, see `../CHANGELOG.md`.
> For planned features, see `TODO.md`. For ideas, see `IDEAS.md`.

## Summary

- **Version:** 0.6.0
- **Workspace:** 6 default crates + 2 optional (Python, GRIB) + 2 separate packages (xarray, zarr)
- **Tests:** 1008 total (283 Rust + 226 Python + 181 xarray + 204 Zarr + 105 C++ + 7 GRIB new + 2 streamer)
- **Quality:** 0 clippy warnings, 90.5% Rust line coverage

## tensogram-benchmarks

9 tests (8 smoke + 1 GRIB, gated on `eccodes` feature). Separate workspace crate.

- `datagen.rs` — Deterministic SplitMix64-based synthetic weather field generator
  (smooth sinusoidal temperature field, base ≈ 280 K, amplitude ≈ 30 K, ±0.1 K noise).
- `report.rs` — ASCII table formatter with reference-relative comparison columns
  (`vs Ref Enc`, `vs Ref Dec`, `Ratio %`, `Size KiB`). `median_ns` helper for timing.
- `codec_matrix.rs` — 24 pipeline combos: baseline + raw f64 + simple_packing × {none,zstd,lz4,blosc2,szip} × {16,24,32 bits} + ZFP (rate 16/24/32) + SZ3 (abs=0.01).
- `grib_comparison.rs` — ecCodes CCSDS (`grid_ccsds`) vs `grid_simple` vs tensogram `sp(24)+szip`. Feature-gated by `eccodes`. Uses raw C API via `extern "C"` declarations.
- Two binaries: `codec-matrix` (default) and `grib-comparison` (requires `--features eccodes`).
- `build.rs` — links `libeccodes` via pkg-config or Homebrew fallback when `eccodes` feature is active.
- Documentation: `docs/src/guide/benchmarks.md`.

## tensogram-core

Unit, integration, adversarial, and edge-case tests.

- `wire.rs` — v2 frame-based wire format: Preamble (24B), FrameHeader (16B), Postamble (16B), FrameType enum (incl. PrecederMetadata type 8), MessageFlags (incl. bit 6 PRECEDER_METADATA), DataObjectFlags
- `framing.rs` — `encode_message()` with two-pass index construction, `decode_message()`, `scan()` for multi-message buffers. Decomposed into 5 focused helpers.
- `metadata.rs` — Deterministic CBOR encoding for GlobalMetadata, DataObjectDescriptor, IndexFrame, HashFrame (three-step: serialize → canonicalize → write). `verify_canonical_cbor()` utility.
- `types.rs` — `GlobalMetadata` (version, base, `_reserved_`, `_extra_`), `DataObjectDescriptor`, `IndexFrame`, `HashFrame`
- `dtype.rs` — All 15 dtypes (float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask)
- `hash.rs` — xxh3 hashing + verification (xxh3 only)
- `encode.rs` — Full encode pipeline: validate → build pipeline config → encode per object → hash → assemble frames. Auto-populates `base[i]._reserved_.tensor` entries. Validates that client code does not write to `_reserved_`.
- `decode.rs` — `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()` (split results by default, `join` parameter)
- `file.rs` — `TensogramFile`: open, create, lazy scan, append, seek-based random access
- `iter.rs` — `MessageIter` (zero-copy buffer), `ObjectIter` (lazy per-object decode), `FileMessageIter` (seek-based file), `objects_metadata()` (descriptor-only)
- `streaming.rs` — `StreamingEncoder<W: Write>`: progressive encode, footer hash/index, no buffering, `write_preceder()` for per-object streaming metadata
- Feature gates: `mmap` (memmap2 zero-copy), `async` (tokio spawn_blocking)
- `DecodePhase` enum for frame ordering validation

## tensogram-encodings

47 tests.

- `simple_packing.rs` — GRIB-style lossy quantization, MSB-first bit packing, 0-64 bits, NaN rejection, `decode_range()` for arbitrary bit offsets
- `shuffle.rs` — Byte-level shuffle/unshuffle (HDF5-style)
- `libaec.rs` — Safe Rust wrapper around libaec: `aec_compress()` with RSI block offset tracking, `aec_decompress()`, `aec_decompress_range()`
- `compression/` — `Compressor` trait + 6 implementations:
  - `szip.rs` — SzipCompressor (CCSDS 121.0-B-3, RSI block random access)
  - `zstd.rs` — ZstdCompressor (Zstandard, stream compressor)
  - `lz4.rs` — Lz4Compressor (LZ4 via lz4_flex, fastest decompression)
  - `blosc2.rs` — Blosc2Compressor (multi-codec, chunk-based random access)
  - `zfp.rs` — ZfpCompressor (lossy float, fixed-rate/precision/accuracy, range decode)
  - `sz3.rs` — Sz3Compressor (SZ3 error-bounded, absolute/relative/PSNR)
- `zfp_ffi.rs` — Safe Rust wrapper around ZFP C library
- `pipeline.rs` — Two-phase dispatch, `decode_range_pipeline()` with random access support
- All codecs feature-gated (default on). `CompressionError::NotAvailable` for disabled features.

## tensogram-cli

12 tests.

- Subcommands: `info`, `ls`, `dump`, `get`, `set`, `copy`, `merge`, `split`, `reshuffle`, `convert-grib` (feature-gated)
- Where-clause filtering (`-w`), key selection (`-p`), JSON output (`-j`)
- Immutable key protection in `set` (shape, strides, dtype, encoding, hash)
- Filename placeholder expansion in `copy` and `split`
- Recursive dot-path key lookup for namespaced MARS keys

## tensogram-ffi (C FFI)

Tested indirectly via C++ wrapper (105 tests).

- 62 C API functions with `tgm_` naming
- Opaque handles: `TgmMessage`, `TgmMetadata`, `TgmFile`, `TgmScanResult`, `TgmStreamingEncoder`
- Error codes: `TGM_ERROR_OK` through `TGM_ERROR_END_OF_ITER`
- Thread-local error messages via `tgm_last_error()`
- Iterator API: `tgm_buffer_iter_*`, `tgm_file_iter_*`, `tgm_object_iter_*`
- Streaming encoder: `tgm_streaming_encoder_create/write/write_preceder/count/finish/free`
- Auto-generated `tensogram.h` (~544 lines) via cbindgen
- Panic safety: `panic = "abort"` in both release and dev profiles
- Vec capacity UB fixed (shrink_to_fit before forget), null pointer validation

## C++ Wrapper

105 GoogleTest tests across 10 files.

- `include/tensogram.hpp` — single-header C++17 wrapper (~934 lines)
- RAII classes: `message`, `metadata`, `file`, `buffer_iterator`, `file_iterator`, `object_iterator`, `streaming_encoder`
- Typed exception hierarchy: `error` → `framing_error`, `metadata_error`, etc.
- `decoded_object` view with `data_as<T>()`, `element_count<T>()`
- Range-based for via `message::iterator`
- C++ Core Guidelines: `[[nodiscard]]`, `noexcept`, `const`-correct, Rule of Five
- CMake build: GoogleTest v1.15.2 via FetchContent

## tensogram-python (PyO3)

226 pytest tests.

- Full Python API with NumPy integration
- `encode()`, `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()`, `scan()`
- `iter_messages()` — iterate decoded messages from a byte buffer
- `Message` namedtuple — `.metadata` and `.objects` attribute access, tuple unpacking
- `TensogramFile` with context manager, `len()`, iterator
  - `for msg in file:` — iterate all messages (owns independent file handle, free-threaded safe)
  - `file[i]`, `file[-1]` — index by position (negative indexing)
  - `file[1:10:2]` — slice returns list of Message namedtuples
- `Metadata` with `version`, `base`, `_reserved_`, `_extra_`, dict-style access
- `DataObjectDescriptor` with all tensor + encoding fields
- All 10 numeric numpy dtypes + float16/bfloat16/complex support
- Zero-copy for u8/i8, safe i128→i64 bounds check
- ruff configured (0 warnings)

## tensogram-grib

17 tests (0 unit + 17 integration).

- `convert_grib_file()` via ecCodes, extracts ~40 MARS keys dynamically
- Grouping modes: `OneToOne`, `MergeAll`
- All MARS keys stored in each `base[i]["mars"]` entry independently (no common/varying partitioning)
- `preserve_all_keys` option: 6 additional ecCodes namespaces under `grib` sub-object in each `base[i]` entry
- 4 real ECMWF opendata GRIB test fixtures (IFS 0.25° operational)

## tensogram-xarray

181 tests, ~98% coverage. Separate pure-Python package.

- xarray backend engine: `engine="tensogram"` for `xr.open_dataset()`
- `TensogramBackendArray` — lazy loading with N-D random-access slice mapping
- Coordinate auto-detection (13 known names: lat, lon, time, level, etc.)
- `open_datasets()` — multi-message auto-merge with hypercube stacking
- `StackedBackendArray` for lazy composition without eager decode
- Ratio-based `range_threshold` heuristic for partial vs full decode

## Examples

### Rust (11 runnable, workspace member)
01 encode_decode, 02 mars_metadata, 03 simple_packing, 04 shuffle_filter, 05 multi_object, 06 hash_verification, 07 scan_buffer, 08 decode_variants, 09 file_api, 10 iterators, 11 streaming

### C++ (5 examples, C++ wrapper API)
01 encode_decode, 02 mars_metadata, 03 simple_packing, 04 file_api, 05 iterators

### Python (11 examples)
01 encode_decode, 02 mars_metadata, 03 simple_packing, 04 multi_object, 05 file_api, 06 hash_and_errors, 07 iterators, 08 xarray_integration, 08 zarr_backend, 09 dask_distributed, 09 streaming_consumer

## Documentation (mdbook)

- `docs/` — mdbook source
- Introduction, Concepts (messages, metadata, objects, pipeline)
- Wire Format (message layout, CBOR schema, dtypes)
- Developer Guide (quickstart, encoding, decoding, file API, iterators, xarray integration, dask integration)
- Encodings (simple_packing, shuffle, compression)
- CLI Reference (all subcommands)
- GRIB conversion overview + MARS key mapping
- xarray integration guide
- Edge Cases and Internals reference

## Golden Test Files

5 canonical `.tgm` files in `crates/tensogram-core/tests/golden/`:
- `simple_f32.tgm`, `multi_object.tgm`, `mars_metadata.tgm`, `multi_message.tgm`, `hash_xxh3.tgm`
- Byte-for-byte deterministic, verified by all 3 languages

## tensogram-zarr (Zarr v3 store backend, 81 tests)
- Zarr v3 Store implementation for `.tgm` files — `zarr.open_group(store=TensogramStore(...))`
- `TensogramStore` — implements `zarr.abc.store.Store` ABC with full async interface
- **Read path**: scans `.tgm` file, builds virtual Zarr key space, serves `get()` from decoded objects
  - Each TGM data object → one Zarr array with single chunk (chunk_shape = array_shape)
  - Root `zarr.json` synthesized from `GlobalMetadata` (`_extra_` → attributes)
  - Per-array `zarr.json` synthesized from `DataObjectDescriptor` (shape, dtype, encoding metadata)
  - Chunk keys use correct Zarr v3 multi-dimensional format (`c/0/0` for 2D, `c/0/0/0` for 3D)
  - Variable naming from metadata (`mars.param`, `name`, `param`) with deduplication suffix
  - Byte-range support: `RangeByteRequest`, `OffsetByteRequest`, `SuffixByteRequest`
- **Write path**: buffers chunk data in memory, assembles into TGM message on `close()`
  - Group attributes → `GlobalMetadata._extra_`
  - Array metadata → `DataObjectDescriptor` with dtype/shape/encoding params
  - Supports `mode="w"` (create) and `mode="a"` (append)
- **Listing**: `list()`, `list_prefix()`, `list_dir()` — full async generators over virtual key space
- **Mapping layer** (`mapping.py`):
  - Bidirectional dtype conversion: TGM ↔ Zarr v3 ↔ NumPy (14 dtypes + bitmask)
  - `build_group_zarr_json()` / `build_array_zarr_json()` — read path metadata synthesis
  - `parse_array_zarr_json()` — write path metadata extraction
  - `resolve_variable_name()` — dotted-path metadata resolution with priority chain
- **Integration**: works with `zarr.open_group()`, `zarr.open_array()`, slicing, scalar indexing
- **Error handling** (hardened):
  - All Rust `tensogram.*` calls wrapped with Python-level context (file path, message index, variable name)
  - `OSError` for file-open failures, `ValueError` for decode/encode errors, `IndexError` for out-of-range
  - Input validation: mode, message_index, path (non-empty string)
  - `close()` exception-safe via `try/finally`; `__exit__` logs flush errors when exception already in flight
  - Byte-count validation on write path; `TypeError` for unknown `ByteRequest` types
  - `deserialize_zarr_json` wraps `json.JSONDecodeError` with byte-count context
  - Silent-skip paths elevated to `WARNING` log level (arrays without chunks, empty flush)
- **Second pass fixes**: file handle leak closed, `_open`/`_open_sync` deduplicated, `parse_array_zarr_json` no longer mutates input, bfloat16 fill value corrected, variable names with `/` sanitized
- **PR review fixes** (Copilot review #12):
  - `serialize_zarr_json()` now converts NaN/Infinity to Zarr v3 string sentinels for RFC 8259 compliance
  - Write path (`_flush_to_tgm`) uses proper `tgm_dtype_to_numpy()` mapping (handles bfloat16, etc.) and honours byte_order from Zarr metadata
  - `_find_chunk_data()` raises `ValueError` on multiple chunk keys instead of silently dropping data
  - `delete("zarr.json")` now clears `_write_group_attrs` to prevent stale state on flush
  - Replaced deprecated `ndarray.newbyteorder()` with `.view(dtype.newbyteorder())` for NumPy 2.x compatibility
- **Edge case coverage**: 34 edge case tests (invalid mode, negative index, slash names, triple duplicates, zero-object messages, chunk key shapes 0D-5D, byte range boundaries, fill values, lifecycle double-open/close, dotted_get paths)
- 204 tests: 43 mapping + 16 store read + 4 store write + 12 round-trip + 16 Zarr integration + 39 edge cases + 74 coverage gap tests
- 0 ruff lint warnings
- Documentation: `docs/src/guide/zarr-backend.md` with mermaid diagram, edge cases section, error handling table
- Example: `examples/python/08_zarr_backend.py`

## CI / Build Tooling

- `astral-sh/setup-uv@v5` in all Python CI jobs (`python`, `python-packages`)
- All Python envs now use `uv venv .venv` + `uv pip install` (replaces `python -m venv` + `pip install`)
- Legacy `ci.yaml` removed; single authoritative `ci.yml` remains
- Local dev instructions updated in `CLAUDE.md`, `README.md`, `CONTRIBUTING.md`, `docs/`, `examples/python/README.md`

## Metadata Major Refactor (v0.6.0)

- **Removed** `common` (shared metadata) and `payload` (per-object metadata array) from `GlobalMetadata`
- **Added** `base` — per-object metadata array where each entry holds ALL metadata for that object independently, no tracking of commonalities
- **Renamed** `reserved` → `_reserved_` in CBOR (library internals: encoder, time, uuid)
- **Renamed** `extra` → `_extra_` in CBOR (client-writable catch-all for message-level annotations)
- Auto-populated keys (ndim/shape/strides/dtype) now live under `base[i]["_reserved_"]["tensor"]` instead of directly in `payload[i]`
- **Added** `compute_common()` utility that extracts common keys from base entries when needed (e.g. for display or merge)
- Commonalities are computed in software, not encoded in the wire format
- Encoder validates that client code does not write to `_reserved_` at any level
- All documentation updated to reflect new metadata structure

## Error Handling Review (post-refactor)

Comprehensive audit of all Rust library code:

- **No panics**: confirmed zero `unwrap()`, `expect()`, `panic!()`, `todo!()`, `unimplemented!()` in non-test library code across all crates
- **Integer overflow**: all `as usize` casts on `total_length` (u64) replaced with `usize::try_from()` + proper error propagation in decode paths; scan paths use `as usize` with subsequent bounds checks (truncation is harmless)
- **Truncation**: `zstd_level` and `blosc2_clevel` i64→i32 casts replaced with `i32::try_from()` + error propagation
- **cbor_offset validation**: added bounds check in `decode_data_object_frame` ensuring offset is within `[FRAME_HEADER_SIZE, cbor_offset_pos]`
- **Buffer underflow**: added `checked_sub()` for `buf.len() - POSTAMBLE_SIZE` in streaming-mode decode to prevent underflow on short buffers
- **Logging**: `eprintln!` in `hash.rs` for unknown hash algorithms replaced with `tracing::warn!` for consistency
- **Comments**: added safety comments on all `as` casts and array indexing in `wire.rs` read helpers, `civil_from_days` doe cast, and `get_f64_param` precision note
- **Error messages**: all errors include what went wrong, where, and relevant values (expected vs actual)
- **Documentation**: `docs/src/guide/error-handling.md` updated with all metadata-refactor error paths (encoding, decoding, streaming, CLI), no-panic guarantee section

## Dependencies

- ciborium 0.2, serde 1, thiserror 2, xxhash-rust 0.8
- libaec-sys (szip), zstd 0.13, lz4_flex 0.11, blosc2 0.2, zfp-sys-cc 0.2, sz3 0.4
- clap 4, serde_json 1, tempfile 3 (dev)
