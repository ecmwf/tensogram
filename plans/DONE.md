# Tensogram — Implementation Path Followed

> **Purpose.** This document exists to give AI agents (and humans) the
> context of *what has already been built and why*, so new work can
> build on the path taken. Read it alongside `DESIGN.md` (the rationale)
> and `WIRE_FORMAT.md` (the canonical spec).
>
> **Do not add version numbers to this file.**
> Release-by-release changes belong in `../CHANGELOG.md`; this file is
> version-agnostic. When you extend it, describe the *shape* of the
> work and the decisions taken, not the tag it shipped under. Never
> write "v0.X.Y" anywhere in this document, and never list test counts
> — both drift and make the file lie over time.
>
> For planned work, see `TODO.md`. For speculative ideas, see `IDEAS.md`.

## Scope summary

- **Workspace.** Default Rust workspace + optional Rust crates
  (`tensogram-grib`, `tensogram-netcdf`, `tensogram-wasm`) + the PyO3
  `python/bindings` crate + two separate Python packages
  (`tensogram-xarray`, `tensogram-zarr`). See `ARCHITECTURE.md` for the
  full crate list and what each one does.
- **Quality bar.** Zero clippy warnings, zero ruff issues,
  `panic = "abort"` on both release and dev profiles. Library code
  avoids `unwrap`/`expect`/`panic!` on any fallible input path;
  remaining `unwrap` calls on library code are confined to provably
  infallible conversions (e.g. `chunk.try_into()` after
  `chunks_exact(N)`). Panics never cross the FFI boundary.

---

## Python async bindings

`AsyncTensogramFile` exposes all read/decode operations as `asyncio`
coroutines via `pyo3-async-runtimes` + tokio. A single handle supports
truly concurrent operations (core async methods take `&self`, no mutex).

| Component | What changed |
|-----------|-------------|
| `tensogram-core/file.rs` | All async methods `&mut self` → `&self`. Added `decode_range_async`, `decode_range_batch_async`, `decode_object_batch_async`, `prefetch_layouts_async`, `message_count_async`. Sync batch: `decode_range_batch`, `decode_object_batch`. |
| `tensogram-core/remote.rs` | Added `read_range_async`, `read_range_batch_async`, `read_object_batch_async`, `ensure_all_layouts_batch_async` (batched layout discovery via `get_ranges`), `message_count_async`. Sync batch: `read_range_batch`, `read_object_batch`. |
| `tensogram-python` | `PyAsyncTensogramFile` (`Arc<TensogramFile>`, no mutex), `PyAsyncTensogramFileIter`, sync `file_decode_range_batch` and `file_decode_object_batch` on `PyTensogramFile`. `pyo3-async-runtimes` dependency. |
| Tests | Async/batch tests in `test_async.py`, shared fixtures in `conftest.py`. |
| Docs | `python-api.md` async section, example `15_async_operations.py`, examples README. |
| CI | `pytest-asyncio` added, `--no-default-features` check. |

## Caller-endianness (native byte order decode)

Decoded data is always returned in the caller's native byte order by
default. The `DecodeOptions.native_byte_order` field (default `true`)
controls this across all interfaces.

| Component | What changed |
|-----------|-------------|
| `tensogram-encodings` | `ByteOrder::native()`, `byteswap()`, `PipelineConfig.swap_unit_size`, `decode_pipeline`/`decode_range_pipeline` gain `native_byte_order` param, ZFP/SZ3 made byte-order-aware |
| `tensogram-core` | `Dtype::swap_unit_size()`, `DecodeOptions.native_byte_order`, threaded through all decode paths + iterators |
| `tensogram-python` | `native_byte_order=True` on `decode()`, `decode_object()`, `decode_range()`, `TensogramFile.decode_message()`. Default `byte_order` → native |
| `tensogram-ffi` | `native_byte_order` param on all decode functions |
| C++ wrapper | `decode_options.native_byte_order` threaded to all decode + iterator calls |
| `tensogram-zarr` | Read-path manual byteswap workaround removed |
| CLI | `reshuffle`, `merge`, `split`, `set` use `native_byte_order=false` to preserve wire layout on re-encode |
| Docs | `decoding.md`, `encode-pre-encoded.md`, `DESIGN.md` updated |

## `tensogram validate`

A full validation API and CLI subcommand, layered across four levels.

- **Level 1 (Structure).** Raw byte walking — magic, preamble, frame
  headers/ENDF, total_length, postamble, first_footer_offset, frame
  ordering, preceder legality, preamble flags vs observed, overflow-safe
  arithmetic.
- **Level 2 (Metadata).** Raw CBOR parsing from frame payloads (before
  `decode_message` normalisation), required keys, dtype/encoding/filter/
  compression recognised, shape/strides/ndim consistency, index/hash
  frame consistency.
- **Level 3 (Integrity).** xxh3 hash verification (descriptor + hash
  frame fallback), decode pipeline execution for compressed objects.
  Unknown hash algorithms produce warnings. `hash_verified` is true only
  when every object is verified.
- **Level 4 (Fidelity).** Full decode, decoded-size check, NaN/Inf scan
  for Float16/Bfloat16/Float32/Float64/Complex64/Complex128. NaN/Inf are
  errors. Reports element index and component (real/imag).

- **API.** Composable `ValidateOptions { max_level, check_canonical,
  checksum_only }`. `ObjectContext` for shared per-object state across
  levels: Level 2 caches descriptors, Level 3 caches decoded bytes,
  Level 4 reuses both.
- **Modular layout.** `validate/types.rs` (types + `IssueCode` enum),
  `validate/structure.rs`, `validate/metadata.rs`,
  `validate/integrity.rs`, `validate/fidelity.rs`, `validate/mod.rs`.
- **CLI.** `tensogram validate [--quick|--checksum|--full]
  [--canonical] [--json] <files>` with mutually exclusive level flags,
  `--canonical` combinable with any level, serde_json batch array
  output, exit code 0/1.
- **Python.** `tensogram.validate(buf, level, check_canonical) -> dict`
  and `tensogram.validate_file(path, level, check_canonical) -> dict`
  via PyO3. Four levels: `"quick"`, `"default"`, `"checksum"`, `"full"`.
- **C FFI.** `tgm_validate` and `tgm_validate_file` return JSON via
  `tgm_bytes_t` out-parameter.
- **C++ wrapper.** `tensogram::validate()` and
  `tensogram::validate_file()` return JSON strings with typed exception
  mapping (`invalid_arg_error`, `io_error`, `encoding_error`).
- **Examples.** `examples/python/13_validate.py`,
  `examples/rust/src/bin/13_validate.rs`.
- **Docs.** `docs/src/cli/validate.md`, `docs/src/guide/python-api.md`.

## `tensogram-netcdf`

Optional crate for converting NetCDF → Tensogram. Excluded from the
default workspace build because it requires `libnetcdf` at the OS level.

- **Supported inputs.** NetCDF-3 classic + NetCDF-4 (HDF5-backed); all
  10 numeric dtypes (i8/i16/i32/i64, u8/u16/u32/u64, f32/f64);
  root-group variables (sub-groups warn and are skipped); scalar
  variables; unlimited dimensions; packed variables with `scale_factor`
  / `add_offset` unpacked to f64; fill values replaced with NaN for
  floats.
- **Skipped inputs.** `char`, `string`, `compound`, `vlen`, `enum`,
  `opaque` — skipped with a stderr warning, never a hard error.
- **Split modes.** `file` (one message with N objects), `variable` (N
  messages each with one object), `record` (one message per step along
  the unlimited dimension; static variables replicated into each).
- **CF metadata.** 16-attribute allow-list (`standard_name`, `long_name`,
  `units`, `calendar`, `cell_methods`, `coordinates`, `axis`, `positive`,
  `valid_min`, `valid_max`, `valid_range`, `bounds`, `grid_mapping`,
  `ancillary_variables`, `flag_values`, `flag_meanings`) lifted into
  `base[i]["cf"]` when `--cf` is set. Full verbose attribute dump always
  lives under `base[i]["netcdf"]`.
- **Pipeline flags.** `--encoding simple_packing --bits N`,
  `--filter shuffle`, `--compression {zstd,lz4,blosc2,szip}`,
  `--compression-level N` — symmetric with `convert-grib`.
  `simple_packing` is f64-only and skipped (with warning) for non-f64
  and NaN-containing variables.
- **Docs.** `docs/src/guide/convert-netcdf.md` user guide,
  `docs/src/reference/netcdf-cf-mapping.md` CF attribute reference.
- **Examples.** `examples/python/12_convert_netcdf.py` (CLI via
  `subprocess`, the pattern used because the Python bindings do not
  expose `convert_netcdf_file` directly),
  `examples/rust/src/bin/12_convert_netcdf.rs` (direct library API,
  gated behind the examples crate's `netcdf` feature).
- **CI.** `netcdf` job runs clippy + crate tests + CLI tests + example
  build on both Ubuntu and macOS. The `grib` job covers the same matrix
  for symmetry. The `python` job installs libnetcdf and runs the e2e
  tests.

## `tensogram-wasm`

WebAssembly bindings for browser-side decode, encode, scan, and
streaming. Compiles to `wasm32-unknown-unknown` via `wasm-pack`.

- **Crate.** `rust/tensogram-wasm/` — `lib.rs`, `convert.rs`,
  `streaming.rs`.
- **Supported compressors.** `lz4`, `szip` (pure-Rust via
  `tensogram-szip`), `zstd` (pure-Rust via `ruzstd`). `blosc2`/`zfp`/
  `sz3` return an error.
- **Decode API.** `decode()`, `decode_metadata()`, `decode_object()`,
  `scan()`.
- **Encode API.** `encode()` — accepts `Uint8Array`, `Float32Array`,
  `Float64Array`, `Int32Array` as data inputs.
- **`DecodedMessage`.** Zero-copy TypedArray views
  (`object_data_f32/f64/i32/u8`) and a safe-copy variant
  (`object_data_copy_f32`); all views handle zero-length payloads
  without UB.
- **`StreamingDecoder`.** Progressive chunk feeding, `feed()` returns
  `Result` (rejects oversized chunks), `last_error()` /
  `skipped_count()` for corrupt-message visibility, configurable
  `set_max_buffer()` (default 256 MiB), `reset()`, `pending_count()`,
  `buffered_bytes()`.
- **`DecodedFrame`.** Per-object streaming output with `data_f32/f64/
  i32/u8`, `descriptor()`, `base_entry()`, `byte_length()`.
- **`tensogram-szip`.** Pure-Rust CCSDS 121.0-B-3 AEC/SZIP codec —
  encode, decode, range-decode; FFI cross-validated against libaec.
- **Feature gates.** `szip-pure` and `zstd-pure` in `tensogram-encodings`
  and `tensogram-core`; mutually exclusive with `szip` / `zstd` (C FFI).
- **Build / test.** `wasm-pack build rust/tensogram-wasm --target web`,
  `wasm-pack test --node rust/tensogram-wasm`.

## `tensogram-benchmarks`

Separate workspace crate providing a codec-matrix benchmark and a
comparison against ecCodes.

- `constants.rs` — shared `AEC_DATA_PREPROCESS` constant.
- `datagen.rs` — deterministic SplitMix64-based synthetic weather field
  generator.
- `report.rs` — `BenchmarkResult` with `TimingStats` (median/min/max),
  `Fidelity` enum (`Exact`, `Lossy{linf, l1, l2}`, `Unchecked`),
  throughput (MB/s), compressed-size variability tracking.
  `compute_fidelity()` compares decoded output to original.
- `codec_matrix.rs` — all pipeline combos. `compute_params` inside the
  timed encode loop for `SimplePacking` cases. Uses
  `encode_pipeline_f64` to avoid bytes↔f64 round-trip. Configurable
  warm-up. Returns `BenchmarkRun` with separate `results` and
  `failures`.
- `grib_comparison.rs` — symmetric end-to-end timing. Uses
  `encode_pipeline_f64`. Returns `BenchmarkRun`.
- `lib.rs` — `BenchmarkError` enum (`Validation`, `Pipeline`),
  `CaseFailure`, `BenchmarkRun` with `all_passed()`. Binaries exit
  non-zero on failures.
- Two binaries: `codec-matrix` and `grib-comparison` (requires
  `--features eccodes`). Both accept `--warmup`.
- `build.rs` — links `libeccodes` via pkg-config or Homebrew fallback
  when the `eccodes` feature is active.
- Docs: `docs/src/guide/benchmarks.md`,
  `docs/src/guide/benchmark-results.md`.

## `tensogram-core`

- `wire.rs` — v2 frame-based wire format: Preamble (24 B), FrameHeader
  (16 B), Postamble (16 B), `FrameType` enum (incl. `PrecederMetadata`
  type 8), `MessageFlags` (incl. bit 6 `PRECEDER_METADATA`),
  `DataObjectFlags`.
- `framing.rs` — `encode_message()` with two-pass index construction,
  `decode_message()`, `scan()` for multi-message buffers. Decomposed
  into focused helpers.
- `metadata.rs` — Deterministic CBOR encoding for `GlobalMetadata`,
  `DataObjectDescriptor`, `IndexFrame`, `HashFrame` (three-step:
  serialize → canonicalize → write). `verify_canonical_cbor()` utility.
- `types.rs` — `GlobalMetadata` (`version`, `base`, `_reserved_`,
  `_extra_`), `DataObjectDescriptor`, `IndexFrame`, `HashFrame`.
- `dtype.rs` — All 15 dtypes (float16/32/64, bfloat16, complex64/128,
  int/uint 8-64, bitmask).
- `hash.rs` — xxh3 hashing + verification (xxh3 only).
- `encode.rs` — Full encode pipeline: validate → build pipeline config →
  encode per object → hash → assemble frames. Auto-populates
  `base[i]._reserved_.tensor` entries. Validates that client code does
  not write to `_reserved_`.
  - `encode_pre_encoded()` — Bypass the encoding pipeline for
    already-encoded payloads. Accepts pre-packed bytes with a descriptor
    declaring encoding/filter/compression. Validates structure (shape,
    dtype, szip block offsets) but skips the pipeline. Available in
    Rust, Python, C FFI, C++.
  - `StreamingEncoder::write_object_pre_encoded()` — Streaming variant
    for progressive encode of pre-encoded objects.
- `decode.rs` — `decode()`, `decode_metadata()`, `decode_descriptors()`,
  `decode_object()`, `decode_range()` (split results by default, `join`
  parameter for concatenated output).
- `file.rs` — `TensogramFile`: open, create, lazy scan, append,
  seek-based random access; remote-aware backend.
- `iter.rs` — `MessageIter` (zero-copy buffer), `ObjectIter` (lazy
  per-object decode), `FileMessageIter` (seek-based file),
  `objects_metadata()` (descriptor-only).
- `pipeline.rs` — Shared `DataPipeline` + `apply_pipeline` helper
  re-exported by `tensogram-grib` and `tensogram-netcdf`, so the CLI
  pipeline flags produce byte-identical descriptors in both converters.
- `streaming.rs` — `StreamingEncoder<W: Write>`: progressive encode,
  footer hash/index, no buffering, `write_preceder()` for per-object
  streaming metadata.
- `remote.rs` — `object_store`-backed remote access for `s3://`,
  `s3a://`, `gs://`, `az://`, `azure://`, `http://`, `https://`. Shared
  tokio runtime via `OnceLock`, `block_on_shared` for sync bridge,
  native async methods when `remote` + `async` are both on, batched
  range reads via `get_ranges`.
- Feature gates: `mmap` (memmap2 zero-copy), `async` (tokio), `remote`
  (object_store + tokio multi-thread).
- `DecodePhase` enum for frame ordering validation.

## `tensogram-encodings`

- `simple_packing.rs` — GRIB-style lossy quantization, MSB-first bit
  packing, 0-64 bits, NaN rejection, `decode_range()` for arbitrary bit
  offsets. Optimised encode/decode: precomputed scale (no per-value
  division), specialised byte-aligned loops for 8/16/24/32 bits, fused
  NaN+min+max scan in `compute_params`.
- `shuffle.rs` — Byte-level shuffle/unshuffle (HDF5-style).
- `libaec.rs` — Safe Rust wrapper around libaec: `aec_compress()` with
  optional RSI block offset tracking (`aec_compress_no_offsets`),
  `aec_decompress()`, `aec_decompress_range()`. Auto-sets
  `AEC_DATA_3BYTE` for 17-24 bit samples (fixes a corruption bug).
- `pipeline.rs` — `encode_pipeline_f64()` variant for callers with typed
  f64 data (avoids bytes→f64 conversion). Auto-sets `AEC_DATA_MSB` for
  szip when encoding is `SimplePacking` so libaec's predictor sees
  most-significant bytes first; compression ratio on 24-bit GRIB data
  matches ecCodes.
- `compression/` — `Compressor` trait + implementations:
  - `szip.rs` — `SzipCompressor` (CCSDS 121.0-B-3, RSI block random
    access).
  - `zstd.rs` — `ZstdCompressor` (Zstandard, stream compressor).
  - `lz4.rs` — `Lz4Compressor` (LZ4 via `lz4_flex`, fastest
    decompression).
  - `blosc2.rs` — `Blosc2Compressor` (multi-codec, chunk-based random
    access).
  - `zfp.rs` — `ZfpCompressor` (lossy float, fixed-rate/precision/
    accuracy, range decode).
  - `sz3.rs` — `Sz3Compressor` (SZ3 error-bounded, absolute/relative/
    PSNR).
- `zfp_ffi.rs` — Safe Rust wrapper around the ZFP C library.
- `pipeline.rs` — Two-phase dispatch, `decode_range_pipeline()` with
  random-access support.
- All codecs feature-gated (default on).
  `CompressionError::NotAvailable` for disabled features. `szip-pure`
  and `zstd-pure` provide C-FFI-free alternatives.

## `tensogram-cli`

- Subcommands: `info`, `ls`, `dump`, `get`, `set`, `copy`, `merge`,
  `split`, `reshuffle`, `validate`; feature-gated `convert-grib`
  (`--features grib`) and `convert-netcdf` (`--features netcdf`).
- Where-clause filtering (`-w`), key selection (`-p`), JSON output
  (`-j`).
- `merge --strategy {first,last,error}` for metadata conflict
  resolution.
- Immutable key protection in `set` (shape, strides, dtype, encoding,
  hash).
- Filename placeholder expansion in `copy` and `split`.
- Recursive dot-path key lookup for namespaced MARS keys.
- Shared `PipelineArgs` (`--encoding/--bits/--filter/--compression/
  --compression-level`) wired through both `convert-grib` and
  `convert-netcdf` via `apply_pipeline`.

## `tensogram-ffi` (C FFI)

- `tgm_`-prefixed C API with opaque handles: `TgmMessage`,
  `TgmMetadata`, `TgmFile`, `TgmScanResult`, `TgmStreamingEncoder`.
- Error codes: `TGM_ERROR_OK` through `TGM_ERROR_END_OF_ITER`.
- Thread-local error messages via `tgm_last_error()`.
- Iterator API: `tgm_buffer_iter_*`, `tgm_file_iter_*`,
  `tgm_object_iter_*`.
- Streaming encoder: `tgm_streaming_encoder_create/write/
  write_preceder/write_pre_encoded/count/finish/free`.
- Validation: `tgm_validate`, `tgm_validate_file` (JSON out via
  `tgm_bytes_t`).
- Auto-generated `tensogram.h` via cbindgen.
- Panic safety: `panic = "abort"` in both release and dev profiles.
  Vec-capacity UB fixed (shrink_to_fit before forget), null pointer
  validation everywhere.

## C++ wrapper

- `cpp/include/tensogram.hpp` — single-header C++17 wrapper.
- RAII classes: `message`, `metadata`, `file`, `buffer_iterator`,
  `file_iterator`, `object_iterator`, `streaming_encoder`.
- `encode_pre_encoded()` — free function for already-encoded payloads.
- `streaming_encoder::write_object_pre_encoded()` — streaming variant.
- Typed exception hierarchy: `error` → `framing_error`,
  `metadata_error`, etc.
- `decoded_object` view with `data_as<T>()`, `element_count<T>()`.
- Range-based for via `message::iterator`.
- C++ Core Guidelines: `[[nodiscard]]`, `noexcept`, `const`-correct,
  Rule of Five.
- `validate()` / `validate_file()` wrappers returning JSON strings.
- CMake build: GoogleTest via FetchContent.

## `tensogram-python` (PyO3)

- Full Python API with NumPy integration.
- `encode()`, `decode()`, `decode_metadata()`, `decode_descriptors()`,
  `decode_object()`, `decode_range()`, `scan()`.
- `encode_pre_encoded()` — bypass pipeline for already-encoded payloads
  (bytes input, not numpy arrays).
- `StreamingEncoder` — progressive encode to file with `write_object()`
  and `write_object_pre_encoded()`.
- `iter_messages()` — iterate decoded messages from a byte buffer.
- `Message` namedtuple — `.metadata` and `.objects` attribute access,
  tuple unpacking.
- `TensogramFile` with context manager, `len()`, iterator:
  - `for msg in file:` — iterate all messages (owns independent file
    handle, free-threaded safe).
  - `file[i]`, `file[-1]` — index by position (negative indexing).
  - `file[1:10:2]` — slice returns list of `Message` namedtuples.
- `AsyncTensogramFile` — asyncio-based coroutines for all read/decode
  operations; true concurrency via `Arc<TensogramFile>` + `&self`
  methods.
- `Metadata` with `version`, `base`, `reserved`, `extra`, dict-style
  access (checks `base` entries then `_extra_`).
- `DataObjectDescriptor` with all tensor + encoding fields.
- All 10 numeric numpy dtypes + float16/bfloat16/complex support.
- Zero-copy for u8/i8, safe i128→i64 bounds check.
- Free-threaded Python 3.13t / 3.14t support: `#[pymodule(gil_used =
  false)]`, `GILOnceCell` replaced with `std::sync::OnceLock`, all hot
  paths release the GIL (`py.allow_threads`).
- Validation: `tensogram.validate`, `tensogram.validate_file`.
- ruff configured (0 warnings).

## `tensogram-grib`

- `convert_grib_file()` via ecCodes, extracts MARS keys dynamically.
- Grouping modes: `OneToOne`, `MergeAll`.
- All MARS keys stored in each `base[i]["mars"]` entry independently (no
  common/varying partitioning).
- `preserve_all_keys` option: additional ecCodes namespaces stored under
  a `grib` sub-object in each `base[i]` entry.
- Real ECMWF opendata GRIB fixtures (IFS 0.25° operational) in the
  integration tests.
- Honours shared `PipelineArgs` via the `apply_pipeline` helper, so
  `--encoding/--bits/--filter/--compression/--compression-level`
  produce descriptors byte-identical to `convert-netcdf`.

## `tensogram-xarray`

- xarray backend engine: `engine="tensogram"` for `xr.open_dataset()`.
- `TensogramBackendArray` — lazy loading with N-D random-access
  slice-to-flat-range mapping.
- Coordinate auto-detection (known names: lat, lon, time, level, etc.).
- `open_datasets()` — multi-message auto-merge with hypercube stacking.
- `StackedBackendArray` for lazy composition without eager decode.
- Ratio-based `range_threshold` heuristic for partial vs full decode.
- Producer metadata dimension hints: `_extra_["dim_names"]` resolved
  with priority chain `user dim_names > coord matching > producer hints
  > dim_N fallback`.
- Remote support: `storage_options` threaded through all modules;
  remote reads go through file-level APIs.

## `tensogram-zarr`

- Zarr v3 Store implementation for `.tgm` files —
  `zarr.open_group(store=TensogramStore(...))`.
- `TensogramStore` implements `zarr.abc.store.Store` ABC with full
  async interface.
- **Read path.** Scans `.tgm` file, builds virtual Zarr key space,
  serves `get()` from decoded objects.
  - Each TGM data object → one Zarr array with single chunk
    (`chunk_shape = array_shape`).
  - Root `zarr.json` synthesised from `GlobalMetadata` (`_extra_` →
    attributes).
  - Per-array `zarr.json` synthesised from `DataObjectDescriptor`.
  - Chunk keys use correct Zarr v3 multi-dimensional format (`c/0/0`
    for 2D, `c/0/0/0` for 3D).
  - Variable naming from metadata (`mars.param`, `name`, `param`) with
    deduplication suffix and slash sanitisation.
  - Byte-range support: `RangeByteRequest`, `OffsetByteRequest`,
    `SuffixByteRequest`.
- **Write path.** Buffers chunk data in memory, assembles into a TGM
  message on `close()`.
  - Group attributes → `GlobalMetadata._extra_`.
  - Array metadata → `DataObjectDescriptor` with dtype/shape/encoding
    params.
  - Supports `mode="w"` (create) and `mode="a"` (append).
- **Listing.** `list()`, `list_prefix()`, `list_dir()` — async
  generators over the virtual key space.
- **Mapping layer** (`mapping.py`):
  - Bidirectional dtype conversion: TGM ↔ Zarr v3 ↔ NumPy (numeric
    dtypes + bitmask).
  - `build_group_zarr_json()`, `build_array_zarr_json()`,
    `parse_array_zarr_json()`, `resolve_variable_name()`.
- **Remote.** `storage_options` accepted; remote writes rejected early;
  lazy chunk reads for remote `.tgm` files via `file_decode_descriptors`
  at scan time and `file_decode_object` per chunk on demand.
- **Error handling.** All Rust calls wrapped with Python-level context
  (file path, message index, variable name). `OSError` for file-open
  failures, `ValueError` for decode/encode, `IndexError` for out-of-range.
  `close()` is exception-safe.

## Metadata structure

- `GlobalMetadata`: `version`, `base` (per-object metadata array, each
  entry fully self-contained), `_reserved_` (library internals:
  encoder, time, uuid — writable by library only), `_extra_`
  (client-writable catch-all).
- Auto-populated tensor metadata (ndim/shape/strides/dtype) lives under
  `base[i]["_reserved_"]["tensor"]`.
- `compute_common()` utility extracts shared keys from `base` entries
  when needed (display, merge) — commonalities are computed in
  software, not encoded on the wire.
- Encoder validates that client code never writes to `_reserved_` at
  any level.
- Preceder Metadata Frames (type 8) for per-object metadata in
  streaming mode; preceder keys override footer keys on merge.

## Error handling

- No panics on any fallible input path: no `unwrap()`, `expect()`,
  `panic!()`, `todo!()` or `unimplemented!()` used to bail out of
  library code on unexpected input. Remaining `unwrap()` calls are
  provably infallible (e.g. `try_into()` after a `chunks_exact(N)`
  length guard). Panics cannot cross the FFI boundary because
  `panic = "abort"` is set on both release and dev profiles.
- Integer overflow: `usize::try_from()` on `total_length` (u64) in
  decode paths; scan paths use `as usize` with subsequent bounds checks
  (truncation is harmless).
- Truncation: `zstd_level` and `blosc2_clevel` use `i32::try_from()` +
  error propagation.
- Bounds checks on `cbor_offset` in `decode_data_object_frame`.
- Buffer underflow: `checked_sub()` for `buf.len() - POSTAMBLE_SIZE` in
  streaming-mode decode.
- Logging: `tracing::warn!` (not `eprintln!`) for unknown hash
  algorithms.
- Comments: safety comments on all `as` casts and array indexing in
  hot paths.
- Error messages include what went wrong, where, and relevant values
  (expected vs actual).
- Docs: `docs/src/guide/error-handling.md` covers all metadata-refactor
  error paths (encoding, decoding, streaming, CLI) and the no-panic
  guarantee.

## Examples

### Rust

`01_encode_decode`, `02_mars_metadata`, `03_simple_packing`,
`04_shuffle_filter`, `05_multi_object`, `06_hash_verification`,
`07_scan_buffer`, `08_decode_variants`, `09_file_api`, `10_iterators`,
`11_encode_pre_encoded`, `11_streaming`, `12_convert_netcdf` (needs
`--features netcdf`), `13_validate`, `14_remote_access` (needs
`--features remote`).

### C++

`01_encode_decode`, `02_mars_metadata`, `03_simple_packing`,
`04_file_api`, `05_iterators`, `11_encode_pre_encoded`.

### Python

`01_encode_decode`, `02_mars_metadata`, `03_simple_packing`,
`04_multi_object`, `05_file_api`, `06_hash_and_errors`, `07_iterators`,
`08_xarray_integration`, `08_zarr_backend`, `09_dask_distributed`,
`09_streaming_consumer`, `11_encode_pre_encoded`, `12_convert_netcdf`,
`13_validate`, `14_remote_access`, `15_async_operations`.

## Documentation (mdBook)

- `docs/` — mdBook source.
- Introduction, Concepts (messages, metadata, objects, pipeline).
- Wire Format (message layout, CBOR schema, dtypes).
- Developer Guide (quickstart, encoding, decoding, file API,
  iterators, Python API, C++ API, xarray integration, Dask
  integration, Zarr v3 backend, free-threaded Python, remote access,
  benchmarks, benchmark results).
- Encodings (simple_packing, shuffle, compression).
- CLI Reference (all subcommands incl. `validate`).
- GRIB conversion overview + MARS key mapping.
- NetCDF conversion overview + CF metadata mapping.
- Reference (error handling, edge cases, internals).

## Golden test files

Canonical `.tgm` files in `rust/tensogram-core/tests/golden/` used for
byte-for-byte cross-language verification:
`simple_f32.tgm`, `multi_object.tgm`, `mars_metadata.tgm`,
`multi_message.tgm`, `hash_xxh3.tgm`.

## CI / build tooling

- `astral-sh/setup-uv@v5` in all Python CI jobs.
- `uv venv .venv` + `uv pip install` everywhere.
- Single authoritative `.github/workflows/ci.yml`.
- Docker CI image with all build deps pre-baked; CI split into parallel
  lint/test/python/C++ jobs.
- Top-level `Makefile`: `make rust-test`, `make python-test`,
  `make cpp-test`, `make lint`, `make fmt`, `make docs-build`,
  `make clean`.
- Self-hosted runners for heavy jobs.

## Open-source preparation

- Apache-2.0 licence headers on every source file.
- `THIRD_PARTY_LICENSES` audit — all dependencies Apache-2.0-compatible.
- Zero GPL code in the dependency tree (clean-room `tensogram-sz3-sys`
  replaces the GPL `sz3-sys` crate).
- `CODE_OF_CONDUCT.md`, `SECURITY.md`, PR template with CLA, ECMWF
  Support Portal link in README, branch protection on `main`.

## Dependencies

- **Metadata & serialisation.** `ciborium`, `serde`, `thiserror`,
  `xxhash-rust`, `uuid`.
- **Compression (C FFI).** `libaec-sys`, `zstd`, `blosc2`, `blosc2-sys`,
  `zfp-sys-cc`; SZ3 via the clean-room `tensogram-sz3-sys` shim.
- **Compression (pure Rust).** `lz4_flex`, `ruzstd`, `tensogram-szip`.
- **CLI / JSON.** `clap`, `serde_json`.
- **Async / remote.** `tokio`, `object_store`, `bytes`, `url`,
  `memmap2`.
- **Tracing.** `tracing`, `tracing-subscriber`.
- **Bindings.** `PyO3`, `pyo3-async-runtimes`, `cbindgen`,
  `wasm-bindgen`, `wasm-pack`.
- **Dev.** `tempfile`, `proptest`, GoogleTest (FetchContent).
