# Tensogram — Current Implementation Status (v0.12.0)

> For historical release notes, see `../CHANGELOG.md`.
> For planned features, see `TODO.md`. For ideas, see `IDEAS.md`.

## TypeScript wrapper — Scope B complete (Phases 0–5)

New top-level `typescript/` package (`@ecmwf/tensogram`) that wraps
`rust/tensogram-wasm` with a typed, idiomatic TypeScript API. Design doc:
`plans/TYPESCRIPT_WRAPPER.md`. User-facing docs:
`docs/src/guide/typescript-api.md`.

| Component | What landed |
|-----------|-------------|
| Rust fix | `rust/tensogram-wasm/src/convert.rs::to_js` now uses `Serializer::json_compatible()` so metadata comes across as plain JS objects rather than ES `Map`. All 134 existing `wasm-bindgen-test` tests still pass. |
| Scaffold | `typescript/` with `package.json`, `tsconfig.json`, `vitest.config.ts`, `README.md`, `.gitignore`. ESM-only, Node ≥ 20. |
| Build wiring | `wasm-pack build --target web` → `typescript/wasm/`, then `tsc` emits `dist/`. |
| `src/types.ts` | Hand-written types: `Dtype`, `ByteOrder`, `Encoding`, `Filter`, `Compression`, `CborValue`, `HashDescriptor`, `DataObjectDescriptor`, `GlobalMetadata`, `BaseEntry`, `DecodedObject`, `DecodedMessage`, `EncodeInput`, `EncodeOptions`, `DecodeOptions`, `TypedArray`, `MessagePosition`. |
| `src/errors.ts` | Abstract `TensogramError` + 8 WASM-mapped subclasses (`FramingError`, `MetadataError`, `EncodingError`, `CompressionError`, `ObjectError`, `IoError`, `RemoteError`, `HashMismatchError`) plus 2 TS-only classes (`InvalidArgumentError`, `StreamingLimitError`). Base-class constructor `(message, rawMessage = message)` lets TS-layer errors pass a single argument while WASM-mapped errors pass the prefixed raw form. `mapTensogramError` parses the Rust-side error prefixes, extracts hex digests from hash-mismatch messages, and strips the prefix from `.message` for consistency with other variants. |
| `src/dtype.ts` | `DTYPE_BYTE_WIDTH`, `payloadByteSize` (handles bitmask `ceil(N/8)`), `shapeElementCount`, `typedArrayFor` (dispatches to all 10 native JS `TypedArray` types plus `Uint16Array` / interleaved `Float32/64Array` surrogates for half-precision and complex dtypes), `isDtype`, `SUPPORTED_DTYPES`. |
| `src/init.ts` | Idempotent async `init()`; explicit Node file read path (avoids depending on Node's experimental `file://` `fetch`); browser `fetch` fallback. Internal `getWbg()` + `_resetForTests()`. |
| `src/encode.ts` | Typed wrapper with client-side validation for `version`, `_reserved_` writes, dtype recognition, byte-order, and descriptor shape. Accepts any `ArrayBufferView` (not just `TypedArray`). |
| `src/decode.ts` | `decode`, `decodeMetadata`, `decodeObject`, `scan`. Returns `DecodedMessage` with safe-copy `data()` + zero-copy `dataView()` per object. Explicit `close()` + `FinalizationRegistry` fallback. |
| `src/metadata.ts` | `getMetaKey(meta, "mars.param")` with first-match-across-`base[*]` + fallback-to-`_extra_` semantics; `_reserved_` always hidden. `computeCommon(meta)` mirrors Rust `compute_common`. `cborValuesEqual` handles NaN bit patterns and positional map/array comparison. |
| `src/index.ts` | Barrel export of the full public surface. |
| `src/streaming.ts` | `decodeStream(stream, opts?)` async generator built on the WASM `StreamingDecoder`. Produces `DecodedFrame` objects with typed `descriptor`, per-object `baseEntry`, safe-copy `data()`, zero-copy `dataView()`. Handles `AbortSignal`, `maxBufferBytes` limits, and corrupt-message skip/report via `onError`. Cleanup on early `break`, thrown exceptions, and signal-fire is covered by the generator's `finally` block. |
| `src/file.ts` | `TensogramFile` class with three factories: `.open(path)` (Node `fs/promises`, lazy-imported so browser bundlers tree-shake it), `.fromUrl(url, opts?)` (any fetch-capable runtime; `fetch`, `headers`, `signal` options), `.fromBytes(bytes)` (in-memory, with defensive copy). Random access via `message(i)` / `messageMetadata(i)` / `rawMessage(i)`; async iteration via `[Symbol.asyncIterator]`; `close()` blocks further access. |
| Tests | **125 vitest tests across 9 files** (after coverage push + PR-review fixes): `smoke` (4), `encode` (21), `decode` (19), `dtype` (16), `metadata` (16), `errors` (16), `streaming` (14), `file` (24), `init` (6). Covers round-trips for float32/float64/int32/int64/uint8, multi-object messages, hash verification with tamper-detect and prefix-stripping, every error-prefix routing path plus fallback heuristics, `_reserved_` rejection, scan tolerance of inter-message garbage, `close()` idempotence, the plain-object-not-ES-Map parity claim, chunk-boundary-tolerant streaming, corrupt-message skip + `onError` observation, `AbortSignal` cancellation, `maxBufferBytes` enforcement, Node temp-file `open()`, `fetch`-mock `fromUrl()`, HTTP-status error handling, explicit-bytes WASM init, frame close-after-use, and defensive copy semantics. |
| Typecheck | Strict `tsc` (noImplicitAny, strictNullChecks, noUnusedLocals/Parameters) passes cleanly on source and tests. |
| Examples | `examples/typescript/01_encode_decode.ts`, `02_mars_metadata.ts`, `03_multi_object.ts`, `05_streaming_fetch.ts`, `06_file_api.ts`, `07_hash_and_errors.ts`. `examples/typescript/package.json` uses a local file: dependency on the `typescript/` package. All six examples run green end-to-end via `npx tsx`. Numbered to stay in step with `examples/python/`. |
| Makefile | Top-level targets: `ts-install`, `ts-build`, `ts-test`, `ts-typecheck`. `make test` runs `ts-test`; `make lint` runs `ts-typecheck`; `make clean` also removes `typescript/dist`, `typescript/wasm`, `typescript/node_modules`, `examples/typescript/node_modules`. |
| CI | New `typescript` job in `.github/workflows/ci.yml` mirroring the existing `wasm` job: installs the full Node distribution, runs `wasm-pack build`, `tsc --noEmit` (src + tests), `vitest run`, `tsc` (emit dist), then smoke-tests all six examples via `npx tsx`. |
| Docs | mdBook page `docs/src/guide/typescript-api.md` now covers the full Scope-B surface (encode/decode, dtype dispatch, metadata helpers, streaming, `TensogramFile`, memory model, error classes, examples); linked from `docs/src/SUMMARY.md`. |
| Agent & contributor docs | `CLAUDE.md` and `CONTRIBUTING.md` covered TS setup in the earlier pass; unchanged here. |

### Deferred to follow-ups (Scope C)

- Range-based lazy backend for `TensogramFile.fromUrl` (current impl downloads the whole file).
- `validate` / `encodePreEncoded` wrappers.
- First-class `float16` / `bfloat16` / `complex*` support (today they round-trip as `Uint16Array` / interleaved `Float32/64Array` surrogates).
- npm publish pipeline (the package is currently local-only).
- Zarr.js integration mirroring `tensogram-zarr`.
- Bundle-size budget (`size-limit`) in CI.

### Follow-up work landed on the same PR (#40)

- **PR-review fixes** (7 Copilot comments): explicit `await` in
  `TensogramFile[Symbol.asyncIterator]`; prefix-stripped
  `HashMismatchError.message`; corrected NaN-equality docstring;
  renumbered example-07 header; replaced flaky `/`-write test with a
  chmod + runtime-write-probe pattern (skips gracefully as root);
  corrected bitmask/zstd comment; extended the read-on-deleted-file
  test to actually exercise `read_message`.
- **Coverage push** (+61 Rust tests, +37 TS tests): previously-unasserted
  IssueCode variants, per-dtype NaN/Inf fidelity scans, FFI null-arg
  paths for `tgm_validate`/`tgm_validate_file`, simple_packing aligned
  and generic bpv edge cases, file.rs create/read error paths.
- **Pass-2 simplification pass** (-59 net lines): base-class
  `rawMessage = message` default removes `new Err(msg, msg)` boilerplate
  at 24 call sites across six files; internal helpers marked
  `@internal`; removed the duplicate `full_opts_local()` Rust test
  helper; corrected docstring drift.

## Python async bindings (completed)

`AsyncTensogramFile` exposes all read/decode operations as `asyncio` coroutines
via `pyo3-async-runtimes` + tokio.  A single handle supports truly concurrent
operations (core async methods take `&self`, no mutex).

| Component | What changed |
|-----------|-------------|
| `tensogram-core/file.rs` | All async methods `&mut self` → `&self`. Added `decode_range_async`, `decode_range_batch_async`, `decode_object_batch_async`, `prefetch_layouts_async`, `message_count_async`. Sync batch: `decode_range_batch`, `decode_object_batch`. |
| `tensogram-core/remote.rs` | Added `read_range_async`, `read_range_batch_async`, `read_object_batch_async`, `ensure_all_layouts_batch_async` (batched layout discovery via `get_ranges`), `message_count_async`. Sync batch: `read_range_batch`, `read_object_batch`. |
| `tensogram-python` | `PyAsyncTensogramFile` (20 methods, `Arc<TensogramFile>`, no mutex), `PyAsyncTensogramFileIter`, sync `file_decode_range_batch` and `file_decode_object_batch` on `PyTensogramFile`. `pyo3-async-runtimes` 0.28 dependency. |
| Tests | 73 async/batch tests (`test_async.py`), shared fixtures (`conftest.py`). 407 total. |
| Docs | `python-api.md` async section, example `15_async_operations.py`, examples README. |
| CI | `pytest-asyncio` added, `--no-default-features` check. |
| Rust examples | Fixed stale `let mut` from `&self` change in `12_convert_netcdf.rs`, `14_remote_access.rs`. |
| Rust tests | Fixed 26 stale `let mut` in `remote_http.rs` and `file.rs` tests. |

## caller-endianness (completed)

Decoded data is now always returned in the caller's native byte order by
default. The `DecodeOptions.native_byte_order` field (default `true`) controls
this across all interfaces.

**Changes across 43 files, 1060+ lines added:**

| Component | What changed |
|-----------|-------------|
| `tensogram-encodings` | `ByteOrder::native()`, `byteswap()`, `PipelineConfig.swap_unit_size`, `decode_pipeline`/`decode_range_pipeline` gain `native_byte_order` param, ZFP/SZ3 made byte-order-aware |
| `tensogram-core` | `Dtype::swap_unit_size()`, `DecodeOptions.native_byte_order`, threaded through all decode paths + iterators |
| `tensogram-python` | `native_byte_order=True` on `decode()`, `decode_object()`, `decode_range()`, `TensogramFile.decode_message()`. Default `byte_order` → native |
| `tensogram-ffi` | `native_byte_order` param on all 5 decode functions |
| C++ wrapper | `decode_options.native_byte_order` threaded to all decode + iterator calls |
| `tensogram-zarr` | Read-path manual byteswap workaround removed |
| CLI | `reshuffle`, `merge`, `split`, `set` use `native_byte_order=false` to preserve wire layout on re-encode |
| Tests | 15+ new tests for byteswap, cross-endian, complex types, wire opt-out, ZFP cross-endian, decode_range cross-endian |
| Docs | `decoding.md`, `encode-pre-encoded.md`, `DESIGN.md` updated |

## Summary

- **Version:** 0.8.0
- **Workspace:** 6 default crates + 3 optional (Python, GRIB, **NetCDF**) + 2 separate packages (xarray, zarr)
- **Tests:** 1050+ total (283+ Rust + 253 Python + 181 xarray + 204 Zarr + 117 C++ + 17 GRIB + 44 NetCDF integration + 5 CLI netcdf pipeline + 8 Python netcdf e2e)
- **Quality:** 0 clippy warnings, 90.5% Rust line coverage

## tensogram validate PR 3 — Python + FFI bindings + examples

- Python bindings via PyO3:
  - `tensogram.validate(buf, level="default", check_canonical=False) -> dict`
    validates a single message buffer. Returns `{"issues": [...], "object_count": int, "hash_verified": bool}`.
  - `tensogram.validate_file(path, level="default", check_canonical=False) -> dict`
    validates a `.tgm` file with streaming I/O. Returns `{"file_issues": [...], "messages": [...]}`.
  - Level parameter: `"quick"` (structure), `"default"` (integrity), `"checksum"` (hash-only), `"full"` (fidelity+NaN/Inf).
  - Reports serialized via `serde_json::to_value()` → recursive Python dict conversion.
- C FFI bindings:
  - `tgm_validate(buf, len, level, check_canonical, *out) -> tgm_error` — returns JSON via `TgmBytes`.
  - `tgm_validate_file(path, level, check_canonical, *out) -> tgm_error` — returns JSON via `TgmBytes`.
  - NULL level defaults to `"default"`. Invalid level returns `TGM_ERROR_INVALID_ARG`.
- C++ wrapper (`include/tensogram.hpp`):
  - `tensogram::validate(buf, len, level, check_canonical) -> std::string` (JSON).
  - `tensogram::validate_file(path, level, check_canonical) -> std::string` (JSON).
- Examples: `examples/python/13_validate.py`, `examples/rust/src/bin/13_validate.rs`.
- 34 pytest tests in `python/tests/test_validate.py` covering all levels, canonical,
  hash verification, NaN/Inf detection, file validation, edge cases (garbage-only,
  garbage between messages, truncated messages, all level+canonical combos).
- 11 C++ GoogleTest tests in `tests/cpp/test_validate.cpp` covering the C++ wrapper
  chain (valid message, empty buffer, corrupted magic, all levels, exception mapping,
  file validation, nonexistent file, empty file).
- 12 FFI unit tests covering level option parsing plus end-to-end
  `tgm_validate`/`tgm_validate_file` validation cases.
- Documentation: `docs/src/guide/python-api.md` added to mdBook.

## tensogram validate PR 2 — Level 4 fidelity + API refactor

- Refactored `ValidateMode` enum into composable `ValidateOptions`
  with `max_level`, `check_canonical`, `checksum_only` fields.
  `--canonical` is now independent and combinable with any level.
- Introduced `ObjectContext` for shared per-object state across
  validation levels: Level 2 caches descriptors, Level 3 caches
  decoded bytes, Level 4 reuses both. No duplicate work.
- New `fidelity.rs` module with Level 4 checks:
  - Full decode of each object (reuses Level 3 cache for non-raw)
  - Decoded-size check: verify decoded bytes match shape * dtype width
  - NaN/Inf scan for Float16, Bfloat16, Float32, Float64, Complex64,
    Complex128 — all are errors (break encoding pipeline computations)
  - Reports element index and component (real/imag) for complex types
  - Raw objects scanned in-place without decode_pipeline
- CLI: `--full` flag (mutually exclusive with `--quick`/`--checksum`),
  `--canonical` combinable with any level.
- Comprehensive fidelity test suite covering all float dtypes, byte orders,
  complex components, size mismatch, and mode combinations.

## tensogram validate (PR 1 of 3)

New `validate` subcommand and library API for checking `.tgm` file
correctness and integrity without consuming the data.

- Modular architecture in `rust/tensogram-core/src/validate/`:
  `types.rs` (public types + `IssueCode` enum), `structure.rs`
  (Level 1 raw byte walking), `metadata.rs` (Level 2 CBOR +
  descriptor checks), `integrity.rs` (Level 3 hash + decompression),
  `mod.rs` (public API entry points).
- `validate_message(buf, options) -> ValidationReport` — single message.
- `validate_file(path, options)` — streaming I/O, one message at a time.
- `validate_buffer(buf, options)` — in-memory multi-message validation.
- Stable `IssueCode` enum (~40 codes) with serde serialization.
- CLI: `tensogram validate [--quick|--checksum|--canonical] [--json] <files>`
  with mutually exclusive mode flags, serde_json batch array output,
  exit code 0/1.
- 25 unit tests in tensogram-core, 10 CLI tests.
- Docs: `docs/src/cli/validate.md`.
- Remaining work (PR 2): Level 4 Fidelity + `--full` flag.
- PR 3 (Python + FFI bindings + examples): completed — see above.

## tensogram-netcdf

New optional crate for converting NetCDF → Tensogram (v0.8.0). Excluded
from the default workspace build because it requires `libnetcdf` at the
OS level.

- `rust/tensogram-netcdf/src/` — `converter.rs` (~830 lines with pipeline
  plumbing), `metadata.rs` (CF allow-list + CBOR attribute conversion),
  `error.rs`, `lib.rs` (re-exports `convert_netcdf_file`, `ConvertOptions`,
  `DataPipeline`, `SplitBy`, `NetcdfError`).
- **Supported inputs:** NetCDF-3 classic + NetCDF-4 (HDF5-backed); all 10
  numeric dtypes (i8/i16/i32/i64, u8/u16/u32/u64, f32/f64); root-group
  variables (sub-groups warn and are skipped); scalar variables; unlimited
  dimensions; packed variables with `scale_factor` / `add_offset` unpacked
  to f64; fill values replaced with NaN for floats.
- **Skipped inputs:** char, string, compound, vlen, enum, opaque — all
  skipped with a stderr warning, never cause a hard error.
- **Split modes:** `file` (one message with N objects), `variable` (N
  messages each with one object), `record` (one message per step along
  the unlimited dimension; static variables replicated into each).
- **CF metadata:** 16-attribute allow-list (`standard_name`, `long_name`,
  `units`, `calendar`, `cell_methods`, `coordinates`, `axis`, `positive`,
  `valid_min`, `valid_max`, `valid_range`, `bounds`, `grid_mapping`,
  `ancillary_variables`, `flag_values`, `flag_meanings`) lifted into
  `base[i]["cf"]` when `--cf` is set. Verbose dump of every attribute
  always lives under `base[i]["netcdf"]`.
- **Pipeline flags (Task 13b):** `--encoding simple_packing --bits N`,
  `--filter shuffle`, `--compression {zstd,lz4,blosc2,szip}`,
  `--compression-level N` — symmetric with `convert-grib`. `simple_packing`
  is f64-only and skipped (with warning) for non-f64 and NaN-containing
  variables so mixed files convert cleanly.
- **Tests:**
  - 44 library integration tests in `rust/tensogram-netcdf/tests/integration.rs`
    (21 baseline + 13 pipeline plumbing + 10 new coverage tests).
  - 5 CLI tests under `rust/tensogram-cli/src/commands/convert_netcdf.rs`
    for pipeline flag plumbing and error paths.
  - 8 Python end-to-end round-trip tests in
    `python/tests/test_convert_netcdf.py` (build CLI with `--features
    netcdf`, run via `subprocess`, decode through Python bindings).
- **Fixtures:** 8 committed `.nc` files under `rust/tensogram-netcdf/testdata/`
  totaling <60 KB, generated by `testdata/generate.py` with fixed seed.
- **Docs:** `docs/src/guide/convert-netcdf.md` user guide,
  `docs/src/reference/netcdf-cf-mapping.md` CF attribute reference, new
  "NetCDF Conversion" section in `docs/src/SUMMARY.md`.
- **Examples:** `examples/python/12_convert_netcdf.py` (CLI via
  `subprocess`), `examples/rust/src/bin/12_convert_netcdf.rs` (direct
  library API, gated behind an `examples-rust` `netcdf` feature).
- **CI:** new `netcdf` job runs clippy + crate tests + CLI tests + example
  build on both Ubuntu and macOS. The `grib` job was extended to the same
  ubuntu+macos matrix for symmetry. The `python` job now installs
  libnetcdf and runs the new Python e2e tests.

## tensogram-wasm (v0.8.0)

WebAssembly bindings for browser-side decode, encode, scan, and streaming.
Compiles to `wasm32-unknown-unknown` via `wasm-pack`.

- **Crate:** `rust/tensogram-wasm/` — `lib.rs`, `convert.rs`, `streaming.rs`
- **Supported compressors:** lz4, szip (pure-Rust via `tensogram-szip`), zstd
  (pure-Rust via `ruzstd`). blosc2/zfp/sz3 return an error.
- **Decode API:** `decode()`, `decode_metadata()`, `decode_object()`, `scan()`
- **Encode API:** `encode()` — accepts `Uint8Array`, `Float32Array`,
  `Float64Array`, `Int32Array` as data inputs
- **DecodedMessage:** zero-copy TypedArray views (`object_data_f32/f64/i32/u8`)
  and safe-copy variant (`object_data_copy_f32`); all views handle zero-length
  payloads without UB
- **StreamingDecoder:** progressive chunk feeding, `feed()` returns `Result`
  (rejects oversized chunks), `last_error()` / `skipped_count()` for corrupt
  message visibility, configurable `set_max_buffer()` (default 256 MiB),
  `reset()`, `pending_count()`, `buffered_bytes()`
- **DecodedFrame:** per-object streaming output with `data_f32/f64/i32/u8`,
  `descriptor()`, `base_entry()`, `byte_length()`
- **tensogram-szip:** pure-Rust CCSDS 121.0-B-3 AEC/SZIP codec — encode,
  decode, range-decode; FFI cross-validated against libaec; signed
  preprocessing fixed (overflow bug found and corrected by tests)
- **Feature gates:** `szip-pure` and `zstd-pure` in `tensogram-encodings` and
  `tensogram-core`; mutually exclusive with `szip` / `zstd` (C FFI) variants
- **Tests:** 134 `wasm-bindgen-test` tests (run via `wasm-pack test --node`);
  50 `tensogram-szip` unit tests + 15 FFI cross-validation tests
- **Build:** `wasm-pack build rust/tensogram-wasm --target web`
- **Test:** `wasm-pack test --node rust/tensogram-wasm`

## tensogram-benchmarks

23 smoke tests + 36 unit tests (+ GRIB tests gated on `eccodes`). Separate workspace crate.

- `constants.rs` — Shared `AEC_DATA_PREPROCESS` constant (value 8, fixed from incorrect value 1).
- `datagen.rs` — Deterministic SplitMix64-based synthetic weather field generator.
- `report.rs` — `BenchmarkResult` with `TimingStats` (median/min/max), `Fidelity` enum
  (Exact, Lossy{linf, l1, l2}, Unchecked), throughput (MB/s), compressed-size
  variability tracking. `compute_fidelity()` compares decoded output to original.
- `codec_matrix.rs` — 24 pipeline combos. `compute_params` inside the timed encode loop
  for SP cases. Uses `encode_pipeline_f64` to avoid bytes→f64 round-trip. Configurable
  warm-up (default 3). Returns `BenchmarkRun` with separate `results` and `failures`.
- `grib_comparison.rs` — Symmetric end-to-end timing. Uses `encode_pipeline_f64`.
  Returns `BenchmarkRun`.
- `lib.rs` — `BenchmarkError` enum (Validation, Pipeline), `CaseFailure`, `BenchmarkRun`
  with `all_passed()`. Binaries exit non-zero on failures.
- Two binaries: `codec-matrix` (default 10 iterations, 3 warmup) and `grib-comparison`
  (requires `--features eccodes`). Both accept `--warmup` flag.
- `build.rs` — links `libeccodes` via pkg-config or Homebrew fallback when `eccodes` feature is active.
- Documentation: `docs/src/guide/benchmarks.md`, `docs/src/guide/benchmark-results.md`.

## tensogram-core

Unit, integration, adversarial, and edge-case tests.

- `wire.rs` — v2 frame-based wire format: Preamble (24B), FrameHeader (16B), Postamble (16B), FrameType enum (incl. PrecederMetadata type 8), MessageFlags (incl. bit 6 PRECEDER_METADATA), DataObjectFlags
- `framing.rs` — `encode_message()` with two-pass index construction, `decode_message()`, `scan()` for multi-message buffers. Decomposed into 5 focused helpers.
- `metadata.rs` — Deterministic CBOR encoding for GlobalMetadata, DataObjectDescriptor, IndexFrame, HashFrame (three-step: serialize → canonicalize → write). `verify_canonical_cbor()` utility.
- `types.rs` — `GlobalMetadata` (version, base, `_reserved_`, `_extra_`), `DataObjectDescriptor`, `IndexFrame`, `HashFrame`
- `dtype.rs` — All 15 dtypes (float16/32/64, bfloat16, complex64/128, int/uint 8-64, bitmask)
- `hash.rs` — xxh3 hashing + verification (xxh3 only)
- `encode.rs` — Full encode pipeline: validate → build pipeline config → encode per object → hash → assemble frames. Auto-populates `base[i]._reserved_.tensor` entries. Validates that client code does not write to `_reserved_`.
  - `encode_pre_encoded()` — Bypass the encoding pipeline for already-encoded payloads. Accepts pre-packed bytes with a descriptor declaring encoding/filter/compression. Validates object structure (shape, dtype, szip block offsets) but skips the pipeline. Available across all bindings: Rust, Python, C FFI, C++.
  - `StreamingEncoder::write_object_pre_encoded()` — Streaming variant for progressive encode of pre-encoded objects.
- `decode.rs` — `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()` (split results by default, `join` parameter)
- `file.rs` — `TensogramFile`: open, create, lazy scan, append, seek-based random access
- `iter.rs` — `MessageIter` (zero-copy buffer), `ObjectIter` (lazy per-object decode), `FileMessageIter` (seek-based file), `objects_metadata()` (descriptor-only)
- `streaming.rs` — `StreamingEncoder<W: Write>`: progressive encode, footer hash/index, no buffering, `write_preceder()` for per-object streaming metadata
- Feature gates: `mmap` (memmap2 zero-copy), `async` (tokio spawn_blocking)
- `DecodePhase` enum for frame ordering validation

## tensogram-encodings

47 tests.

- `simple_packing.rs` — GRIB-style lossy quantization, MSB-first bit packing, 0-64 bits, NaN rejection, `decode_range()` for arbitrary bit offsets. Optimized encode/decode: precomputed scale (no per-value division), specialized byte-aligned loops for 8/16/24/32 bits, fused NaN+min+max scan in `compute_params`.
- `shuffle.rs` — Byte-level shuffle/unshuffle (HDF5-style)
- `libaec.rs` — Safe Rust wrapper around libaec: `aec_compress()` with optional RSI block offset tracking (`aec_compress_no_offsets`), `aec_decompress()`, `aec_decompress_range()`. Auto-sets `AEC_DATA_3BYTE` for 17-24 bit samples (fixes corruption bug). 
- `pipeline.rs` — `encode_pipeline_f64()` variant for callers with typed f64 data (avoids bytes→f64 conversion). Auto-sets `AEC_DATA_MSB` for szip when encoding is SimplePacking (fixes byte-order mismatch so libaec's predictor sees most-significant bytes first; compression ratio now matches ecCodes at ~27% on 24-bit GRIB data — see `docs/src/guide/benchmark-results.md`).
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
- Streaming encoder: `tgm_streaming_encoder_create/write/write_preceder/write_pre_encoded/count/finish/free`
- Auto-generated `tensogram.h` (~544 lines) via cbindgen
- Panic safety: `panic = "abort"` in both release and dev profiles
- Vec capacity UB fixed (shrink_to_fit before forget), null pointer validation

## C++ Wrapper

117 GoogleTest tests across 11 files.

- `cpp/include/tensogram.hpp` — single-header C++17 wrapper (~934 lines)
- RAII classes: `message`, `metadata`, `file`, `buffer_iterator`, `file_iterator`, `object_iterator`, `streaming_encoder`
- `encode_pre_encoded()` — free function for already-encoded payloads
- `streaming_encoder::write_object_pre_encoded()` — streaming variant
- Typed exception hierarchy: `error` → `framing_error`, `metadata_error`, etc.
- `decoded_object` view with `data_as<T>()`, `element_count<T>()`
- Range-based for via `message::iterator`
- C++ Core Guidelines: `[[nodiscard]]`, `noexcept`, `const`-correct, Rule of Five
- CMake build: GoogleTest v1.15.2 via FetchContent

## tensogram-python (PyO3)

226 pytest tests.

- Full Python API with NumPy integration
- `encode()`, `decode()`, `decode_metadata()`, `decode_object()`, `decode_range()`, `scan()`
- `encode_pre_encoded()` — bypass pipeline for already-encoded payloads (bytes input, not numpy arrays)
- `StreamingEncoder` — progressive encode to file with `write_object()` and `write_object_pre_encoded()`
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

### Rust (12 runnable, workspace member)
01 encode_decode, 02 mars_metadata, 03 simple_packing, 04 shuffle_filter, 05 multi_object, 06 hash_verification, 07 scan_buffer, 08 decode_variants, 09 file_api, 10 iterators, 11 streaming, 11 encode_pre_encoded

### C++ (6 examples, C++ wrapper API)
01 encode_decode, 02 mars_metadata, 03 simple_packing, 04 file_api, 05 iterators, 11 encode_pre_encoded

### Python (12 examples)
01 encode_decode, 02 mars_metadata, 03 simple_packing, 04 multi_object, 05 file_api, 06 hash_and_errors, 07 iterators, 08 xarray_integration, 08 zarr_backend, 09 dask_distributed, 09 streaming_consumer, 11 encode_pre_encoded

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

5 canonical `.tgm` files in `rust/tensogram-core/tests/golden/`:
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

## record-benchmark-results (2026-04-07, v0.6.0)

Ran both benchmark binaries on this machine and published results as a static docs page.

- **Machine**: MacBook Pro (Mac16,1), Apple M4 10-core (4P+6E), 16 GB, macOS 26.3.1
- **Rust**: 1.94.1 (e408947bf 2026-03-25)
- **ecCodes**: 2.46.0 (Homebrew)
- **Codec matrix**: 24 encoder×compressor×bit-width combos on 16M float64 values, 5 iterations, seed 42 — all 24 combos succeeded, no `[ERROR]` rows
- **GRIB comparison**: 3 methods (eccodes grid_ccsds, eccodes grid_simple, tensogram sp(24)+szip) on 10M float64 values, 24-bit, 5 iterations, seed 42
- **New file**: `docs/src/guide/benchmark-results.md` — metadata table, exact commands, verbatim output tables, portability note
- **Updated**: `docs/src/SUMMARY.md` — added `[Benchmark Results](guide/benchmark-results.md)` after `[Benchmarks]`
- **Docs build**: `mdbook build` passes with no errors or warnings

## Dependencies

- ciborium 0.2, serde 1, thiserror 2, xxhash-rust 0.8
- libaec-sys (szip), zstd 0.13, lz4_flex 0.11, blosc2 0.2, zfp-sys-cc 0.2, sz3 0.4
- clap 4, serde_json 1, tempfile 3 (dev)
