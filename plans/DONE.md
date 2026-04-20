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

## Hash-while-encoding

Inline xxh3-64 hashing fused into the encoding pipeline, eliminating a
second pass over the encoded payload whenever `EncodeOptions.hash_algorithm
= Some(Xxh3)` (the default).  Output bytes, descriptor contents, and wire
format are unchanged — every `.tgm` produced after this change is
byte-identical to the pre-change output for every codec and every thread
count.

| Component | What changed |
|-----------|-------------|
| `tensogram-encodings/pipeline.rs` | New `PipelineConfig.compute_hash: bool` flag and `PipelineResult.hash: Option<u64>`; new `copy_and_hash` helper fuses the passthrough copy with `Xxh3Default::update` in 64 KiB chunks; hash is produced at each codec exit point while the buffer is cache-hot.  Added `xxhash-rust` workspace dependency. |
| `tensogram-core/encode.rs` | `encode_one_object` sets `config.compute_hash = options.hash_algorithm.is_some()` in `EncodeMode::Raw` and populates the descriptor's `HashDescriptor` from `PipelineResult.hash`.  `EncodeMode::PreEncoded` unchanged (no pipeline to fuse with). |
| `tensogram-core/hash.rs` | New `pub(crate) format_xxh3_digest(u64) -> String` helper — single source of truth for `"{digest:016x}"` formatting, used by both the pipeline-inline path and the legacy post-hoc `compute_hash`. |
| `tensogram-core/streaming.rs` | `write_object_inner` refactored to use new `write_data_object_frame_hashed` helper that writes directly to the `W: Write` sink, hashing the payload in 64 KiB chunks during the single pass.  Reduces streaming payload reads from three (hash → memcpy → write) to one.  CBOR size is pre-computed via a placeholder digest (xxh3 is fixed-width), guarded by a debug assert. |
| Tests | New `pipeline.rs` unit tests: `streaming_and_oneshot_xxh3_agree`, `pipeline_hash_none_when_disabled`, `pipeline_hash_matches_post_hoc_for_{passthrough,simple_packing,lz4}`, `pipeline_f64_hash_matches_post_hoc`, `pipeline_hash_byte_identical_across_threads_transparent`.  New integration suite `tensogram-core/tests/hash_while_encoding.rs` covering buffered/streaming hash parity, no-hash passthrough, multi-thread determinism, and `verify_hash` round-trip. |
| Benchmarks | New `hash_overhead.rs` criterion bench comparing `no_hash`, `two_pass_hash`, `fused_inline_hash` across `{none+none, none+lz4, sp24+szip, sp24+zstd3}` on 16 Mi float64 (128 MiB).  Passthrough case: ~11% total encode speedup, recovering ~24% of hash overhead.  Heavy pipelines (sp24+szip): within noise, as expected — encode dominates. |

### Determinism contract

Unchanged from the multi-threaded pipeline contract.  The hash is a pure
function of the final encoded bytes and follows the same rule:
transparent codecs produce byte-identical hashes across thread counts;
opaque codecs (blosc2, zstd with workers) hash their own
worker-completion-ordered output, which round-trips losslessly.

### What did NOT change

- Public APIs of `tensogram-core`, `tensogram-ffi`, `tensogram-cli`,
  `tensogram-wasm`, and every language binding (Python, C++, WASM, TS).
- Wire format — every `.tgm` byte is identical to pre-change output.
- Golden files — verified by the existing `tests/golden_files.rs` suite.
- `compute_hash` and `verify_hash` public functions — still used by
  `encode_pre_encoded`, validation, and external callers.
- `encode_pre_encoded` hashing — still a single pass over caller bytes
  (no pipeline to fuse with; already optimal).

## Multi-threaded coding pipeline

Caller-controlled `threads: u32` budget on `EncodeOptions` and
`DecodeOptions`, off by default (`threads=0` matches the sequential
path byte-for-byte — golden files unchanged).  When the caller opts
in, a scoped rayon pool is built for the call and work is dispatched
**axis-B-first** so a small number of very large messages benefits
most:

| Stage | Axis B mechanism |
|-------|------------------|
| `simple_packing` encode/decode | chunked `par_iter`, byte-aligned chunks (LCM-chunked for non-aligned widths) |
| `shuffle` / `unshuffle` | parallel byte-plane outer loop (shuffle), output-chunk scatter (unshuffle) |
| `blosc2` | `CParams/DParams::nthreads` on the compress path (decompress stays sequential — safe-wrapper limitation) |
| `zstd` FFI | `NbWorkers` via `bulk::Compressor` (requires `zstdmt` feature) |

Axis A (`par_iter` across objects) is the fallback when no object has
an axis-B-friendly codec, to avoid N×M thread over-subscription.

| Component | What changed |
|-----------|-------------|
| `tensogram-core/parallel.rs` | new private module: `resolve_budget`, `with_pool`, `run_maybe_pooled`, `is_axis_b_friendly`, `use_axis_a`, `should_parallelise`; constants `DEFAULT_PARALLEL_THRESHOLD_BYTES` (64 KiB) and `ENV_THREADS="TENSOGRAM_THREADS"` re-exported from the crate root |
| `tensogram-core/encode.rs` | `encode_one_object` extracted; axis-A/B dispatch at the top of `encode_inner` |
| `tensogram-core/decode.rs` | axis-A/B dispatch in `decode`, `decode_object`, `decode_range_from_payload`; `decode_single_object_with_backend` gains `intra_codec_threads` arg |
| `tensogram-core/streaming.rs` | `StreamingEncoder` captures `EncodeOptions.threads` at construction and forwards to every `write_object` pipeline call (axis B only — streaming semantics preclude axis A) |
| `tensogram-encodings/pipeline.rs` | `PipelineConfig.intra_codec_threads` threaded into every codec builder |
| `tensogram-encodings/simple_packing.rs` | `encode_with_threads`, `decode_with_threads`, `compute_params_with_threads`; parallel aligned + generic (LCM-chunked) paths; shared `splat_aligned`/`gather_aligned` helpers |
| `tensogram-encodings/shuffle.rs` | `shuffle_with_threads`, `unshuffle_with_threads`; 64 KiB threshold for parallel path |
| `tensogram-encodings/compression/blosc2.rs` | `nthreads` field + `build_cparams`/`build_dparams` helpers |
| `tensogram-encodings/compression/zstd.rs` | `nb_workers` field + `bulk::Compressor` with `NbWorkers`; `zstdmt` workspace feature enabled on the zstd dep |
| Python (PyO3) | `threads` kwarg on `encode`, `encode_pre_encoded`, `decode`, `decode_object`, `decode_range`, `TensogramFile.append`/`decode_message`/`file_decode_*`, `AsyncTensogramFile` mirrors, `StreamingEncoder.__init__` |
| C FFI | `threads: u32` parameter added to `tgm_encode`, `tgm_encode_pre_encoded`, `tgm_decode`, `tgm_decode_object`, `tgm_decode_range`, `tgm_file_append`, `tgm_file_decode_message`, `tgm_streaming_encoder_create`; `tensogram.h` regenerated |
| C++ wrapper | `encode_options.threads`, `decode_options.threads` (default 0); all free functions and member methods forward them |
| CLI | global `--threads N` (env `TENSOGRAM_THREADS` fallback) on every subcommand; honoured by `merge`, `split`, `reshuffle`, `convert-grib`, `convert-netcdf`; metadata-only commands ignore it |
| Benchmark | new `rust/benchmarks/src/threads_scaling.rs` module + `threads-scaling` binary; 7 codec configurations × thread sweep |
| Docs | `docs/src/guide/multi-threaded-pipeline.md` (API, axis-A/B policy, determinism contract, env-var semantics, cross-language parity matrix, free-threaded Python notes, tuning recommendations); Threading Scaling section in `benchmark-results.md` |
| Examples | `examples/rust/src/bin/16_multi_threaded_pipeline.rs`, `examples/python/16_multi_threaded_pipeline.py` |
| Feature gate | new `threads` cargo feature (default-on native, off on `wasm32`) pulls `rayon`; when disabled, any `threads > 0` request logs a one-time `tracing::warn!` and falls back to sequential |

**Determinism contract.**  Two classes of codec behave differently:

- **Transparent codecs** (`none`, `lz4`, `szip`, `zfp`, `sz3`,
  `simple_packing`, `shuffle`) produce **byte-identical** encoded
  payloads across all `threads` values.
- **Opaque codecs** (`blosc2` with `nthreads>0`, `zstd` with
  `nb_workers>0`) may produce compressed bytes that differ from the
  sequential path (block offsets land in worker completion order) but
  always **round-trip losslessly** regardless of how they were
  encoded.

`threads=0` (the default) is byte-identical to the pre-feature
sequential path on every codec, so golden `.tgm` files continue to
validate unchanged.

**Policy.**  The choice between axis A and axis B is taken once per
call based on the descriptors — there is no run-time tunable beyond
`threads` and `parallel_threshold_bytes` (small-message cutoff,
default 64 KiB).  The env var `TENSOGRAM_THREADS` fills in when
`threads=0`; explicit option beats environment.

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

## TypeScript wrapper (`@ecmwf/tensogram`)

Ergonomic typed layer over `tensogram-wasm` for browser + Node consumers.
Design doc: `plans/TYPESCRIPT_WRAPPER.md`. User guide:
`docs/src/guide/typescript-api.md`.

- **Package.** `typescript/` — ESM-only, strict TS, Node ≥ 20, built
  via `wasm-pack build --target web` + `tsc`. Package name
  `@ecmwf/tensogram`.
- **WASM side-change.** `rust/tensogram-wasm/src/convert.rs::to_js`
  uses `Serializer::json_compatible()` so CBOR metadata surfaces as
  plain JS objects rather than ES `Map`. Backwards-compatible at the
  WASM-test level because existing tests use `from_value`, which
  accepts both shapes.
- **Public surface.** `init()` (idempotent async WASM load); `encode`
  / `decode` / `decodeMetadata` / `decodeObject` / `scan`; dtype-aware
  payload views (`data()` safe-copy, `dataView()` zero-copy);
  `decodeStream(readable, opts?)` over Web Streams; `TensogramFile`
  with `.open(path)` (Node), `.fromUrl(url)` (fetch), `.fromBytes()`
  factories; `getMetaKey`, `computeCommon`, `cborValuesEqual`
  metadata helpers; dtype introspection (`typedArrayFor`,
  `payloadByteSize`, `shapeElementCount`, `DTYPE_BYTE_WIDTH`,
  `SUPPORTED_DTYPES`, `isDtype`).
- **Error hierarchy.** Abstract `TensogramError` base + 8 WASM-mapped
  subclasses (`FramingError`, `MetadataError`, `EncodingError`,
  `CompressionError`, `ObjectError`, `IoError`, `RemoteError`,
  `HashMismatchError`) + 2 TS-only (`InvalidArgumentError`,
  `StreamingLimitError`). Base constructor
  `(message, rawMessage = message)` lets TS-side errors pass a single
  argument. `mapTensogramError` parses WASM error strings, strips
  variant prefixes from `.message` for consistency, and extracts
  `expected` / `actual` hex digests for hash mismatches.
- **C++ parity.** A `remote_error` class was added to the header-only
  C++ wrapper to cover the `TGM_ERROR_REMOTE` code that previously
  fell through to the generic `error` base.
- **Memory model.** Safe-copy by default (`data()` allocates on the
  JS heap, survives WASM memory growth). Zero-copy opt-in (`dataView()`
  — invalidated on the next WASM call that grows memory). Explicit
  `close()` + `FinalizationRegistry` fallback for all handles.
- **Streaming.** `decodeStream` wraps the existing WASM
  `StreamingDecoder` as an async generator; handles `AbortSignal`,
  `maxBufferBytes`, and per-message corruption via `onError` without
  breaking iteration. Cleans up on early `break`, throws, and aborts.
- **Cross-language parity.** TS decodes the same
  `rust/tensogram-core/tests/golden/*.tgm` fixtures that Rust,
  Python, and C++ verify — full parity matrix in
  `plans/TYPESCRIPT_WRAPPER.md`.
- **Property-based robustness.** `fast-check` invariants pin
  `mapTensogramError` totality, `encode → decode` bit-exactness across
  random shapes + MARS metadata, and the "no panic on random bytes"
  invariant for `decode`.
- **Build / test.** `make ts-build` (wasm-pack + tsc), `make ts-test`
  (vitest), `make ts-typecheck` (strict tsc on src + tests). `make
  test` / `make lint` / `make clean` include the TS lanes. The
  `typescript` CI job mirrors the `wasm` lane.
- **Examples.** `examples/typescript/` carries runnable `.ts` files
  numbered in step with `examples/python/` (`01_encode_decode`,
  `02_mars_metadata`, `03_multi_object`, `05_streaming_fetch`,
  `06_file_api`, `07_hash_and_errors`). The package references the
  local `typescript/` package via a `file:` dependency.

## TypeScript wrapper — Scope C.1 (API-surface parity)

Closes the parity gap with Rust / Python / FFI / C++ for every
concept on the cross-language matrix in
`plans/TYPESCRIPT_WRAPPER.md`.  Changes are additive on the WASM
side; the only breaking change on the TS side is that
`TensogramFile#rawMessage` is now async (needed for the lazy HTTP
backend).

| Component | What changed |
|-----------|-------------|
| `tensogram-wasm/src/encoder.rs` | New `StreamingEncoder` class backed by a `Vec<u8>` sink, with `write_object`, `write_object_pre_encoded`, `write_preceder`, `object_count`, `bytes_written`, `finish`.  Hand-off from Rust core `StreamingEncoder<Vec<u8>>`. |
| `tensogram-wasm/src/extras.rs` | New module with `decode_range`, `encode_pre_encoded`, `compute_hash`, `simple_packing_compute_params`, `validate_buffer` exports.  `decode_range` accepts `BigUint64Array` pairs on the boundary; `validate_buffer` returns a JSON string so large integers are lossless. |
| `tensogram-wasm/src/convert.rs` | `typed_array_or_u8_to_bytes` covers every `ArrayBufferView` + `DataView` + `Uint8Array`, used by every new write path.  Old `typed_array_to_bytes` removed from `lib.rs`. |
| `tensogram-wasm/Cargo.toml` | Adds `tensogram-encodings` (for `simple_packing::compute_params`) and `serde_json` (for the JSON-returning validate export). |
| `rust/tensogram-wasm/tests/wasm_tests.rs` | 22 new `wasm_bindgen_test`s covering every new export and the StreamingEncoder lifecycle. |
| `typescript/src/range.ts` | `decodeRange(buf, objIndex, ranges, opts?)`.  Packs `number`/`bigint` pairs into `BigUint64Array`, returns dtype-typed `parts` views.  `join: true` concatenates at the byte level before dtype wrap. |
| `typescript/src/hash.ts` | `computeHash(bytes, algo?)` returning the hex digest string.  Unknown algorithm → `MetadataError`. |
| `typescript/src/simplePacking.ts` | `simplePackingComputeParams(values, bits, decScale?)` returning snake-cased params that spread directly into a descriptor. |
| `typescript/src/validate.ts` | `validate(buf, opts?)` — single message.  `validateBuffer(buf, opts?)` — multi-message with gap detection.  `validateFile(path, opts?)` — Node-only convenience (reads file via `node:fs/promises`, then delegates to `validateBuffer`).  All three parse the JSON the WASM side returns. |
| `typescript/src/encodePreEncoded.ts` | `encodePreEncoded(meta, objects, opts?)`.  Validates descriptor shape before the WASM call, forwards the pre-encoded bytes verbatim. |
| `typescript/src/streamingEncoder.ts` | `StreamingEncoder` class (single-use) with `writeObject`, `writePreceder`, `writeObjectPreEncoded`, `finish`, `close`, `objectCount`, `bytesWritten`.  `FinalizationRegistry` cleanup fallback for missed `close()`. |
| `typescript/src/file.ts` | Rewritten backend model: `InMemoryBackend` (fromBytes), `LocalFileBackend` (open, with path stored for append), `LazyHttpBackend` (fromUrl, HTTP Range).  `rawMessage` is now async — was sync in Scope B.  `append(meta, objects, opts?)` appends to the on-disk file, refreshes the mirror + position index from disk, only permitted on `LocalFileBackend`.  `fromUrl` does a `HEAD` probe; when Accept-Ranges + Content-Length are advertised it uses Range reads per preamble during scan and on-demand Range fetches per message afterwards (with a 32-entry LRU).  Streaming-mode messages (`total_length == 0`) and non-Range servers transparently fall back to eager GET. |
| `typescript/src/index.ts` | Re-exports the new functions, classes, and types. |
| `typescript/tests/` | Per-module test files: `range.test.ts`, `hash.test.ts`, `simplePacking.test.ts`, `validate.test.ts`, `encodePreEncoded.test.ts`, `streamingEncoder.test.ts`, `append.test.ts`, `lazyFromUrl.test.ts`.  Golden-file parity: the validate suite decodes every golden `.tgm` fixture. |
| `examples/typescript/` | `04_decode_range.ts`, `08_validate.ts`, `11_encode_pre_encoded.ts`, `12_streaming_encoder.ts`, `13_range_access.ts`.  `06_file_api.ts` updated for the async `rawMessage` signature. |
| `docs/src/guide/typescript-api.md` | New sections on pre-encoded bytes, validate, streaming encoder, append, lazy Range access, and the Scope-C API additions table.  Cross-language parity matrix refreshed. |
| `plans/TYPESCRIPT_WRAPPER.md` | Parity matrix now shows every Scope-C.1 entry as ✓; Scope-C.3 / C.4 items moved into the follow-ups section. |

## TypeScript wrapper — Scope C.2 (half-precision + complex dtypes)

Upgrades `typedArrayFor(dtype)` so `float16`, `bfloat16`, `complex64`,
`complex128` return first-class view classes — callers no longer
need to know the raw bit layout or interleaving.

| Component | What changed |
|-----------|-------------|
| `typescript/src/float16.ts` | `Float16Polyfill` with the observable behaviour of the TC39 Stage-3 `Float16Array` proposal: round-ties-to-even narrow, NaN / ±Inf / ±0 / subnormal preservation.  Storage is a `Uint16Array` of bits (WeakMap-backed private slot so `wrapBits` can build zero-copy instances).  `hasNativeFloat16Array()` / `getFloat16ArrayCtor()` let callers control native-vs-polyfill.  `float16FromBytes(bytes)` zero-copies on aligned input, falls back to an aligned copy for odd-offset buffers. |
| `typescript/src/bfloat16.ts` | `Bfloat16Array` — 1-8-7 layout matching ML frameworks.  Widen is "shift left by 16 into float32"; narrow uses round-to-nearest-even on the 16 dropped bits.  Mirror of the Float16Polyfill API. |
| `typescript/src/complex.ts` | `ComplexArray(dtype, storage)` view over an interleaved `Float32Array` (complex64) or `Float64Array` (complex128).  `.real(i)`, `.imag(i)`, `.get(i) → {re, im}`, `.set(i, re, im)`, iteration, `.toArray()`.  `complexFromBytes(dtype, bytes)` zero-copies on aligned input. |
| `typescript/src/dtype.ts` | Routing updated — `float16` → native `Float16Array` when present, polyfill otherwise; `bfloat16` → `Bfloat16Array`; `complex64` / `complex128` → `ComplexArray`. |
| `typescript/src/types.ts` | `TypedArray` union extended with structural aliases for the three view classes.  Structural aliases avoid a circular import. |
| `typescript/tests/float16.test.ts` | Bit-conversion invariants (±0, ±Inf, NaN, subnormals, range-saturation, round-to-nearest-even), array API (constructor shapes, `.bits`, `.set`, `.fill`, `.slice`, `.subarray`, `.toFloat32Array`, iteration), native-vs-polyfill detection, `encode → decode` bit-exact round-trip for the dtype, `fast-check` property: `f32 → f16 → f32` within half-precision ulp. |
| `typescript/tests/bfloat16.test.ts` | Same shape as float16, adapted for bfloat16 layout + ulp. |
| `typescript/tests/complex.test.ts` | `.real` / `.imag` / `.get` / `.set` / iteration / `.toArray()`; round-trip through `encode → decode` for complex64 and complex128; property-based invariant that interleaved storage round-trips byte-exactly. |
| `typescript/tests/dtype.test.ts` | Updated assertions — `typedArrayFor('float16')` now returns `Float16Polyfill` or native; `bfloat16` returns `Bfloat16Array`; `complex*` returns `ComplexArray`. |
| `typescript/src/index.ts` | Exports the new classes + factories + detection helpers. |
| `docs/src/guide/typescript-api.md` | "First-class half-precision and complex dtypes" section documents the new view types and the Scope-B → C.2 migration note. |

### What did NOT change

- Wire format — every `.tgm` byte is identical to pre-change output
  across every Scope-C addition.
- Existing golden `.tgm` fixtures in `rust/tensogram/tests/golden/`
  continue to validate and decode unchanged across Rust, Python, C++,
  and TypeScript.
- Scope-B TS tests (145 of them) still pass — the only breakage was
  the intentional `rawMessage` async change, which was mechanical.

## TypeScript wrapper — streaming `StreamingEncoder` (Pass 6)

Closes the "callback-per-frame" gap flagged in Pass 4's focus list.
Consumers that need true no-buffering streaming (browser upload,
WebSocket push, any sink that needs bytes as soon as they're
produced) now pass an `onBytes` callback at construction time.

| Component | What changed |
|-----------|-------------|
| `rust/tensogram-wasm/src/encoder.rs` | New `JsCallbackWriter` struct that implements `std::io::Write` by calling into a held `js_sys::Function`; errors thrown by the JS callback surface as `std::io::Error::other(...)` which the core wraps into `TensogramError::Io`.  New `Inner` enum with `Buffered(StreamingEncoder<Vec<u8>>)` / `Streaming(StreamingEncoder<JsCallbackWriter>)` variants; the exported class dispatches every method through the enum.  Constructor gained a third optional argument `on_bytes: Option<js_sys::Function>`. |
| `typescript/src/types.ts` | `StreamingEncoderOptions.onBytes` field added with full docstring covering the synchronous-only contract, chunk-ownership rules (copy before next write), and error-propagation semantics. |
| `typescript/src/streamingEncoder.ts` | Constructor validates that `opts.onBytes` (if supplied) is callable, forwards it to the WASM layer, and records the mode in a `#streaming` private field.  New `streaming: boolean` getter for consumers that need to branch on mode.  `finish()` docstring updated to explain the empty-`Uint8Array` return in streaming mode. |
| `typescript/tests/streamingEncoder.test.ts` | 9 new tests: construction-time byte delivery (preamble magic check), empty `finish()` return in streaming mode, decoded-bytes parity between buffered and streaming outputs, `bytesWritten` tracking, multi-object delivery, callback-throw propagation, non-function rejection, `streaming` getter, `hash: false` compatibility. |
| `rust/tensogram-wasm/tests/wasm_tests.rs` | 5 new `wasm_bindgen_test`s: round-trip, construction-time delivery + magic check, bytes-written tracking, callback-throw propagation, non-function rejection.  All existing tests updated to pass `None` for the new constructor arg. |
| `examples/typescript/14_streaming_callback.ts` | Runnable example: collects chunks into a JS array, reassembles via concatenation, decodes to prove semantic equivalence with the buffered path. |
| `docs/src/guide/typescript-api.md` | New "Streaming `StreamingEncoder` (no full-message buffering)" section covering the full contract (synchronous, chunk ownership, error propagation, mode detection). |
| `plans/TYPESCRIPT_WRAPPER.md` | Follow-up list notes that Python / FFI / C++ remain buffered-only and documents the extension pattern for future parity. |
| `CHANGELOG.md` | New entry under "Added — TypeScript wrapper: streaming `StreamingEncoder`". |

## `simple_packing` — Infinity rejection in the Rust core

The TypeScript wrapper's `simplePackingComputeParams` Pass-3 guard
against ±Infinity was a band-aid over a real core-level gap:
`tensogram_encodings::simple_packing::compute_params` and `encode`
silently produced `binary_scale_factor = i32::MAX` (garbage) when fed
infinite samples.  Pass 5 pushed the guard upstream so every binding
benefits uniformly.

| Component | What changed |
|-----------|-------------|
| `rust/tensogram-encodings/src/simple_packing.rs` | New `PackingError::InfiniteValue(usize)` variant reports the first infinite sample's index (matching the NaN contract).  A new shared `validate_sample(v, i) -> Result<(), PackingError>` helper lives next to `scan_min_max`; both the min/max scan and the encode loops (sequential, parallel-aligned, parallel-generic, sequential-tail) now use it as the single source of truth for the "is this value encodable?" predicate. |
| `rust/tensogram-encodings/src/simple_packing.rs` tests | New `test_positive_infinity_rejected_in_compute_params`, `test_negative_infinity_rejected_in_compute_params`, `test_infinity_rejected_in_encode` — cover both f64 infinity polarities through both the compute and encode paths. |
| `python/tests/test_tensogram.py::TestPackingParamsCoverage` | `test_inf_accepted_but_nonsensical` (which documented the old buggy behaviour) replaced by `test_positive_infinity_rejected` + `test_negative_infinity_rejected` — mirrors the Rust core tests and pins the cross-language contract. |
| TS wrapper | Existing Pass-3 guard in `typescript/src/simplePacking.ts` retained as defence-in-depth — callers still see a clean `InvalidArgumentError` without a WASM round-trip, and the check now acts as a belt-and-braces backup over the core guarantee. |
| CHANGELOG | Entry under "Changed — Rust core (affects all language bindings)". |

## Python bindings — `compute_hash` parity

Adds `tensogram.compute_hash(data, algo="xxh3")` to close the
cross-language parity gap — Rust, WASM, FFI, C++, and TypeScript all
exposed equivalent functionality; Python was the only binding without
it.

| Component | What changed |
|-----------|-------------|
| `python/bindings/src/lib.rs` | New `py_compute_hash` function wrapping `tensogram_lib::compute_hash`.  Accepts `PyBackedBytes` (zero-copy over `bytes` / `bytearray`); unknown algorithm names route through the existing `to_py_err` for a clean `ValueError`. |
| `python/tests/test_compute_hash.py` | 18 tests covering shape (16-char lowercase hex), determinism, known vector (xxh3 of `b"hello world"`), empty buffer (known constant `2d06800538d394c2`), `bytes` / `bytearray` acceptance, explicit rejection of `memoryview` / `numpy.ndarray` with a documented conversion path, unknown-algo `ValueError`, and byte-level parity with the hash stamped by `encode` on a no-pipeline object.  The parity test pins Python's `compute_hash` to the same byte-level contract as the Rust core, WASM, FFI, and TypeScript implementations — any drift fails the test simultaneously. |
| `CHANGELOG.md` | New entry under Python bindings. |
| `plans/TYPESCRIPT_WRAPPER.md` | Parity matrix now shows ✓ for Python's `compute_hash`. |

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

## Tensoscope (web viewer)

- React SPA (`tensoscope/`) for exploring `.tgm` files in the browser.
- Depends on `@ecmwf/tensogram` WASM package (`typescript/`) for all
  decode — no server-side component.
- **File loading.** Drag-and-drop or URL fetch; WASM scan builds an
  in-memory index of messages and objects without decoding payloads.
- **Field browser.** Sidebar lists every decodable field; selecting one
  triggers WASM decode and map render.
- **Map rendering.** deck.gl `BitmapLayer` over MapLibre GL JS.
  Regridding runs in a dedicated web worker (`regrid.worker.ts`) to
  keep the main thread free.
- **Colour scale.** Per-field min/max with chromatic colour maps;
  `ColorBar` overlay for reference.
- **Animation.** `StepSlider` + play/pause over the step/time dimension;
  `AnimationControls` handles frame timing.
- **Projections.** Equirectangular (flat) and globe.
- **Globe renderer.** CesiumJS (open-source, no Ion token, OSM base
  tiles via `OpenStreetMapImageryProvider`) replaces MapLibre for the
  globe projection mode. MapLibre remains for the flat/Mercator mode.
  The projection toggle in the toolbar switches renderers; both share
  the same decoded field data.
- **Render mode.** A `RenderModePicker` toggle in the map toolbar
  switches between heatmap and filled-contours display. In
  filled-contours mode the regrid worker quantises decoded pixel values
  into N discrete colour bands (N = colour-scale step count; default 10
  for continuous palettes, stop count for custom palettes). The same
  canvas-based pipeline is used for both renderers -- no GeoJSON or
  marching-squares algorithm is required.
- **State.** Zustand store (`useAppStore.ts`) owns selected file, field,
  step, and colour scale.
- **Deployment.** nginx Docker image; `BASE_PATH` env var for subpath
  deployments. Makefile targets: `build`, `push`, `run`, `stop`.

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
