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
  `python/bindings` crate + three separate Python packages
  (`tensogram-xarray`, `tensogram-zarr`, `tensogram-anemoi`). See
  `ARCHITECTURE.md` for the full crate list and what each one does.
- **Quality bar.** Zero clippy warnings, zero ruff issues,
  `panic = "abort"` on both release and dev profiles. Library code
  avoids `unwrap`/`expect`/`panic!` on any fallible input path;
  remaining `unwrap` calls on library code are confined to provably
  infallible conversions (e.g. `chunk.try_into()` after
  `chunks_exact(N)`). Panics never cross the FFI boundary.

---

## Remote bidirectional walker — pipelined and on by default

The remote backend now uses a pipelined bidirectional walker out of
the box across Rust, Python, and TypeScript.  Each iteration fetches
the next forward preamble, the next backward postamble, AND the
previous iteration's candidate-preamble validation in parallel,
collapsing the per-round critical path from 2 RTTs to 1 RTT.  On
real-network workloads (LUMI, ~47 ms RTT, 301 messages) this halves
wall-clock for full layout discovery and produces 1.3× to 3.2× wins
across `message_count`, `decode_metadata` at any index, and
`prefetch_layouts`.

| Component | Shape |
|-----------|-------|
| `rust/tensogram/src/scan_opts.rs` | `RemoteScanOptions::default()` returns `bidirectional: true`; doctest pinned to that value |
| `rust/tensogram/src/remote.rs` | `scan_bidir_round_locked` and `scan_bidir_round_async` use two independent `get_range` futures joined via `tokio::join!` instead of `store.get_ranges(&[fwd, bwd])` (avoiding `object_store`'s 1 MiB coalescer) |
| `rust/tensogram/src/remote.rs` | New `scan_pipelined_async(target_fwd_count: Option<usize>)` runs the full pipelined walk off the state mutex, accumulates layouts in local `Vec`s, and commits via `record_forward_hop` / `record_backward_hop` only on success.  `Ok(false)` on snapshot drift so concurrent callers never expose partial state.  `target_fwd_count = Some(idx + 1)` lets `decode_metadata(idx)` short-circuit once forward reaches the target |
| `rust/tensogram/src/remote.rs` | `message_count_async`, `message_layouts_async`, `ensure_layout_eager_async`, `ensure_message_async`, `ensure_all_layouts_batch_async` route through `scan_pipelined_async` first, falling back to the per-round walker on bail |
| `python/bindings/src/lib.rs` | Four PyO3 entry points (`PyTensogramFile.open` / `.open_remote`, `PyAsyncTensogramFile.open` / `.open_remote`) default `bidirectional=True`; `scan_opts_for` always emits explicit options so the opt-out survives any future default change |
| `typescript/src/file.ts` | `lazyScanMessages` runs `runPipelinedBidirectional` (replaces the old per-round `tryBidirectionalRound`); same snapshot/commit pattern, same-message-meet detection, eager footer fetch gated on the just-validated preamble's flags before issuing the GET |
| `typescript/src/file.ts` `effectiveBidirectional` | Default is `true`; `concurrency: 1` is illegal alongside the bidirectional default and rejects synchronously |
| `tests/remote-parity/drivers/rust_driver` | Always passes explicit `RemoteScanOptions { bidirectional: args.bidirectional }` |
| `rust/benchmarks/src/mock_http.rs` | `std`-only TCP listener with Range support and per-request counters; bench-local, no `hyper`/`tokio` |
| `rust/benchmarks/src/remote_scan_bench.rs`, `bin/remote_scan_metrics.rs`, `benches/remote_scan.rs` | Shared scenario runner + deterministic NDJSON emitter + Criterion wall-clock |
| `rust/benchmarks/python/bench_remote_scan.py` | Custom argparse harness (`--quick`, `--headline`, `--mode {sync,async}`, `--json`) reusing `tests/remote-parity/mock_server.py` |
| `typescript/tests/run_remote_bench.ts`, `tests/remote_scan.bench.ts` | Node runner spawning `mock_server.py` + Vitest `bench()` headline cells; `vitest.config.ts` gains `benchmark.include` |
| `tests/remote-parity/tools/gen_fixtures.py` | New `_STREAMING_TAIL` kind plus `thousand-msg`, `hundred-msg-footer`, `thousand-msg-footer`, `streaming-tail` fixtures; validator requires `preamble.total_length == postamble.total_length` on every non-streaming message |
| `examples/rust/src/bin/18_remote_scan_trace.rs`, `examples/typescript/18_remote_scan_trace.ts` | Subscribe to `tensogram::remote_scan` tracing events / `console.debug` walker events |
| `plans/WIRE_FORMAT.md §9.3` | Rewritten with format-vs-transport taxonomy: in-memory `ScanOptions` vs remote `RemoteScanOptions` |
| `docs/src/guide/remote-access.md`, `typescript-api.md` | Updated to reflect new defaults |

Architectural finding behind the work: the original per-round walker
was paying 2 RTTs per round (paired primary fetch + sequential
candidate-preamble validation), so the design's "halves discovery
hops" claim was undelivered — both walkers paid roughly the same
wall-clock.  Pipelining round k's validation with round k+1's
primary fetches is what makes the design actually deliver.  An
earlier round of localhost benchmarks measured request count and
bytes (the criterion the TODO entry called out), but the actual
operator-visible benefit lives in wall-clock under non-zero RTT,
which the criterion explicitly demoted to informational.

---

## `tensogram doctor` — environment diagnostics subcommand

A top-level CLI subcommand that reports compiled-in features, backend library
versions, and runs a self-test of the encode/decode pipeline plus the GRIB and
NetCDF converters.  Designed so support tickets can include a single
`tensogram doctor --json` dump that captures the full environment.

| Component | What was built |
|-----------|---------------|
| `tensogram-encodings/src/version.rs` | Feature-gated `BackendVersion` lookups for every codec: runtime FFI calls for C libraries (`ZSTD_versionString`, `blosc2_get_version_string`, `zfp_version_string`), a tiny C shim for libaec (`AEC_VERSION_STR`), and compile-time `Cargo.lock` snapshots via the `built` crate for pure-Rust deps |
| `tensogram-encodings/build.rs` | Compiles the libaec version shim via `cc::Build` and invokes `built::write_built_file()` |
| `tensogram-encodings/build_shim/libaec_version.c` | Exposes `AEC_VERSION_STR` as `tensogram_libaec_version()` |
| `tensogram/build.rs` | Invokes `built::write_built_file()` for tensogram-core deps |
| `tensogram/src/doctor/version.rs` | Feature-gated lookups for `remote` (object_store), `mmap` (memmap2), `async` (tokio), `grib` (`codes_get_api_version`), `netcdf` (`nc_inq_libvers`) |
| `tensogram/src/doctor/mod.rs` | Public data model (`DoctorReport`, `BuildInfo`, `FeatureStatus`, `FeatureState`, `SelfTestResult`, `SelfTestOutcome`), `run_diagnostics()`, `run_self_test()` with core and per-codec rows |
| `tensogram-cli/src/commands/doctor.rs` | `TempFileGuard` drop-guard, embedded fixtures via `include_bytes!`, converter self-tests, human and JSON renderers, `DoctorFailed` sentinel |
| `tensogram-cli/src/main.rs` | `Doctor` variant in `Commands`, dispatch arm, `DoctorFailed` → exit code 1 |
| `tensogram-cli/tests/doctor_cli.rs` | Integration tests: exit 0 + `Status: HEALTHY`, `--json` produces valid JSON with expected keys |
| `share/tensogram/doctor/` | `sanity.grib2` (4×4 GRIB2 2t), `sanity-classic.nc` (2×2 f32 NetCDF-3), `sanity-hdf5.nc` (2×2 f32 NetCDF-4), `regenerate.py`, `README.md` |
| `docs/src/cli/doctor.md` | Usage, flag list, output sections, JSON schema, exit codes, library API, examples |

Key design choices: no external probes (compiled-in features only), `Skipped`
rows for absent features (never `Failed`), exit code 1 only on actual test
failures, fixtures embedded in the binary via `include_bytes!`.

---

## Python async bindings

`AsyncTensogramFile` exposes all read/decode operations as `asyncio`
coroutines via `pyo3-async-runtimes` + tokio. A single handle supports
truly concurrent operations (core async methods take `&self`, no mutex).

| Component | What changed |
|-----------|-------------|
| `tensogram/file.rs` | All async methods `&mut self` → `&self`. Added `decode_range_async`, `decode_range_batch_async`, `decode_object_batch_async`, `prefetch_layouts_async`, `message_count_async`. Sync batch: `decode_range_batch`, `decode_object_batch`. |
| `tensogram/remote.rs` | Added `read_range_async`, `read_range_batch_async`, `read_object_batch_async`, `ensure_all_layouts_batch_async` (batched layout discovery via `get_ranges`), `message_count_async`. Sync batch: `read_range_batch`, `read_object_batch`. |
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
| `tensogram` | `Dtype::swap_unit_size()`, `DecodeOptions.native_byte_order`, threaded through all decode paths + iterators |
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
| `tensogram/encode.rs` | `encode_one_object` sets `config.compute_hash = options.hash_algorithm.is_some()` in `EncodeMode::Raw` and populates the descriptor's `HashDescriptor` from `PipelineResult.hash`.  `EncodeMode::PreEncoded` unchanged (no pipeline to fuse with). |
| `tensogram/hash.rs` | New `pub(crate) format_xxh3_digest(u64) -> String` helper — single source of truth for `"{digest:016x}"` formatting, used by both the pipeline-inline path and the legacy post-hoc `compute_hash`. |
| `tensogram/streaming.rs` | `write_object_inner` refactored to use new `write_data_object_frame_hashed` helper that writes directly to the `W: Write` sink, hashing the payload in 64 KiB chunks during the single pass.  Reduces streaming payload reads from three (hash → memcpy → write) to one.  CBOR size is pre-computed via a placeholder digest (xxh3 is fixed-width), guarded by a debug assert. |
| Tests | New `pipeline.rs` unit tests: `streaming_and_oneshot_xxh3_agree`, `pipeline_hash_none_when_disabled`, `pipeline_hash_matches_post_hoc_for_{passthrough,simple_packing,lz4}`, `pipeline_f64_hash_matches_post_hoc`, `pipeline_hash_byte_identical_across_threads_transparent`.  New integration suite `tensogram/tests/hash_while_encoding.rs` covering buffered/streaming hash parity, no-hash passthrough, multi-thread determinism, and `verify_hash` round-trip. |
| Benchmarks | New `hash_overhead.rs` criterion bench comparing `no_hash`, `two_pass_hash`, `fused_inline_hash` across `{none+none, none+lz4, sp24+szip, sp24+zstd3}` on 16 Mi float64 (128 MiB).  Passthrough case: ~11% total encode speedup, recovering ~24% of hash overhead.  Heavy pipelines (sp24+szip): within noise, as expected — encode dominates. |

### Determinism contract

Unchanged from the multi-threaded pipeline contract.  The hash is a pure
function of the final encoded bytes and follows the same rule:
transparent codecs produce byte-identical hashes across thread counts;
opaque codecs (blosc2, zstd with workers) hash their own
worker-completion-ordered output, which round-trips losslessly.

### What did NOT change

- Public APIs of `tensogram`, `tensogram-ffi`, `tensogram-cli`,
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
| `tensogram/parallel.rs` | new private module: `resolve_budget`, `with_pool`, `run_maybe_pooled`, `is_axis_b_friendly`, `use_axis_a`, `should_parallelise`; constants `DEFAULT_PARALLEL_THRESHOLD_BYTES` (64 KiB) and `ENV_THREADS="TENSOGRAM_THREADS"` re-exported from the crate root |
| `tensogram/encode.rs` | `encode_one_object` extracted; axis-A/B dispatch at the top of `encode_inner` |
| `tensogram/decode.rs` | axis-A/B dispatch in `decode`, `decode_object`, `decode_range_from_payload`; `decode_single_object_with_backend` gains `intra_codec_threads` arg |
| `tensogram/streaming.rs` | `StreamingEncoder` captures `EncodeOptions.threads` at construction and forwards to every `write_object` pipeline call (axis B only — streaming semantics preclude axis A) |
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
  and `tensogram`; mutually exclusive with `szip` / `zstd` (C FFI).
- **Build / test.** `wasm-pack build rust/tensogram-wasm --target web`,
  `wasm-pack test --node rust/tensogram-wasm`.

## TypeScript wrapper (`@ecmwf.int/tensogram`)

Ergonomic typed layer over `tensogram-wasm` for browser + Node consumers.
User guide: `docs/src/guide/typescript-api.md`.

- **Package.** `typescript/` — ESM-only, strict TS, Node ≥ 20, built
  via `wasm-pack build --target web` + `tsc`. Package name
  `@ecmwf.int/tensogram`.
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
  `rust/tensogram/tests/golden/*.tgm` fixtures that Rust,
  Python, and C++ verify; see the parity section of
  `docs/src/guide/typescript-api.md`.
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
concept on the cross-language matrix (see the parity section of
`docs/src/guide/typescript-api.md`).  Changes are additive on the
WASM side; the only breaking change on the TS side is that
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

## TypeScript wrapper — Scope C.5 (browser-usable remote + async parity)

Brings the TypeScript wrapper, the WASM crate, and the Tensoscope
viewer to remote-access parity with the Rust `tensogram` crate's
`object_store` integration over HTTP(S) and AWS-signed HTTPS.  The
Rust core already had S3 / GCS / Azure / HTTP via `object_store`;
this scope ports the user-facing pieces that matter for browser
consumers — without re-implementing wire-format parsing in TS.

| Component | What changed |
|-----------|-------------|
| `rust/tensogram/src/decode.rs` | New public `decode_object_from_frame(frame_bytes, options)` and `decode_range_from_frame(frame_bytes, ranges, options)` — same axis-A/axis-B dispatch and mask-aware NaN/Inf restoration as `decode_object` / `decode_range`, but take one frame's bytes in isolation.  Used by the WASM single-frame helpers and any consumer that has fetched one indexed frame via Range. |
| `rust/tensogram/tests/decode_object_from_frame.rs` | New parity suite: 4 tests covering uncompressed and zstd round-trips against `decode::decode_object` / `decode::decode_range`, plus a non-frame rejection check.  Reads index frame manually so the test reproduces the lazy backend's actual fetch shape. |
| `rust/tensogram-wasm/src/layout.rs` | New module exporting nine `#[wasm_bindgen]` helpers: `read_preamble_info`, `read_postamble_info`, `parse_header_chunk`, `parse_footer_chunk`, `read_data_object_frame_header`, `read_data_object_frame_footer`, `parse_descriptor_cbor`, `decode_object_from_frame`, `decode_range_from_frame`.  Each routes errors through `convert::js_err` and (for chunk parsers) routes metadata through `convert::metadata_to_js` so the synthesised wire-format `version` field matches the eager-backend `decode_metadata` path. |
| `rust/tensogram-wasm/src/lib.rs` | Re-exports the layout helpers; adds `DecodedMessage::from_single_object(desc, data)` constructor used by `layout::decode_object_from_frame`. |
| `rust/tensogram-wasm/tests/layout_tests.rs` | 11 new `wasm_bindgen_test`s covering boundary normalisation (number↔bigint), preamble/postamble parsing, header + footer chunk parse round-trips, frame-header + frame-footer reads, descriptor-CBOR round-trip, and `decode_object_from_frame` / `decode_range_from_frame` parity against the encoded fixture. |
| `typescript/src/file.ts` | `POSTAMBLE_BYTES` corrected from 16 (v2) to 24 (v3).  `LazyHttpBackend` gains a per-message `MessageLayout` cache (preamble + optional metadata + index + descriptors) mirroring Rust `RemoteBackend.state`.  New public methods: `messageMetadata(i)` (header chunk or footer suffix only); `messageDescriptors(i)` (index + per-frame descriptor-prefix optimisation); `messageObject(i, j)` (one Range per frame); `messageObjectRange(i, j, ranges)` (partial sub-tensor decode against one fetched frame); `messageObjectBatch` / `messageObjectRangeBatch` (parallel fan-out with bounded concurrency); `prefetchLayouts(msgIndices)` (pre-warm the layout cache).  All Range fetches inside descriptor logic route through `b.limit` so the per-host cap is never bypassed; the outer per-task limiter is removed because nesting limiter slots would deadlock at low caps. |
| `typescript/src/internal/layout.ts` | Numeric-safety boundary: `safeNumberFromBigint` rejects WASM-returned u64 values above `Number.MAX_SAFE_INTEGER` (TS file positions are JS `number` throughout — files larger than 2^53 − 1 must use the Rust or Python bindings).  Normalisers for `PreambleInfo`, `PostambleInfo`, `FrameFooterInfo`, `FrameIndex`. |
| `typescript/src/internal/pool.ts` | FIFO bounded-concurrency limiter (`createLimiter(concurrency)`).  Default 6, matching typical browser per-host connection limits.  Exact in-flight cap; rejects do not starve subsequent tasks. |
| `typescript/src/internal/httpRange.ts` | Reusable `fetchRange(ctx, start, end, requireRange?)` helper.  Routes through any caller-supplied `fetch` implementation; tolerates `200 OK` with full body when `requireRange = false`. |
| `typescript/src/internal/wbgWrap.ts` | Shared adapter that wraps a wasm-bindgen `DecodedMessage` handle in the public `DecodedMessage` interface.  `FinalizationRegistry` cleanup keyed on the returned identity; new `metadataOverride` parameter lets the lazy backend's `messageObject` substitute the message's real cached metadata in place (mutating the wrapped object instead of spreading, which would have leaked the WASM handle). |
| `typescript/src/internal/rangePack.ts` | `flattenRangePairs(ranges)` and `concatBytes(parts)` — extracted from `range.ts` so both buffer-level `decodeRange` and per-message `messageObjectRange` paths share the same range-pair packing and byte-concat routines. |
| `typescript/src/decode.ts`, `typescript/src/range.ts` | Refactored to use the shared `wbgWrap` and `rangePack` helpers.  Public surface unchanged. |
| `typescript/src/types.ts` | `FromUrlOptions.concurrency` field added (default 6). |
| `typescript/src/auth/signAwsV4.ts` | Pure AWS Signature V4 signer.  Web Crypto only.  Byte-for-byte against `aws-sig-v4-test-suite` vectors (get-vanilla, query-string canonicalisation incl. duplicate-key ordering, header value trim and inner-whitespace collapse, session token, pre-encoded path round-trip). |
| `typescript/src/auth/awsSigV4Fetch.ts` | `createAwsSigV4Fetch(creds, opts?)` — `fetch`-compatible wrapper pluggable into `FromUrlOptions.fetch`.  Read-only (signs over the SHA-256 of an empty body); for write paths use a presigned URL.  Header merge for `Request` inputs follows standard `fetch` semantics — `init.headers` overrides `input.headers`. |
| `typescript/src/index.ts` | Re-exports `signAwsV4Request`, `createAwsSigV4Fetch`, and the SigV4 type aliases. |
| `typescript/tests/layout.test.ts` | 16 tests pinning the WASM↔TS numeric boundary (number / bigint / 2^53 / u64::MAX / negative / NaN / fractional). |
| `typescript/tests/pool.test.ts` | 7 tests covering FIFO order, exact in-flight cap, single-slot serialisation, rejection propagation without starvation, the never-resolving-task-doesn't-starve case, invalid-concurrency rejection, and the `DEFAULT_CONCURRENCY = 6` constant. |
| `typescript/tests/lazyFromUrl.test.ts` | New test cases on top of the existing Scope-C.1 suite: `messageMetadata` fetches header chunk only; `messageObject` fetches one frame only; `messageObject` returns the real cached metadata (not a default-constructed placeholder); `messageObjectRange` and `messageObjectRangeBatch` round-trip; `messageObjectBatch` fans out within concurrency; `prefetchLayouts` warms cache; `messageDescriptors` lazy + eager paths (the eager path used to throw on stray `FR` byte sequences inside payloads); descriptor fan-out respects the concurrency cap; nested-pool deadlock regression at concurrency = 1; `total_length < PREAMBLE_BYTES + POSTAMBLE_BYTES` bails to eager (the v3 24-byte postamble fix); MAX_SAFE_INTEGER fallback. |
| `typescript/tests/signAwsV4.test.ts` | 12 tests covering AWS test vectors, anti-replay, query canonicalisation, and URI path encoding (literal-space vs pre-encoded `%20` round-trip; `%2F` not folded into `/`). |
| `typescript/tests/awsSigV4Fetch.test.ts` | 5 integration tests: signed requests accepted by a strict-auth mock S3, unsigned rejected, no-fetch-impl rejection, `init.headers` merges over `Request` headers (regression test for the fix that put Range back on signed requests), and Request-only Range header survives signing. |
| `tensoscope/src/tensogram/index.ts` | Migrated to layout-aware reads.  `buildIndex` calls `prefetchLayouts` in chunks of 64 (concurrency 6) so very large files don't queue thousands of Promises at once.  `decodeField` uses `messageObject(msgIdx, objIdx)` — one Range GET for the target object's frame instead of a whole-message download. |
| `tensoscope/src/components/file-browser/{FileOpenDialog,WelcomeModal}.tsx` | URL placeholder copy: replaced misleading `s3://` suggestion with "https:// URL (use a presigned URL for private S3/GCS/Azure)" — TypeScript scheme adapters are out of scope for this package. |
| `tensoscope/src/tensogram/__tests__/viewer.test.ts` | Lightweight structural-guard tests asserting the wrapper imports `TensogramFile` and not `decodeObject`, `buildIndex` calls `prefetchLayouts` before its `messageMetadata` loop, and `decodeField` uses `messageObject` (not `rawMessage` + `decodeObject`). |
| `examples/typescript/15_remote_access.ts` | End-to-end remote-access demo: in-process Range-capable HTTP server, then `messageMetadata` / `messageDescriptors` / `messageObject` / `messageObjectRange` against it.  Mirrors `examples/python/14_remote_access.py` and `examples/rust/src/bin/14_remote_access.rs`. |
| `examples/typescript/16_remote_batch.ts` | Demonstrates `prefetchLayouts` + `messageObjectBatch` with bounded concurrency.  Mock server records peak in-flight to prove the per-host cap is honoured. |
| `examples/typescript/17_remote_s3_signed_fetch.ts` | `createAwsSigV4Fetch` against a mock S3 that enforces `Authorization: AWS4-HMAC-SHA256 …`.  Demonstrates the unsigned→401, signed→200 transition. |
| `typescript/README.md`, `docs/src/guide/typescript-api.md` | Document the new layout-aware methods, the concurrency option, and the AWS SigV4 helpers. |
| `plans/TODO.md` | Adds `remote 7 — TS lazy scan: 256 KB forward-chunk variant` to the accepted backlog (deferred pending a benchmark that shows the round-trip saving outweighs the larger per-message fetches). |

### What's intentionally not in scope

- Direct `s3://` / `gs://` / `az://` URL adapters in TS (would require
  full protocol clients per scheme — separate scope).
- Azure Shared Key signing, Google HMAC signing — use presigned URLs.
- The 256 KB forward-chunk-during-scan variant — tracked as
  `remote 7` in `plans/TODO.md` pending a benchmark.

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

## `tensogram`

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
- For `regular_ll` grids with standard scan mode, lifts the four
  corner-point keys from the `geography` namespace into a canonical
  `mars.area = [N, W, S, E]` via the pure helper
  `compute_regular_ll_area` (module `area.rs`).  The only normalisation
  applied is the full-global dateline-first case (raw ecCodes
  `lon_first = 180, lon_last = 179.75` on a 0.25° grid, as used by
  ECMWF open-data) which is shifted to `west = -180, east = 179.75`;
  non-standard scan, dateline-crossing regional subdomains,
  non-`regular_ll` grids, and degenerate inputs are all rejected (no
  `mars.area` emitted), leaving downstream consumers free to either
  fall back to a compat default or refuse to render.

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
- Depends on `@ecmwf.int/tensogram` WASM package (`typescript/`) for all
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
- **Coordinate inference.** For GRIB-derived files that ship no
  explicit latitude / longitude coordinate objects,
  `inferAxesFromMarsGrid` synthesises 1-D axes from
  `mars.grid = "regular_ll"` + `mars.area` (emitted by
  `tensogram-grib` for supported grids) or, failing that, from a
  documented compatibility-bridge default — `[N=90, W=-180, S=-90,
  E=180]`, chosen to match the ECMWF open-data dateline-first
  convention that is the common source of such files.  A one-shot
  `console.warn` fires when the bridge default is used so the
  provenance gap is visible.  `expandAxesIfRectangularGrid`
  meshgrids the 1-D axes into per-point arrays for the regrid
  worker.  Non-`regular_ll` grids (`reduced_gg`, octahedral `O*`,
  Gaussian `N*`) return `null` — their geometry cannot be inferred
  from shape alone.
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

## Preallocation hardening across decode paths

A malformed `.tgm` descriptor whose tensor-shape product drove
`num_values × dtype_byte_width` into the terabyte range previously
aborted the decode process via infallible allocations scattered across
the codec, encoding, filter, and bitmask layers. Every descriptor-
derived allocation on the decode path is now fallible, and every
size-arithmetic step that could wrap `usize` on hostile input is
guarded.

| Component | What changed |
|-----------|-------------|
| `tensogram-encodings/src/libaec.rs` | `aec_decompress` and `aec_decompress_range` use `Vec::new()` + `try_reserve_exact` + `set_len(decoded_len)` in place of `vec![0u8; N]`. Trust-model doc comments on both entry points. `expected_size == 0` short-circuits to avoid the `NonNull::dangling` edge case. `checked_sub` on `expected_size - avail_out` guards `set_len`. A new `validate_params` helper rejects invalid `AecParams` (matching the pure-Rust crate): `bits_per_sample` outside `1..=32`, `block_size == 0`, invalid block sizes without `AEC_NOT_ENFORCE`, `rsi` outside `1..=4096`, `AEC_RESTRICTED` combined with `bits_per_sample > 4`. |
| `tensogram-szip/src/` | Decoder output, `decode_rsi` scratch, and `write_samples` output all use `try_reserve_exact`. `checked_mul` on `rsi_blocks × block_size` and on `samples × byte_width`. `params::validate` tightened to reject `block_size == 0` under `AEC_NOT_ENFORCE`. |
| `tensogram-encodings/src/simple_packing.rs` | Every decode-side allocation (`decode`, `decode_aligned`, `decode_aligned_par`, `decode_generic`, `decode_generic_par`, `decode_range`, and both `bits_per_value == 0` fast paths) uses a shared `try_reserve_f64` helper. `num_values × bpv` arithmetic promoted to `u128`, surfaced as a new `PackingError::BitCountOverflow` variant. Byte-count conversion uses `usize::try_from` producing `OutputSizeOverflow`. New `PackingError::AllocationFailed { bytes, reason }` variant. Parallel helpers now return `Result<Vec<f64>, PackingError>` and reserve before rayon dispatch (no partial output on failure). |
| `tensogram-encodings/src/bitmask/` | `try_reserve_mask` shared helper in `mod.rs`. `packing::unpack`, `rle::decode`, `roaring::decode` all use it. New `MaskError::AllocationFailed`. RLE's run-length overflow check reworked from `out.len() + run > n_elements` (itself overflow-prone) to `run > n_elements - out.len()` (safe after the guaranteed `out.len() <= n_elements` invariant). |
| `tensogram-encodings/src/zfp_ffi.rs` | `zfp_decompress_f64` uses `try_reserve_exact` + `resize`. |
| `tensogram-encodings/src/compression/{zfp,sz3}.rs` | The per-codec `f64_to_bytes` helpers now return `Result<Vec<u8>, CompressionError>`, with `checked_mul` on `values.len() × 8` and fallible byte-buffer reservation. |
| `tensogram-encodings/src/pipeline.rs` | `bytes_to_f64` and `f64_to_bytes` return `Result<_, PipelineError>` with the same guards. |
| `tensogram-encodings/src/shuffle.rs` | Output buffers use `try_reserve_shuffle` helper; new `ShuffleError::AllocationFailed`. |
| Tests | Helper-level tests (`try_reserve_f64`, `try_reserve_mask`, `aec_decompress` / `aec_decompress_range` with `usize::MAX`), public-API tests (simple_packing `decode` / `decode_range` with `usize::MAX` both at `bpv=0` and `bpv=64`, `zfp_decompress_f64` with `usize::MAX`, roaring with `u32::MAX + 1`), FFI parameter-validator parity tests covering all invalid combinations, and end-to-end `decode_pipeline` integration tests against both szip backends, simple_packing, and zfp. |
| Docs | `docs/src/guide/error-handling.md` — *Malformed Descriptor — Pathological Tensor Size* subsection with the per-codec / per-stage error-variant table; the No-Panic Guarantee bullet list now calls out `checked_mul`/`u128` promotion alongside `try_reserve_exact`. |

Design choices worth recording:

- **No static cap.** Imposing a documented maximum decompressed size
  was considered and rejected — the library's target workloads (ERA5,
  ML weights, high-res climate simulations) routinely produce
  legitimate multi-GiB single objects, so any cap that fits them is
  too permissive to block attackers, and any cap that blocks
  attackers breaks honest use. Fallible allocation is sufficient on
  its own.
- **Sound FFI buffer pattern.** `aec_decompress` keeps `len == 0`
  during the FFI call and only `set_len(decoded_len)` after libaec
  reports how many bytes it wrote. No uninitialised memory is ever
  observable through the `Vec` API; this avoids the UB trap of
  calling `set_len(expected_size)` before the bytes are written.
- **Checked arithmetic at every width-conversion step.** `u128`
  promotion for `num_values × bits_per_value`, `checked_mul` for
  sample-count → byte-count, `checked_sub` for the `expected_size -
  avail_out` math that drives `set_len`, `checked_add` (via
  rearrangement) for the RLE run-length accumulator.
- **Parallel path ordering.** All rayon-parallel decoders reserve the
  output buffer and resize it to length `num_values` *before* any
  `par_iter_mut` / `par_chunks_mut` call. A reservation failure
  cannot leave rayon with a partial output buffer — it errors out
  before dispatch.
- **FFI / pure parity contract.** The szip parameter validator in
  `libaec.rs` is a mirror of `params::validate` in `tensogram-szip`.
  Both are now tightened to reject `block_size == 0` even under
  `AEC_NOT_ENFORCE` (the previous policy would have led to
  divide-by-zero / infinite loops later in the decoder). Parity is
  asserted by tests on the FFI side; the pure-Rust side already had
  coverage.

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

Canonical `.tgm` files in `rust/tensogram/tests/golden/` used for
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

## anemoi-inference output plugin (`tensogram-anemoi`)

Standalone pip package at `python/tensogram-anemoi/` that registers
`TensogramOutput` as an anemoi-inference output plugin.

| Component | What was built |
|-----------|---------------|
| `src/tensogram_anemoi/output.py` | `TensogramOutput(Output)` — writes each forecast step as one tensogram message |
| `pyproject.toml` | `setuptools-scm` build, entry point `anemoi.inference.outputs = tensogram` |
| `tests/test_tensogram_output.py` | Round-trip, encoding, stacking, remote fsspec, metadata |

**Design decisions:**
- Registered via `[project.entry-points."anemoi.inference.outputs"]` so anemoi-inference discovers it on install — no fork required.
- All parameters after `path` are keyword-only (enforced with `*`).
- Per-object MARS keys (`date`, `time`, `step`) live in `base[i]["mars"]`; no redundant copy in `_extra_`.
- Pressure-level stacking produces 2-D `(n_grid, n_levels)` objects with `"levelist": [...]` in the `"anemoi"` namespace (MARS convention).
- `_extra_["dim_names"]` carries axis-size→name hints for the tensogram-xarray backend.
- Coordinate arrays (lat/lon) use `"grid_latitude"` / `"grid_longitude"` names, outside `KNOWN_COORD_NAMES`, so all objects share one flat dimension in xarray.

## earthkit-data integration (`tensogram-earthkit`)

Standalone pip package at `python/tensogram-earthkit/` that registers
tensogram as a first-class source **and** encoder with
[`ecmwf/earthkit-data`](https://github.com/ecmwf/earthkit-data), so
users load and write `.tgm` content through the same surface they use
for GRIB, NetCDF, BUFR, etc.

| Component | What was built |
|-----------|---------------|
| `src/tensogram_earthkit/source.py` | `TensogramSource(FileSource)` — entry point `earthkit.data.sources.tensogram`. Accepts local path, remote URL, bytes, bytearray, memoryview. Bytes inputs are materialised to a temp file that's unlinked via `weakref.finalize`. |
| `src/tensogram_earthkit/encoder.py` | `TensogramEncoder(Encoder)` — entry point `earthkit.data.encoders.tensogram`. Encodes FieldLists and xarray Datasets to lossless `.tgm` bytes via `tensogram.encode`. Exposes `TensogramEncodedData` for target consumers. |
| `src/tensogram_earthkit/detection.py` | `_match_magic(buf, deeper_check)` and `is_mars_tensogram(meta)` — magic-byte detection and per-object MARS discriminator. |
| `src/tensogram_earthkit/readers/{file,memory,stream}.py` | Reader-shaped modules exposing `reader` / `memory_reader` / `stream_reader` callables matching earthkit-data's internal registry contract, so a future upstream PR is a verbatim directory copy. |
| `src/tensogram_earthkit/fieldlist.py` | `TensogramSimpleFieldList` — MARS FieldList whose `to_xarray()` delegates to `tensogram-xarray`. |
| `src/tensogram_earthkit/field.py` + `ArrayField` | Array-namespace interop (numpy / torch / cupy / jax) inherited from `earthkit.data.sources.array_list.ArrayField`. |
| `src/tensogram_earthkit/mars.py` | `field_to_base_entry` / `base_entry_to_usermetadata` / `has_mars_namespace` — the bidirectional MARS ↔ tensogram `base[i]` mapping. |
| `src/tensogram_earthkit/data.py` | `TensogramData` — thin facade with `available_types = ("xarray", "numpy", "fieldlist")`. |
| `tests/` | 86 pytest cases: detection, source discovery, MARS / non-MARS / memory / stream / remote / encoder / array-namespace paths, with parity checks against tensogram-xarray for every xarray output. |
| `docs/src/guide/earthkit-integration.md` | User guide with mermaid architecture diagrams, decode / encode snippets, edge cases. Linked from `SUMMARY.md`. |
| `examples/python/18_earthkit_integration.py` | End-to-end example exercising MARS FieldList + xarray + encoder round-trip + torch namespace. |
| `Makefile` | `earthkit-install`, `earthkit-test`, `earthkit-lint` targets. |
| `.github/workflows/ci.yml` | New `test-earthkit` job on Linux (self-hosted).  An equivalent step is also captured in the (currently disabled) `main-macos` job so the macOS surface comes back automatically when that runner is re-enabled. |

**Design decisions:**

- Dual decode paths — MARS tensograms produce a FieldList, non-MARS tensograms produce xarray directly. The per-object MARS discriminator (`base[i]["mars"]` non-empty) is the single branch point.
- FieldList `to_xarray()` always delegates to `tensogram-xarray`'s `TensogramBackendEntrypoint.open_dataset` so coordinate detection, dim-name resolution, and lazy backing arrays live in exactly one place.
- Source-plugin only (not a reader-plugin): we do not register with earthkit-data's internal reader registry (which has no entry-point support) and instead use the `source.reader` hook. The reader-shaped module layout makes a future upstream PR mechanical.
- Bytes / bytearray / memoryview inputs go through a temp-file that's unlinked automatically via `weakref.finalize` when the source is collected. `close()` can also trigger early cleanup.
- Byte streams are drained and dispatched via the memory path. True progressive (yield-as-each-message-arrives) streaming is a follow-up — blocked by the xarray backend needing a concrete file and the FieldList contract requiring `__len__` up-front.
- Remote URLs (`http://`, `https://`, `s3://`, `gs://`, `az://`) are forwarded to `tensogram.TensogramFile.open_remote`; `storage_options` is threaded through to both the xarray backend and the FieldList builder.
- The encoder writes a lossless pass-through message (encoding / filter / compression all `"none"`). Callers who want a tuned pipeline use the `tensogram` Python API directly; the earthkit path prioritises round-trip fidelity.
- Array-namespace (torch / cupy / jax) interop is inherited from `earthkit.data.sources.array_list.ArrayField`, which uses `earthkit-utils.array` internally — no duplication here.

## simple_packing ergonomics: auto-compute + `sp_*` key rename

Two paired changes to the `simple_packing` encoding that together
drop the pre-encode `compute_packing_params` step from 99% of user
code.

### Wire-format rename to `sp_*` prefix

Every other codec's descriptor params use a codec-prefix convention
(`szip_rsi`, `zstd_level`, `blosc2_codec`, `zfp_rate`,
`shuffle_element_size`).  `simple_packing` is now consistent:

| Old                  | New                      |
|----------------------|--------------------------|
| `reference_value`    | `sp_reference_value`     |
| `binary_scale_factor`| `sp_binary_scale_factor` |
| `decimal_scale_factor`| `sp_decimal_scale_factor`|
| `bits_per_value`     | `sp_bits_per_value`      |

Preamble version stays at 3 (the structural wire layout is unchanged
— only CBOR descriptor keys rename).  Pre-rename v3 messages with
unprefixed simple_packing keys are no longer decodable; no
deprecation path because the software is pre-public.

### Encoder auto-compute

When the descriptor carries `encoding: "simple_packing"` +
`sp_bits_per_value` and the derived keys are absent, the encoder
calls `simple_packing::compute_params` on the input bytes and stamps
the full four-key set into the descriptor.  The encoded message
therefore remains fully self-describing — decoders see no difference
from a hand-written explicit descriptor.

| Knob                       | Default                     |
|----------------------------|-----------------------------|
| `sp_bits_per_value`        | required (no default)       |
| `sp_decimal_scale_factor`  | `0` when absent             |
| `sp_reference_value`       | auto-computed when absent   |
| `sp_binary_scale_factor`   | auto-computed when absent   |

When the caller supplies explicit `sp_reference_value` +
`sp_binary_scale_factor`, the encoder trusts them verbatim (Q2(b)
from the design plan).  This supports advanced workflows that need
pinned reference values across a time-series.

### Components touched

| Layer | What changed |
|-------|-------------|
| `rust/tensogram/src/encode.rs` | New `resolve_simple_packing_params(&mut desc, data_bytes)` called at the top of `encode_one_object` (Raw mode only).  Helper `bytes_as_f64_vec` respects the descriptor's `byte_order`.  `extract_simple_packing_params` reads the new `sp_*` keys. |
| `rust/tensogram/src/streaming.rs` | `StreamingEncoder::write_object` invokes the same resolver before building the pipeline config; `write_object_pre_encoded` is unchanged (pre-encoded bytes are opaque). |
| `rust/tensogram/src/pipeline.rs::apply_pipeline` | Stamps the four `sp_*` keys; automatically propagates the rename through `convert-grib` and `convert-netcdf`. |
| `python/bindings/src/lib.rs::compute_packing_params` | Returns a dict with `sp_*`-prefixed keys so `desc = {..., **params}` continues to work. |
| `rust/tensogram-wasm/src/extras.rs::simple_packing_compute_params` | WASM export returns an object with `sp_*` keys. |
| `typescript/src/types.ts::SimplePackingParams` | Interface fields renamed to `sp_*`. |
| `rust/tensogram-cli/src/commands/set.rs` | Immutable-key list covers the new names. |
| Test suites (Rust / Python / C++ / TS) | Every test literal using the old names has been updated.  New BDD suite `rust/tensogram/tests/simple_packing_auto_compute.rs` + mirrored Python suite `python/tests/test_simple_packing_auto_compute.py` exercise happy path, explicit-wins, missing knob (auto-compute and explicit paths), NaN rejection, streaming, defaults, non-f64 dtype (incl. 8-byte non-float dtypes), and byte-for-byte parity with the explicit path. |
| Examples | `examples/python/03_simple_packing.py` leads with the ergonomic form and demonstrates the explicit path as a secondary option. `examples/python/11_encode_pre_encoded.py`, `examples/rust/src/bin/{03,11}_*.rs`, `examples/cpp/{03,11}_*.cpp` updated. |
| Docs | `docs/src/encodings/simple-packing.md` gains a "Quick start — auto-compute" section; `docs/src/guide/encode-pre-encoded.md`, `docs/src/guide/encoding.md`, `docs/src/format/cbor-metadata.md`, `plans/WIRE_FORMAT.md` and the concept pages use the new key names. |

### What did NOT change

- Preamble wire version (stays at 3).
- Software version (will be bumped at release time).
- `simple_packing::compute_params` Rust signature, `SimplePackingParams` struct fields (those are internal Rust identifiers, separate from wire keys).
- `PackingError::InvalidParams { field }` internal field names (still the Rust struct field identifiers; tests assert on those).
- Golden `.tgm` fixtures (none use `simple_packing`).
- Any non-`simple_packing` codec.

### Hardening fixes

Three follow-ups landed alongside the rename:

| Fix | What changed |
|-----|-------------|
| **Strict float64 guard** | `resolve_simple_packing_params` and `build_pipeline_config_with_backend` now check `dtype == Dtype::Float64` exactly, not `byte_width() == 8`.  `Int64` / `Uint64` / `Complex64` are 8 bytes and would otherwise have been silently re-interpreted as `f64`, producing bogus packing params.  Error message now names the offending dtype. |
| **Pre-flight `sp_bits_per_value` check** | The presence check is hoisted ahead of the explicit-params short-circuit so missing-bpv produces a single, action-oriented diagnostic on every code path (auto-compute, ref+bsf-only, full explicit) instead of failing later with a generic "missing required" string from the param accessor. |
| **Fallible byte→f64 allocation** | `bytes_as_f64_vec` uses `try_reserve_exact` + push loop (matching `tensogram_encodings::pipeline::bytes_to_f64`) instead of an infallible `.collect()`, so encode-time auto-compute on very large inputs returns a structured `TensogramError::Encoding` on OOM rather than aborting the process. |

Test additions: `s3b_explicit_path_missing_bits_per_value_is_a_clear_error` and `s8_eight_byte_non_float_dtypes_rejected` (auto + explicit paths) in `rust/tensogram/tests/simple_packing_auto_compute.rs`.  Existing `edge_cases::missing_required_param` updated to assert on the new key-named diagnostic.

## Remote bidirectional scan — internal walker

Internal-only restructuring of the remote backend to support a
two-cursor "meet-in-the-middle" scan over HTTP Range requests.  The
public API stays forward-only by default; the bidirectional path is
reachable via a `pub(crate)` constructor used by tests until the
public API is extended.

| Component | What changed |
|-----------|-------------|
| `tensogram/remote.rs` — state | `RemoteState` extended with `suffix_rev`, `prev_scan_offset`, `bwd_active`, `fwd_terminated`, `gap_closed`, plus a `scan_epoch` race-detection counter.  Old `scan_complete: bool` replaced by computed `scan_complete()` accessor — collapses to `fwd_terminated` when `bwd_active=false && suffix_rev.is_empty()` so forward-only mode is byte-identical to the forward-only baseline. |
| `tensogram/remote.rs` — options | New `pub(crate) RemoteScanOptions { bidirectional }` (default `false`).  `RemoteBackend::open_with_scan_opts` / `open_async_with_scan_opts` accept it; `open` / `open_async` delegate with `Default`. |
| `tensogram/remote.rs` — algorithm | `scan_bidir_round_*` issues one forward-preamble + one backward-postamble fetch via `store.get_ranges(&[fwd, bwd])`; when the postamble parses cleanly, a candidate-preamble fetch validates the backward-discovered offset.  Pure parsers `parse_backward_postamble`, `validate_backward_preamble`, and `parse_forward_preamble` keep state-mutation under one lock acquisition. |
| `tensogram/remote.rs` — race detection | `ScanSnapshot { next, prev, epoch }` snapshotted before the paired fetch and validated for full-equality on reacquire in the async path; the sync path holds the mutex throughout `block_on_shared` so no recheck is needed.  `scan_epoch` bumps on every state-machine transition (forward/backward hop, fallback, terminate, close) so concurrent state changes — which can leave offsets unchanged — still invalidate stale paired fetches.  `scan_bidir_round_async` re-checks `bwd_active && !fwd_terminated` under the initial lock to defeat stale dispatch from a sibling that disabled backward in flight. |
| `tensogram/remote.rs` — commit decision | `apply_round_outcomes` + `commit_or_yield_backward` encode the commit table: forward Hit non-overlapping → record forward; same-message (1-msg file or odd-count middle meet) → forward only, leave `suffix_rev` intact; forward Hit overlapping → `disable_backward("backward-overlaps-forward")`; forward `ExceedsBound` (corrupt suffix offset) → `disable_backward("forward-exceeds-backward-bound")` then commit forward; backward format/streaming → `disable_backward(reason)`; forward format → `terminate_forward(reason)` which itself discards `suffix_rev` (bidirectional is never recovery). |
| `tensogram/remote.rs` — overflow safety | EOF and overlap guards use `pos.saturating_add(min_message_size) > bound` and `snap.prev < snap.next.saturating_add(min_message_size)` so wraparound on `u64::MAX`-class offsets correctly forces termination / forward-fallback.  Once the saturating guard passes, the unchecked `pos + PREAMBLE_SIZE` and `snap.prev - POSTAMBLE_SIZE` range constructions cannot wrap. |
| `tensogram/remote.rs` — bound dispatch | `forward_bound(state)` returns `prev_scan_offset` when `suffix_rev` is non-empty, else `file_size`.  `scan_fwd_step_*` accepts the bound as a parameter so the bidirectional dispatcher can prevent forward overrun into backward-claimed suffix.  `scan_next_*` keep their old signatures and pass `file_size`. |
| `tensogram/remote.rs` — accessors | `scan_step_*` central dispatcher routes between forward-only (`scan_fwd_step_*`) and bidirectional (`scan_bidir_round_*`).  `ensure_message_*` / `scan_all_locked` / `message_count_async` funnel through `scan_step_*`.  `ensure_layout_eager_*` / `ensure_all_layouts_batch_async` dispatch via an `EagerAction` tag so forward-only mode keeps its combined-chunk discovery (preamble + frames in one round trip). |
| `tensogram/remote.rs` — tracing | One-shot `tensogram::remote_scan` event in `open_with_scan_opts*` carrying the active scan `mode`.  Per-hop event with direction / offset / length on every `record_*_hop`.  Existing fallback / fwd_terminated / gap_closed events on the same target.  Filterable via `RUST_LOG=tensogram::remote_scan=debug`. |
| Tests | Truth-table tests pin the `scan_complete()` byte-identical equivalence with the previous `scan_complete: bool` and the helper invariants (`record_forward_hop`, `disable_backward` idempotency, `terminate_forward` clears `suffix_rev`).  Pure parser tests cover every backward taxonomy row (short fetch, bad END_MAGIC, postamble parse error, arithmetic underflow, overlap with forward, bad MAGIC at candidate, streaming preamble at non-tail).  End-to-end hyper-mock-server tests exercise the bidirectional path: 1-message no-duplicate (same-message overlap), 2-message clean meet, 3-message odd-count meet at the middle message, 10-message full traversal, streaming-postamble yield, streaming-in-the-middle no-recovery, corrupt END_MAGIC graceful degradation, postamble length below minimum, preamble/postamble length mismatch, concurrent readers consistency. |

**Design decisions:**
- Forward-only path is byte-identical.  Default `RemoteScanOptions { bidirectional: false }` means existing call sites land on the same code path with the same Range request sequence.  The remote-parity harness produces byte-identical scan logs to the forward-only baseline.
- Backward yields without modifying forward state.  Format errors on the backward path (bad END_MAGIC, postamble parse error, length below minimum, arithmetic underflow, overlap with forward, preamble mismatch) call `disable_backward(reason)`, which sets `bwd_active=false` and clears `suffix_rev` — the backend reverts to forward-only for the rest of its lifetime.
- Forward termination cascades into backward.  `terminate_forward` calls `disable_backward` so the invariant `fwd_terminated => suffix_rev.is_empty() && !bwd_active` always holds.  This preserves remote semantics: bidirectional may not surface a message that forward-only scanning couldn't reach.
- Same-message detection avoids double-commit.  When a paired round produces both forward Hit and backward Layout for identical `(offset, length)` (1-message file or odd-count meet at the middle), the forward layout commits once and the backward record yields silently — `suffix_rev` from earlier hops stays intact.
- Forward stays canonical when backward state is corrupt.  `ExceedsBound` covers the case where forward parses a clean preamble whose `msg_end` exceeds `prev_scan_offset` while still fitting inside `file_size`; backward must be wrong, so disable backward and commit forward.  Without this, bidirectional regressed vs forward-only on corrupt-backward inputs.
- Lock-around-await with full snapshot.  Offset-only validation isn't race-complete because `disable_backward` / `terminate_forward` / `close_gap` change state without changing offsets; the `scan_epoch` counter closes that hole.  The sync path holds the mutex across `block_on_shared` so no recheck is needed there.
- Saturating arithmetic on offset guards.  EOF and overlap checks use `saturating_add` so wraparound on `u64::MAX`-class offsets correctly forces termination; downstream range constructions are safe-by-construction once the saturating guard passes.

## Remote bidirectional scan — public API surface

Surfaces the bidirectional walker (built in the previous entry) as a
public Rust + Python option, plus extends the parity harness to assert
walker-mode equivalence on the discovered layouts.

| Component | What changed |
|-----------|-------------|
| `tensogram/scan_opts.rs` (new) | `RemoteScanOptions { bidirectional: bool }` lifted out of `remote.rs` into a cfg-independent module so `open_source` (which is not `remote`-gated) can accept `Option<RemoteScanOptions>` whether or not the `remote` Cargo feature is enabled.  Re-exported from the crate root. |
| `tensogram/file.rs` — open methods | `open_source`, `open_remote`, `open_source_async`, `open_remote_async` gain `scan_opts: Option<RemoteScanOptions>` (default `None` collapses to forward-only).  `RemoteScanOptions` is `Copy`, so passing by value is identical in cost to passing a reference and avoids the `Some(&opts)` boilerplate at every call site.  Local-file backends accept the value and silently ignore it.  The redundant `RemoteBackend::open` / `open_async` wrappers (which only forwarded to the `_with_scan_opts` siblings with `Default`) are dropped. |
| `tensogram/file.rs` — layout accessor | New `pub struct MessageLayout { offset, length }` with `Debug + Clone + Copy + PartialEq + Eq`.  New `pub fn message_layouts() -> Result<Vec<MessageLayout>>` (and async sibling) returns per-message offset+length in scan order, so any caller can compare layouts produced by either walker without reading payload.  Internal `remote::MessageLayout` (richer struct) renamed to `remote::CachedLayout` to free the public name. |
| `python/bindings/src/lib.rs` | `PyTensogramFile.open` / `open_remote` and the matching `PyAsyncTensogramFile` siblings gain a keyword-only `bidirectional: bool` argument (default `False`).  PyO3's strict `bool` extractor surfaces `bidirectional=1` / `="yes"` as `TypeError` at the call site (before any I/O), satisfying the eager-validation requirement.  Shared helpers `parse_storage_options` and `scan_opts_for` deduplicate the kwarg-handling between the four entry points. |
| `tests/remote-parity/drivers/rust_driver/src/main.rs` | New `--bidirectional` CLI flag (manual arg parsing kept).  New op `dump-layout` that emits per-message `[{"offset": ..., "length": ...}, ...]` JSON to stdout. |
| `tests/remote-parity/run_parity.py` | `DriverCase.mode` widened to `Literal["forward", "bidirectional"]`.  `collect_events` returns `dict[str, RunResult]` where `RunResult { events, stdout }` so the orchestrator now passes driver stdout through to the assertion layer.  A separate Rust-only generator emits four fixtures × `dump-layout` × `{forward, bidirectional}` cases; existing shared-language `_OPS` matrix unchanged. |
| `tests/remote-parity/test_parity.py` | New `test_rust_forward_vs_bidirectional_layouts_equal(fixture)` per fixture: parses the JSON layouts from each `RunResult.stdout`, asserts equality across modes — request patterns may differ, but the discovered layouts must match exactly.  Existing forward-only TS-vs-Rust scan-event parity tests migrated mechanically to `result.events`. |
| Tests | Rust: `open_remote_with_bidirectional_layouts_match_forward` (sync + async), `open_remote_some_false_equivalent_to_none`, `open_source_local_path_ignores_scan_opts`, `message_layouts_local_and_remote_agree`.  Python: `bidirectional` default-False, kwarg-True succeeds, layouts match forward-only, local-path no-op, eager `TypeError` on int / str / non-bool, keyword-only enforcement.  Async: same suite covering `AsyncTensogramFile`. |
| Build matrix | New `cargo check -p tensogram --no-default-features` step in the `rust-check` Makefile target guards the cfg-independent build path. |
| Docs | `docs/src/guide/remote-access.md` gains a `## Bidirectional Scan` section with the Rust + Python opt-in and the trade-off framing.  `examples/rust/src/bin/14_remote_access.rs` and `examples/python/14_remote_access.py` show the opt-in inline. |

**Design decisions:**
- `RemoteScanOptions` is cfg-independent.  The struct lives in `rust/tensogram/src/scan_opts.rs` (no `#[cfg]`) so `open_source` — which is unconditionally public — can accept `Option<RemoteScanOptions>` with or without the `remote` feature.  The struct is harmless without `remote`: `open_source` always falls through to `open()` and the parameter is dropped.
- `Option<RemoteScanOptions>` over a builder or borrow.  The struct is `Copy`, so the value form has zero overhead and reads cleanly at the call site (`Some(RemoteScanOptions { bidirectional: true })` rather than `Some(&opts)` after a separate `let opts = ...`).  `None` collapses to `Default::default()` in one line and the API remains a single function instead of fanning out into `open_remote_builder().bidirectional(true).build()`.
- `MessageLayout` is a slim struct, not a tuple.  Public tuple fields harden into the API surface and read poorly at the call site.  The struct adds a name to the contract (`offset`, `length`) and derives the obvious traits (`Debug + Clone + Copy + PartialEq + Eq`).
- Local-file `bidirectional=true` is silently no-op.  Mirrors how a hypothetical `storage_options` would behave on a local path; raising would force generic caller code to know the URL scheme before passing options through.
- Python `TypeError` over `ValueError` for non-bool input.  Python convention reserves `ValueError` for valid-typed values that fall outside the accepted range; type mismatches are `TypeError`.  PyO3 gives `TypeError` for free with strict `bool` extraction at the synchronous call site (before any `await`), which satisfies the eager-validation requirement.
- Driver stdout is captured separately from `ScanEvent`s.  The new `RunResult { events, stdout }` keeps HTTP request-pattern data and logical-output data on different channels — `dump-layout` cases produce stdout that other ops do not, and merging into `ScanEvent` would have forced every existing test to carry an unused stdout field.
- `dump-layout` is Rust-only for this sub-task.  TS bidirectional support is a separate sub-task; the harness adds Rust-only `mode="bidirectional"` cases without polluting the shared-language `_OPS` matrix that the existing TS-vs-Rust parity tests parametrise over.

## Remote bidirectional scan — TypeScript walker

Brings the bidirectional walker to TypeScript by promoting the pure
parsers to a public Rust module (and re-exposing them via four
`#[wasm_bindgen]` outcome functions) and rewriting the TS lazy-scan
dispatcher as a state machine that mirrors Rust's `scan_step_locked`
exactly.  Parsing logic and reason-string taxonomy live in **one**
source of truth; TypeScript keeps only the JS-shaped pieces (paired
`Promise.allSettled` fetches, `AbortController` cascade, dispatch).

| Component | What changed |
|-----------|-------------|
| `tensogram/remote_scan_parse.rs` (new) | Cfg-independent public module with three slim outcome enums (`BackwardOutcome`, `BackwardCommit`, `ForwardOutcome`) using `#[serde(tag = "kind", rename_all_fields = "camelCase")]` so the WASM-side JS shape is `{ kind: "Hit", offset, length, msgEnd }` and field names are camelCase but variant tags stay PascalCase.  Public reason-string constants document every Format/Terminate identifier.  Four pure parsers — `parse_backward_postamble`, `validate_backward_preamble`, `parse_forward_preamble`, `same_message_check` — re-exported from the crate root.  Internal `remote.rs` keeps its rich `Preamble`-carrying parsers unchanged for byte-output equivalence with the existing test suite. |
| `tensogram-wasm/src/remote_scan.rs` (new) | Four `#[wasm_bindgen]` thin wrappers — `parse_backward_postamble_outcome`, `validate_backward_preamble_outcome`, `parse_forward_preamble_outcome`, `same_message_check` — that delegate to the public Rust parsers and serialise the slim outcomes via `to_js`.  Pinned by per-variant `wasm-bindgen-test` round-trips in `rust/tensogram-wasm/tests/remote_scan_tests.rs` so JS shape drift surfaces as a Rust test failure rather than a silent TS-side parse error. |
| `typescript/src/types.ts` | `FromUrlOptions` gains `bidirectional?: boolean` (mirrors Rust `RemoteScanOptions { bidirectional }` and Python `bidirectional=True`) and `debug?: boolean` for `console.debug` instrumentation. |
| `typescript/src/internal/httpRange.ts` | `fetchRange` gains an optional fourth-positional `overrideSignal: AbortSignal` that takes precedence over `ctx.signal` for the underlying `fetch` — the per-round `AbortController` cascade in the bidirectional walker uses this to cancel sibling Range fetches without leaking into post-open Range requests. |
| `typescript/src/file.ts` — `fromUrl` | Eager validation: `bidirectional: true && concurrency: 1` throws `InvalidArgumentError` synchronously before any HEAD probe.  The limiter built from `concurrency` is now passed into `lazyScanMessages` so paired fetches honour the user's per-host cap. |
| `typescript/src/file.ts` — `messageLayouts` getter | Public passive accessor returning `readonly { offset: number; length: number }[]` — mirrors Rust `TensogramFile::message_layouts` and Python `PyTensogramFile.message_layouts`.  No additional fetches; reads the already-populated `positions` array. |
| `typescript/src/file.ts` — `lazyScanMessages` | Rewritten as a state-machine dispatcher matching Rust's `scan_step_locked`: a single `while (!scanComplete(state))` loop that dispatches each step to either `tryBidirectionalRound` (when `bwdActive && !fwdTerminated`) or `tryForwardStep`.  Backward yields (format error, streaming preamble, gap-below-min, overlap, exceeds-bound) transition the loop into forward-only continuation from `state.next` — never bail-to-eager, never restart from offset 0.  Returns `null` only on unrecoverable transport failures so the caller can fall back to eager download. |
| `typescript/src/file.ts` — `tryBidirectionalRound` | Pre-round min-gap guard mirrors Rust `scan_bidir_round_locked`: when `prev < next + MIN_MESSAGE_BYTES`, either `closeGap` (cursors meet) or `disableBackward("gap-below-min-message-size")` (gap too small, fall through to forward-only).  Per-iteration child `AbortController` derives from the user signal via `addEventListener('abort', …, { once: true })` and an explicit `removeEventListener` after the round to avoid listener accumulation across many rounds.  `Promise.allSettled` runs the paired forward / backward fetches through the limiter; one-side failure aborts the sibling and bails to eager.  When the postamble outcome is `NeedPreambleValidation`, fetches the candidate preamble and calls `validate_backward_preamble_outcome` before dispatching. |
| `typescript/src/file.ts` — `applyRoundOutcomes` + `commitOrYieldBackward` | Five mutually-exclusive cases mirror Rust `apply_round_outcomes` + `commit_or_yield_backward` exactly: backward Format / Streaming → `disableBackward`; backward Layout matching forward Hit (`same_message_check`) → silent yield; backward Layout overlapping forward Hit → `disableBackward("backward-overlaps-forward")`; forward `ExceedsBound` → `disableBackward("forward-exceeds-backward-bound")` + commit forward; otherwise → record. |
| `typescript/src/file.ts` — `tryForwardStep` | Forward-only step with EOF guard.  On any non-Hit outcome (`Streaming`, `Terminate`, `ExceedsBound`) before any layout has been committed, returns `false` (bail to eager) — preserves the pre-existing forward-only behaviour where streaming-mode preambles or length-too-small at the file start force eager fallback so the END_MAGIC walker can recover.  After at least one layout is committed, accepts the partial result via `terminateForward` (matches Rust `scan_fwd_step_*`). |
| `typescript/src/file.ts` — debug instrumentation | `tensogram:scan:mode`, `tensogram:scan:hop`, `tensogram:scan:fallback`, `tensogram:scan:fwd-terminated`, `tensogram:scan:gap-closed` events emitted via `console.debug` when `debug: true` — same kebab-case vocabulary as the Rust `tracing::debug!` events at `target = "tensogram::remote_scan"`. |
| `typescript/tests/lazyBidirectional.test.ts` (new) | Vitest suite covering 1/2/3/10-message walks, forward-vs-bidirectional layout equality, corrupt `END_MAGIC`, length mismatch, streaming-at-start eager fallback, `concurrency: 1` rejection, abort cascade, paired-round Range count invariants, debug-event vocabulary, and outcome determinism under reordered `Promise.allSettled` resolution.  Existing `lazyFromUrl.test.ts` regression suite stays green — forward-only path is byte-identical to before. |
| `tests/remote-parity/drivers/ts_driver.ts` | Gains `--bidirectional` flag (mirroring the Rust driver) and `dump-layout` op that prints per-message `[{"offset", "length"}, ...]` JSON to stdout.  The TS driver no longer rejects `mode = "bidirectional"`. |
| `tests/remote-parity/run_parity.py` | Drops the Rust-only-bidirectional restriction in `_run_driver`; `dump-layout` cases generated for both `rust` and `ts`.  New `is_layout_dump_case(case)` helper documents the parity-classifier exclusion: the existing `classifier.py` + `RoundBuilder` are forward-only, so `dump-layout` cases ship in this sub-task with stdout JSON comparison only — event-based cross-language assertions land in the next sub-task once the classifier learns about backward scan events. |
| `tests/remote-parity/test_parity.py` | `test_forward_vs_bidirectional_layouts_equal` parametrised over `(rust, ts)` × fixtures so a layout-equivalence regression in either language gets its own named failure.  Existing forward-only TS-vs-Rust scan-event parity tests stay unchanged. |
| `tensoscope/src/tensogram/__tests__/bidirectional.test.ts` (new) | Smoke test pinning the Vite dev-proxy contract: opens a 3-message file with `bidirectional: true` through a fetch that simulates the proxy's pass-through forwarding, verifies `messageCount` and `messageLayouts`.  Cheap regression guard against future proxy changes that would serialise paired Range requests or strip the Range header. |
| `examples/typescript/15_remote_access.ts` | Adds an opt-in demonstration: re-opens the in-process Range-capable HTTP server with `{ bidirectional: true }` and prints `messageCount` + `messageLayouts`. |
| `docs/src/guide/typescript-api.md` | New `### Bidirectional scan` subsection under the File API; the `FromUrlOptions` table grows `bidirectional` and `debug` rows with cross-references. |
| `docs/src/guide/remote-access.md` | The existing `## Bidirectional Scan` section gains the TypeScript opt-in alongside Rust and Python so all three bindings are visible in one place. |

**Design decisions:**

- Slim public outcome enums plus rich internal parsers (no replication of validation control flow).  The public `remote_scan_parse` module returns slim outcomes (no `Preamble` / `CachedLayout` payloads); the internal `remote.rs` keeps its existing rich parsers unchanged.  Both call the same `Preamble::read_from()` / `Postamble::read_from()` core, and the internal apply-round step uses the rich `Preamble` directly without re-parsing.  This avoids exposing `Preamble` through the public outcome enums (which would leak many internal flags) while keeping the existing 60+ remote tests byte-output equivalent.  Reason strings are exported as `pub const` from `remote_scan_parse` so a future consolidation can replace literals with constant references without changing semantics.
- Internally-tagged enum serialisation (`#[serde(tag = "kind", rename_all_fields = "camelCase")]`) over externally tagged.  Externally tagged would have produced `{ Hit: { offset, length, msg_end } }`; internally tagged gives `{ kind: "Hit", offset, length, msgEnd }` — a single discriminant field the TS dispatch tests for, and camelCase field names that match the rest of the wasm-side serialisation (e.g. `read_preamble_info`).  `rename_all_fields` (rather than `rename_all`) keeps variant tags PascalCase so the TS `kind === "Hit"` check stays readable.  Pinned by per-variant `wasm-bindgen-test` round-trips so any drift surfaces immediately.
- State-machine dispatcher with explicit forward-only fallback.  Oracle plan-review flagged that a separate `bidirectionalScan` function would have missed the recovery path: when backward yields, the walker must continue forward-only from `state.next` rather than bail-to-eager or restart from offset 0.  The single-loop dispatcher mirrors Rust's `scan_step_locked` byte-for-byte: each iteration inspects `state.bwdActive && !state.fwdTerminated` and routes to one of two helpers; backward-disable transitions are observed by the next iteration without any extra plumbing.
- Paired fetches go through the existing concurrency limiter.  The user's `concurrency` cap applies to the open-time scan in bidirectional mode, not just to post-open fan-out, so `concurrency: 1` would serialise the pair (defeating the purpose).  This makes the synchronous `bidirectional: true && concurrency: 1` rejection meaningful — the validation reflects an actual constraint of the implementation rather than a wishful-thinking guard.
- `fetchRange` gains an optional override-signal parameter rather than a separate API.  Per-iteration child `AbortController`s need to flow through `fetch(init)` without disturbing the post-open path that still honours the user's top-level `signal`.  An override-signal parameter is a one-line API addition with backwards-compatible default behaviour; any other approach (parent-signal mutation, separate `fetchRangeWithSignal` function, signal composition via `AbortSignal.any` on every call) would have been more invasive.
- `addEventListener` cascade with `{ once: true }` AND explicit `removeEventListener`.  Either alone leaks listeners on long scans: `{ once: true }` only fires when the parent aborts (so a never-aborted parent accumulates orphans across many rounds), while explicit `removeEventListener` after the round handles the no-abort case.  Belt-and-braces.
- Bigint discipline at the WASM boundary.  `serde-wasm-bindgen::Serializer::json_compatible()` demotes safe-range u64 to JS `number`; the TS dispatch must use `Number(...)` for arithmetic comparisons (mixed bigint-number `===` is always `false` in JS) and `BigInt(...)` for fields that flow back into WASM-bigint-typed exports (e.g. `validate_backward_preamble_outcome`'s `msg_start` / `length` arguments).  All conversion sites are explicit; no implicit narrowing.
- Forward-only fallback preserves the pre-existing eager-bail behaviour at the file start.  When `tryForwardStep` encounters `Streaming` or `Terminate` AND no layouts have been committed yet, it returns `false` to bail to eager.  Without this, a malformed file (`total_length = 0` or below-minimum length at offset 0) would terminate gracefully with empty layouts instead of letting the eager `END_MAGIC`-walking scanner recover the message.  After at least one layout is committed, `terminateForward` accepts the partial result — matching Rust's `scan_fwd_step_*` behaviour.
- TS bidirectional cases excluded from event-based parity until next sub-task.  The existing `classifier.py` + `RoundBuilder` recognise forward scan events only; backward Range fetches would be mis-classified as out-of-order forward scans and break the strict scan-round invariants.  The new `is_layout_dump_case` helper documents the exclusion; the next sub-task ("parity harness: bidirectional assertions") extends the classifier and adds the cross-language event-parity assertions.

## Remote bidirectional scan — parity harness + eager footer-indexed backward discovery

Lands two interlocking pieces together because they share fixtures
and the same harness extension surface: the parity harness gains
role-aware classification + a bidirectional `RoundBuilder`, and the
bidirectional walker (Rust + TypeScript) folds an eager footer-
region fetch into the same paired round as the candidate-preamble
validation when a backward-discovered message turns out to be
footer-indexed.

| Component | What changed |
|-----------|-------------|
| `tensogram/remote_scan_parse.rs` (public parser) | `BackwardOutcome::NeedPreambleValidation` gains `first_footer_offset: u64` so dispatchers can decide whether to issue an eager footer-region fetch alongside the candidate-preamble validation.  New `footer_region_present(first_footer_offset, length) -> bool` helper uses `checked_sub` so any malformed public input returns `false` rather than panicking.  Pure parser kept cfg-independent. |
| `tensogram/remote.rs` (private parser + dispatcher) | The private duplicate `BackwardOutcome::NeedPreambleValidation` mirrors the public parser's new `first_footer_offset` field in lockstep (both must stay in parity; the parity harness fixtures surface drift as a Rust↔TS layout divergence).  `scan_bidir_round_locked` and `scan_bidir_round_async` issue the candidate-preamble fetch alongside a best-effort footer-region fetch when `footer_region_present` holds; the async path uses `tokio::join!` (NOT `try_join!` — that would short-circuit on footer error and break the best-effort contract).  New `try_populate_eager_footer` helper applies the parsed footer chunk only when the validated preamble's flags carry `FOOTER_METADATA | FOOTER_INDEX`; header-indexed messages with footer hash frames have their speculative footer bytes discarded harmlessly.  `parse_footer_frames` refactored into a thin wrapper over a new pure `parse_footer_frames_into(buf) -> Result<(Option<GlobalMetadata>, Option<IndexFrame>)>` so the eager-footer path can parse footer bytes without touching backend state. |
| `tensogram-wasm/src/remote_scan.rs` | JSDoc on `parse_backward_postamble_outcome` documents the new `firstFooterOffset` bigint field on the `NeedPreambleValidation` JS shape.  `wasm-bindgen-test` `parse_backward_postamble_need_validation_propagates_first_footer_offset` pins the field round-trips through the WASM boundary preserving the bigint shape `serde-wasm-bindgen` demotes to JS `number` for safe-range u64 values. |
| `python/bindings/src/lib.rs` | `PyStreamingEncoder` switches its inner writer from `Vec<u8>` to `std::io::Cursor<Vec<u8>>` so the seek-back-and-patch pattern of `StreamingEncoder::finish_with_backfill` becomes available through PyO3.  New `finish_backfilled() -> bytes` method exposes the back-filled wire format that satisfies the backward-locatability invariant from wire-format §7 (mirrored `total_length` in both preamble and postamble).  Required for the parity harness's footer-indexed fixtures; the existing `finish()` method (streaming-mode `total_length = 0`) still works for forward-only readers. |
| `typescript/src/file.ts` | TS `BackwardOutcome` discriminated-union variant gains `firstFooterOffset: bigint`.  `ScanState.layouts` and `suffixRev` widen from a narrow `{ offset, length, preamble }` shape to the full `MessageLayout` shape (with optional `metadata?` and `index?` fields) so eager-populated data flows through `finaliseScan` into the lazy backend's `b.layouts` map.  `tryBidirectionalRound` issues the candidate-preamble fetch (required) and a sequential best-effort footer-region fetch when `footerRegionPresent` holds; failure on the footer fetch is silently swallowed.  New `tryApplyEagerFooter` populates `layout.metadata` and `layout.index` from the parsed footer chunk via the existing `parse_footer_chunk` WASM export, gated on `FOOTER_METADATA | FOOTER_INDEX` flags.  New `tensogram:scan:footer-eager` debug event emitted with `{ offset, footerBytes }` when eager populate succeeds. |
| `tests/remote-parity/tools/gen_fixtures.py` | New `_encode_footer_indexed` builder using `StreamingEncoder.finish_backfilled` produces fixtures whose preamble + postamble both carry the real `total_length`.  Two new specs: `single-msg-footer.tgm` (1 msg) and `ten-msg-footer.tgm` (10 msgs).  Post-generation `_assert_fixture_well_formed` re-scans the produced file and asserts the index-location flag matches the requested kind; without this guard a silent encoder regression could ship fixtures the bidirectional walker could never discover backward. |
| `tests/remote-parity/classifier.py` | `classify(observation, layout)` extends from a forward-only direction label to a four-valued `ScanRole` (`fwd_preamble`, `bwd_postamble`, `bwd_preamble_validation`, `none`).  Layout-aware role assignment for 24-byte explicit Range fetches: a range starting at a known `message_start` labels as `fwd_preamble`; a range ending at a known `message_end` labels as `bwd_postamble`.  When no layout is supplied, fallback labels every 24-byte fetch `fwd_preamble` (forward-only contract).  `RoundBuilder` gains a `forward_only: bool` knob; bidirectional mode runs a state machine that puts at most one of each role per round and lets either of the paired requests start a round (`Promise.allSettled` and `store.get_ranges` parallelise the pair, so observation order is unspecified). |
| `tests/remote-parity/run_parity.py` | `_FIXTURES` extends to include the two new footer-indexed fixtures.  New `_BIDIR_OPS` set (`message-count`, `read-last`, `read-metadata`) generates forward + bidirectional driver cases for both languages over every fixture.  `_all_cases` de-duplicates via run_id so the existing forward-mode cases for `message-count` / `read-last` are not double-generated.  `load_fixture_layout(fixture)` pre-computes message starts and ends per fixture from `tensogram.scan(fixture_bytes)`; `normalise_log` threads `FixtureLayout` and `forward_only` flag into the classifier. |
| `tests/remote-parity/drivers/{rust_driver,ts_driver}.{rs,ts}` | New `read-metadata` operation calls `decode_metadata(N-1)` (Rust) / `messageMetadata(N-1)` (TS).  Forces metadata-populate on a backward-discovered message in bidirectional mode; on a footer-indexed file this exercises the eager-footer fast path tightly. |
| `tests/remote-parity/test_parity.py` | New `test_eager_footer_one_fetch_per_backward_discovered_message` parametrised over `(footer_fixture, language)`.  On a footer-indexed fixture in bidirectional mode, asserts `footer_region_payloads_count == floor(N/2)` (the number of backward-discovered messages) and zero post-scan lazy fetches.  The 1-message edge case allows up to one footer-region fetch (same-message detection drops the eager-footer bytes silently; lazy populate fires once on accessor call).  Existing tests (`test_no_fallback_or_error_events`, `test_open_divergence_*`, `test_read_first_*`, `test_forward_vs_bidirectional_layouts_equal`) extend automatically to the new fixtures. |
| `tests/remote-parity/schema.json` | `role` field added to `ScanEvent` schema.  `category` description updated to mention eager-footer chunk fetches falling under `payload`.  `scan_round` description extended to cover the bidirectional contract. |
| `tests/remote-parity/test_unit.py` | Schema test updated for the new required `role` field. |
| Tests | Rust: `bidir_eager_footer_populates_metadata_index_on_footer_indexed_file`, `bidir_header_indexed_skips_eager_footer_apply`, `bidir_corrupt_footer_falls_back_to_lazy_without_poisoning_layout` (sync end-to-end via hyper-mock-server).  TS: `header-indexed bidirectional walk issues no footer-region fetches`, `synthetic FOOTER_INDEX flag triggers eager-footer fetch attempt then falls through to lazy`.  Pure parser: `footer_region_present_*` boundary cases (offset at postamble, before preamble end, in valid range, no panic on underflow inputs).  Python: `test_finish_writes_streaming_mode_zero_lengths`, `test_finish_backfilled_writes_real_length_in_both_slots`, `test_finish_backfilled_round_trips_through_decode`, double-finish errors in both directions, two-message concatenation pattern. |

**Design decisions:**

- Eager footer is best-effort, never poisons preamble validation.  A paired multi-range `store.get_ranges([preamble, footer])` would have rejected the entire round on either-side failure.  Splitting into two separate fetches (preamble required, footer optional with `.ok()` / `try/catch`) preserves preamble validation — the footer fetch failure path falls through to the existing lazy `ensure_layout` route.  Cost: at most one extra round-trip per footer-indexed message, paid only when present (`footer_region_present` is `false` for header-indexed messages with no footer frames).  Async dispatcher uses `tokio::join!` (not `try_join!`) so one future failing doesn't abort the other.  TypeScript walker awaits preamble then footer sequentially — symmetric in correctness with the Rust async path, asymmetric in latency: TS pays one extra round-trip per backward-discovered footer-indexed message in exchange for simpler abort-propagation semantics.
- Speculative footer fetch when `first_footer_offset < length - POSTAMBLE_SIZE`.  Header-indexed messages can carry a `FooterHash` frame (no `FooterIndex`), so a non-empty footer region does not guarantee footer-indexing.  We fetch the bytes anyway and gate the parse on `FOOTER_METADATA | FOOTER_INDEX` flags after preamble validation.  Worst-case waste: a few hundred bytes per header-indexed-with-hash message; saved-GET benefit: every footer-indexed message backward-discovered.
- Pure parser carries `first_footer_offset`; predicate is a separate helper.  Putting the predicate inside `BackwardOutcome` would have forced every consumer to pattern-match the variant just to compute the boolean.  A free function (`footer_region_present`) keeps the dispatcher code direct and lets the predicate be tested independently.  Predicate uses `checked_sub` so a malicious or corrupt postamble cannot panic the walker.
- Private duplicate parser kept in lockstep, not deduplicated.  The `Preamble` / `CachedLayout` types in `remote.rs` cannot cross the WASM boundary, so the public `remote_scan_parse` module returns slim outcomes and the private path keeps a richer parser.  Both must propagate `first_footer_offset`.  A future refactor could share more between the two; out of scope for this work.  The parity harness fixtures act as a drift sentinel — any divergence between the two parsers surfaces as a Rust↔TS layout divergence on the footer-indexed fixtures.
- `StreamingEncoder.finish_backfilled` exposed in Python rather than generating fixtures via subprocess.  The streaming encoder needed a seekable writer to satisfy the backward-locatability invariant (mirrored `total_length` in postamble); switching `PyStreamingEncoder`'s inner writer from `Vec<u8>` to `Cursor<Vec<u8>>` was a one-line change that unlocked the existing core API.  The two finish modes (streaming vs backfilled) co-exist so users can choose; documentation makes the wire-format consequence explicit.
- Fixture generator validates the produced bytes.  The post-generation `_assert_fixture_well_formed` re-scans every generated file via `tensogram.scan`, asserts non-zero `total_length`, and checks the FOOTER_INDEX flag for footer-indexed kinds.  Without this guard, a silent encoder regression could ship fixtures the bidirectional walker could never discover backward — the parity harness would still run but the eager-footer code path would never fire, masking real correctness regressions.
- Role-based classifier instead of direction-only.  The classifier emits `fwd_preamble` for ranges at known `message_starts` and `bwd_postamble` for ranges at known `message_ends`.  The `bwd_preamble_validation` role is RESERVED on the schema for future analysis paths with access to dispatcher hooks: at the wire level alone, a candidate-preamble validation fetch is indistinguishable from a paired forward-preamble (both are 24-byte fetches at `[msg_start, msg_start + 24)`).  The `RoundBuilder` therefore does NOT attempt to relabel forward preambles; it groups events with distinct roles into one round and opens a fresh round on role-collision.  Either of the paired forward / backward fetches can arrive first; both share the round.  This matches the harness consumers — every shipping test counts events by `(category, role, range)` or by language-internal structural invariants, none rely on validation-detection.
- Cross-language bidirectional event-parity at the wire level is impossible; layout-equality is sufficient.  Rust's `object_store.get_ranges` coalesces paired forward + backward fetches into a single multi-byte HTTP request when the gap is below its coalesce threshold (every fixture in this harness falls under that threshold), while TypeScript's `Promise.allSettled` issues each Range as a separate request.  The per-round wire shapes therefore differ structurally.  Cross-language correctness is instead asserted via existing `test_forward_vs_bidirectional_layouts_equal` (per-language layout equality between forward and bidirectional modes) plus `test_rust_ts_parity_on_full_scan_ops` (cross-language forward-only event parity, which catches scan-walker drift) plus the new `test_eager_footer_one_fetch_per_backward_discovered_message` (structural per-fixture eager-footer behaviour parametrised over both languages).
- Eager-footer GET-shape test counts derive from the fixture, not from observed events.  Counting `role == "bwd_postamble"` events would undercount on Rust because the coalesced paired fetches are tagged as `payload` (not `scan`).  The fixture-derived `floor(N/2)` count is portable across both languages and matches the structural invariant: every backward-discovered message triggers exactly one eager-footer fetch.  Footer-region filter rejects coalesced paired fetches (size > L) and raw-message fallbacks (size == L) by requiring `PREAMBLE_BYTES < size < message_length`.

## Tensoscope — click-to-inspect

Interactive point inspection added to the web viewer. Clicking anywhere
on the map (flat or globe) snaps to the nearest grid point, fetches the
time series for that point, and displays a chart overlay.

| Component | What changed |
|-----------|-------------|
| `tensoscope/src/components/map/usePointInspection.ts` | New hook: resolves the nearest grid lat/lon from a raw click position, fetches the decoded time series for that location, returns `{ selectedPoint, inspectionResult }`. Normalises longitude to `[-180, 180]` before nearest-point lookup to handle wrap-around at the antimeridian. |
| `tensoscope/src/components/map/PointInspector.tsx` | New component: floating chart overlay positioned at the snapped grid point's screen coordinates. Y-axis ticks format by value range (exponential notation for very large or very small ranges). Auto-closes when the marker leaves the viewport. |
| `tensoscope/src/components/map/MapView.tsx` | Added `onMapClick`, `onSelectedPointScreen`, `onSelectedPointOutOfView`, `selectedPoint`, `selectedPointGridSpacing` props. Flat map (MapLibre) renders the selected point as a GeoJSON circle layer that scales with zoom; circle radius is derived from `selectedPointGridSpacing`. `captureViewport` recomputes marker screen position on move and zoom end. |
| `tensoscope/src/components/map/CesiumView.tsx` | Added corresponding props for the 3-D globe. Selected point is rendered as a Cesium point entity (white outline, transparent fill, size scaled by grid spacing via `markerPixelSize`). Screen position is tracked via `SceneTransforms.worldToWindowCoordinates` on every camera change. Off-screen detection uses CSS-pixel bounds (`clientWidth/clientHeight`) consistent with Cesium's window coordinate space. |
| `tensoscope/src/App.tsx` | Wires `usePointInspection` into the top level. `onMapClick` simultaneously records the raw click point and seeds `inspectedScreenPos` from the click's screen coordinates so the inspector appears immediately before the async fetch returns. |
| `tensoscope/src/App.css` | Map picker buttons gain `flex: 1` and `text-align: center` for equal-width, centred labels. |
| Docs | `docs/src/guide/tensoscope.md` updated with click-to-inspect usage. |

## Tensoscope — high-resolution viewport rendering and display config panel

Re-enables viewport-resolution field rendering on both the flat map and
the globe, restyles the basemap so coastlines and place labels render
on top of the field instead of being obscured by it, and consolidates
user-facing toggles into a single collapsible "Display" panel in the
top-left controls bar.

| Component | What changed |
|-----------|-------------|
| `tensoscope/src/components/map/FieldOverlay.tsx` | Cache stores raw RGBA + grid params instead of dataUrls so a mask-only update re-runs `applyExcludeMask` over cached bytes without invoking the worker. `resolveBounds` signature fixed: takes `(mapProjection, bounds, vw, vh)` instead of the previous `resolveBounds(props)` shape that silently routed geographic renders through the mercator branch. `FieldImage` now carries `width` and `height` so consumers (notably Cesium's `SingleTileImageryProvider`) can pass `tileWidth`/`tileHeight` without re-decoding the dataUrl. Helpers exposed via a `__test` namespace for unit testing without enlarging the public surface. |
| `tensoscope/src/components/map/MapView.tsx` | Mounts back + front layers per view: low-resolution full-globe back at fixed dimensions (`BACK_FLAT_W × BACK_FLAT_H`, `BACK_GEO_W × BACK_GEO_H`) and screen-resolution viewport front buffered by `FRONT_BUFFER_FACTOR = 1.5`. Front is skipped when the buffered span exceeds `FRONT_SKIP_LON_SPAN_DEG = 330`. Both layers render at full opacity to avoid the dimming and seam artefacts that semi-transparent stacking produced. `maxZoom` capped by `computeMaxZoom(nPoints)` so the user cannot zoom past the data's resolution. On map load the basemap style is walked: the field layer is anchored beneath the first line/symbol layer so coastlines, country borders and place labels render on top; water fills are zeroed and a synthetic coastline line layer is added using the basemap's water source; line and symbol paint is forced to white-on-dark-halo for legibility. New "Display" pill (collapsible dropdown) with three toggles — high-res viewport, place labels, borders & coastlines — persisted to localStorage via a `usePersistedState` helper; place labels default to off, the others to on. |
| `tensoscope/src/components/map/CesiumView.tsx` | `Rectangle` entity replaced with `SingleTileImageryProvider` for both back and front field layers so additional imagery can stack on top. Basemap switched from `dark_all` to `dark_nolabels`; a separate `dark_only_labels` imagery layer is added above the field for place names. Borders and coastlines loaded from Natural Earth 50 m via JSDelivr (~500 KB / ~150 KB, cached) as `GeoJsonDataSource`s with `clampToGround: true` so they render above all imagery layers. `showLabels` and `showLines` props toggle the labels imagery layer's `show` and the GeoJSON data sources' `show`; ref-mirrored versions handle the async-load race so toggles set before the GeoJSON parse completes are honoured when the load lands. Ground-atmosphere and sky-atmosphere passes disabled (they washed the field with a white sheen at zoom-out). Click handler wraps `event.position` in `Cartesian2(x, y)` to satisfy `pickEllipsoid`'s typed signature. |
| `tensoscope/src/App.css` | New `.map-config-pill` and `.map-config-panel` classes — dark-glass dropdown matching the existing pill aesthetic, with rows containing checkbox, label and an optional dim hint. |
| `tensoscope/src/components/map/__tests__/FieldOverlay.test.ts` | New test suite covering `applyExcludeMask` (interior alpha=0, boundary rounding via `Math.floor`/`Math.ceil`, out-of-extent clamp), the raw-RGBA cache round-trip, and a regression test locking the `resolveBounds` positional fix so a future revert to `resolveBounds(props)` is caught at test time. |

**Design decisions:**
- Two layers, not one composite canvas. A single client-side composite would cap front resolution at the canvas's dimensions; a dual-layer arrangement preserves the front's screen-resolution gain without paying memory for a globe-sized high-res buffer.
- Both layers at opacity 1.0. Stacked semi-transparent layers produce a `2X − X²` opacity discontinuity at the front rect boundary; opacity 1.0 collapses to single-layer per pixel and removes the seam entirely. Basemap detail is restored as overlays (labels, borders, coastlines) rendered on top, not via see-through transparency.
- Skip the front when `bufferedSpan > 330°`. At that zoom the back is already at or above the front's effective per-pixel density on screen, so the front would only contribute a rasterised-quad seam at its outer edge for no resolution gain.
- Cache raw RGBA (not dataUrl). The previous dataUrl cache forced the worker to re-run any time `excludeBounds` changed, because masking needed access to the underlying pixels. Caching the RGBA decouples mask updates from worker calls; `rgbaToDataUrl` is ~5 ms for 2K×1K and runs every render unconditionally.
- Globe field as imagery, not entity rectangle. Cesium primitives draw after the imagery pass, so an entity-rectangle field would always cover the basemap's labels. Promoting the field to a `SingleTileImageryProvider` puts it back into the imagery stack where additional layers (`dark_only_labels`, plus ground-clamped `GeoJsonDataSource` polylines) can be ordered above it.
- Anchor field below first line/symbol layer (flat). Country borders, coastlines (synthesised), roads and place labels all live in the line+symbol stack of `dark-matter-gl-style`. Anchoring at this point keeps every detail layer above the field while putting all fills (water, landcover, landuse) below it — preserving the "field shows over the basemap fills, basemap detail shows over the field" goal in a single ordering decision.
- `localStorage` for toggle persistence. Same effective scope as cookies, simpler API, no server round-trip; gracefully falls back to in-memory when storage is unavailable (Safari private mode, quota errors). Initial state is read synchronously inside `useState`'s lazy initialiser so the first paint already reflects the persisted value — no flash of defaults.

## Tensoscope — frame-loading spinner and client-side frame cache

Shows a loading spinner in the bottom-left of the map during frame decode and
renders, and prefetches the next N frames during animation via an in-process
LRU cache with generation-based invalidation.

| Component | What changed |
|-----------|-------------|
| `tensoscope/src/tensogram/frameCache.ts` | New `FrameCache` class: LRU cache keyed by `msgIdx:objIdx`, configurable look-ahead (3 for remote, 8 for local), configurable concurrency (2 for remote, 4 for local), generation-based flush, eviction window of `currentIndex ± keepBehind`. `prefetch()` drains a FIFO queue up to `maxConcurrent` in-flight decodes, pruning stale entries via generation check. |
| `tensoscope/src/store/useAppStore.ts` | `frameLoading: boolean` state added. `initFrameCache(isRemote)` called during file open. `getFrameCache()?.flush()` wired into the `selectedLevel` effect so stale cached frames are discarded when the level changes. `loading` moved to a `finally` block for reliable reset. |
| `tensoscope/src/App.tsx` | `decodeFrameForCache()` helper decodes a field slice with the correct `sliceDim`, `selectedLevel` awareness, and shape preservation. `prefetch()` called on frame change in `handleFrameChange`. Loading spinner shown via `frameLoading` state. |
| `tensoscope/src/App.css` | `.map-frame-spinner` class — 18 px spinning circle in the bottom-left, white semi-transparent border, 0.7 s linear infinite animation, `pointer-events: none` so it does not block map interaction. |
