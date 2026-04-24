# Features Decided to Implement

Accepted features that are planned but not yet implemented.
They should contain some notes about the intended aims and direction of implementation.
Code Agents are very much encouraged to ask questions to get the design correct, seeking also  clarifications and sorting out ambiguities.

For speculative ideas, see `IDEAS.md`.

## API

- [x] ~~api-pre-encoded~~ → `encode.rs:encode_pre_encoded()` + bindings (Python, C FFI, C++) + `docs/src/guide/encode-pre-encoded.md` + benchmarks + examples

- [x] ~~Populate `reserved` metadata field with provenance information~~ → `encode.rs:populate_reserved_provenance()`

- [x] ~~caller-endianess~~ (caller endianness) → Decoded data is now always returned in the caller's native byte order by default. `DecodeOptions.native_byte_order` (default `true`) controls this across all interfaces: Rust, Python, C FFI, C++. ZFP/SZ3 lossy codecs made byte-order-aware for uniform pipeline behaviour. Zarr read-path byteswap workaround removed. `Dtype::swap_unit_size()` handles complex types correctly. Documentation updated.

## CLI

- [x] ~~`tensogram merge` strategies~~ → `--strategy first|last|error` flag added to CLI merge command

- [x] ~~tensogram-convert-netcdf~~ → v0.7.0. New `rust/tensogram-netcdf/`
  crate + `tensogram convert-netcdf` CLI (feature-gated behind `netcdf`);
  NetCDF-3 classic + NetCDF-4 (HDF5) inputs; 10 native dtypes; packed
  `scale_factor`/`add_offset` unpacked to f64; `--cf` lifts 16 CF
  allow-list attributes into `base[i]["cf"]`; `--split-by
  file|variable|record`; shared `PipelineArgs` (`--encoding/--bits/
  --filter/--compression/--compression-level`) retrofitted onto BOTH
  `convert-grib` and `convert-netcdf`; docs, Python + Rust examples,
  Python e2e tests; Ubuntu+macOS CI via new `netcdf` job; `grib` CI
  extended to macOS for symmetry.

## Multi-Language Support

  - [x] ~~**wasm-decoder**~~ → v0.8.0. `rust/tensogram-wasm/` — full decode,
    encode, scan, and streaming API via `wasm-bindgen`. Compressors: lz4,
    szip (pure-Rust `tensogram-szip` crate, CCSDS 121.0-B-3), zstd (pure-Rust
    `ruzstd`). 134 `wasm-bindgen-test` tests. `wasm-pack build --target web`.
    StreamingDecoder with buffer limit, error visibility, and reset.
    Zero-copy TypedArray views for f32/f64/i32/u8 with zero-length safety.

  - [x] ~~**typescript-wrapper (Scope B)**~~ — see `DONE.md` and
    `docs/src/guide/typescript-api.md`. New `typescript/` package,
    typed encode / decode / scan, dtype dispatch, metadata helpers,
    streaming (`decodeStream`), `TensogramFile.open/fromUrl/fromBytes`,
    87 tests, 6 examples, CI job, mdBook page, Makefile targets.

  - [x] ~~**typescript-wrapper (Scope C.1) — API-surface parity**~~
    Closed the gap between the TS wrapper and the other language
    surfaces.  New WASM exports (`decode_range`, `compute_hash`,
    `simple_packing_compute_params`, `encode_pre_encoded`,
    `validate_buffer`, `StreamingEncoder`) in `rust/tensogram-wasm`
    backed by new TS wrappers (`decodeRange`, `computeHash`,
    `simplePackingComputeParams`, `encodePreEncoded`, `validate` /
    `validateBuffer` / `validateFile`, `StreamingEncoder`).
    `TensogramFile#append(meta, objects, opts?)` gained for files
    opened via `TensogramFile.open(path)` (Node local paths only —
    matches the Rust / Python / FFI / C++ contract).
    `TensogramFile.fromUrl` now auto-detects HTTP Range support via a
    `HEAD` probe and switches to a lazy backend that fetches messages
    on demand; falls back transparently to the eager Scope-B download
    when the server omits `Accept-Ranges` or returns a streaming-mode
    message.  `TensogramFile#rawMessage` is now async (was sync) to
    accommodate the lazy backend.  Five new examples
    (`04_decode_range`, `08_validate`, `11_encode_pre_encoded`,
    `12_streaming_encoder`, `13_range_access`), per-module test suites,
    and the TS API guide / parity matrix updated.  See
    `plans/DONE.md` → *TypeScript Scope C.1* for the full breakdown.

  - [x] ~~**typescript-wrapper (Scope C.2) — first-class half-precision + complex dtypes**~~
    `typedArrayFor(dtype)` now returns `Float16Array` (native) or
    `Float16Polyfill` for `float16`, `Bfloat16Array` for `bfloat16`,
    and `ComplexArray` for `complex64` / `complex128`.  The polyfill
    matches the TC39 Stage-3 `Float16Array` observable behaviour
    (round-ties-to-even narrow, NaN / ±Inf / subnormal / ±0
    preservation).  `ComplexArray` exposes numpy-flavoured accessors
    (`.real(i)`, `.imag(i)`, `.get(i)`, iteration).  Raw bits / raw
    interleaved storage are still reachable through `.bits` / `.data`.
    fast-check round-trip property tests cover every new dtype;
    `encode → decode` keeps the bytes bit-exact.  Breaking vs Scope B:
    `obj.data()` for these four dtypes no longer returns a raw
    `Uint16Array` / interleaved `Float32Array` — consumers wanting the
    raw shape use `.bits` / `.data` on the returned view.

  - [ ] **typescript-wrapper (Scope C.3) — distribution & CI maturity**
    Three intertwined tasks that all touch the build, pack, and publish
    pipeline. Best done together so we don't re-open CI config three
    times.
    - **npm publish pipeline.** Choose an npm org (`@ecmwf.int/tensogram`
      already exists in `package.json`), wire a GitHub Actions job that
      publishes on tagged releases and enforces semver lock-step with
      the root `VERSION` file (add `typescript/package.json` to the
      VERSION-sync list in CLAUDE.md — already noted but not wired).
    - **Browser-environment CI.** Today the `typescript` CI job runs
      Vitest in Node only. Add a browser lane (Vitest `browser` mode
      via Playwright, or a dedicated Playwright job) so regressions
      that only surface in a real browser (`import.meta.url` resolution,
      streams API, `fetch` semantics) can't slip through.
    - **Bundle-size budget via `size-limit`.** Track the produced
      `dist/*.js` + `wasm/tensogram_wasm_bg.wasm` size deltas on every
      PR. Fail CI if the growth exceeds a documented threshold
      (e.g. WASM ≤ 1.5 MiB, JS glue ≤ 30 KiB).
    - **Micro-benchmarks on hot paths.** Add `vitest bench` runs for
      `encode`, `decode`, and `decodeStream` so allocation regressions
      can be spotted without running full load-test scenarios.

  - [ ] **typescript-wrapper (Scope C.4) — Zarr.js integration**
    JavaScript mirror of the `tensogram-zarr` package. Lets downstream
    JS tooling (browser-based data explorers, notebook viewers)
    consume Tensogram-backed Zarr stores without re-implementing the
    chunk-store contract. Ships as a separate `@ecmwf.int/tensogram-zarr`
    npm package that depends on `@ecmwf.int/tensogram`.
    - Implement the Zarr v3 chunk-store trait against a `TensogramFile`
      backend.
    - 14 bidirectional dtype mappings to match the Python
      `tensogram-zarr` test matrix.
    - Smoke-test from a notebook-style fetch scenario
      (`zarr-js` reading a `.tgm`-backed store over HTTPS).
    - Standalone CI job; does not block the main `@ecmwf.int/tensogram`
      pipeline.

## Metadata

- [x] ~~metadata-major-refactor~~ → v0.6.0. Removed `common`/`payload`, added `base` (per-object independent entries), renamed `reserved` → `_reserved_`, `extra` → `_extra_`. Auto-populated keys under `base[i]._reserved_.tensor`. Added `compute_common()` utility. All docs updated.

## Integration with other software

- [ ] **earthkit-data-integration**:
    - data loader inside earthkit-data
    - research the code of ecmwf/earthkit-data on github, latest verison.
    - develop a loader for earthkit-data to load tensogram data
    - support loading files and streaming 
    - support decoding and encoding
    - support integration with Python array interface as Earthkit data does (see also earthkit-utils)
    - add this code to the tensogram-xarray module, such that the earthkit extension can use to export to xarray
- [ ] **torch**
    - convenience methods for tensogram as/from torch, to avoid the numpy intermediary. Wilder ideas and optimizations are additionally given in IDEAS.md
- [ ] **nvidia stack**
    - cuFile or similar interface
- [ ] **arrow/parquet**
    - analyze where we can integrate with arrow/parquet stack and implement: file reading, streaming data conversion (in both directions -- offering tensogram via arrow streaming api, as well as converting arrow streams into tensogram messages)
- [ ] **mlx**
    - similarly to torch, conveninece methods for tensogram as/from mlx frame, to avoid the numpy intermediary


## Documentation

- [x] ~~Document all error paths in docs/ (error handling reference page)~~ → `docs/src/guide/error-handling.md`

- [ ] interactive-docs:
  - make the docs interactive. 
  - since we have WASM, you can embed demos in the docs

- [x] ~~jupyter-notebooks~~ → `examples/jupyter/` with five narrative
  notebooks (quickstart/MARS, encoding fidelity, GRIB, NetCDF+xarray,
  validation+threads). CI-tested via `pytest --nbval-lax`. See
  `docs/src/guide/jupyter-notebooks.md`.

## Builds

- [x] ~~**restructure-repo**~~:
  - moved code to sub-folders with languages as names
  - rust code in rust/
  - python code in python/
  - cpp code in cpp/
  - examples/<lang> stays separate

- [x] ~~CI matrix~~ → `.github/workflows/ci.yml` — Rust (ubuntu+macos), Python (3.12+3.13, ubuntu+macos), xarray, zarr, C++ (ubuntu+macos), docs. GRIB gated on ecCodes.

- [x] ~~change-to-uv~~ → `uv venv` + `uv pip install` everywhere; CI uses `astral-sh/setup-uv@v5`; legacy `ci.yaml` removed; all docs and CONTRIBUTING.md updated

## Tests and Examples

- [x] ~~consumer-side-streaming~~ → `examples/python/09_streaming_consumer.py` — mock HTTP server, chunked download, progressive scan+decode, xarray Dataset assembly

## Optimisation

- [x] ~~**multi-threaded-coding-pipeline**~~ → v0.13.0.  Caller-controlled
  `threads: u32` option on `EncodeOptions`/`DecodeOptions` (off by
  default).  `TENSOGRAM_THREADS` env fallback.  Axis-B-first dispatch
  (intra-codec parallelism for blosc2 `nthreads`, zstd `NbWorkers`,
  `simple_packing` chunked, `shuffle`/`unshuffle` chunked) with axis A
  (`par_iter` across objects) as fallback when no codec is axis-B
  friendly — avoids N×M thread over-subscription.  Threaded through
  Python (kwarg), C FFI (new `uint32_t` parameter), C++ wrapper
  (`options.threads` field), and the CLI (global `--threads N`).
  Determinism: transparent codecs byte-identical across thread counts,
  opaque codecs lossless round-trip.  New `threads-scaling` benchmark,
  new docs page, runnable Rust+Python examples, ~20 new determinism
  tests.  See `DONE.md` for the full breakdown and
  `docs/src/guide/multi-threaded-pipeline.md` for the API reference.

- [x] ~~hash-while-encoding~~ → inline xxh3-64 hashing fused into the
  encoding pipeline. Eliminates a second pass over the encoded payload
  for transparent codecs; passthrough workloads see ~11% faster encode.
  Output bytes, descriptor contents, and wire format unchanged — every
  `.tgm` byte-identical to pre-change output. See `DONE.md` →
  *Hash-while-encoding*.

- [x] ~~minimise-mem-alloc~~ → documented in DESIGN.md "Memory Strategy" section. Pipeline uses `Cow` for zero-copy when no encoding/filter/compression. Metadata-only ops never touch payloads. xarray/zarr use lazy loading.

- [x] ~~add-benchmarks~~:
  - think about how we could have a series of benchmarks in the repo that could be used to iterate development and reliably improve the performance of the software library.
  - make proposals of how this could be achieved. iterate with the user ideas.
  - create a benchmarks/ dir where multiple benchmarks for this library will be added
  - the benchmarks should always report against a reference.
  - add a benchmark that compares encoding large runtime auto-generated entries (10M float64 packed to 24 bit) to GRIB (feature gated by eccodes) using grid_ccsds packing and comparing it with the simple_packing (also 24 bit) + szip compression by tensogram. eccodes implementation is the reference.
  - add a benchmark that compares all combinations of encoders + compressors. none+none is the reference. include the speed of compression (ms), decompression (ms), and the rate compression (in % and KiB). Use large runtime auto-generated entries of 16M points starting in float64. Vary also the packing to 16, 24 and 32 bits.

- [x] ~~*record-benchmark-results*~~ → `docs/src/guide/benchmark-results.md` — ran both benchmarks (codec-matrix 24 combos, grib-comparison 3 methods) on Apple M4 / macOS 26.3.1 / Rust 1.94.1 / ecCodes 2.46.0; results page added to docs with date, version, machine metadata; `mdbook build` passes.

## Validation

- [x] **tensogram-validate PR 1** — core library API + CLI (Levels 1-3):
  - `validate_message(buf, options) -> ValidationReport` and `validate_file(path, options)` in tensogram-core.
  - Level 1 (Structure): raw byte walking — magic, preamble, frame headers/ENDF, total_length, postamble, first_footer_offset, frame ordering, preceder legality, preamble flags vs observed, overflow-safe arithmetic.
  - Level 2 (Metadata): raw CBOR parsing from frame payloads (before decode_message normalization), required keys, dtype/encoding/filter/compression recognized, shape/strides/ndim consistency, index/hash frame consistency.
  - Level 3 (Integrity): xxh3 hash verification (descriptor + hash frame fallback), decode pipeline execution for compressed objects. Unknown hash algorithms produce warnings. `hash_verified` only true when ALL objects verified.
  - CLI: `tensogram validate [--quick|--checksum|--canonical] [--json] <files>`, mutually exclusive modes, batch JSON array output, exit code 0/1.
  - File-level validation: detects garbage bytes, truncated messages, trailing data between messages. Streaming message scan validates postamble candidates.
  - Canonical CBOR check opt-in via `--canonical` flag (warnings, not errors).
  - Modular architecture: `validate/types.rs` (types + `IssueCode` enum), `validate/structure.rs`, `validate/metadata.rs`, `validate/integrity.rs`, `validate/mod.rs` (public API).
  - Stable machine-readable `IssueCode` enum with serde serialization. CLI JSON output via serde_json (not hand-built).
  - Streaming file validation via `scan_file` + per-message reads (O(1 message) memory).
  - Index offset validation: verifies offsets point to actual data object frame positions.
  - 35 tests (25 core + 10 CLI). Docs at `docs/src/cli/validate.md`.

- [x] **tensogram-validate PR 2** — Level 4 fidelity + API refactor:
  - Refactored `ValidateMode` enum into composable `ValidateOptions { max_level, check_canonical, checksum_only }`. `--canonical` now combinable with `--full`/`--quick`.
  - Added `ObjectContext` for shared per-object state: Level 2 caches descriptors, Level 3 caches decoded bytes, Level 4 reuses both. No duplicate parsing or decoding.
  - Level 4 (Fidelity, `--full`): full decode, decoded-size check, NaN/Inf scan for Float16/Bfloat16/Float32/Float64/Complex64/Complex128. NaN/Inf are errors. Reports element index and component (real/imag).
  - Dropped encode.rs panic cleanup (test-only assertions, not meaningful to convert).

- [x] ~~**tensogram-validate PR 3**~~ — Python + FFI bindings + examples:
  - Python: `tensogram.validate(buf, level="default") -> dict` and `tensogram.validate_file(path, level="default") -> dict` via PyO3.
  - C FFI: `tgm_validate(buf, len, level, check_canonical, *out) -> tgm_error` and `tgm_validate_file(path, level, check_canonical, *out) -> tgm_error` returning JSON via `TgmBytes` out-parameter.
  - C++ wrapper: `tensogram::validate()` and `tensogram::validate_file()` returning JSON strings.
  - Examples in `examples/python/13_validate.py` and `examples/rust/src/bin/13_validate.rs`.
  - 34 Python tests in `python/tests/test_validate.py`. 12 FFI unit tests. 11 C++ GoogleTest tests.

## Remote Access

- [x] **remote 1 — Rust core (header-indexed)**:
  - `remote` feature gate with `object_store` 0.13, `Backend` enum (Local | Remote), `remote.rs` module
  - APIs: `open_source()`, `open_remote()`, `decode_metadata()`, `decode_descriptors()`, `decode_object()`, `is_remote()`, `source()`
  - schemes: `s3://`, `s3a://`, `gs://`, `az://`, `azure://`, `http://`, `https://`
  - sync bridge: `std::thread::scope` + per-call tokio runtime
  - 21 Rust tests with mock HTTP server; docs: `docs/src/guide/remote-access.md`
- [x] **remote 2 — footer-indexed (streaming) support**:
  - fixed `StreamingEncoder` index lengths (payload → frame), `scan_messages` handles `total_length=0`
  - `discover_footer_layout` + `parse_footer_frames`, streaming must be last in multi-message files
  - 5 new tests: streaming open/decode, local parity, multi-object, mixed, index lengths
- [x] **remote 3 — Python + xarray + zarr integration**:
  - Python `open()` auto-detects remote, `open_remote(source, storage_options)`, file-level decode APIs
  - GIL released during all I/O via `py.allow_threads()`
  - xarray: `storage_options` threaded through all 5 modules, remote reads use file-level APIs
  - zarr: `storage_options`, remote writes rejected early
  - 12 Python tests with mock HTTP server
- [x] **remote 4 — async + shared runtime**:
  - shared `OnceLock<Runtime>` replaces per-call `block_on_thread` (thread + runtime per I/O call)
  - `block_on_shared`: direct `handle.block_on()` when not in async context, scoped thread fallback when in async context
  - native async methods when both `remote` and `async` features enabled: `open_source_async`, `open_remote_async`, `decode_metadata_async`, `decode_descriptors_async`, `decode_object_async`, `read_message_async` (remote-aware)
  - descriptor-only reads: `read_descriptor_only` fetches only CBOR prefix for large frames (> 64 KB), full frame for small ones
  - `rt-multi-thread` tokio feature gated to `remote` only
  - 7 new tests (concurrent reads, sync context, descriptor parity, async open/decode/descriptors/streaming/parity/errors)
- [x] **remote 5 — polish (examples, CI, zarr lazy reads)**:
  - Python example `14_remote_access.py`: self-contained HTTP server, open_remote, file-level decode APIs
  - Rust example `14_remote_access.rs`: TcpListener-based HTTP server, open_source, decode APIs
  - CI: added `pytest python/tests/test_remote.py` to Python CI job
  - zarr lazy reads: remote files use `file_decode_descriptors()` at scan time (metadata-only), `file_decode_object()` per chunk on demand; local files unchanged (eager decode)
  - 9 new zarr remote tests: lazy open, on-demand decode, close cleanup, cached repeat access, exists on lazy chunk, list includes lazy chunks, no duplicates after cache, exception cleanup, local-still-eager parity
- [x] **remote 6 — range reads**:
  - extracted `decode_range_from_payload` from `decode_range` (takes descriptor + payload, no message parsing)
  - `RemoteBackend::read_range`: fetches object frame via index, extracts payload, runs range decode pipeline
  - `TensogramFile::decode_range(msg_idx, obj_idx, ranges, options)`: dispatches to remote or local
  - Python `file_decode_range(msg_index, obj_index, ranges, join, verify_hash)`: file-level range decode binding
  - xarray `array.py`: uses `file_decode_range` for both local and remote (replaces buffer-level `decode_range` and remote `file_decode_object` fallback)
  - 3 Rust tests: single range, remote-vs-local parity, out-of-range error
- [ ] **remote 7 — TS lazy scan: 256 KB forward-chunk variant**:
  - during `lazyScanMessages`, fetch one 256 KB chunk per message instead of 24 bytes
  - for header-indexed messages, parse metadata + index inline via `parse_header_chunk`
    (saves the second round trip `#ensureLayout` would otherwise do)
  - gated by a benchmark that shows the round-trip saving outweighs the larger
    per-message fetches on realistic server latencies (RTT ≥ 20 ms)
  - requires the current preamble-only walk to remain as the fallback when any
    chunk-parse fails (bail-to-eager path should stay identical)

## Python Async Bindings

- [x] **Python asyncio support**:
  - `AsyncTensogramFile` class via `pyo3-async-runtimes` + tokio bridge
  - all core async methods changed from `&mut self` to `&self` (no mutex needed)
  - `file_decode_range_batch` and `file_decode_object_batch` for batched HTTP via `object_store::get_ranges`
  - `prefetch_layouts` for layout warmup
  - `asyncio.gather` achieves real I/O overlap on a single handle
  - async context manager, async iteration, 20 methods total
  - 73 tests, example `15_async_operations.py`

## Free-Threaded Python

- [x] **free-threaded Python (parallelism)** *(merged in v0.10.0)*:
  - `#[pymodule(gil_used = false)]` declared
  - `GILOnceCell` replaced with `std::sync::OnceLock`
  - CI tests with Python 3.13t / 3.14t
  - enables true parallel decode/encode across threads

## Code Quality

- [x] ~~code coverage~~ → All CLI subcommands have dedicated tests (ls, dump, get, set, copy, merge, split, reshuffle, validate, convert-grib, convert-netcdf). Encodings: `simple_packing` and `zfp` covered. FFI exercised through the C++ wrapper test suite.
- [x] ~~add logging trace~~ → `tracing` crate instrumented on encode/decode/scan/file/pipeline. Activate with `TENSOGRAM_LOG=debug`

- [x] ~~**cross-codec `expected_size` preallocation hardening (szip)**~~ →
  Fallible allocation across both szip backends (FFI `aec_decompress` +
  `aec_decompress_range` in `libaec.rs`; pure-Rust `decode`, `decode_rsi`,
  `write_samples` in `tensogram-szip`). `checked_mul` added to
  `samples_per_rsi` and `write_samples` sample-count → byte-count math.
  `checked_sub` added to the `decoded_len = expected_size - avail_out`
  arithmetic that feeds `set_len`. Hostile `num_values` now surfaces as
  `CompressionError::Szip` / `AecError::Data` with a `"failed to
  reserve"` prefix instead of aborting the process.
  Static cap deliberately NOT introduced — rejected because legitimate
  multi-GiB scientific workloads make any usable cap too permissive to
  block attackers. See `DONE.md` → *Preallocation hardening for szip
  decode paths* for the full rationale and soundness notes on the FFI
  `set_len` pattern. Closes the szip half of issue #68 / PR #69.

- [ ] **preallocation hardening — `simple_packing` decode paths**:
    - `simple_packing::decode_aligned`, `decode_generic`, `decode_range`,
      and the parallel variants in
      `rust/tensogram-encodings/src/simple_packing.rs` all call
      `Vec::with_capacity(num_values)` on descriptor-derived
      `num_values` for the output `Vec<f64>`, giving 8× amplification.
    - Reachable without any compression codec in play
      (`CompressionType::None + EncodingType::SimplePacking`), so a
      malformed `.tgm` can still abort the process through this path
      after the szip hardening landed.
    - Fix shape mirrors the szip fix: switch `Vec::with_capacity` to
      `Vec::new()` + `try_reserve_exact` and surface allocation
      failures as `PackingError::AllocationFailed` (new variant).

- [ ] **preallocation hardening — bitmask decode paths**:
    - `bitmask::packing::unpack`, `bitmask::rle::decode`,
      `bitmask::roaring::decode` each call `Vec::with_capacity(n_elements)`
      (or `vec![false; n_elements]`) on descriptor-derived `n_elements`.
    - RLE is especially cheap to attack: a tiny compressed blob can
      claim a huge run count.
    - Fix: same `try_reserve_exact` pattern; new `MaskError::AllocationFailed`
      variant.

- [ ] **preallocation hardening — `zfp_decompress_f64`**:
    - `rust/tensogram-encodings/src/zfp_ffi.rs:82` —
      `vec![0.0f64; num_values]` on `num_values` sourced from
      `ZfpCompressor` which reads from the descriptor. 8× amplification,
      eager zero-fill forces commit under Linux overcommit.
    - Fix: `Vec::try_reserve_exact(num_values)` + avoid the eager
      zero-fill (libzfp writes every f64 it produces into the output
      slice passed via `zfp_field_1d`).

- [ ] **FFI-vs-pure szip parameter validation asymmetry**:
    - `rust/tensogram-szip/src/lib.rs` calls `params::validate()` on
      every entry point.
    - `rust/tensogram-encodings/src/libaec.rs` (FFI path) does not
      apply the same pre-validation; it relies on libaec returning
      non-zero status codes.
    - Add a shared validation step (or lift `params::validate` into
      `tensogram-encodings` as a small helper) so the two backends
      reject malformed AEC parameters consistently before touching the
      FFI boundary.

## Viewer

- [ ] Loading spinner/skeleton on map while field is being regridded
- [ ] Wire LevelSelector into the UI for 3D pressure-level fields
- [ ] Cache rendered frames client-side for instant scrubbing through previously viewed steps
- [ ] OffscreenCanvas in worker to avoid main-thread canvas.toDataURL
- [ ] Cache decoded Float32Arrays to skip WASM decode when revisiting fields
- [ ] Pre-fetch next N frames during animation playback
- [ ] URL state persistence (selected file, field, colour scale)
- [ ] Keyboard shortcuts: space play/pause, arrow keys step
- [ ] Resizable sidebar (drag handle)
- [ ] Handle polar stereographic projections and single-point fields
