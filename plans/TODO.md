# Features Decided to Implement

Accepted features that are planned but not yet implemented.
They should contain some notes about the intended aims and direction of implementation.
Code Agents are very much encouraged to ask questions to get the design correct, seeking also  clarifications and sorting out ambiguities.

For speculative ideas, see `IDEAS.md`.

## API

- [x] ~~api-pre-encoded~~ → `encode.rs:encode_pre_encoded()` + bindings (Python, C FFI, C++) + `docs/src/guide/encode-pre-encoded.md` + benchmarks + examples

- [x] ~~Populate `reserved` metadata field with provenance information~~ → `encode.rs:populate_reserved_provenance()`

- [ ] **caller-endianess**:
  - the Docs in byte-order section clarify that payload bytes are returned **verbatim** for `encoding="none"` and callers must interpret/byteswap according to `byte_order`.
  - review the code and plan for the data to ALWAYS be returned to the user in the endianess of the local execution environment so user does not have to byte-swap. the library does it for him.
  - add an option to the API to return the message endianess if the user wants exactly as it was coded internally. Note this should be rare.
  - apply this to all interfaces in all languages
  - document it thouroughly. users should not need to know in what endianess the data was encoded.

## CLI

- [x] ~~`tensogram merge` strategies~~ → `--strategy first|last|error` flag added to CLI merge command

- [x] ~~tensogram-convert-netcdf~~ → v0.7.0. New `crates/tensogram-netcdf/`
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

  - [ ] **wasm-decoder**:
    - aim is to enable the usage of tensogram as the comms message format for a remote web visualiser of scientific data (eg earth science data)
    - research about wasm in particular decoding streams of messages with data
    - limited implementation in wasm of the decode API and decode pipelines
    - support a limited amount of compressors - limit to what is available as packages
    - if no package available for szip, implement it outright  
   
## Metadata

- [x] ~~metadata-major-refactor~~ → v0.6.0. Removed `common`/`payload`, added `base` (per-object independent entries), renamed `reserved` → `_reserved_`, `extra` → `_extra_`. Auto-populated keys under `base[i]._reserved_.tensor`. Added `compute_common()` utility. All docs updated.

## Integration with other software

- [ ] **earthkit-data-integration**:
    - data loader inside earthkit-data
    - research the code of ecmwf/earthkit-data on github, latest verison.
    - develop a loader for earthkit-data to load tensogram data
    - support loading files and streaming 
    - support decoding and encoding
    - add this code to the tensogram-xarray module, such that the earthkit extension can use to export to xarray

## Documentation

- [x] ~~Document all error paths in docs/ (error handling reference page)~~ → `docs/src/guide/error-handling.md`

## Builds

- [x] ~~CI matrix~~ → `.github/workflows/ci.yml` — Rust (ubuntu+macos), Python (3.12+3.13, ubuntu+macos), xarray, zarr, C++ (ubuntu+macos), docs. GRIB gated on ecCodes.

- [x] ~~change-to-uv~~ → `uv venv` + `uv pip install` everywhere; CI uses `astral-sh/setup-uv@v5`; legacy `ci.yaml` removed; all docs and CONTRIBUTING.md updated

## Tests and Examples

- [x] ~~consumer-side-streaming~~ → `examples/python/09_streaming_consumer.py` — mock HTTP server, chunked download, progressive scan+decode, xarray Dataset assembly

## Optimisation

- [ ] **multi-threaded-coding-pipeline**:
  - add multi-threaded support for the encoding and decoding pipelines
  - this should always be an option controled by the caller, off by default, similar to the hash check
  - caller can control how many threads are spawn simultaneously
  - option can be integer such that:
    - 0 means off
    - 1 means spawn a singe thread and execute the pipeline there
    - N means spawn that many and parallelise where possible
  - parallelisation can be done in 2 ways:
    - async coding of data objects simultaneously
    - sync coding of a single data object using multiple threads, where the algorithms are parallelisable.
  - when on, consider the theads as a pool of workers, and the main thread as a broker of the work
 
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

- [ ] **tensogram-validate PR 2** — Level 4 fidelity + encode.rs cleanup:
  - Add `fidelity.rs` module to `validate/` with Level 4 checks.
  - Level 4 (Fidelity, `--full`): full decode of each object succeeds, NaN/Inf in float arrays reported as warnings (not errors — scientific data may legitimately contain them).
  - Note: lossy error-budget verification is NOT feasible from .tgm alone (wire format stores encoded payload, not original values). Accepted scope reduction.
  - NaN/Inf as warnings (not errors) is an accepted spec change — scientific data legitimately contains them.
  - Add `--full` flag to CLI (mutually exclusive with existing mode flags).
  - Convert 12 `panic!()` calls in encode.rs test functions to proper `assert!`/`assert_eq!` with descriptive messages. Note: these are test-only assertions, not runtime validation panics — the original "proper error returns" request is already satisfied by the production code.
  - Add missing test coverage: NaN/Inf policy tests, full-mode decode failures, mixed batch results with exit-code assertions.
  - Add `IssueCode` variants for Level 4: `DecodeObjectFailed`, `NanDetected`, `InfDetected`.

- [ ] **tensogram-validate PR 3** — Python + FFI bindings + examples:
  - Python: `tensogram.validate(buf, level="default") -> dict` and `tensogram.validate_file(path, level="default") -> list[dict]` via PyO3.
  - C FFI: `tgm_validate(buf, len, level, *out) -> tgm_error` and `tgm_validate_file(path, level, *out) -> tgm_error` returning JSON via `TgmBytes` out-parameter (matches existing ABI pattern).
  - Examples in `examples/python/` and `examples/rust/`.
  - Python tests in `tests/python/`.

## Remote Access

- [ ] **remote-object-store**:
  - add the ability to open `.tgm` files on remote object stores (S3, GCS, Azure, HTTP) and read individual objects without downloading the whole file.
  - the wire format already supports efficient remote access:
    - header-indexed files (non-streaming): index is at the beginning, right after the preamble. A single range read gets metadata + index + all object offsets. One more range read fetches the target object. 2 HTTP requests total.
    - footer-indexed files (streaming): the postamble (last 16 bytes) contains `first_footer_offset`. One small range read (last 16 bytes) finds the footer. One more fetches the footer (index + hashes). One more fetches the target object. 2-3 HTTP requests total.
  - what's missing is the transport layer. `TensogramFile` currently only works with local paths via `std::fs::File` + `Seek`.
  - implementation approach:
    - add an async `ReadAt` trait (or use `object_store` crate's `ObjectStore` trait) supporting range reads: `read_range(offset, length) -> Bytes`
    - implement for local files (existing, via seek+read), HTTP/HTTPS (range requests), S3 (via `object_store` or `opendal` crate)
    - add `TensogramFile::open_remote(url)` that auto-detects the protocol
    - the scan/index/decode logic already works with byte offsets — wire it to use range reads instead of seeking a local file
    - expose in Python: `tensogram.open("s3://bucket/file.tgm")`
    - expose in xarray: `xr.open_dataset("s3://bucket/file.tgm", engine="tensogram")`
  - tests: unit tests with mock HTTP server returning range responses, read single object from remote file and verify data matches local decode, latency test measuring request count (should be 2 for header-indexed files), integration test with real S3 bucket (CI-optional), test both header-index and footer-index paths.

## Code Quality

- [x] ~~code coverage~~ → 86 new Rust tests (376 total). All CLI commands tested (ls 98%, dump 97%, get 97%, convert_grib 99%, output 96%, merge 94%, copy 94%, reshuffle 94%, set 91%, split 89%). Encodings: simple_packing 97%, zfp 92%. Remaining: FFI at 0% (tested by 109 C++ tests). Total 974 tests.
- [x] ~~add logging trace~~ → `tracing` crate instrumented on encode/decode/scan/file/pipeline. Activate with `TENSOGRAM_LOG=debug`
