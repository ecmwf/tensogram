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

## Documentation

- [x] ~~Document all error paths in docs/ (error handling reference page)~~ → `docs/src/guide/error-handling.md`

## Builds

- [x] ~~CI matrix~~ → `.github/workflows/ci.yml` — Rust (ubuntu+macos), Python (3.12+3.13, ubuntu+macos), xarray, zarr, C++ (ubuntu+macos), docs. GRIB gated on ecCodes.

- [x] ~~change-to-uv~~ → `uv venv` + `uv pip install` everywhere; CI uses `astral-sh/setup-uv@v5`; legacy `ci.yaml` removed; all docs and CONTRIBUTING.md updated

## Tests and Examples

- [x] ~~consumer-side-streaming~~ → `examples/python/09_streaming_consumer.py` — mock HTTP server, chunked download, progressive scan+decode, xarray Dataset assembly

## Optimisation
 
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

- [ ] **tensogram-validate**:
  - add a `validate` CLI subcommand that checks whether a `.tgm` file is well-formed and intact without using the data. Like `grib_check` or `h5check`.
  - implement 4 validation levels, selectable by flag:
    - **Level 1 — Structure** (`--quick`): verify magic bytes (`TENSOGRM` preamble, `39277777` postamble), frame headers parse with valid types and lengths within file bounds, no overlapping or gap frames, index frame present and points to valid offsets, total message length matches actual data.
    - **Level 2 — Metadata** (included in default): CBOR parses without error and is canonical (deterministic key ordering), required keys present (`_reserved_.tensor.shape`, `dtype`, etc.), dtype is one of the 15 supported types, encoding and compression types are recognized, object count matches number of data object frames, shape/strides/dtype are internally consistent.
    - **Level 3 — Integrity** (included in default): xxh3 hash in hash frame matches recomputed hash over data payloads, each compressed payload decompresses without error (no value interpretation needed).
    - **Level 4 — Fidelity** (`--full`): full decode of each object succeeds, no NaN/Inf in decoded float arrays, for lossy codecs with error-budget metadata: actual error within declared tolerance.
  - CLI interface: `tensogram validate file.tgm` (levels 1-3 by default), `--quick` (level 1 only), `--full` (levels 1-4), `--checksum` (level 3 only), `--json` (machine-parseable output). Batch mode: `tensogram validate data/*.tgm`.
  - output format: `file.tgm: OK (3 messages, 47 objects, hash verified)` or `bad.tgm: FAILED — hash mismatch in message 2 (expected a3f7..., got 91c2...)`. Exit code 0 if all pass, 1 if any fail.
  - most building blocks already exist: `scan()` validates framing, `decode_metadata()` validates CBOR, hash verification exists in decode path, full decode exists. The work is wiring these into a CLI command with precise error reporting (which frame, which byte offset, what's wrong).
  - add a `validate_message(data, options) -> Vec<ValidationIssue>` function in tensogram-core as the library-level API. The CLI subcommand calls this.
  - also as part of this work: convert the 12 panics in `encode.rs` metadata validation to proper error returns. These are the same validation checks that `validate` needs.
  - tests: unit tests for each validation level, test with truncated files, corrupted hashes, invalid CBOR, wrong magic bytes, batch mode, JSON output, and that valid files pass all 4 levels.

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
