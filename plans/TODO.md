# Features Decided to Implement

Accepted features that are planned but not yet implemented.
For speculative ideas, see `IDEAS.md`.

## API

- [x] ~~api-pre-encoded~~ → `encode.rs:encode_pre_encoded()` + bindings (Python, C FFI, C++) + `docs/src/guide/encode-pre-encoded.md` + benchmarks + examples

- [x] ~~Populate `reserved` metadata field with provenance information~~ → `encode.rs:populate_reserved_provenance()`

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

## Code Quality

- [x] ~~code coverage~~ → 86 new Rust tests (376 total). All CLI commands tested (ls 98%, dump 97%, get 97%, convert_grib 99%, output 96%, merge 94%, copy 94%, reshuffle 94%, set 91%, split 89%). Encodings: simple_packing 97%, zfp 92%. Remaining: FFI at 0% (tested by 109 C++ tests). Total 974 tests.
- [x] ~~add logging trace~~ → `tracing` crate instrumented on encode/decode/scan/file/pipeline. Activate with `TENSOGRAM_LOG=debug`
