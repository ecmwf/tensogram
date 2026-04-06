# Features Decided to Implement

Accepted features that are planned but not yet implemented.
For speculative ideas, see `IDEAS.md`.

## API

- [x] ~~Populate `reserved` metadata field with provenance information~~ → `encode.rs:populate_reserved_provenance()`

## CLI

- [x] ~~`tensogram merge` strategies~~ → `--strategy first|last|error` flag added to CLI merge command

- [ ] `tensogram convert-netcdf`:
  - research about netcdf and CF conventions
  - implement a similar tool to convert-grib to convert netcdf's to tensogram (one-way only)
  - add an option to try to map CF convention

## Metadata

- [ ] *metadata-major-refactor*:
  - currently 'extra' is the serde catch-all for forward/backward compatibility. It ensures any CBOR key in a message survives round-trip even if the decoder doesn't know about it. In practice, most structured metadata currently should go into common (shared) or payload (per-object), with extra used only for ad-hoc annotations or unknown keys.
  - refactor to remove 'common' and 'payload' from metadata objects. all structured will go into a single 'base' map.
  - 'base' becomes an array per data object. all entries are independent.
  - commonalities are computed in software after decoding cbor and only if requested or needed for algorithmic purposes (like merges and splits).
  - namespaces still exist but not hardcoded and used for semantics.
  - only special maps that remain are:
    - '_reserved_' formerly 'reserved', used for internals of the library
    - '_extra_' formerly 'extra', continues to serve as a catch-all
  - make all this explicity and updated in the docs/ and wire format.

## Documentation

- [x] ~~Document all error paths in docs/ (error handling reference page)~~ → `docs/src/guide/error-handling.md`

## Builds

- [x] ~~CI matrix~~ → `.github/workflows/ci.yml` — Rust (ubuntu+macos), Python (3.12+3.13, ubuntu+macos), xarray, zarr, C++ (ubuntu+macos), docs. GRIB gated on ecCodes.

- [ ] *change-to-uv*
  - change to use uv to manage python environements and the dependencies
  - update all the python builds on CI
  - update all the python tests
  - always try first to find if uv is installed in the system and use that

## Tests and Examples

- [x] ~~consumer-side-streaming~~ → `examples/python/09_streaming_consumer.py` — mock HTTP server, chunked download, progressive scan+decode, xarray Dataset assembly

## Optimisation
 
- [x] ~~minimise-mem-alloc~~ → documented in DESIGN.md "Memory Strategy" section. Pipeline uses `Cow` for zero-copy when no encoding/filter/compression. Metadata-only ops never touch payloads. xarray/zarr use lazy loading.

- [ ] *add-benchmarks*:
  - create a benchmarks/ dir where multiple benchmarks for this library will be added
  - the benchmarks should always report against a reference.
  - add a benchmark that compares encoding large runtime auto-generated entries (10M float64 packed to 24 bit) to GRIB (feature gated by eccodes) using grib_ccsds packing and comparing it with the simple_packing (also 24 bit) + szip compression by tensogram. eccodes implementation is the reference.
  - add a benchmark that compares all combinations of encoders + compressors. none+none is the reference. include the speed of compression (ms), decompression (ms), and the rate compression (in % and KiB). Use large runtime auto-generated entries of 16M points starting in float64. Vary also the packing to 16, 24 and 32 bits.

## Code Quality

- [x] ~~code coverage~~ → 86 new Rust tests (376 total). All CLI commands tested (ls 98%, dump 97%, get 97%, convert_grib 99%, output 96%, merge 94%, copy 94%, reshuffle 94%, set 91%, split 89%). Encodings: simple_packing 97%, zfp 92%. Remaining: FFI at 0% (tested by 109 C++ tests). Total 974 tests.
- [x] ~~add logging trace~~ → `tracing` crate instrumented on encode/decode/scan/file/pipeline. Activate with `TENSOGRAM_LOG=debug`
