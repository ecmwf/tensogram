# Features Decided to Implement

Accepted features that are planned but not yet implemented.
For speculative ideas, see `IDEAS.md`.

## API

- [x] ~~Populate `reserved` metadata field with provenance information~~ → `encode.rs:populate_reserved_provenance()`

## CLI

- [x] ~~`tensogram merge` strategies~~ → `--strategy first|last|error` flag added to CLI merge command

## Metadata

- [ ] *metadata-major-refactor*:
  - currently 'extra' is the serde catch-all for forward/backward compatibility. It ensures any CBOR key in a message survives round-trip even if the decoder doesn't know about it. In practice, most structured metadata currently should go into common (shared) or payload (per-object), with extra used only for ad-hoc annotations or unknown keys.
  - refactor to remove 'common' and 'payload' from metadata objects. all structured will go into a single 'metadata' map.
  - 'metadata' becomes an array per data object. all entries are independent.
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

## Tests and Examples

- [x] ~~consumer-side-streaming~~ → `examples/python/09_streaming_consumer.py` — mock HTTP server, chunked download, progressive scan+decode, xarray Dataset assembly

## Optimisation
 
- [x] ~~minimise-mem-alloc~~ → documented in DESIGN.md "Memory Strategy" section. Pipeline uses `Cow` for zero-copy when no encoding/filter/compression. Metadata-only ops never touch payloads. xarray/zarr use lazy loading.

## Code Quality

- [~] code coverage: Rust core ~91%, Python 200 tests, C++ 109 tests. Remaining gaps are szip/zfp edge cases and CLI/FFI (tested via C++/integration, not measurable by cargo-llvm-cov). Total 888 tests.
- [x] ~~add logging trace~~ → `tracing` crate instrumented on encode/decode/scan/file/pipeline. Activate with `TENSOGRAM_LOG=debug`
