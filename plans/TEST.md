# Test Plan

Repo: ecmwf/tensogram

> This document describes the *shape* of the test suite — what is
> tested, where, and why. It deliberately avoids concrete test counts
> and version numbers because both drift with every release. For
> release-by-release changes, see `../CHANGELOG.md`. For the
> implementation path followed, see `DONE.md`.

## Coverage shape

| Component | Where tests live | Shape |
|-----------|------------------|-------|
| `tensogram-core` | `src/*.rs` unit + `tests/{integration,adversarial,edge_cases,golden_files,integration_pre_encoded,cross_language_pre_encoded,remote_http}.rs` | Unit + integration + adversarial + edge case + golden-file + remote HTTP |
| `tensogram-encodings` | `src/*.rs` unit | Unit — packing, shuffle, pipeline dispatch |
| `tensogram-szip` | `src/*.rs` unit + `tests/{libaec_parity,proptest_roundtrip,stress,error_paths,ffi_crosscheck}.rs` | Unit + property-based + stress + FFI cross-validation against libaec |
| `tensogram-sz3` | `src/*.rs` unit | Unit for the SZ3 Rust API over the clean-room shim |
| `tensogram-sz3-sys` | (no crate-level tests; exercised via `tensogram-sz3` and `tensogram-encodings`) | FFI shim |
| `tensogram-wasm` | `wasm-bindgen-test` in-crate | Browser / Node.js decode paths |
| `tensogram-cli` | `src/**/*.rs` unit (per-subcommand) | CLI argument parsing + subcommand behaviour, incl. `--features netcdf,grib` for converter coverage |
| `tensogram-ffi` | `src/lib.rs` unit (indirect via C++ wrapper) | FFI handle behaviour, null-pointer safety, error-code mapping |
| C++ wrapper | `cpp/tests/*.cpp` (GoogleTest) | RAII handle behaviour, typed exception mapping, cross-language round-trip |
| `tensogram-python` | `python/tests/test_*.py` (pytest) | PyO3 bindings, NumPy round-trip, async, validation, remote, free-threaded |
| `tensogram-grib` | `tests/integration.rs` | Real ECMWF opendata GRIB fixtures |
| `tensogram-netcdf` | `tests/integration.rs` (+ Python e2e via `subprocess` in `python/tests/test_convert_netcdf.py`) | NetCDF-3 + NetCDF-4 round-trips, split modes, CF lifting |
| `tensogram-xarray` | `python/tensogram-xarray/tests/*.py` | Backend engine, coordinate detection, hypercube stacking, remote |
| `tensogram-zarr` | `python/tensogram-zarr/tests/*.py` | Zarr v3 store read + write, mapping, edge cases, remote |
| `tensogram-benchmarks` | `tests/smoke.rs` + per-binary unit | Smoke coverage so benchmarks do not silently rot |

To see current counts, run `cargo test --workspace`, `cargo test` in
each opt-in crate directory, and `pytest` / `ctest` for the other
languages — these numbers are the source of truth.

## Affected components

- **`tensogram-core`**: encode, decode, `decode_metadata`,
  `decode_descriptors`, `decode_object`, `decode_range`, scan,
  streaming, file, iterators, validation (Levels 1-4), remote access.
- **`tensogram-encodings`**: `simple_packing`, shuffle filter,
  compression codecs (szip, zstd, lz4, blosc2, zfp, sz3), pipeline
  dispatch; `szip-pure` and `zstd-pure` feature variants.
- **`tensogram-szip`**: pure-Rust AEC/SZIP encode/decode/range-decode,
  signed preprocessing, FFI cross-validation.
- **`tensogram-wasm`**: decode, `decode_metadata`, `decode_object`,
  scan, encode, `StreamingDecoder`, `DecodedFrame`, zero-copy
  TypedArray views.
- **`tensogram-ffi`**: C API opaque handles, typed getters, error
  codes, iterators, streaming encoder, `tgm_validate`,
  `tgm_validate_file`.
- **`tensogram-python`**: PyO3 bindings, NumPy round-trip,
  `TensogramFile`, `AsyncTensogramFile`, metadata access,
  `tensogram.validate`, free-threaded operation.
- **`tensogram-cli`**: `info`, `ls`, `dump`, `get`, `set`, `copy`,
  `merge`, `split`, `reshuffle`, `validate`, `convert-grib` (opt-in),
  `convert-netcdf` (opt-in).
- **`tensogram-xarray`**: xarray backend engine, lazy loading,
  coordinate detection, producer `dim_names` hints, hypercube merging,
  remote.
- **`tensogram-zarr`**: read path, write path, mapping layer, remote
  lazy reads, byte-range requests.

## Key interactions to verify

- Round-trip encode → decode for every dtype (float16/32/64, bfloat16,
  complex64/128, int8-64, uint8-64, bitmask).
- Round-trip for every encoding × filter × compression combo.
- Cross-language byte-identical output (Rust, Python, C++) via golden
  files.
- Partial range decode via szip block offsets, blosc2 chunks, zfp
  fixed-rate blocks.
- CLI subcommands with `-w` filtering, `-p` key selection, `-j` JSON
  output.
- Frame-based index: header/footer index frames with correct offsets;
  O(1) random access.
- Streaming mode: `total_length=0`, footer-based index, Preceder
  Metadata Frames.
- Native byte-order decode on / off on all decode paths.
- mmap-based file access: zero-copy scan + read.
- Async encode/decode paths (feature-gated): `spawn_blocking` for FFI
  calls; `AsyncTensogramFile` concurrency via `Arc<TensogramFile>`.
- Remote access via `object_store`: S3/GCS/Azure/HTTP; batched range
  reads; footer-indexed streaming files; remote-vs-local parity.
- GRIB conversion: MARS key extraction, per-object independent
  `base[i]` entries, grouping modes, shared pipeline flags.
- NetCDF conversion: NetCDF-3 + NetCDF-4 inputs, split modes, CF
  lifting, shared pipeline flags.
- xarray backend: lazy loading, N-D slicing, coordinate detection,
  multi-message merge, producer `dim_names` hints.
- Zarr backend: store read/write, byte-range requests, remote lazy
  reads.
- Validation Levels 1-4: structure, metadata, integrity, fidelity,
  canonical CBOR, NaN/Inf scan.

## Edge cases

- NaN input to `simple_packing` → must reject with `EncodingError`.
- Zero-object message (metadata-only).
- Streaming mode (`total_length=0`, footer-based metadata and index).
- Payload > 4 GiB (uint64 offsets in index frames).
- Non-byte-aligned bit packing (12-bit, 24-bit, 1-bit bitmask).
- Corrupted message mid-file → scan recovers to next valid `TENSOGRM`
  marker.
- `tensogram set` with immutable key (shape, strides, dtype, encoding,
  hash) → must reject.
- Empty tensor (ndim=0, scalar value).
- Frame marker corruption (FR/ENDF) in one data object → that object
  rejected, others accessible.
- Inter-frame padding for 64-bit alignment.
- Shuffle + partial range decode → rejected (documented as unsupported
  combination).
- Ciborium canonical encoding: same metadata map serialises to
  byte-identical CBOR on every call.
- Big-endian byte order round-trips.
- Wire-format determinism (idempotent encode→decode→encode).
- Preceder + footer metadata merge (preceder wins on conflict).
- Remote ranges that straddle object boundaries.
- Free-threaded Python: concurrent encode/decode on multiple OS
  threads without data races.

## Critical paths

- Message encode → network transmit → decode must be bit-exact
  (lossless) or within tolerance (lossy).
- File append → scan → random access by message index → decode object
  by index.
- Streaming encode → footer index → random access to data objects.
- mmap → seek to message → decode partial range via block offsets.
- Golden binary `.tgm` files decoded identically by Rust, Python, and
  C++ on every CI run.
- `tensogram copy -w mars.param=2t input.tgm output.tgm` → only
  matching messages, byte-identical.
- xarray `open_dataset()` → lazy load → partial decode on slice
  access.
- Remote `open_remote(url)` → descriptor-only prefetch → range decode
  per chunk.
- `tensogram validate --full` → all four levels pass on canonical
  fixtures; `--checksum` isolates hash-only failures.

## Golden test files

Canonical `.tgm` files in `rust/tensogram-core/tests/golden/`:

- `simple_f32.tgm` — single float32 object.
- `multi_object.tgm` — 3 dtypes: u8, i32, f64.
- `mars_metadata.tgm` — MARS keys in per-object `base` entries.
- `multi_message.tgm` — 3 independent messages.
- `hash_xxh3.tgm` — hash verification.

All files are byte-for-byte deterministic. Cross-language tests verify
decode correctness and encode determinism.
