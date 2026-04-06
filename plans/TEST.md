# Test Plan

Repo: ecmwf/tensogram

> For release-by-release changes, see `../CHANGELOG.md`.

## Test Coverage Summary (v0.6.0)

| Component | Tests | Type | Coverage |
|-----------|-------|------|----------|
| tensogram-core | ~283 | Unit + integration + adversarial + edge-case | ~90% |
| tensogram-encodings | 47 | Unit | ~85% |
| tensogram-cli | 12 | Unit | ~70% |
| tensogram-ffi | (via C++ wrapper) | Indirect | — |
| C++ wrapper | 109 | GoogleTest | — |
| tensogram-python | 226 | pytest (parametrized) | ~88% |
| tensogram-grib | 17 | Integration | ~80% |
| tensogram-xarray | 179 | pytest | ~98% |
| tensogram-zarr | 204 | pytest | ~95% |
| **Total** | **~1010** | | **90.5% (Rust)** |

## Affected Components

- tensogram-core: encode, decode, decode_metadata, decode_object, decode_range, scan, streaming, file, iterators
- tensogram-encodings: simple_packing, shuffle filter, 6 compression codecs (szip, zstd, lz4, blosc2, zfp, sz3), pipeline dispatch
- tensogram-ffi: C API opaque handles, typed getters, error codes, iterators, streaming encoder
- tensogram-python: PyO3 bindings, NumPy round-trip, TensogramFile, metadata access
- tensogram-cli: info, ls, dump, get, set, copy, merge, split, reshuffle, convert-grib
- tensogram-xarray: xarray backend engine, lazy loading, coordinate detection, hypercube merging

## Key Interactions to Verify

- Round-trip encode→decode for every dtype (float16/32/64, bfloat16, complex64/128, int8-64, uint8-64, bitmask)
- Round-trip for every encoding×filter×compression combo
- Cross-language byte-identical output (Rust, Python, C++) via golden files
- Partial range decode via szip block offsets, blosc2 chunks, zfp fixed-rate blocks
- CLI subcommands with -w filtering, -p key selection, -j JSON output
- Frame-based index: header/footer index frames with correct offsets; O(1) random access
- Streaming mode: total_length=0, footer-based index
- mmap-based file access: zero-copy scan + read
- Async encode/decode paths (feature-gated): spawn_blocking for FFI calls
- GRIB conversion: MARS key extraction, per-object independent `base[i]` entries, grouping modes
- xarray backend: lazy loading, N-D slicing, coordinate detection, multi-message merge

## Edge Cases

- NaN input to simple_packing → must reject with EncodingError
- Zero-object message (metadata-only)
- Streaming mode (total_length=0, footer-based metadata and index)
- Payload > 4 GiB (uint64 offsets in index frames)
- Non-byte-aligned bit packing (12-bit, 24-bit, 1-bit bitmask)
- Corrupted message mid-file → scan recovers to next valid TENSOGRM marker
- `tensogram set` with immutable key (shape, strides, dtype, encoding, hash) → must reject
- Empty tensor (ndim=0, scalar value)
- Frame marker corruption (FR/ENDF) in one data object → that object rejected, others accessible
- Inter-frame padding for 64-bit alignment
- Shuffle + partial range decode → rejected (documented as unsupported combination)
- ciborium canonical encoding: same metadata map serializes to byte-identical CBOR on every call
- Big-endian byte order round-trips
- Wire format determinism (idempotent encode→decode→encode)

## Critical Paths

- Message encode → network transmit → decode must be bit-exact (lossless) or within tolerance (lossy)
- File append → scan → random access by message index → decode object by index
- Streaming encode → footer index → random access to data objects
- mmap → seek to message → decode partial range via block offsets
- Golden binary .tgm files decoded identically by Rust, Python, and C++ on every CI run
- `tensogram copy -w mars.param=2t input.tgm output.tgm` → only matching messages, byte-identical
- xarray `open_dataset()` → lazy load → partial decode on slice access

## Golden Test Files

5 canonical `.tgm` files in `crates/tensogram-core/tests/golden/`:
- `simple_f32.tgm` — single float32 object
- `multi_object.tgm` — 3 dtypes: u8, i32, f64
- `mars_metadata.tgm` — MARS keys in per-object `base` entries
- `multi_message.tgm` — 3 independent messages
- `hash_xxh3.tgm` — hash verification

All files are byte-for-byte deterministic. Cross-language tests verify decode correctness and encode determinism.
