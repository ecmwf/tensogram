# Ideas

Speculative ideas for possible future work. Not yet accepted. Do not
implement until promoted to `TODO.md`.

## Feedback

Add here feedback ideas:

- [ ] Ipsum Verbum


## Tools

- [ ] `tensogram filter` subcommand (v2 rules engine)
- [ ] Inspectors/Viewers: Command-line or GUI tools to peer inside the file without writing a custom script. -- a TUI for tensograms?

## Features

- [ ] Progressive precision decoding (maybe you already have that)
  - this needs more discussion. how would you see this progression? in the same message? each data frame contains further precision?

- [ ] Metadata schemas and self-registration of schema definitions -- see CovJSON
  - there's a standard for registering your own metadata. 
  - you describe something like "$schema: "ecmwf.int/coveragejson/schema" which standard tools know how to fetch
  - research how is done


## Bridges

- [ ] Tensogram as a storage backend for NetCDF

## Languages

- [ ] Go interface over Rust
- [ ] Fortran interface over Rust
- [ ] Mojo integration (one already available through python, but we could have a direct one)

## encoding / decoding

- (shipped) ~~NaN bitmask companion object~~ — delivered as the v3
  `NTensorFrame` mask region with `allow_nan` / `allow_inf` encode
  opt-ins and `restore_non_finite` on decode.  See
  `plans/WIRE_FORMAT.md` §6.5 and `docs/src/guide/nan-inf-handling.md`.

## Optimisations

- [ ] GPU encoding/decoding. check nvidia nvcomp
    - also check compute-shader decoding (maybe encoding too) -- wonder how fast that would go vs threads

- [ ] GPU direct loading
    - check out the https://github.com/rapidsai/kvikio library -- possibly we want to utilize GDS/RDMA, for direct IO to GPU. There are 2-3x gains even on a puny laptop when comparing kvikio to vanilla torch.load, and reportedly much bigger ones on H100/A100 systems. We should be able to move tensogram messages to GPU avoiding the CPU (fully or partially, depending on HW capabilities) in a manner similar to kvikio, resulting in a cuFile/tensor

- [ ] SIMD payload alignment: optional padding for 16 / 32 / 64-byte
  aligned payloads.

- [ ] Parallel batch validation: `validate` file loop with rayon
  `par_iter` for thousands-of-files use cases. Architecture already
  supports it (`validate_message` is `&[u8] -> Report`, no shared
  state).

- [ ] 4-byte AEC containers for 24-bit szip: zero-padded 4-byte
  containers may improve compression ratio for 17-24 bit data.
  Requires padding/unpadding in the szip compressor and is a wire
  format change.
- [ ] `ValidateOptions.threads` (follow-up to the multi-threaded
  coding pipeline): level 3/4 validation runs the full decode
  pipeline, so adding `threads` to `ValidateOptions` would let the
  `--threads` CLI flag accelerate `tensogram validate --full`.  The
  CLI already plumbs the flag but currently drops it on the floor
  before `validate::run`.  Needs a small API change; blocked only by
  appetite.
- [ ] blosc2 `decompress` parallelism: currently our
  `Blosc2Compressor` applies `nthreads` only on the compress path
  because blosc2's safe `SChunk::from_buffer` ignores runtime
  dparams overrides.  A `Chunk::set_dparams`-based rewrite
  (unsafe-free but lower-level) would let axis-B benefit the decode
  path too.  Gains are modest (decompress is memory-bound) but it
  would close the symmetry gap.

- [ ] blosc2 per-chunk `block_offsets`: populate
  `CompressResult.block_offsets` from blosc2's frame offsets
  (`blosc2_frame_get_offsets` in `c-blosc2/include/blosc2.h`,
  exposed via the `blosc2_rs_sys` FFI crate) and serialise them in
  the object descriptor as `"blosc2_chunk_offsets"`, parallel to
  szip's existing `"szip_block_offsets"` key.  Unlocks two wins:
  amortises the `SChunk::from_buffer` header-parse cost on hot
  `decompress_range` loops (xarray/dask workloads), and enables
  byte-ranged remote reads — fetch the small offsets metadata from
  S3/GCS first, then issue targeted `Range:` GETs for only the
  chunks that cover the requested slice instead of downloading the
  whole multi-GiB compressed blob.  The SChunk frame already stores
  these offsets in its trailer, so this is about hoisting them up
  to tensogram's metadata layer for free random access without
  re-parsing the frame on every call.  Defer until a concrete use
  case or microbenchmark motivates the wire-format addition;
  `blosc2_frame_get_offsets` has been a public `BLOSC_EXPORT` since
  at least v2.7.1, so the upstream surface is stable.  Spun out of
  the issue #68 multi-chunk fix review (PR #69).

## CI

- [ ] Integrate CI with ECMWF workers
    - Testing on ECMWF platforms
    - Testing on macstadium

## Viewer

- [ ] Tiled overlay image for partial updates on zoom
- [ ] Field search grouped by param, filtered by levtype
- [ ] Responsive layout for smaller screens / mobile
- [ ] Server-side rendering mode: optional backend for environments where WASM is blocked
