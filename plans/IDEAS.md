# Ideas

Speculative ideas for possible future work. Not yet accepted. Do not
implement until promoted to `TODO.md`.

## Tools

- [ ] `tensogram filter` subcommand (v2 rules engine)

## Bridges

- [ ] Tensogram as a storage backend for NetCDF

## NaN handling

- [ ] NaN bitmask companion object: use a reserved preamble flag to
  signal that a data object has NaN values, with a succeeding bitmask
  data object marking NaN positions. This avoids NaN in the float
  payload (which breaks packing) while preserving missing-value
  semantics. Needs a concrete use case before implementing.

## Optimisations

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

## CI

- [ ] Integrate CI with ECMWF workers
    - Testing on ECMWF platforms
    - Testing on macstadium
