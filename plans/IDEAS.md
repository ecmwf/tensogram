# Ideas

Ideas for possible future implementation. Do not implement these yet.

- Tools
    - [ ] `tensogram filter` subcommand (v2 rules engine)

- Bridges
	- [x] tensogram as backend for Zarr v3 (as storage backend)
	- [x] xarray loader from tensogram (DONE — tensogram-xarray, 113 tests, shipped in 0.4.0)
	- [ ] tensogram as a storage backend for netcdf

- it should support transforming this data to xarray in earthkit-data

- Optimisations:
    - [ ] SIMD payload alignment: optional padding for 16/32/64-byte aligned payloads. 

- Performance
    - [ ] parallel batch validation: `validate` file loop with rayon `par_iter` for thousands-of-files use cases. Architecture already supports it (`validate_message` is `&[u8] -> Report`, no shared state).
    - [x] benchmark suite
    - [x] compare with eccodes simple+szip
    - [ ] 4-byte AEC containers for 24-bit szip: zero-padded 4-byte containers may improve compression ratio for 17-24 bit data. Requires padding/unpadding in the szip compressor and is a wire format change.

- CI
    - [ ] integrate CI with ECMWF workers
        - testing our platforms
        - testing in macstadium

