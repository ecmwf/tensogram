# Ideas

Ideas for possible future implementation. Do not implement these yet.

- Tools
    - [ ] `tensogram filter` subcommand (v2 rules engine)

- Bridges
	- [ ] tensogram as backend for Zarr v3 (as storage backend)
	- [x] xarray loader from tensogram (DONE — tensogram-xarray, 113 tests, shipped in 0.4.0)
	- [ ] tensogram as a storage backend for netcdf

- Optimisations:
    - [ ] SIMD payload alignment: optional padding for 16/32/64-byte aligned payloads. 

- Performance
    - [ ] benchmark suite?
	- [ ] add performance tests
	- [ ] compare with eccodes simple+szip