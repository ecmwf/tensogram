# Ideas

Ideas for possible future implementation. Do not implement these yet.

- Tools
    - [ ] `tensogram filter` subcommand (v2 rules engine)

- Bridges
	- [ ] tensogram as backend for Zarr v3 (as storage backend)
	- [ ] xarray loader form tensogram
	- [ ] tensogram as a storageg backend for netcdf
    - [ ] GRIB converter + benchmark suite: `tensogram from-grib` + `tensogram-bench`

- Optimisations:
    - [ ] SIMD payload alignment: optional padding for 16/32/64-byte aligned payloads. 

- Performance
	- [ ] add performance tests
	- [ ] compare with eccodes simple+szip