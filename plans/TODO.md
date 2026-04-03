# Features decided to implemented
- `async` feature gate (tokio + spawn_blocking for libaec FFI)
- `mmap` feature gate (memmap2 for memory-mapped file access)
- Streaming mode (may need revisit of Wire Format)
- Cross-language golden binary test files
- ciborium canonical encoding verification (current two-step approach works but should be validated against a reference implementation)
- ciborium canonical encoding verification: confirm RFC 8949 Section 4.2 deterministic output; build BTreeMap wrapper if needed

# Deferred TODO list (not yet decided)
- `tensogram filter` subcommand (v2 rules engine)
- Zarr v3 backend: Tensogram as storage backend for Zarr v3
- GRIB converter + benchmark suite: `tensogram from-grib` + `tensogram-bench`
- SIMD payload alignment: optional padding for 16/32/64-byte aligned payloads
