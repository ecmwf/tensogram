# Features decided to implement
- `async` feature gate (tokio + spawn_blocking for libaec FFI)
- `mmap` feature gate (memmap2 for memory-mapped file access)
- Cross-language golden binary test files
- ciborium canonical encoding verification (current two-step approach works but should be validated against a reference implementation)

# Code quality improvements (from code review)
- Decompose `encode_message` in framing.rs (~198 lines) into smaller helper functions
- Add stricter frame sync validation in `decode_message` (verify frame types appear in expected order)
- Optimize `ensure_scanned` in file.rs to avoid loading full file into memory (use streaming scan)

# Deferred TODO list (not yet decided)
- `tensogram filter` subcommand (v2 rules engine)
- Zarr v3 backend: Tensogram as storage backend for Zarr v3
- GRIB converter + benchmark suite: `tensogram from-grib` + `tensogram-bench`
- SIMD payload alignment: optional padding for 16/32/64-byte aligned payloads
