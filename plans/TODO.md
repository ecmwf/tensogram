# Deferred TODO list
- Zarr v3 backend: Tensogram as storage backend for Zarr v3
- GRIB converter + benchmark suite: `tensogram from-grib` + `tensogram-bench`
- Metadata integrity hash: hash over CBOR metadata block (not just payload)
- SIMD payload alignment: optional padding for 16/32/64-byte aligned payloads
- ciborium canonical encoding verification: confirm RFC 8949 Section 4.2 deterministic output; build BTreeMap wrapper if needed
