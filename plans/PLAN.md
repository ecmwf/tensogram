---
status: HISTORICAL
---
# Iteration 0 Plan (Historical)

> **Note:** This document records the initial planning decisions made at project inception (2026-04-01).
> It is preserved as historical context — the framing below is ECMWF-internal and
> reflects the original scope. The project's scope has since broadened to
> position Tensogram as a general-purpose N-tensor message format for
> scientific computing at scale, with ECMWF weather-forecasting workloads as
> one important validated use case. For the current state of the project, see:
> - `MOTIVATION.md` — why Tensogram exists and what we're building
> - `DESIGN.md` — design rationale and key decisions
> - `DONE.md` — current implementation status
> - `../CHANGELOG.md` — release history

## Original Vision

### 10x Check
The 10x version of Tensogram isn't just a wire format library. It's ECMWF's internal data plane, replacing GRIB, NetCDF, BUFR, and ODB for all internal data movement. Every pipeline component speaks Tensogram. FDB stores Tensogram natively. Zarr v3 uses Tensogram as a backend store. The CLI becomes the swiss army knife that ecCodes' grib_* tools are today. ML training pipelines read Tensogram directly via PyTorch/NumPy integration. Partial range decode means FDB can serve sub-field extractions without reading full messages. The encoding pipeline evolves to include ZFP and SZ3 for error-bounded lossy compression of ML model outputs.

## Original Scope Decisions

| # | Proposal | Effort | Original Decision | Current Status |
|---|----------|--------|-------------------|----------------|
| 1 | Zarr v3 backend | M | DEFERRED | Still deferred (see IDEAS.md) |
| 2 | GRIB-to-Tensogram converter | M | DEFERRED | **DONE** (tensogram-grib, shipped 0.2.0) |
| 3 | Async I/O (tokio, feature-gated) | M | ACCEPTED | **DONE** (shipped 0.1.0) |
| 4 | Memory-mapped file access (memmap2) | S | ACCEPTED | **DONE** (shipped 0.1.0) |
| 5 | Benchmark suite vs ecCodes | S-M | DEFERRED | Still deferred (see IDEAS.md) |
| 6 | Streaming mode (total_length=0) | S | ACCEPTED | **DONE** (shipped 0.2.0) |

## Original Implementation Decisions

These were the decisions made before coding began. Some evolved during implementation:

- **CBOR crate:** ciborium (built-in canonical encoding, serde integration) — *still current*
- **Approach:** Full Stack (all 5 crates from day 1) — *baseline still current; the workspace has since grown with additional opt-in crates; see `ARCHITECTURE.md`*
- **Feature gates:** `async` for tokio, `mmap` for memory-mapped access — *still current*
- **NaN handling:** simple_packing rejects NaN with EncodingError — *still current*
- **Zero-object messages:** Valid (metadata-only for pipeline signaling) — *still current*
- **Payload order:** Always C-order (row-major) — *still current*
- **Shuffle + partial range decode:** Don't compose — *still current, documented*
- **`tensogram set` safety:** Mutable vs immutable key distinction — *still current*
- **Async + compression:** `spawn_blocking` for libaec FFI calls — *still current*
- **Binary header index:** Originally planned as fixed-size binary prefix — *evolved to frame-based index in v2 wire format*
- **Parallel arrays:** objects[] and payload[] originally separate — *evolved: data object descriptors now embedded in frames, payload is Vec of per-object metadata*
