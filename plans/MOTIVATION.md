# Motivation: Why Tensogram Exists

## The Problem

ECMWF's operational weather forecasting pipelines rely primarily on GRIB (1 and 2) for encoding and transmitting meteorological data. GRIB is an international standard developed by WMO, which means vocabulary evolution — new parameter types, metadata concepts, level types — requires international negotiation. This process takes months. The result: concrete delays in implementing new processed products that don't match existing WMO descriptions.

Additionally, GRIB is fundamentally limited to 1-Tensors (single fields). Multi-dimensional data like sea wave spectra — which should be represented as a 3-tensor (spatial lat-lon, parameter, frequency) — must be flattened into sequences of GRIB messages with ad-hoc conventions, creating persistent technical debt. New AI/ML weather models produce N-dimensional tensors that don't map cleanly to GRIB's 1-tensor model either.

## Why Not Extend Existing Formats?

- **GRIB 1 & 2:** Primary workhorse for all operational data. Good tooling ecosystem (ecCodes). But vocabulary is WMO-regulated and structurally limited to 1-tensors.
- **NetCDF / HDF5:** Minimal internal use, mostly for static data. Metadata is flexible but the format is file-oriented, not message-oriented.
- **BUFR / ODB:** Separate column formats for observations. Possible future unification target, but explicitly deferred.

GRIB remains the right choice for external clients and archival (strict international format). Tensogram is for the internal operational pipeline where speed of vocabulary evolution and N-dimensional tensor support matter.

## Demand Evidence

- **Concrete delays:** Implementation of new processed products blocked by WMO vocabulary evolution wait times.
- **Technical debt:** Developers work around GRIB's limitations, creating accumulating workarounds that become permanent fixtures.
- **Developer consensus:** A retreat and brainstorming session among developers concluded that a flexible internal format is needed.
- **Structural limitation:** Sea wave spectra (a 3-tensor) are currently split across multiple GRIB messages with ad-hoc conventions.
- **ML model outputs:** New AI/ML weather models produce N-dimensional tensors that don't map to GRIB's 1-tensor model.

## What We're Building

A **binary message format library** for N-dimensional scientific tensors. Not a file format — a network-transmissible message format that can also be appended to files.

### Core Properties

1. **N-dimensional tensors** — encode any number of dimensions, not just 2D fields
2. **Multiple tensors per message** — each with its own shape, dtype, and encoding pipeline
3. **Self-describing messages** — CBOR metadata carries everything needed to interpret the data
4. **Vocabulary-agnostic** — the library doesn't validate or interpret metadata semantics; ECMWF's MARS vocabulary is the application layer's concern
5. **Encoding pipeline per object** — encode → filter → compress, each step configurable per tensor
6. **Partial range decode** — extract sub-tensors without decoding the full payload
7. **Cross-language** — Rust core with C FFI, C++ wrapper, and Python/NumPy bindings

### Target Users

ECMWF developers working on operational weather forecasting pipelines — the teams producing, processing, and consuming gridded forecast data internally.

### Adoption Wedge (Priority Order)

1. **Existing products with structural limitations** — data like sea wave spectra that GRIB flattens. This is the sharpest wedge: existing data, existing pain, existing customers.
2. **ML model outputs** — N-dimensional tensors from AI weather models.
3. **New processed products** — data products with parameters that lack WMO vocabulary definitions.

### Constraints

- Must be production-ready for ECMWF operational pipelines
- Network-transmissible message format (but appendable to files)
- Limited dependencies — lightweight enough to embed in pipeline components
- Must support existing encoding approaches (simple_packing, szip/libaec) for compatibility
- Vocabulary remains externally managed — library is vocabulary-agnostic
- Must support forward evolution via version metadata

## What We're Not Building (Yet)

- **BUFR/ODB replacement** — observation data unification is deferred
- **Zarr v3 backend** — possible future direction
- **External format** — Tensogram is internal; GRIB remains for external distribution

## Success Criteria

1. **Correctness:** Round-trip encoding/decoding with bit-exact results (lossless) or verified quantization tolerance (lossy)
2. **Flexibility demonstrated:** Encode a sea wave spectrum as a 3-tensor, an ML output as an N-tensor, and a 2D field — all using the same library
3. **Cross-language interop:** Messages encoded in Rust decode in Python and C++, and vice versa
4. **Production readiness:** No memory safety issues, comprehensive test suite, documented API
5. **Performance baseline:** Encode/decode throughput measured for reference data sizes
