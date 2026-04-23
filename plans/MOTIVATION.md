# Motivation: Why Tensogram Exists

## The Problem

Scientific computing routinely produces and moves N-dimensional tensor data at
scale — weather and climate forecasts, Earth observation, CFD, satellite imagery, 
machine-learning inputs, weights and outputs.
Moving that data between producers and consumers hits three recurring friction
points:

1. **Metadata vocabulary is tightly coupled to the transport.** Adding a new
   field type, coordinate, or provenance annotation often means waiting on a
   standards body or negotiating a convention with collaborators. Pipelines
   stall while the vocabulary catches up.
2. **Tensor shape is constrained by the format, not the data.** Formats
   designed around 1-D columns or 2-D fields force higher-dimensional data
   into sequences of messages with ad-hoc external conventions that every
   consumer must re-implement. Sea wave spectra, multi-echo MRI acquisitions,
   ensemble forecast members, and hyperspectral cubes all live naturally as
   N-tensors.
3. **Transport, file, and partial-access use different code paths.** A single
   format that works equally well over a TCP stream, as an appendable file,
   and as a byte-range read against a cloud object store is rare.

Tensogram is a binary message format designed around these three frictions.
Metadata ships with the data. Any number of tensor dimensions is supported
per object. The same bytes work on a socket, as a `.tgm` file, or as a range
read from S3 / GCS / Azure / HTTP.

## What we are building

A **binary message format library** for N-dimensional scientific tensors. Not
a file format — a network-transmissible message format that can also be
appended to files.

### Core properties

1. **N-dimensional tensors** — encode any number of dimensions, not just 1-D
   or 2-D fields.
2. **Multiple tensors per message** — each with its own shape, dtype, and
   encoding pipeline.
3. **Self-describing messages** — CBOR metadata carries everything needed to
   interpret the data.
4. **Vocabulary-agnostic** — the library does not validate or interpret
   metadata semantics. Domain vocabularies (MARS at ECMWF, CF conventions,
   BIDS in neuroimaging, DICOM in medical imaging, or in-house namespaces)
   are the application layer's concern.
5. **Encoding pipeline per object** — encode → filter → compress, each step
   configurable per tensor.
6. **Partial range decode** — extract sub-tensors without decoding the full
   payload.
7. **Cross-language** — Rust core with C FFI, C++ wrapper, Python/NumPy
   bindings, and WebAssembly/TypeScript bindings.

## Target communities

Anywhere N-dimensional scientific tensors are produced, transported, or
archived:

- Numerical weather prediction and climate modelling
- Earth observation and remote sensing
- Medical imaging pipelines (volumetric and time-resolved)
- Genomics and omics tensors
- Particle physics and accelerator facility data
- AI / ML model inputs, weights, and outputs
- Computational chemistry and materials simulation
- Any pipeline moving large N-tensors between producers and consumers

The library makes no assumption about domain. ECMWF provides initial
production validation through operational weather-forecasting workloads, but
nothing in the format is weather-specific.

## Adoption wedges (priority order)

1. **Existing data that is awkward to transport in current formats** —
   sequences of 2-D fields that are naturally 3-D or higher tensors,
   time-resolved imaging stacks, spectra, ensemble members. This is the
   sharpest wedge: existing data, existing pain.
2. **AI/ML pipelines** where inputs, weights, and outputs are native
   N-tensors that do not fit neatly into legacy scientific formats.
3. **New data products** whose metadata vocabularies are not yet standardised
   and benefit from self-describing CBOR.

## Constraints

- Production-ready for high-throughput scientific pipelines.
- Transportable over network and appendable to files.
- Minimal mandatory dependencies; heavy codecs are optional and feature-gated.
- Vocabulary externally managed by the application layer.
- Forward-compatible via a single version integer.

## Relation to existing formats

Tensogram complements rather than replaces existing formats. It provides
**importers** for GRIB (via ecCodes) and NetCDF (via libnetcdf) so that data
in those formats can be brought into Tensogram pipelines without a lossy
re-modelling step. Going the other way — Tensogram → other — is currently out of scope
because Tensogram's data model is a superset of 1-D and 2-D formats and a
faithful down-conversion is often difficult if not impossible.

We believe that existing formats remain appropriate for distribution, archival, and
standards-driven exchange. Tensogram targets the internal pipeline layer
where speed of vocabulary evolution and N-dimensional tensor support matter.

## What we are not building (yet)

- **Tensogram → other-format writers.** Initial scope is one-way import.
- **Observation / column formats** (BUFR, ODB, Arrow-like tabular data).
  Potentially a future direction.
- **Domain-specific vocabularies.** WMO/GRIB, CF conventions, and
  so on live above the library.

## Success criteria

1. **Correctness:** round-trip encoding/decoding with bit-exact results
   (lossless) or verified quantisation tolerance (lossy).
2. **Flexibility demonstrated:** encode an N-dimensional scientific tensor
   (wave spectrum, imaging volume, ML output, 2-D field) with the same
   library and the same API.
3. **Cross-language interop:** messages encoded in Rust decode identically
   in Python, C++, and TypeScript/WASM.
4. **Production readiness:** no memory safety issues, no undefined behaviour or library aborts, comprehensive test
   suite, documented API.
5. **Performance baseline:** encode/decode throughput measured against
   established references for representative data sizes.
