# Ideas

Speculative ideas for possible future work. Not yet accepted. Do not
implement until promoted to `TODO.md`.

This file has two parts: a list of concrete speculative ideas
(immediately below), and a longer-form **Horizon** brainstorm of
ambitious directions (at the end) exploring whether Tensogram could
become a shared cross-domain tensor message format. Ideas that mature
into committed work move up to `TODO.md`.

## Tools

- [ ] `tensogram filter` subcommand (v2 rules engine)
- [ ] Inspectors/Viewers: Command-line or GUI tools to peer inside the file without writing a custom script. -- a TUI for tensograms?
  - this is a nice parquet tool which could be an inspiration https://github.com/raulcd/datanomy
  - it makes sense to also develop a GUI-in-browser, to manage a directory structure with partition columns -- and use it standalone or eg within forecast in abox

## Features

- [ ] Progressive precision decoding (maybe you already have that)
  - this needs more discussion. how would you see this progression? in the same message? each data frame contains further precision?

- [ ] Metadata schemas and self-registration of schema definitions -- see CovJSON
  - there's a standard for registering your own metadata. 
  - you describe something like "$schema: "ecmwf.int/coveragejson/schema" which standard tools know how to fetch
  - research how is done


## Bridges

- [ ] Tensogram as a storage backend for NetCDF

## Python API & Serialisation

- [ ] **Raw bytes retention on decoded `Message` — `RawMessage` type**

  When `tensogram.decode(raw_bytes)` is called today the original wire bytes are discarded after
  parsing; only the decoded `Metadata` / `DataObjectDescriptor` / NumPy arrays are kept.  Retaining
  the raw bytes would make a decoded `Message` a lossless handle on the original wire representation,
  enabling efficient round-trips in any system that serialises, routes, or caches tensogram data
  without re-encoding.

  Two implementation options:

  - Add an optional `_raw: bytes | None = None` field to the `Message` `NamedTuple` (set by
    `decode()`, `None` for programmatically constructed messages).
  - Introduce a separate **`RawMessage`** type that wraps `bytes` and exposes `.decoded: Message` as
    a lazy cached property — a cleaner API that does not pollute the primary `Message` type and makes
    the intent explicit.

  The `RawMessage` variant is arguably the right design: it is a first-class type for *uninterpreted*
  tensogram bytes, useful as a routing token in message brokers, workflow engines, and caches that
  must forward data without ever decoding it.  A workflow scheduler, for example, moves data between
  machines but never inspects tensor values — handing it a `RawMessage` means it pays zero decode
  cost.  `RawMessage` would be the natural input/output type for the `message_to_bytes` function
  and for the `__reduce__` protocol described below.

  **Note:** `RawMessage` is also the enabling primitive for true zero-copy NumPy (and Arrow)
  deserialisation — the source buffer must outlive the arrays that view it, and `RawMessage`
  provides the Python object whose lifetime NumPy's `.base` reference can anchor to.  See the
  zero-copy NumPy deserialisation idea below.

- [ ] **`__reduce__` / pickle support via wire bytes**

  `Metadata` and `DataObjectDescriptor` are PyO3 `#[pyclass]` objects with no custom `__reduce__`.
  Cloudpickle handles some PyO3 types via bytecode introspection, but this is fragile, version-
  dependent, and may silently produce wrong results or error at unpredictable times.  If raw bytes
  are retained (see above), a correct and efficient `__reduce__` becomes trivial:

  ```python
  def __reduce__(self):
      return (tensogram.decode, (self._raw,))
  ```

  This makes cloudpickling a `Message` equivalent to: pickle-protocol overhead + one copy of already-
  serialised wire bytes — far cheaper than pickling NumPy arrays by value (which copies all tensor
  data twice: once when the array is allocated by `decode`, once when pickle serialises it).

  The `tensogram-xarray` `TensogramBackendArray` is a direct precedent: it already implements
  `__getstate__` / `__setstate__` that drops open file handles and reconstructs lazily from
  `(file_path, msg_index, obj_index)`.  The same pattern applied to in-memory wire bytes covers
  the in-process / over-the-wire case.

- [ ] **`message_to_bytes(msg: Message) -> bytes` convenience function**

  A canonical public function to obtain wire bytes from any `Message`, regardless of how it was
  constructed.  If the message carries retained raw bytes (from `decode()`), this is O(1).  If the
  message was built programmatically, it falls back to a fresh `tensogram.encode(...)` call.

  Having a single stable API surface decouples callers from the internal representation and avoids
  forcing callers to replicate the `encode()` call with the correct options.  It also acts as the
  natural serialisation hook for third-party integrations (serde registries, object stores, IPC
  transports) that need "give me bytes I can hand to `decode()`" without caring whether the source
  was a decode round-trip or a fresh encode.

- [ ] **Guarantee that `decode()` accepts buffer-protocol objects (including `memoryview`) without copying**

  Currently `decode()` accepts bytes-like / `PyBackedBytes` input.  Whether a `memoryview` is
  accepted *zero-copy* — i.e. the Rust core parses directly from the provided buffer without first
  copying into a Rust-owned `Vec<u8>` — is undocumented and may not hold.

  Explicitly supporting and documenting zero-copy `memoryview` input matters for systems that hold
  tensogram bytes in POSIX shared memory (e.g. `multiprocessing.shared_memory`) and want to decode
  in a receiving process without a userspace memcpy.  For example: producer task serialises once
  into shared memory; all consumer tasks on the same host call `tensogram.decode(shm_memoryview)`
  and get their `Message` with zero additional copies of the wire bytes.  The NumPy arrays produced
  by decode will always involve a copy (they own their memory), but the metadata-parsing and
  descriptor-reading phases are read-only and need not copy the input buffer at all.

- [ ] **Zero-copy NumPy deserialisation for plain-layout objects**

  *Requires the `RawMessage` idea.*

  NumPy supports viewing an external buffer as an array without any copy via
  `numpy.frombuffer(buffer, dtype=dtype).reshape(shape)`.  The resulting array holds a reference to
  the source object as its `.base`, keeping it alive for as long as the array exists.  Tensogram
  does not currently exploit this: every decode path copies tensor data into NumPy-owned memory,
  and the `float16` / `complex` path (`numpy_via_frombuffer` in the Python bindings) even copies
  *twice* — once from the Rust decode buffer into a temporary `PyBytes`, and again via an explicit
  `.copy()` call whose sole purpose is to break the lifetime dependency on that temporary.

  For objects whose encoding pipeline is a no-op (no compression, no filter, no encoding
  transformation), the decoded payload bytes are a contiguous slice directly into the original wire
  buffer — already in the correct in-memory layout for NumPy.  If the source wire buffer is kept
  alive as a `RawMessage` Python object, `numpy.frombuffer` can create a zero-copy array view of
  those bytes.  The array's `.base` reference to the `RawMessage` prevents premature collection.
  No data movement at all is required.

  For compressed or filtered objects the decompression / filter-reversal output is a freshly
  allocated `Vec<u8>` that did not exist in the source buffer; one copy is genuinely unavoidable.
  However, even there the current code copies *again* from the decompressed Vec into a separate
  NumPy buffer — a second unnecessary copy that the same `frombuffer`-with-base technique would
  eliminate.

  **Optional extension — `DataObjectDescriptor.allows_zerocopy_deser() -> bool`.**
  Whether an object qualifies for the zero-copy path is entirely determinable from its descriptor
  metadata, which is already fully exposed to Python:

  ```python
  import sys

  def allows_zerocopy_deser(descriptor) -> bool:
      return (
          descriptor.encoding    == "none" and
          descriptor.filter      == "none" and
          descriptor.compression == "none" and
          descriptor.byte_order  == sys.byteorder   # "little" or "big"
      )
  ```

  Exposing this as a method on `DataObjectDescriptor` lets callers (library code, integrations,
  benchmarks) make an informed decision about whether to use the zero-copy path or fall back to a
  full decode, without reimplementing the check themselves.  It also serves as a stable contract:
  if new encoding / filter types are added in the future, the method's implementation is the single
  place that must be updated.

## Languages

- [ ] Go interface over Rust
- [ ] Fortran interface over Rust
- [ ] Mojo integration (one already available through python, but we could have a direct one)

## encoding / decoding

- (shipped) ~~NaN bitmask companion object~~ — delivered as the v3
  `NTensorFrame` mask region with `allow_nan` / `allow_inf` encode
  opt-ins and `restore_non_finite` on decode.  See
  `plans/WIRE_FORMAT.md` §6.5 and `docs/src/guide/nan-inf-handling.md`.

## Optimisations

- [ ] GPU encoding/decoding. check nvidia nvcomp
    - also check compute-shader decoding (maybe encoding too) -- wonder how fast that would go vs threads

- [ ] GPU direct loading
    - check out the https://github.com/rapidsai/kvikio library -- possibly we want to utilize GDS/RDMA, for direct IO to GPU. There are 2-3x gains even on a puny laptop when comparing kvikio to vanilla torch.load, and reportedly much bigger ones on H100/A100 systems. We should be able to move tensogram messages to GPU avoiding the CPU (fully or partially, depending on HW capabilities) in a manner similar to kvikio, resulting in a cuFile/tensor

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
- [ ] `ValidateOptions.threads` (follow-up to the multi-threaded
  coding pipeline): level 3/4 validation runs the full decode
  pipeline, so adding `threads` to `ValidateOptions` would let the
  `--threads` CLI flag accelerate `tensogram validate --full`.  The
  CLI already plumbs the flag but currently drops it on the floor
  before `validate::run`.  Needs a small API change; blocked only by
  appetite.
- [ ] blosc2 `decompress` parallelism: currently our
  `Blosc2Compressor` applies `nthreads` only on the compress path
  because blosc2's safe `SChunk::from_buffer` ignores runtime
  dparams overrides.  A `Chunk::set_dparams`-based rewrite
  (unsafe-free but lower-level) would let axis-B benefit the decode
  path too.  Gains are modest (decompress is memory-bound) but it
  would close the symmetry gap.

- [ ] blosc2 per-chunk `block_offsets`: populate
  `CompressResult.block_offsets` from blosc2's frame offsets
  (`blosc2_frame_get_offsets` in `c-blosc2/include/blosc2.h`,
  exposed via the `blosc2_rs_sys` FFI crate) and serialise them in
  the object descriptor as `"blosc2_chunk_offsets"`, parallel to
  szip's existing `"szip_block_offsets"` key.  Unlocks two wins:
  amortises the `SChunk::from_buffer` header-parse cost on hot
  `decompress_range` loops (xarray/dask workloads), and enables
  byte-ranged remote reads — fetch the small offsets metadata from
  S3/GCS first, then issue targeted `Range:` GETs for only the
  chunks that cover the requested slice instead of downloading the
  whole multi-GiB compressed blob.  The SChunk frame already stores
  these offsets in its trailer, so this is about hoisting them up
  to tensogram's metadata layer for free random access without
  re-parsing the frame on every call.  Defer until a concrete use
  case or microbenchmark motivates the wire-format addition;
  `blosc2_frame_get_offsets` has been a public `BLOSC_EXPORT` since
  at least v2.7.1, so the upstream surface is stable.  Spun out of
  the issue #68 multi-chunk fix review (PR #69).

## CI

- [ ] Integrate CI with ECMWF workers
    - Testing on ECMWF platforms
    - Testing on macstadium

## Viewer

- [ ] Tiled overlay image for partial updates on zoom
- [ ] Field search grouped by param, filtered by levtype
- [ ] Responsive layout for smaller screens / mobile
- [ ] Server-side rendering mode: optional backend for environments where WASM is blocked

---

## Horizon: Ambitious Future Directions

> This section captures speculative, exploratory, and ambitious ideas
> for where Tensogram could go. Nothing here is accepted for
> implementation. Ideas that mature and get decided move up to `TODO.md`.
>
> The central question framing this document:
> **Can Tensogram become a shared binary tensor message format across
> scientific domains — weather, imaging, genomics, physics, materials,
> ML — rather than one more domain-specific artefact?**
>
> The format is technically ready for that ambition. What it lacks is
> discoverability, tooling, and community. The ideas below address all
> three, alongside deeper technical directions.

---

## A — Tensogram as a Protocol, not just a Format

Tensogram is currently a push format: a producer encodes, a consumer
decodes. The network-transmissible message design and the remote access
infrastructure suggest a richer interaction model.

### A1. Semantic Query Layer (Request/Response Tensogram)

A lightweight *request message* — itself a valid Tensogram message —
that describes what the requester wants: which metadata keys, which
object indices, which sample ranges, at what precision level. The
server responds with a Tensogram message containing exactly what was
asked for. Essentially HTTP range requests elevated to the
tensor/metadata semantic level.

The remote access infrastructure (`object_store`, batched range reads,
`decode_range`) is almost the right foundation. The missing piece is a
server-side query evaluator and a request wire format.

Use case: a pipeline component asks "give me only the sea surface
temperature objects from this message, at 24-bit precision, for this
bounding box" and gets back a valid self-describing Tensogram message.

Questions to resolve:
- What does the request message schema look like? (CBOR naturally)
- Is this a new frame type or a convention in `_extra_`?
- Does the server need to understand the vocabulary (MARS keys), or
  can queries be purely structural (object index + range)?

### A2. Tensogram over Message Queues

First-class support for streaming Tensogram messages over standard
message infrastructure (Kafka, AMQP, NATS, ActiveMQ). The key problem:
a large tensor may exceed the maximum message size of the queue broker.

Protocol sketch:
- A *chunked Tensogram stream*: a large message is split into N
  consecutive queue messages, each carrying a sequence number,
  a stream ID (UUID), a total chunk count, and a byte slice of the
  original Tensogram wire bytes.
- The receiver reassembles in sequence-number order and hands off to
  the standard decoder.
- Partial delivery is detectable (missing sequence numbers); the
  streaming encoder + xxh3 hash frame provides end-to-end integrity.

The streaming encoder already writes frames progressively — the
missing piece is the chunked-reassembly protocol and reference
implementations for at least one broker.

Questions to resolve:
- Which broker(s) are most common in target communities (Kafka in
  data-platform shops, AMQP in scientific facilities, NATS in
  emerging stacks)?
- Is there value in a pure-Rust Kafka/AMQP adapter crate, or is this
  better handled as a Python integration?

### A3. Pub/Sub Tensogram Streams

An extension of A2: a subscription model where consumers register
interest in Tensogram messages matching a metadata filter
(e.g. `mars.param=167, mars.step=0..12`) and receive a live stream of
matching messages. The vocabulary-agnostic metadata filtering already
implemented in `tensogram copy -w` is the right primitive — a pub/sub
broker that evaluates the same `-w` where-clause syntax against
incoming messages.

---

## B — Metadata as a First-Class Asset

### B1. Lineage and Provenance Graph

Every encoded Tensogram message already has a UUID and timestamp in
`_reserved_`. The natural extension: a `_reserved_.lineage` entry that
lists the UUIDs of the messages this one was derived from.

A forecast field derived from a model output derived from observations
→ a DAG of UUIDs. Tools:

- `tensogram lineage <file>` — print the lineage graph of a message
- A lineage store (a small SQLite or key-value store indexed by UUID)
  that maps UUIDs to file paths/URLs

This costs nothing in the wire format (it's just a CBOR list in
`_reserved_`) and gives reproducibility and audit trails for free,
which matters across many scientific domains (reproducible weather
forecasts, medical imaging audits, genomics pipelines with FDA
traceability, etc.).

Questions to resolve:
- Should lineage be in `_reserved_` (library-managed) or `_extra_`
  (application-managed)?
- Is the granularity per-message or per-object?

### B2. Message Signing and Authenticity

The xxh3 hash proves integrity (did the bytes change?) but not
authenticity (did this come from who I think?). A detached signature
in `_extra_` — or a dedicated `SignatureFrame` (frame type 9) — would
let producers sign messages with an institutional key. Consumers verify
without trusting the transport.

Relevant if Tensogram messages cross institutional boundaries or are
published as official data products. The signature covers the
postamble hash (which already commits to all payload bytes).

Implementation options:
- Ed25519 signature over the footer hash bytes (simple, fast, small)
- A `tensogram sign` / `tensogram verify` CLI pair
- Optional: a detached `.tgm.sig` file for cases where the wire
  format must stay unchanged

Questions to resolve:
- Which communities have a real requirement for data authenticity (published
  weather products, medical imaging, archival scientific records)?
- Key-holder model — institutional, per-pipeline, per-instrument?

### B3. Inline Schema Definition (Schema Frame)

Rather than just a URI reference (the CovJSON approach in IDEAS.md),
a Tensogram message could carry its own schema *inline* in a new
`SchemaFrame` (or in `_extra_["$schema"]`). The schema describes:
- Required metadata keys and their types
- Shape/dtype constraints per object
- Vocabulary definitions (human-readable descriptions, units, valid
  ranges)

The schema travels with the data — no network fetch needed. Producers
include it for bootstrap/discovery; once a protocol is established
between producer and consumer, it can be omitted for efficiency.

Format candidates: JSON Schema (widely tooled), CBOR schema (compact),
or a bespoke simple schema language.

This enables Tensogram to be truly self-describing not just in
structure but in *meaning* — a file handed to a stranger contains
everything needed to understand it.

### B4. Tensogram as a Data Contract Format

A *Tensogram Schema* file (`.tgms`) defines what a valid Tensogram
message looks like for a specific data product:
- Required metadata keys with types
- Allowed dtypes and shape constraints
- Expected encoding pipeline
- Version compatibility rules

Producers validate against the schema before sending; consumers
validate on receipt. Think of it as a typed interface definition for
data products — like a Protocol Buffer schema, but for scientific
tensors with encoding pipeline semantics.

CLI integration: `tensogram validate --schema product.tgms <file>`

---

## C — Encoding Pipeline Extensions

### C1. Delta Encoding

Simple packing quantizes absolute values. For time-series or spatially
correlated fields — which describes gridded weather data, sensor time
series, imaging slices, spectra, and most scientific fields —
storing *differences* between adjacent elements before packing can
dramatically improve compression ratio.

A `delta` encoding option applied before `simple_packing`:
- 1D delta: `d[i] = x[i] - x[i-1]`
- 2D delta: Lorenzo predictor (used by ZFP internally)

The implementation is a prefix scan (O(N), trivially parallelisable).
The interaction with partial range decode needs careful design: to
decode element i, you need elements 0..i. This breaks random access
unless you checkpoint at RSI boundaries.

GRIB's grid_second_order uses spatial differencing for exactly this
reason. This would be Tensogram's equivalent.

### C2. Mixed-Precision / Region-of-Interest Encoding

A weather field near a storm needs float64 precision; the surrounding
calm region can be float16. Tensogram's multi-object-per-message makes
this natural without any wire format change:

- Object 0: full field at float16 (background)
- Object 1: high-precision patch at float64 (region of interest)
- Object 2: bitmask indicating which elements are covered by Object 1

The `base[i]` metadata records the relationship between objects via
convention keys. The library doesn't need to understand the semantics
— producers and consumers agree on the convention.

This is a pattern/convention more than a feature, but a reference
implementation and documentation would make it real.

### C3. Learned Compression Codec

A `learned` compression type that uses a small neural network
(stored inline as a companion object in the same message) for
domain-specific lossy compression. Weather data compresses extremely
well with learned models — orders of magnitude better than generic
compressors for the same quality level.

Architecture sketch:
- Object 0: compressed tensor (residuals after model prediction)
- Object 1: the model weights (ONNX format or custom)
- `_extra_["learned_codec_version"]` for forward compatibility

The codec trait in `tensogram-encodings` already defines the interface
— adding a `learned` variant that calls out to an ONNX Runtime or
`candle` (Rust ML library) binding is architecturally clean.

Very experimental. Needs a concrete partner team with a trained model
to test against. But Tensogram's pluggable pipeline is one of the very
few formats where this is architecturally plausible without format
surgery.

### C4. Entropy Coding Layer

Between compression and the wire: an arithmetic coder or ANS (Asymmetric
Numeral Systems) entropy coder as an optional final pass. ANS is what
zstd uses internally — but as a standalone pipeline step it could be
applied after `simple_packing` without any other compression, giving
better ratio than zstd on data that's already quantized. `rANS` is
available in pure Rust.

---

## D — The File Format Side

### D1. Multi-File Virtual Datasets (Tensogram Catalogue)

A *catalogue* — a small Tensogram message containing only metadata and
a list of `(url, byte_offset, byte_length)` tuples for each logical
object. Readers can construct a virtual dataset spanning petabytes
across multiple files and locations without moving any data.

Like the Zarr manifest concept but simpler: the catalogue *is* a valid
Tensogram message; the xarray backend reads it and lazy-loads each
object from its referenced URL via the existing `object_store` remote
backend.

Enables: federated data archives, efficient multi-site access, data
products that reference source data without copying.

### D2. Append-Only Log with Compaction

If Tensogram files are used as append-only logs (forecast runs appended
over time), there is a natural need for compaction: merging multiple
small `.tgm` files into a larger one, re-indexing, and optionally
re-encoding at a different pipeline setting.

`tensogram compact <files>... -o <out>` — generalised multi-file
merge with optional re-encoding. This is `reshuffle` (streaming →
random-access) extended to multi-file merging.

Useful for time-series archives: daily files compacted to monthly
archives with improved index density.

### D3. Content-Addressed Object Store

The per-object xxh3 hash is already computed. A content-addressed
store where the hash *is* the object's identity would allow:
- Deduplication of identical objects across messages (ensemble runs
  often produce the same static fields — land-sea mask, orography,
  etc. — in every member)
- Delta storage: store only the diff between nearly-identical objects
- Efficient caching: a local store can skip re-fetching any object
  whose hash is already present

The hash frame already carries the information. The missing piece is
a store layer (e.g. a local directory keyed by hash prefix, like Git
objects) and a `tensogram store` CLI subcommand.

### D4. Time-Series Optimised Layout

A specialised message layout for time-series data where many objects
share the same spatial structure but differ in time:
- A single shared spatial descriptor
- Per-timestep objects with compact per-object metadata (only the
  differing keys, resolved against the shared descriptor)
- Interleaved or grouped layout options (all-at-once vs. time-first)

This is an application-level convention today, but a first-class
`tensogram pack --layout timeseries` CLI command that auto-detects
the shared structure and produces the compact layout would make it
real.

---

## E — Developer Experience and Adoption

### E1. Zero-Install Browser Explorer

A static web app (HTML + WebAssembly, no server required):
- Drag-and-drop a `.tgm` file
- Explore the metadata tree interactively
- See object shapes, dtypes, encoding pipeline
- Histogram of values for any object
- Simple 2D map plot for lat-lon fields (using Canvas or WebGL)
- Export a slice as CSV or JSON

Built on the existing `tensogram-wasm` binding. Deployable as a
GitHub Pages site at `ecmwf.github.io/tensogram/explorer`. Zero
installation: share a URL, the person uploads a file in the browser,
done.

This is the single highest-leverage adoption tool for scientists who
are not programmers. It makes Tensogram *tangible*.

### E2. `tensogram doctor` — Environment Diagnostics

A CLI subcommand that checks the local environment and reports:
- Which optional features are available (eccodes, netcdf, wasm-pack)
- Which compression codecs are built in vs. missing
- Whether the Python bindings are installed and what version
- A self-test: encode + decode a small synthetic tensor and verify
  round-trip

Useful for onboarding, CI debugging, and support tickets. One command
that tells you exactly what's working and what isn't.

### E3. Language Server Protocol for Tensogram Metadata

An LSP server that understands a registered vocabulary (e.g. MARS)
and provides:
- Autocomplete for known metadata keys as you write Python
- Inline validation of key/value types
- Hover documentation for known parameters
- Warnings for unknown keys (when a schema is registered)

When you type `base[0]["mars"]["param"] = ` in your editor, it
suggests valid parameter codes with descriptions. Built on the schema
system from B3/B4.

This would make Tensogram metadata as ergonomic as a typed API — the
vocabulary layer becomes editor-tooled even though the library itself
remains vocabulary-agnostic.

### E4. Tensogram Playground (cloud-hosted notebook)

A hosted Jupyter/Marimo notebook environment where anyone can:
- Write Python code that encodes and decodes Tensogram messages
- Load real open datasets (ECMWF IFS 0.25° weather data, sample
  medical imaging volumes, reference genomics tensors, etc.)
- Explore the xarray and Zarr backends interactively
- Share notebook links

No local installation. Frictionless entry point for external adopters.
The existing Python examples are almost the right content — they need
a zero-friction hosting environment around them.

### E5. Tensogram in Observable / Quarto / Jupyter Book

Integration with the scientific publishing ecosystem:
- An Observable Plot plugin that renders Tensogram data directly in
  notebooks (uses the WASM binding in the browser)
- A Quarto extension for embedding Tensogram-backed charts in
  scientific papers
- A Jupyter widget (`tensogram-widget`) for interactive exploration
  inline in notebooks

Scientists publish papers that reference their data. If the data is
in Tensogram format and the paper links to an interactive figure
powered by the WASM explorer, Tensogram becomes visible in the
research literature.

---

## F — Community and Ecosystem Building

### F1. Tensogram Improvement Proposals (TIPs)

A lightweight RFC process — a `tips/` directory in the repo where
numbered TIP documents propose wire format changes, new codec
interfaces, vocabulary conventions, or tooling directions. Any
community member can open a TIP, discuss it, and have it accepted
or rejected with a written rationale.

This is how Python (PEPs), Rust (RFCs), and Zarr (ZEPs) govern
evolution. It signals that the project is open to external
contribution at the design level, not just the code level.

### F2. Reference Implementations as Specification Tests

A *conformance test suite* — a set of canonical `.tgm` files covering
every feature of the wire format, with a machine-readable manifest
describing what each file tests and what the expected decode outputs
are. Any implementation in any language passes or fails the suite.

This is different from the golden files in `tensogram/tests/golden/`
(which test the Rust implementation) — it's a language-neutral
specification artifact. It would lower the barrier to writing a new
implementation in Julia, R, Java, etc., and would prove that
Tensogram is a real open standard, not just a Rust library.

### F3. Tensogram Adopter Program

A lightweight programme where external teams (research institutes,
other NWP centres, ML weather labs) are invited to trial Tensogram
and provide structured feedback:
- A dedicated channel for adopter questions
- Commitment to respond to issues within N days
- A public adopters page that builds social proof
- Input into the TIP process (F1)

The hardest part of open-source adoption is the first external user.
Making that step explicit and supported removes the friction.

### F4. Bindings Expansion

Each new language binding multiplies the potential adopter pool.
Priority order based on who does scientific computing:

| Language | Why it matters | Complexity |
|----------|---------------|------------|
| **Julia** | Growing ML/scientific community, native array types, active use in NWP, astronomy, and computational biology | Medium (Julia C FFI is straightforward) |
| **R** | Large statistical, climate, and bioinformatics communities | Medium (Rcpp or direct C FFI) |
| **Fortran** | Legacy numerical scientific code at weather centres, national labs, HPC sites | Low (ISO_C_BINDING over existing C FFI) |
| **Go** | Cloud infrastructure, emerging scientific tools | Medium (cgo over C FFI) |
| **Java/Kotlin** | Scientific-data web services and portals at research institutions | Medium (JNI or JNA over C FFI) |

The C FFI already exists. Each of these is essentially a thin wrapper
layer — the hard Rust work is already done.

---

## G — Scientific Computing Integrations

### G1. PyTorch / JAX Tensor Protocol

A `__dlpack__` and `__array_interface__` implementation on the Python
`DataObject` type so that decoded tensors can be passed zero-copy to
PyTorch, JAX, CuPy, and any other array library that supports the
DLPack protocol. Currently users call `numpy.frombuffer()` — DLPack
would enable GPU-side zero-copy if the decoded buffer is on a CUDA
device.

This is a small, high-value change to the Python bindings.

### G2. Dask-Native Chunked Decode

A `tensogram.open_dask(path, chunk_by="object")` function that
returns a Dask graph where each Tensogram object is a lazy Dask chunk.
Currently the xarray backend supports Dask through the standard
BackendArray mechanism, but a native Dask integration would support
finer-grained parallelism (per-range chunks, not just per-object) and
better integration with Dask's distributed scheduler.

### G3. Apache Arrow Integration

Tensogram objects can be mapped to Arrow arrays (numeric arrays with
metadata). An `to_arrow()` method on decoded objects would enable
zero-copy interop with the entire Arrow ecosystem: Parquet, DuckDB,
DataFusion, Polars, etc. Arrow's tensor extension type maps naturally
to Tensogram's N-dimensional objects.

This opens Tensogram data to SQL-style query tools without any
conversion step.

*Note: Arrow's buffer model supports the same zero-copy-from-external-buffer pattern as NumPy's
`frombuffer`.  When implementing `to_arrow()`, ensure that plain-layout objects (no compression,
no filter, native byte order) produce Arrow arrays that view the original wire bytes directly
rather than copying into new Arrow-managed memory — analogous to the zero-copy NumPy
deserialisation idea in `IDEAS.md`.  This requires the `RawMessage` idea (also in `IDEAS.md`) so
that the source buffer has a Python object lifetime that Arrow's buffer can anchor to.*

### G4. ONNX Runtime Direct Feed

If an ONNX model expects a specific input tensor shape and dtype, and
a Tensogram message carries exactly that tensor, a direct
`tensogram.decode_into_onnx_input(buf, session)` binding that
decodes directly into the pre-allocated ONNX Runtime input buffer
without an intermediate NumPy array. Zero-copy ML inference from
Tensogram messages.

Relevant wherever ML models are deployed as ONNX graphs and consume
Tensogram-encoded tensor inputs (AI weather models, medical-imaging
classifiers, genomics feature pipelines, etc.).

### G5. Kerchunk / Virtualizarr Compatibility

[Kerchunk](https://fsspec.github.io/kerchunk/) and
[VirtualiZarr](https://virtualizarr.readthedocs.io) build virtual
Zarr stores over arbitrary file formats by describing byte ranges.
A Kerchunk plugin for Tensogram would make `.tgm` files readable by
the entire Zarr/xarray ecosystem *without* the Tensogram Python
package — just the byte-range descriptions and a standard Zarr reader.

This is the maximum-compatibility path for external adopters who
already use Zarr tooling.

---

## H — Operational and Infrastructure Ideas

### H1. Tensogram Metrics and Observability

A structured observability layer: encode/decode/validation operations
emit metrics (latency, bytes, compression ratio, object count) to a
configurable sink (Prometheus, OpenTelemetry, or just logs). The
`tracing` crate instrumentation is already in place — the missing piece
is structured metric export with well-defined metric names.

For a production pipeline processing millions of messages per day,
this is essential for capacity planning and anomaly detection.

### H2. Differential/Incremental Updates

A Tensogram *patch message* that describes changes to a previous
message (identified by UUID from lineage, B1): which objects changed,
which metadata keys changed, and optionally a binary diff of the
payload. Consumers who have the base message reconstruct the new one
by applying the patch.

Relevant for ensemble forecasts where most members share large amounts
of data, or for rolling updates where only a few fields change between
cycles.

### H3. Encryption at Rest

An encrypted Tensogram variant where payload bytes are encrypted with
a symmetric key (AES-256-GCM), with the key ID stored in `_extra_`.
The key management is external (a KMS), but Tensogram defines the
envelope: frame type, key ID field, nonce, authentication tag.

Relevant for classified forecast products or commercially sensitive
derived data.

---

## The Central Bet

All of the above ideas share a common enabling assumption: that
**Tensogram is adopted broadly across scientific-computing
communities**, not just in any one institution or domain. The
technical quality of the format is already there. The things that
would make that real:

1. **Zero-install browser explorer (E1)** — makes the format tangible
   to anyone in seconds, no installation required.
2. **Conformance test suite (F2)** — signals that this is a real
   standard, not a Rust library with documentation.
3. **Julia and R bindings (F4)** — reaches the scientific-computing
   communities most likely to adopt a new format.
4. **TIP process (F1)** — signals that outside voices shape the roadmap.
5. **DLPack / Arrow integration (G1, G3)** — plugs into the ML and
   data-tool ecosystems that have momentum right now.

The Tensogram wire format is elegant, the implementation is robust,
and the vocabulary-agnostic design means no domain-specific vocabulary
is baked into the bytes. The format is genuinely general-purpose. The
question is whether the surrounding ecosystem (tooling, bindings,
community process) gets built out to match.
