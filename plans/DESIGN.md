# Design: Tensogram — Binary N-Tensor Message Format Library

Repo: ecmwf/tensogram

> For **why** Tensogram exists and what problem it solves, see `MOTIVATION.md`.
> For the **wire format specification**, see `WIRE_FORMAT.md`.
> For **release history and merged-but-unreleased work**, see `../CHANGELOG.md`.

## Design Premises

1. **CBOR for all metadata** — free-form string keys, strongly-typed values (RFC 8949). The library is vocabulary-agnostic; domain vocabularies (MARS at ECMWF, CF conventions, BIDS in neuroimaging, DICOM in medical imaging, in-house taxonomies) are the application layer's concern. CBOR chosen for flexibility over fixed binary structs because messages are tens to hundreds of MiB where CBOR parsing overhead is negligible compared to payload decode. The CBOR metadata frame has **no required top-level keys**; the wire-format version lives exclusively in the 24-byte preamble (see `WIRE_FORMAT.md` §3) and is never written to CBOR. A stray legacy `"version"` key from a pre-0.17 producer is tolerated and routed into `_extra_` on decode.

2. **Self-describing messages, externally-managed vocabulary** — each message carries enough CBOR metadata to be interpreted, but the library doesn't validate or interpret vocabulary semantics.

3. **Encoding pipeline per object** — tensor payloads flow through a configurable per-object pipeline: encode → filter → compress. Each step is independently selectable and fully described in the DataObjectDescriptor metadata.

4. **Frame-based message framing** — preamble opens with `TENSOGRM` (8 bytes), postamble closes with `39277777` (8 bytes). Internal structure uses typed frames (`FR` + type marker / `ENDF`). Mixed-pattern terminator for corruption resistance. Each message is independently parseable. Multiple messages are appendable to files.

5. **Self-contained data objects** — each data object frame carries its own CBOR DataObjectDescriptor alongside the binary-encoded payload. No external index is needed to decode an individual object.

6. **Minimise memory allocations** — Tensogram should minimise large memory allocations or unnecessary decoding of data where possible. Decoding of actual data into tensors should be delayed until absolutely necessary (when data is actually accessed for caller usage). Use the metadata for dims, sizes, and shapes to prepare lazy objects where necessary. Zero-copy where possible (mmap, buffer iterators), streaming encode without full-message buffering. See the [Memory Strategy](#memory-strategy) section below for the concrete implementation of this premise.

## Approach Decision

Three approaches were considered:

- **A: Pure C99** (tinycbor + libaec) — maximum portability but tedious buffer management and no higher-level language support.
- **B: C++17 core with C API** — RAII buffer management, easy extensibility; comfortable for teams already invested in C++ build infrastructure. Adds C++ build complexity.
- **C: Rust core with multi-language APIs** — memory safety guaranteed by compiler, excellent tooling, exposes Rust native + C FFI + Python (PyO3) APIs.

**Chosen: Approach C (Rust core).** The value of memory safety for a binary format library running in production pipelines outweighs adoption friction. The multi-language API surface (Rust + C FFI for C++ interop + Python via PyO3, plus WebAssembly/TypeScript) covers common scientific-computing language stacks.

**Adoption risk mitigation:** The wire format and CBOR metadata schema are language-agnostic. A reimplementation in C++17 or any other language would produce identical output if it passes the cross-language golden test files shipped alongside the Rust implementation.

## Architecture

### Crate Structure

Default Rust workspace:

- `tensogram` — message encode/decode, CBOR metadata, framing, buffer + file API, iterators, validation (Levels 1-4), remote object-store access
- `tensogram-encodings` — encoding pipeline, filters, compression codecs (all feature-gated, both C-FFI and pure-Rust variants)
- `tensogram-ffi` — C-compatible FFI surface, auto-generated `tensogram.h` via cbindgen
- `tensogram-cli` — CLI binary (`tensogram` command with subcommands)
- `tensogram-szip` — pure-Rust CCSDS 121.0-B-3 szip codec (used via the `szip-pure` feature)
- `tensogram-sz3` — high-level Rust API for SZ3
- `tensogram-sz3-sys` — clean-room FFI shim wrapping the BSD-licensed SZ3 C++ library (Apache-2.0 / MIT)
- `tensogram-benchmarks` — benchmark suite (under `rust/benchmarks/`)
- `tensogram-rust-examples` — runnable Rust examples (under `examples/rust/`)

Excluded from the default workspace (opt-in):

- `python/bindings` (Cargo `tensogram-python`, PyPI `tensogram`) — PyO3/maturin Python bindings with NumPy integration
- `tensogram-grib` — GRIB-to-Tensogram converter via ecCodes (needs `libeccodes`)
- `tensogram-netcdf` — NetCDF-to-Tensogram converter via libnetcdf (needs `libnetcdf`)
- `tensogram-wasm` — WebAssembly bindings via `wasm-pack`

Separate pure-Python packages (not part of the Cargo workspace):

- `tensogram-xarray` — xarray backend engine
- `tensogram-zarr` — Zarr v3 store backend

See `ARCHITECTURE.md` for the full layout and build recipes.

### Core API — Buffer Interface (Primary)

The buffer interface operates on in-memory byte slices — the most common usage path for pipeline components receiving data over the network or from shared memory.

- `encode(metadata, data_objects) → Vec<u8>` — encode a complete message
- `decode(bytes) → (metadata, data_objects)` — decode all data objects
- `decode_metadata(bytes) → metadata` — decode only GlobalMetadata, skip payloads
- `decode_object(bytes, index) → data_object` — decode a single object by index (O(1) via index frame)
- `decode_range(bytes, object_index, ranges, options) → Vec<Vec<u8>>` — decode partial sample ranges. Returns one buffer per range. Python bindings add an extra `join` parameter for concatenation.
- `scan(bytes) → Vec<(usize, usize)>` — scan a multi-message buffer, returning (offset, length) pairs

### Core API — File Interface

- `File::open/create(path) → TensogramFile` — open or create a file
- `file.message_count()` — count messages without decoding (lazy scan on first call)
- `file.read_message(index)` — random access to raw message bytes by index
- `file.decode_message(index)` — decode a specific message by index
- `file.append(metadata, data_objects)` — encode and append a message

### Core API — Streaming Encoder

`StreamingEncoder<W: Write>` writes frames progressively to any sink without buffering the full message in memory. Writes header metadata immediately, accepts objects one-at-a-time, then writes footer index/hash on `finish()`.

### CLI Tool — `tensogram`

A single binary with subcommands, inspired by ecCodes' grib_* tools.

Common options:
- `-w key=value` — where-clause filter (supports `!=`, `/` for OR, dot-notation for namespaced keys)
- `-p key1,key2,...` — select metadata keys to display
- `-j` — JSON output

| Subcommand | Description |
|-----------|-------------|
| `info <file>` | File-level summary: message count, size, version |
| `ls [options] <files>` | List metadata in tabular format |
| `dump [options] <files>` | Full dump of all metadata keys and object descriptors |
| `get [options] <files>` | Extract specific key values (strict: errors on missing key) |
| `set -s key=val,... <in> <out>` | Modify metadata (immutable key protection for shape/dtype/encoding/hash) |
| `copy [options] <in> <out>` | Copy/split with `[keyName]` filename placeholders |
| `merge <files>... -o <out> [--strategy first\|last\|error]` | Merge messages from multiple files, with per-key conflict resolution |
| `split [options] <in> <out>` | Split multi-object messages into singles |
| `reshuffle [options] <in> <out>` | Convert streaming to random-access mode |
| `validate [--quick\|--checksum\|--full] [--canonical] [--json] <files>` | Validate `.tgm` files (Levels 1-4) |
| `convert-grib [--all-keys] <in> -o <out>` | GRIB-to-Tensogram conversion (feature-gated on `grib`) |
| `convert-netcdf [--cf] [--split-by file\|variable\|record] <in> -o <out>` | NetCDF-to-Tensogram conversion (feature-gated on `netcdf`) |

`convert-grib` and `convert-netcdf` share a common pipeline flag set
(`--encoding/--bits/--filter/--compression/--compression-level`) via
`tensogram::pipeline::apply_pipeline`, so both converters produce
byte-identical descriptors for the same options.

## Cross-Language Interface Symmetry

**Goal: every language binding exposes the same user-facing capabilities, with
the Rust core (`rust/tensogram/src/lib.rs`) as the single reference.** A user
should be able to reach for tensogram in Rust, C, C++, Python, TypeScript, or
Fortran and find the *same* feature set, differing only in idiom (e.g. Python
`Mapping`, TS `Promise`, C out-params). Where a capability genuinely cannot be
expressed in a language, the omission must be **explicit and documented here** —
not an accident.

Symmetry is a first-class requirement, not a nice-to-have: divergence is how the
surface rots. Bindings grew organically around their primary consumers (Python
for data science, TS for web/remote got the deepest investment; C/C++ a broad
sync+async core; Fortran a minimal sync core), and without an enforced contract
each binding covered only what its author needed. The 0.24.0 metadata-access
parity work fixed *one* feature area across all six languages; this section
extends that discipline to the *whole* surface.

### The C ABI is the symmetry bottleneck for the C-family

Python and TypeScript bind the Rust **core** directly (PyO3 / wasm-bindgen), so
they can expose anything the core has. C, C++, and Fortran go through the C ABI
(`rust/tensogram-ffi`): **C++ and Fortran can only be as complete as the C
ABI.** Consequently many C/C++/Fortran gaps are not binding bugs — they are
capabilities that were never lowered into the FFI. Closing C-family gaps usually
means *first* widening the C ABI, then wrapping it.

### Gap taxonomy

Every missing cell is classified as one of:

- **[O] Omission** — the backend it binds (Rust core for Py/TS; the C ABI for
  C++/Fortran) already provides it; the binding just never exposed it. **Fixable
  in the binding.** This is the default and the largest bucket.
- **[B] Backend gap** — the layer it binds does not provide it either (e.g. the
  C ABI has no `scan_file`, no typed validation model, no convert, no sync
  remote). **Fix upstream first** (usually the FFI), then wrap.
- **[L] Language limit** — cannot be expressed idiomatically in that language.
  **Accepted; must be listed under "Documented exceptions" below.**

### Feature × language matrix (audited 0.24.0)

Legend: ● full · ◐ partial · ○ absent · — not in the core crate (sibling-crate
or CLI-only). Feature-gated capabilities (async, remote, mmap) are still `●`
where the reference provides them; gating is noted in prose. Reference = Rust
core.

| # | Capability | Rust | C | C++ | Python | TS | Fortran |
|---|------------|:----:|:-:|:---:|:------:|:--:|:-------:|
| 1 | Encode (buffer) | ● | ● | ● | ● | ● | ● |
| 2 | Encode pre-encoded | ● | ● | ● | ● | ● | ○ |
| 3 | Encode options (backend/threads/agg-hash/byte-order/SP-bits) | ● | ◐ | ◐ | ● | ◐ | ◐ |
| 4 | Decode (full) | ● | ● | ● | ● | ● | ● |
| 5 | Decode variants (object/range/descriptors/with-masks/metadata) | ● | ◐ | ◐ | ● | ◐ | ○ |
| 6 | Metadata read (cursor + dot-path) | ● | ● | ● | ● | ● | ● |
| 7 | Metadata build/write | ● | ● | ● | ● | ● | ◐ |
| 8 | Object access (shape/dtype/strides/byte-order/hash) | ● | ● | ● | ● | ● | ◐ |
| 9 | Dtype incl. exotic (f16/bf16/complex) | ● | ● | ◐ | ● | ● | ◐ |
| 10 | File API | ● | ◐ | ● | ● | ● | ● |
| 11 | Streaming encoder | ● | ● | ● | ● | ● | ◐ |
| 12 | Streaming decode/consumer | ● | ◐ | ◐ | ● | ● | ◐ |
| 13 | Async streaming (feat) | ● | ● | ◐ | ◐ | ● | ○ |
| 14 | Iterators (messages/objects/objects_metadata) | ● | ◐ | ◐ | ◐ | ◐ | ○ |
| 15 | Scan (buffer/file/options/inline-hashes) | ● | ◐ | ◐ | ◐ | ◐ | ○ |
| 16 | Hash (compute + inline-hash read) | ● | ● | ◐ | ◐ | ◐ | ◐ |
| 17 | Validate (message/file/buffer, typed report) | ● | ◐ | ◐ | ● | ● | ○ |
| 18 | Doctor (diagnostics) | ● | ● | ○ | ● | ● | ○ |
| 19 | Remote access (feat) | ● | ◐ | ◐ | ● | ● | ○ |
| 20 | Convert GRIB/NetCDF | — | ○ | ○ | ● | ○ | ○ |
| 21 | Masks (encode + `decode_with_masks` introspection) | ● | ◐ | ◐ | ● | ◐ | ○ |
| 22 | Version (wire + package) | ● | ◐ | ◐ | ● | ● | ○ |
| 23 | Error handling (typed) | ● | ◐ | ● | ● | ● | ● |
| 24 | Wire/framing introspection (frame type/flags/layout) | ● | ◐ | ◐ | ◐ | ◐ | ○ |

Where the bindings stand today: **Python** and **TypeScript** are the most
complete (Python uniquely ships GRIB/NetCDF convert and `decode_with_masks`; TS
exceeds the reference in remote, dtype view classes, and typed errors).
**C/C++** have a broad sync+async core but are stringly-typed for options and
inherit every C-ABI backend gap. **Fortran** is the smallest — a clean
synchronous core that binds only ~36% of the C ABI; the overwhelming majority of
its gaps are plain **[O]** omissions (iterators, scan, validate, doctor, hash
utilities, masks, version, pre-encoded, decode variants, object descriptor
accessors, the `threads` argument), not language limits.

### Backend gaps to close first (the C ABI / core is the blocker)

These block the whole C-family (and sometimes TS) and must be fixed upstream
before wrapping:

- **Convert (GRIB/NetCDF)** lives only in sibling crates + the CLI — no core/FFI
  surface. Python reaches the crates directly; C/C++/TS/Fortran cannot.
- **C ABI missing:** `decode_descriptors`, `decode_with_masks` (+ mask-set
  introspection `DecodedMaskSet`/`MaskDescriptor`/`MasksMetadata`),
  `decode_range_from_payload`; `scan_file`/`scan_with_options`/`ScanOptions`/
  `data_object_inline_hashes`; `validate_buffer` and the **typed** validation
  model (`ValidationReport`/`ValidationIssue`/`IssueCode`/…, currently JSON-only);
  `compute_common`, `verify_canonical_cbor`; sync remote + `is_remote_url` +
  `RemoteScanOptions` (only *async* remote open exists); typed wire introspection
  (`FrameType`, `MessageFlags`, `MessageLayout`); typed `Dtype`, `ByteOrder`,
  `AggregateHashPolicy`, `CompressionBackend` (all stringly-typed today).

### Documented exceptions (accepted asymmetries — [L] language limits)

These are the *only* sanctioned gaps. Everything else is a bug to fix.

- **Fortran — async (cat 13, 19):** no coroutines/futures/closures/event loop in
  Fortran 2008. The C ABI async *joins* are blocking and technically bindable,
  but a Fortran caller can only immediately block-join, which collapses to the
  sync API it already has. Accepted omission.
- **Fortran — `float16`/`bfloat16` dtypes (cat 9):** no native half-precision
  type; would require raw `int16` bit-manipulation with no arithmetic. Accepted.
- **Fortran — unsigned dtypes `uint8/16/32/64` (cat 9):** Fortran has no unsigned
  integer kind; round-tripping unsigned *tensors* is semantically lossy.
  Accepted (scalar bit-pattern smuggling via `as_uint`→i64 is provided where it
  makes sense).
- **Fortran — metadata build beyond `base` (cat 7):** no stdlib JSON; the native
  builder is a zero-dependency escaper limited to `base` entries. MARS/`_extra_`
  must be authored as raw JSON. Accepted convenience-layer limit.
- **TypeScript / WASM — `threads` (cat 3/4):** WASM is single-threaded; the
  `threads`/`parallel_threshold` knobs are inert. The *option surface* should
  still be accepted (and ignored) for source symmetry, but true parallelism is a
  platform limit.
- **C / C++ — typed value cursor vs Python/TS native:** Python returns native
  `dict`/`list`; TS returns plain objects. This is idiomatic divergence, not a
  gap — the *capability* (existence, typed read, nesting) is symmetric.

### Interface defects found in the 0.24.0 audit (bugs, not by-design)

Track and fix (see `plans/TODO.md`):

- **C ABI:** the header documents `tgm_last_error_object_index()` (in the
  `TGM_ERROR_MISSING_HASH` note) but the function **does not exist** — the
  offending-object index is unreachable from C. Either add it or fix the doc.
- **Python:** `DataObjectDescriptor.hash` is a permanent `None` stub whose
  docstring points to `Message.object_inline_hashes()`/`Message.object_hash(i)`,
  but `Message` is a bare 2-field `namedtuple` with **no such methods** — reading
  v3 inline hashes is impossible from Python (dangling docs → dead end).
- **Python:** `AsyncStreamingEncoder` (async streaming *encoder*) is entirely
  absent, though the async *decode* surface is rich — a surprising [O] gap.
- **TypeScript:** `TensogramFile.append` declares `allowNan`/`allowInf`/
  `*MaskMethod`/`smallMaskThresholdBytes` in `AppendOptions` but `file.ts`
  forwards **only `hash`** to `encode()` — the mask options are silently dropped.
- **C ABI:** `tgm_doctor_to_json` exists but has **no C++ wrapper** (whole
  `doctor` category missing from C++), and is unused by any C++ test/example.

### Symmetry discipline (enforced going forward)

- When you **add or change a user-facing capability**, mirror it across **all**
  bindings in the same PR, or file follow-up gaps with the [O]/[B]/[L]
  classification. The Rust core is the reference for names and semantics.
- Prefer **widening the C ABI** for any capability the C-family lacks — that
  unblocks C, C++, and Fortran at once.
- Every capability must have a **runnable example in every language that
  supports it** (`examples/<lang>/`), and the per-language example set should
  cover the full public surface (see `# Examples` in AGENTS.md). The audit found
  large example-coverage holes (e.g. the entire precise metadata cursor is
  exercised by *no* C++/Python/Rust example; exotic dtypes by none).
- This matrix is re-audited each release; the accepted-exceptions list above is
  the sole source of truth for "intended" asymmetry.

## Key Design Decisions

### Per-Object Metadata

- **GlobalMetadata** (in header/footer metadata frame):
  - `base` — per-object metadata array, one entry per data object, each entry holds ALL structured metadata for that object independently (no tracking of commonalities)
  - `_reserved_` — library internals (provenance: encoder info, time, uuid). Client code can read but MUST NOT write.
  - `_extra_` — client-writable catch-all for ad-hoc message-level annotations, plus a catch-all for any unknown top-level CBOR keys routed here on decode.
  - The CBOR metadata frame has **no required top-level keys**. The wire-format version lives exclusively in the preamble (see `WIRE_FORMAT.md` §3) — it is never written to CBOR.

- **DataObjectDescriptor** (per data object frame) — encoding parameters only:
  - Tensor description: `obj_type`, `ndim`, `shape`, `strides`, `dtype`, `byte_order`
  - Encoding pipeline: `encoding`, `filter`, `compression`
  - Encoding-specific parameters (in `params` map)
  - Optional integrity `hash`

Application metadata (for example MARS keys, CF attributes, BIDS entities, or any domain-specific namespace) lives in `base[i][<namespace>]` per-object. The encoder auto-populates `_reserved_.tensor` (ndim/shape/strides/dtype) in each `base[i]` entry. The descriptor carries only what's needed to decode the payload. A `compute_common()` utility can extract shared keys from base entries when needed (e.g. for display or merge operations) — commonalities are computed in software, not encoded in the wire format.

### Data Types

15 supported types, unbounded number of dimensions (`ndim` is `u64`):

| dtype | Byte width | Description |
|-------|-----------|-------------|
| `float16` | 2 | IEEE 754 half precision |
| `bfloat16` | 2 | Brain floating point |
| `float32` | 4 | IEEE 754 single precision |
| `float64` | 8 | IEEE 754 double precision |
| `complex64` | 8 | Two float32 (real, imaginary) |
| `complex128` | 16 | Two float64 (real, imaginary) |
| `int8`..`int64` | 1-8 | Signed integers |
| `uint8`..`uint64` | 1-8 | Unsigned integers |
| `bitmask` | 1/8 | Packed bits, MSB-first |

Data objects may exceed 4 GiB — all offsets and lengths are uint64.

### Encoding Pipeline

Each data object specifies an independent pipeline: **encode → filter → compress**. Every step is fully described in the DataObjectDescriptor.

**Encoding:**
- `none` — raw bytes in the logical dtype
- `simple_packing` — GRIB-style lossy quantization: `value = reference_value + 2^E * 10^(-D) * packed_integer`. Supports 0-64 bits per value. NaN / Inf inputs rejected with EncodingError. Descriptor params use the `sp_*` prefix (`sp_bits_per_value`, `sp_reference_value`, `sp_binary_scale_factor`, `sp_decimal_scale_factor`); supplying only `sp_bits_per_value` makes the encoder auto-compute the reference value and scale factors and stamp the full set, so the encoded message stays self-describing.

**Filters:**
- `none` — no pre-processing
- `shuffle` — byte-level shuffle (HDF5-style). Groups first byte of every element together, etc. Dramatically improves compressibility for float types.

**Compression (6 codecs, all feature-gated):**

| Codec | Type | Partial Range Decode | Notes |
|-------|------|---------------------|-------|
| `szip` | Lossless | Yes (RSI block offsets) | CCSDS 121.0-B-3 via libaec |
| `zstd` | Lossless | No (stream) | Excellent ratio/speed tradeoff |
| `lz4` | Lossless | No (stream) | Fastest decompression |
| `blosc2` | Lossless | Yes (SChunk random access) | Multi-codec meta-compressor |
| `zfp` | Lossy | Yes (fixed-rate mode) | Purpose-built for floating-point |
| `sz3` | Lossy | No (stream) | Error-bounded scientific compression |

**Important limitation:** Shuffle and partial range decode do not compose. Byte rearrangement means RSI/chunk block boundaries don't correspond to contiguous sample ranges. Partial range decode works for: `simple_packing+szip`, `blosc2`, `zfp` (fixed-rate), and uncompressed data.

### Integrity Hashing

- **Algorithm:** xxh3 (64-bit, non-cryptographic, ~30 GB/s)
- **Scope:** Per-object payload, computed over final encoded bytes (after full pipeline)
- **Encoding:** Hash computation is on by default (negligible overhead vs compression).  The hash is produced **inline with encoding** — the pipeline drives an `Xxh3Default` streaming hasher in lockstep with codec output, so the encoded payload is walked exactly once for both encoding and hashing.  `encode_pre_encoded` hashes the caller's opaque bytes in a single pass (no pipeline to fuse with).
- **Decoding:** Hash verification is **off by default** and **opt-in** via `DecodeOptions::verify_hash = true` (or its equivalent kwarg/field in every binding).  Off by default because most transports already provide error correction (TCP, HTTPS, object-store ETag validation); enable it when the consumer wants end-to-end integrity.

  Verification is fused with decode — bytes are hashed while hot in cache/buffer, so the cost is one extra walk over the post-encoding payload (typically a small fraction of decode time, dominated by decompression).  This is materially cheaper than running `tensogram validate --checksum` as a pre-flight pass and then decoding, which reads every byte twice.

  **Per-frame contract.** Hash presence is signalled by the `FrameFlags::HASH_PRESENT` bit in the frame header's `flags` field (bit 1; common to all frame types — see `plans/WIRE_FORMAT.md` §2.5).  When set, the inline 8-byte slot in the frame footer holds the xxh3-64 digest of the frame body, including legitimate zero digests.  When clear, the slot is *undefined* — encoders write `0` by convention, but decoders MUST NOT inspect the value.  The preamble's `MessageFlags::HASHES_PRESENT` is retained as a coarse-grained advisory ("every frame in this message has a hash" by encoder invariant); per-frame `HASH_PRESENT` is authoritative for individual decoder decisions.

  **Strict-input rules** (errors include `object_index` so the caller can act on the failure):
  * `verify_hash=true` and `HASH_PRESENT` is clear on object *i* → `TensogramError::MissingHash { object_index: i }`.
  * `verify_hash=true` and `HASH_PRESENT` is set on object *i* but the slot disagrees with the recomputed digest → `TensogramError::HashMismatch { object_index: Some(i), expected, actual }`.
  * `verify_hash=false` (default) → no verification, decode is pure deserialisation; the per-frame flag and the slot value are both ignored.

  **`decode_range` is unverified by construction.**  The inline hash covers the whole post-encoding payload of the source frame; verifying it would require reading every byte that the range-decode optimisation is designed to avoid.  No binding's `decode_range` accepts a `verify_hash` flag.  When integrity matters, use `decode_object(buf, idx, { verify_hash: true })` — that path materialises the full body anyway, so the verification is free.
- **Threading invariant:** hashing runs in the calling thread *after* any intra-codec parallelism (axis B) has joined, and each object's hasher is owned by one thread (axis A).  Transparent codecs produce byte-identical hashes across thread counts; opaque codecs (blosc2, zstd with workers) hash their worker-completion-ordered output and round-trip losslessly.  This matches the determinism contract of the multi-threaded pipeline itself.

### Multi-Threaded Pipeline

Encoding and decoding accept an optional `threads: u32` budget on
`EncodeOptions` / `DecodeOptions`, off by default (the env var
`TENSOGRAM_THREADS` fills in when `threads = 0`).  `threads = 0` matches
the sequential path **byte-for-byte**, so golden files are unchanged.

Work is dispatched along one of two axes, chosen once per call from the
object descriptors:

- **Axis B (intra-object)** — parallelise *within* an object's codec
  (chunked `simple_packing`, byte-plane `shuffle`, blosc2 / zstd worker
  threads).  Preferred so that a few very large objects still scale.
- **Axis A (inter-object)** — parallelise *across* objects; the fallback
  when no object has an axis-B-friendly codec, to avoid N×M thread
  over-subscription.

**Determinism contract.** Transparent codecs (`none`, `lz4`, `szip`,
`zfp`, `sz3`, `simple_packing`, `shuffle`) produce **byte-identical**
output at any thread count.  Opaque codecs (`blosc2`, `zstd` with
workers) may emit different compressed bytes — block boundaries land in
worker-completion order — but always **round-trip losslessly**.  Hashing
follows the same rule (see the Integrity Hashing threading invariant
above).  This is what lets the hash-while-encoding fusion and the
golden-file tests stay valid across thread counts.

### Per-Object Byte Order

Each data object specifies `byte_order` (`big` or `little`) in the wire format, declaring the endianness of the stored payload bytes. Framing fields remain big-endian (network byte order).

**On decode**, the library **automatically converts decoded bytes to the caller's native byte order** by default (`native_byte_order: true` in `DecodeOptions`). Callers never need to inspect `byte_order` or manually byteswap — they can use `from_ne_bytes()`, `data_as<T>()`, or numpy arrays directly. The `native_byte_order: false` opt-out returns raw wire-order bytes for zero-copy forwarding.

**On encode**, callers provide data in their native byte order. The `byte_order` field in the descriptor should match the byte order of the provided bytes (defaults to native in Python bindings).

### Deterministic CBOR

All CBOR output uses RFC 8949 Section 4.2 deterministic encoding with canonical key ordering. Two messages with identical content produce byte-identical bytes, enabling deduplication and cross-language golden file tests.

### Strided Memory Layout

Each data object describes its tensor via `shape` + `strides`, separating logical structure from memory layout (as in NumPy/PyTorch/ndarray). Flat index: `sum(I[k] * strides[k] for k in 0..ndim)`. Enables non-contiguous views, transposed tensors, and sliced sub-tensors without copying.

## Version Compatibility

- Version is a single unsigned 16-bit integer carried **only in the 24-byte
  preamble** (see `WIRE_FORMAT.md` §3).  Currently `3`.  The CBOR metadata
  frame is free-form and never carries `version`.
- **Minor evolution (same version):** New CBOR keys or encoding types don't bump the version. Decoders ignore unknown keys (including a stray legacy `"version"` in old CBOR, which is routed to `_extra_`) and reject unknown encodings gracefully.
- **Version bump:** Only for wire-format structural changes. Decoders reject unrecognized preamble versions.
- **No backward compatibility obligation across version bumps.** The preamble version field exists to fail fast.

## Error Handling & Recovery

- **Framing validation:** Decoder verifies `TENSOGRM` magic. If missing, scanner advances byte-by-byte to find next marker (skip-to-next-marker recovery).
- **Corrupted messages:** Inconsistent total_length or missing end magic → message rejected, scanner continues.
- **Partial reads:** Messages without postamble are never valid. Partial messages at end-of-stream are reported as truncated.
- **Unknown CBOR keys:** Decoders MUST ignore unknown keys (forward compatibility). Unknown encoding types are rejected with clear error.
- **Frame-level corruption:** Each frame's FR/ENDF markers and length are verified. Corrupted object frames are rejected but other objects remain accessible via index frame.
- **Hostile descriptor sizes:** every decode-path allocation derived from a descriptor's claimed tensor size is *fallible* (`try_reserve`), and every size multiplication is checked (`u128` promotion / `checked_mul`). A malformed descriptor claiming a terabyte-scale tensor returns a structured error instead of aborting the process via OOM. No static size cap is imposed — legitimate multi-GiB objects (ERA5, ML weights) must pass — so fallible allocation, not a ceiling, is the guard.
- **Descriptor ↔ payload size consistency:** for the two decode pipelines whose decoded size is *exactly* determined by the descriptor — uncompressed `encoding=none` (`num_values × dtype_width`, or `ceil(num_values / 8)` for `bitmask`) and uncompressed `encoding=simple_packing` (`ceil(num_values × bits_per_value / 8)`) — the decoder checks the claimed size against the actual payload length **before** any decode work and rejects a mismatch in either direction (`PipelineError::DescriptorSizeMismatch`). This catches truncation, trailing junk, and hostile descriptors structurally, with a precise error, and never allocates for an impossible size. **Compressed codecs are deliberately not gated by a size heuristic.** Their decoded size is not derivable from the payload without decompressing, and a fixed compression-ratio ceiling was rejected as a design option: scientific data (constant fields, all-zero masks) can be legitimately compressible past any chosen ratio, so a ceiling risks false-rejecting valid operational data — a worse failure than the one it prevents. The fallible-allocation guard above is the backstop for compressed pipelines: the contract is "fail gracefully with a correct error", not "predict the decoded size".
- **API error surface:** All functions return Result (Rust), error codes (C FFI), exceptions (C++/Python) with categories: Framing, Metadata, Encoding, Compression, Object, IO, HashMismatch.

## Testing Strategy

- **Round-trip correctness:** Bit-exact for lossless, quantization tolerance for lossy
- **Cross-language golden files:** canonical `.tgm` files decoded identically by Rust, C++, Python, and TypeScript
- **Multi-object, multi-message, multi-dtype** parametrized tests
- **Corruption injection:** Framing marker corruption, payload corruption, hash mismatch
- **Decode-time integrity:** every binding asserts the same `MissingHash` / `HashMismatch` semantics under `verify_hash=true`, including the `object_index` payload that names *which* object failed
- **Edge cases:** Zero-object messages, scalars (ndim=0), non-byte-aligned bit packing, >4 GiB offsets, shuffle + partial range rejection
- **CI:** All language test suites run on every commit

## Encoding Capability Note

Producers must not assume receivers support encodings, filters, or compressions added after the initial release. These are capability concerns, not version concerns. Out-of-band capability negotiation (or sticking to baseline encodings) is the producer's responsibility.

## Dependencies

- **ciborium** — CBOR encode/decode (RFC 8949)
- **serde** — serialization framework
- **thiserror** — error derive macros
- **xxhash-rust** — xxh3 payload integrity hashing (pure Rust)
- **libaec-sys** — szip compression (CCSDS 121.0-B-3)
- **zstd** — Zstandard compression
- **lz4_flex** — LZ4 compression (pure Rust)
- **blosc2** — Blosc2 meta-compressor
- **zfp-sys-cc** — ZFP floating-point compression
- **sz3** — SZ3 error-bounded compression
- **clap** — CLI argument parsing
- **PyO3 + maturin** — Python bindings
- **cbindgen** — C header generation

## Memory Strategy

Tensogram minimises large memory allocations as a strategic design choice.
Decoding of actual data into tensors is delayed until absolutely necessary.

### Current Allocation Patterns

| Operation | Allocation Strategy |
|-----------|-------------------|
| `decode()` | Decodes all objects into owned `Vec<u8>`. Pipeline uses `Cow<[u8]>` — zero-copy when encoding=none, filter=none, compression=none. |
| `decode_metadata()` | Parses only the CBOR metadata frame. Does not touch payload bytes. |
| `decode_descriptors()` | Reads metadata + per-object CBOR descriptors. No payload decode. |
| `decode_object()` | Decodes a single object by index. Other objects' payloads are skipped. |
| `decode_range()` | Decodes a sub-range of a single object. Avoids full payload decode when possible. |
| `scan()` / `scan_file()` | Scans message boundaries by reading magic/terminator bytes without decoding payloads. Worst-case time is linear in the scanned buffer/file region, with a fast-path when total message length is available. |
| `TensogramFile::read_message()` | Reads raw bytes for one message. |
| `TensogramFile::open()` | Reads file metadata only. Message data stays on disk. |
| Mmap (`feature = "mmap"`) | Memory-mapped I/O — no buffer allocation for file contents. OS pages in on demand. |
| `iter_messages()` | Copies the input buffer. For large files, use `TensogramFile` iteration instead. |
| Streaming encoder | Writes frames directly to the output. No full-message buffer. |

### xarray / Zarr Lazy Loading

- **xarray backend**: `BackendArray` wraps tensogram decode. Data is read lazily on `.values` access, not at `open_dataset()` time. Slice-to-range mapping converts N-D array slices to flat byte ranges for partial decode.
- **Zarr backend**: `TensogramStore` maps Zarr array chunks to tensogram messages. Chunk data is decoded on demand.

### Design Rules

1. **Metadata-only operations** (`decode_metadata`, `decode_descriptors`, `scan`) must never touch payload bytes.
2. **Partial decode** (`decode_object`, `decode_range`) must skip non-requested objects.
3. **Pipeline zero-copy**: when no encoding, filtering, or compression is applied, the decode pipeline borrows the input buffer via `Cow::Borrowed` rather than allocating.
4. **Streaming encode** must never buffer the full message — each object frame is written immediately.
5. **File iteration** opens an independent file handle per iterator — no shared mutable state.

## Distribution

- **Rust crate:** [crates.io](https://crates.io/crates/tensogram)
- **Python package:** PyPI via maturin (wheel with compiled Rust core)
- **C/C++ library:** Static/shared library + headers via CMake
- **CLI binary:** cargo install, GitHub Releases, optionally Homebrew
- **CI/CD:** GitHub Actions for multi-platform builds + automated testing
