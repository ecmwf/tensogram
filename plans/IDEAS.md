# Ideas

Speculative ideas for possible future work. Not yet accepted. Do not
implement until promoted to `TODO.md`.

## Tools

- [ ] `tensogram filter` subcommand (v2 rules engine)
- [ ] Inspectors/Viewers: Command-line or GUI tools to peer inside the file without writing a custom script. -- a TUI for tensograms?

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
