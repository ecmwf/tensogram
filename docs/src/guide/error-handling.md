# Error Handling

Tensogram uses typed errors across all language bindings. Every fallible
operation returns a `Result` (Rust), raises an exception (Python / C++ /
TypeScript), or returns an error code (C). No library code panics.

## Error Categories

| Category | Trigger | Rust | Python | C++ | TypeScript | C Code |
|----------|---------|------|--------|-----|------------|--------|
| **Framing** | Invalid magic bytes, truncated message, bad terminator | `TensogramError::Framing` | `ValueError` | `framing_error` | `FramingError` | `TGM_ERROR_FRAMING (1)` |
| **Metadata** | CBOR parse failure, missing required field, schema violation | `TensogramError::Metadata` | `ValueError` | `metadata_error` | `MetadataError` | `TGM_ERROR_METADATA (2)` |
| **Encoding** | Encoding pipeline failure (e.g. NaN in simple\_packing) | `TensogramError::Encoding` | `ValueError` | `encoding_error` | `EncodingError` | `TGM_ERROR_ENCODING (3)` |
| **Compression** | Decompression failure, unknown codec | `TensogramError::Compression` | `ValueError` | `compression_error` | `CompressionError` | `TGM_ERROR_COMPRESSION (4)` |
| **Object** | Invalid descriptor, object index out of range | `TensogramError::Object` | `ValueError` | `object_error` | `ObjectError` | `TGM_ERROR_OBJECT (5)` |
| **I/O** | File not found, permission denied, disk full | `TensogramError::Io` | `OSError` | `io_error` | `IoError` | `TGM_ERROR_IO (6)` |
| **Hash Mismatch** | Payload integrity check fails on `verify_hash=True` | `TensogramError::HashMismatch` | `RuntimeError` | `hash_mismatch_error` | `HashMismatchError` | `TGM_ERROR_HASH_MISMATCH (7)` |
| **Invalid Arg** | NULL pointer or invalid argument at the API boundary | — | `ValueError` | `invalid_arg_error` | `InvalidArgumentError` | `TGM_ERROR_INVALID_ARG (8)` |
| **Remote** | S3 / GCS / Azure / HTTP(S) object-store failure | `TensogramError::Remote` | `OSError` | `remote_error` | `RemoteError` | `TGM_ERROR_REMOTE (10)` |
| **Streaming Limit** | `decodeStream` internal buffer exceeded the configured maximum | — | — | — | `StreamingLimitError` | — |

Notes on the TypeScript column:

- All TypeScript errors extend the abstract `TensogramError` base class,
  so a single `catch (err) { if (err instanceof TensogramError) … }`
  handles every library-raised error.
- `HashMismatchError` in TypeScript additionally carries parsed
  `expected` and `actual` hex digests when the underlying Rust message
  is in the canonical `"hash mismatch: expected X, got Y"` form.
- `StreamingLimitError` is TS-specific and is raised only from
  `decodeStream` when the internal buffer would grow past
  `maxBufferBytes` (default 256 MiB).

## Error Paths by Operation

### Encoding

```
Input data + metadata dict
  │
  ├─ Missing 'version' ──────────► Metadata error
  ├─ Missing 'type'/'shape'/'dtype' ► Metadata error
  ├─ Unknown dtype string ────────► Metadata error
  ├─ Unknown byte_order ──────────► Metadata error
  ├─ Data size ≠ shape × dtype ───► Metadata error
  ├─ Shape product overflow ──────► Metadata error
  ├─ NaN in simple_packing ───────► Encoding error
  ├─ Inf reference_value ─────────► Metadata error
  ├─ Client wrote _reserved_ ─────► Metadata error (message or base[i])
  ├─ base.len() > descriptors ────► Metadata error (extra entries would be lost)
  ├─ emit_preceders in buffered ──► Encoding error (use StreamingEncoder)
  ├─ Param out of range (i32/u32) ► Metadata error (zstd_level, szip_rsi, etc.)
  ├─ Unknown compression codec ───► Encoding error
  ├─ Compression codec failure ───► Compression error
  └─ File I/O failure ────────────► I/O error
```

### Decoding

```
Raw bytes
  │
  ├─ No magic bytes / truncated ──► Framing error
  ├─ Bad frame type codes ────────► Framing error
  ├─ Frame total_length overflow ─► Framing error
  ├─ Frame ordering violation ────► Framing error (header→data→footer)
  ├─ cbor_offset out of range ────► Framing error
  ├─ CBOR parse failure ──────────► Metadata error
  ├─ Preceder base ≠ 1 entry ─────► Metadata error
  ├─ Dangling preceder (no obj) ──► Framing error
  ├─ Consecutive preceders ────────► Framing error
  ├─ base.len() > object count ───► Metadata error
  ├─ Object index out of range ───► Object error
  ├─ Shape product overflow ──────► Metadata error
  ├─ Decompression failure ───────► Compression error
  ├─ Decoding pipeline failure ───► Encoding error
  └─ Hash verification mismatch ──► HashMismatch error
```

### File Operations

```
TensogramFile.open(path)
  │
  ├─ File not found ──────────────► I/O error
  ├─ Permission denied ───────────► I/O error
  └─ Invalid file content ────────► Framing error

TensogramFile.decode_message(index)
  │
  ├─ Index out of range ──────────► Object error / IndexError
  └─ Corrupt message at offset ───► Framing error
```

### Streaming Encoder

```
StreamingEncoder
  │
  ├─ write_preceder(_reserved_) ──► Metadata error
  ├─ write_preceder twice ─────────► Framing error (no intervening write_object)
  ├─ finish() with pending prec ──► Framing error (dangling preceder)
  ├─ write_object invalid shape ──► Metadata error
  ├─ Encoding pipeline failure ───► Encoding error
  ├─ Variable-length hash algo ───► Framing error (see below)
  └─ I/O write failure ───────────► I/O error
```

The streaming path writes the frame header before the payload has been
hashed, so it needs to know the final CBOR descriptor length up front.
This works only when the configured `HashAlgorithm` produces a digest
whose hex representation has a fixed length — currently only `Xxh3`
(always 16 hex chars).  If a future hash algorithm with variable-length
output is used, `StreamingEncoder::write_object` returns
`TensogramError::Framing` **before writing any bytes**, so the caller's
sink is never corrupted.  Use the buffered `encode()` API for such
algorithms.

### CLI Operations

```
set command
  │
  ├─ Immutable key (shape, dtype) ► Error (cannot modify structural key)
  ├─ _reserved_ namespace ────────► Error (library-managed)
  └─ Invalid object index ────────► Error (out of range)

merge command
  │
  ├─ No input files ──────────────► Error
  ├─ Invalid strategy name ───────► Error
  └─ Conflicting keys (error mode) ► Error (use first/last to resolve)

split command
  │
  └─ Single-object: pass through; multi-object: split per-object base metadata
```

### Importer Operations (convert-grib / convert-netcdf)

Both importer crates (`tensogram-grib`, `tensogram-netcdf`) use typed
error enums and never panic on invalid or exotic input. Anything the
importer can't represent cleanly is either surfaced as a typed error
or skipped with a `warning: …` line on stderr so the operator can see
what was dropped.

```
tensogram-netcdf errors (rust/tensogram-netcdf/src/error.rs)
  │
  ├─ NetcdfError::Netcdf(netcdf::Error)
  │     Low-level failure from libnetcdf — file missing, permission
  │     denied, format error, truncated file, HDF5 error.
  │
  ├─ NetcdfError::NoVariables
  │     Input file has zero supported numeric variables after skipping
  │     char/string/compound/vlen. Empty files also hit this.
  │
  ├─ NetcdfError::NoUnlimitedDimension { file }
  │     --split-by=record requested but the file has no unlimited
  │     dimension. Contains the file path for diagnostics.
  │
  ├─ NetcdfError::UnsupportedType { name, reason }
  │     Variable has a type we can't represent (e.g. compound,
  │     enum, opaque, vlen). Currently only the char / string
  │     variants hit this path — the other complex types are
  │     downgraded to a stderr warning and skipped because they
  │     frequently coexist with valid numeric variables.
  │
  ├─ NetcdfError::InvalidData(String)
  │     Catch-all for:
  │       - low-level read errors on a specific variable
  │       - unknown --encoding / --filter / --compression names
  │       - simple_packing compute_params failures on edge-case data
  │       - extract_variable_record invariant violations (should be
  │         unreachable; if it fires the importer is buggy)
  │
  ├─ NetcdfError::Encode(String)
  │     tensogram rejected the pipeline. Common cause:
  │     szip on raw f64 (bits_per_sample=64 exceeds libaec's
  │     32-bit cap). Fix: add --filter shuffle or --encoding
  │     simple_packing first.
  │
  └─ NetcdfError::Io(std::io::Error)
        Reserved for future use — the current importer reads
        through libnetcdf and writes through the CLI wrapper, so
        stdlib I/O errors don't currently reach this variant.
```

Soft warnings (stderr, exit 0):

```
warning: {file}: sub-groups found; only root-group variables are converted
warning: skipping variable '{name}': Char variables are not supported
warning: skipping variable '{name}': complex type Compound(_) is not supported
warning: skipping simple_packing for variable '{name}' (not a float64 payload)
warning: variable '{name}': failed to read attribute '{attr}': {cause}
warning: failed to read global attribute '{name}': {cause}
```

Note: NaN/Inf in a variable that targets `simple_packing` now
**hard-fails** the conversion (see
[NetCDF Importer — simple_packing on Mixed-dtype Files](#netcdf-importer--simple_packing-on-mixed-dtype-files)
below).  The previous "warning: skipping simple_packing ... NaN value
encountered" line no longer fires; that case is an error rather than
a warning.

The last two lines above are rare — they only fire on corrupt
attribute values or unsupported upstream AttributeValue variants —
but they surface instead of dropping data silently so operators can
trace unexpected missing metadata.

```
tensogram-grib errors (rust/tensogram-grib/src/error.rs)
  │
  ├─ GribError::Eccodes(String) — ecCodes C library error
  ├─ GribError::NoMessages — empty GRIB file
  ├─ GribError::MissingKey — required ecCodes/MARS namespace key absent
  ├─ GribError::InvalidShape — grid dimension mismatch
  └─ GribError::Encode — tensogram encode failure
```

## Language-Specific Patterns

### Rust

```rust
use tensogram::{decode, DecodeOptions, TensogramError};

match decode(&buffer, &DecodeOptions::default()) {
    Ok((meta, objects)) => { /* use data */ }
    Err(TensogramError::Framing(msg)) => eprintln!("bad format: {msg}"),
    Err(TensogramError::HashMismatch { expected, actual }) =>
        eprintln!("integrity: {expected} ≠ {actual}"),
    Err(e) => eprintln!("error: {e}"),
}
```

### Python

```python
import tensogram

# Decode errors
try:
    msg = tensogram.decode(buf, verify_hash=True)
except ValueError as e:
    # Framing, Metadata, Encoding, Compression, Object errors
    print(f"decode failed: {e}")
except RuntimeError as e:
    # Hash verification mismatch
    print(f"integrity error: {e}")
except OSError as e:
    # File I/O and Remote (S3/GCS/Azure/HTTP) errors
    print(f"I/O error: {e}")

# File errors
try:
    f = tensogram.TensogramFile.open("missing.tgm")
except OSError:
    print("file not found")

# Index errors
with tensogram.TensogramFile.open("data.tgm") as f:
    try:
        msg = f[999]
    except IndexError:
        print("message index out of range")

# Packing errors
try:
    tensogram.compute_packing_params(nan_array, 16, 0)
except ValueError as e:
    print(f"NaN rejected: {e}")
```

### C++

```cpp
#include <tensogram.hpp>

try {
    auto msg = tensogram::decode(buf, len);
} catch (const tensogram::framing_error& e) {
    // Invalid message structure
    std::cerr << "framing: " << e.what() << " (code " << e.code() << ")\n";
} catch (const tensogram::hash_mismatch_error& e) {
    // Payload integrity failure
    std::cerr << "hash: " << e.what() << "\n";
} catch (const tensogram::error& e) {
    // Any Tensogram error (base class)
    std::cerr << "error: " << e.what() << "\n";
}
```

### C

```c
#include "tensogram.h"

tgm_message* msg = tgm_decode(buf, len, 0);
if (!msg) {
    tgm_error code = tgm_last_error_code();
    const char* message = tgm_last_error();
    fprintf(stderr, "%s (%d): %s\n",
            tgm_error_string(code), code, message);
}
```

> **Note:** `tgm_last_error()` returns a thread-local string valid until the
> next FFI call on the same thread. Copy it if you need to keep it.

### TypeScript

Every error thrown by `@ecmwf.int/tensogram` is an instance of the abstract
`TensogramError` base class. The concrete subclasses match the Rust
variants one-to-one, plus a TS-specific `InvalidArgumentError` and
`StreamingLimitError`.

```ts
import {
  decode,
  TensogramError,
  FramingError,
  HashMismatchError,
  ObjectError,
  StreamingLimitError,
} from '@ecmwf.int/tensogram';

try {
  const { metadata, objects } = decode(buf, { verifyHash: true });
  // ...
} catch (err) {
  if (err instanceof HashMismatchError) {
    // Structured fields are parsed from the Rust-side message.
    console.error('integrity failure:', err.expected, err.actual);
  } else if (err instanceof FramingError) {
    console.error('bad wire format:', err.message);
  } else if (err instanceof ObjectError) {
    console.error('object index error:', err.message);
  } else if (err instanceof TensogramError) {
    console.error('tensogram error:', err.name, err.message);
  } else {
    throw err;
  }
}
```

All concrete classes expose:

- `err.rawMessage` — the untruncated string from the WASM / Rust side,
  including any error-variant prefix (`"framing error: ..."`).
- `err.message` — the human-readable message with the prefix stripped.
- `err.name` — stable string name (`"FramingError"`, etc.).

`HashMismatchError` additionally exposes parsed `expected` and `actual`
hex digests when the underlying message follows the canonical
`"hash mismatch: expected X, got Y"` form.

Streaming decode does **not** throw on a single corrupt message — the
iterator skips and continues. Register an `onError` callback to observe
the skips:

```ts
import { decodeStream, StreamingLimitError } from '@ecmwf.int/tensogram';

try {
  for await (const frame of decodeStream(res.body!, {
    maxBufferBytes: 64 * 1024 * 1024,
    onError: ({ message, skippedCount }) => {
      console.warn(`skipped corrupt message (#${skippedCount}): ${message}`);
    },
  })) {
    render(frame.descriptor.shape, frame.data());
    frame.close();
  }
} catch (err) {
  if (err instanceof StreamingLimitError) {
    // Stream exceeded maxBufferBytes; configure a larger limit or split.
  } else {
    throw err;
  }
}
```

> **Note:** `decodeStream` does throw for infrastructure-level failures
> (buffer limit exceeded, `AbortSignal` fired, non-`ReadableStream`
> input). Only per-message corruption is routed through `onError`.

## Common Error Scenarios

### Garbage or Truncated Input

Any non-Tensogram bytes passed to `decode()` produce a **Framing error**.
The decoder looks for the 8-byte magic `TENSOGRM` and a matching terminator.

### Hash Mismatch After Corruption

**v3 note.** Frame-level integrity moved from the decoder to the
validator.  `verify_hash=True` (Python `DecodeOptions`) or
`TGM_DECODE_VERIFY_HASH` (C) is retained for source compatibility
but is a **no-op on the decode path** in v3.

To detect corruption in a v3 message, run the message through
`tensogram validate --checksum` (CLI), `validate_message` (Rust),
`tgm_validate` (C), or the equivalent Python / TypeScript helpers.
The validator:

1. Walks every frame and recomputes the xxh3-64 of its body
   (payload + masks + CBOR; `cbor_offset`, the hash slot, and
   ENDF are excluded — see `plans/WIRE_FORMAT.md` §2.4).
2. Compares the recomputed digest to the inline hash slot at
   `frame_end − 12`.  A mismatch emits a **HashMismatch**
   validation issue carrying the `expected` and `actual` hex
   values plus the frame offset.
3. When both a `HeaderHash` and a `FooterHash` aggregate frame
   are present, cross-checks them against each other and against
   the inline slots.  Disagreement also surfaces as a
   **HashMismatch**.
4. An `UnknownHashAlgorithm` warning fires when the aggregate
   `HashFrame.algorithm` is not `"xxh3"` — the inline slots are
   still verified (they're authoritative); only the aggregate's
   algorithm identifier is advisory.

Messages encoded with `hash_algorithm=None` clear the
`HASHES_PRESENT` preamble flag and leave every inline slot at
`0x00…00`.  On such messages, `validate --checksum` emits
`NoHashAvailable` at warning level and cannot detect corruption
beyond structural errors — re-encode with `hash_algorithm =
Some(Xxh3)` to enable integrity checking.

### Object Index Out of Range

Accessing `decode_object(buf, index=N)` where N ≥ number of objects
produces an **Object error** (Rust/C/C++) or **ValueError** (Python).
File indexing `file[N]` raises **IndexError** for out-of-range N.

### NaN / Inf in Simple Packing

`compute_packing_params()` rejects both **NaN** and **±Inf** values
with a **ValueError** that includes the index of the first offending
sample. `simple_packing`'s scale-factor derivation has no meaningful
value for non-finite input — rejecting them up front prevents the
silent corruption path where an `i32::MAX`-saturated
`binary_scale_factor` decodes to NaN everywhere.

0.17+ extends this contract to every pipeline: `encoding="none"`
(and every compressor) rejects NaN / ±Inf input by default.  The
[NaN / Inf Handling](nan-inf-handling.md) guide covers the
`allow_nan` / `allow_inf` opt-in that substitutes non-finite values
with `0.0` and records their positions in a bitmask companion
section.

### Malformed Descriptor — Pathological Tensor Size (szip)

A corrupted or hostile `.tgm` file whose tensor descriptor declares an
unrealistic element count (for example, `shape: [2^40]`) drives
`num_values × dtype_byte_width` to a multi-terabyte value on decode.

For the **szip compression codec** — both the C FFI backend (via
libaec) and the pure-Rust `tensogram-szip` crate — every allocation on
the decode path that derives from this untrusted size is fallible:
`Vec::try_reserve_exact` on the output buffer, the per-RSI scratch
buffer and the sample-serialisation buffer, `checked_mul` on the
sample-count → byte-count conversion, and `checked_sub` on the
bytes-written arithmetic the FFI path reports back. A hostile size
therefore surfaces as a **Compression error** whose message begins
with `"failed to reserve"` (or, for the multiplication overflow case,
`"overflows usize"`) rather than aborting the process.

Blosc2 is likewise hardened (see the PR #69 fix notes): it ignores the
caller-supplied hint entirely and reserves fallibly per chunk against
its own frame-trailer metadata.

**Not yet hardened — known limitations:** The other codec (`zfp`) and
two non-compression allocation sites still use infallible
preallocation from descriptor-derived element counts. A `.tgm` crafted
to route through any of the following paths can still trigger an
allocation abort rather than a typed error:

- `EncodingType::SimplePacking` on the decode path —
  `simple_packing::decode*` uses `Vec::with_capacity(num_values)` for
  the output `Vec<f64>`.
- Bitmask companion frames — `bitmask::{packing,rle,roaring}::decode`
  use `Vec::with_capacity(n_elements)`.
- `CompressionType::Zfp` — `zfp_ffi::zfp_decompress_f64` does
  `vec![0.0f64; num_values]` on a descriptor-derived `num_values`.

These are tracked in `plans/TODO.md` as separate follow-up items. For
now, callers that accept `.tgm` input from untrusted sources should
either validate the descriptor shape before decode (structure-level
`tensogram validate --quick` confirms the shape is well-formed) or
restrict the accepted pipelines to szip and the other
already-hardened codecs.

### File Not Found / Permission Denied

`TensogramFile.open()` raises **OSError** (Python), **io\_error** (C++),
or returns `TGM_ERROR_IO` (C) for any file system failure.

### NetCDF Importer — `--split-by=record` on Files Without Unlimited Dim

`tensogram convert-netcdf --split-by record foo.nc` where `foo.nc` has
no unlimited dimension hard-errors with
`NetcdfError::NoUnlimitedDimension { file }` (exit code 1). The error
message includes the path so the caller can identify which file in a
multi-input batch triggered it.

### NetCDF Importer — `simple_packing` on Mixed-dtype Files

`--encoding simple_packing` is f64-only by design. Mixed files (a
typical CF temperature file has `f32` lat/lon coordinates alongside
`f64` data) are handled gracefully: non-f64 variables emit a stderr
warning and pass through with `encoding="none"`, and the conversion
overall succeeds.

**NaN or Inf in a targeted f64 variable is now a hard error** (0.17+).
The importer fails with
`NetcdfError::InvalidData("simple_packing failed for {var}: ...")`
and a recovery hint, rather than silently downgrading the variable
to `encoding="none"`. Pre-0.17 soft-downgrade hid data-quality
problems; the new behaviour surfaces them at conversion time.
Callers relying on the old fallback should either pick a
non-simple_packing encoding up front, opt into the NaN / Inf
bitmask companion via `--allow-nan` / `--allow-inf` (see
[NaN / Inf Handling](nan-inf-handling.md)), pre-process NaN / Inf
out of the data, or use `--split-by variable` and choose
per-variable encodings.

### NetCDF Importer — Unknown Codec Name

`--encoding foo`, `--filter bar`, `--compression baz` all hard-error
with `NetcdfError::InvalidData` listing the expected values. The
pre-validation fires inside `apply_pipeline` so the error surfaces
immediately, before any data is read from disk.

### NetCDF Importer — szip on Raw f64

libaec szip caps at 32 bits per sample, but raw `f64` gives
`bits_per_sample = 64`, so `--compression szip` on unencoded f64
produces a low-level `aec_encode_init failed` error from
`tensogram` wrapped in `NetcdfError::Encode`. Fix:

- Combine with `--encoding simple_packing --bits N` (N ≤ 32), or
- Combine with `--filter shuffle` (which makes the element size 8 bits).

### Unknown Hash Algorithm (Forward Compatibility)

When the decoder encounters a hash algorithm string it doesn't recognize
(e.g. a future `"sha256"` hash), it logs a warning via `tracing::warn!`
and **skips** verification rather than failing. This ensures forward
compatibility: older decoders can still read messages produced by newer
encoders that use new hash algorithms.

## No-Panic Guarantee

All Rust library code in `tensogram`, `tensogram-encodings`, and
`tensogram-ffi` is free from `panic!()`, `unwrap()`, `expect()`, `todo!()`,
and `unimplemented!()` in non-test code paths. The library guarantees:

- All fallible operations return `Result<T, TensogramError>`.
- Integer arithmetic uses checked operations (`checked_mul`, `try_from`)
  to prevent overflow and truncation.
- `u64 → usize` conversions use `usize::try_from()` to prevent truncation
  on 32-bit platforms.
- Array indexing is guarded by prior bounds checks.
- **Untrusted sizes in the szip and blosc2 decode paths** (the
  descriptor's `num_values × dtype_byte_width`, range-decode lengths,
  per-codec internal size hints) are reserved via
  `Vec::try_reserve_exact`, so allocation failure surfaces as a typed
  error rather than aborting the process. Other decode paths
  (`simple_packing`, bitmask decoders, `zfp`) still use infallible
  preallocation — see [Malformed Descriptor — Pathological Tensor
  Size (szip)](#malformed-descriptor--pathological-tensor-size-szip)
  for the known limitations and the remediation plan.
- FFI boundary code returns error codes instead of panicking, and uses
  `unwrap_or_default()` only for `CString::new()` (interior null fallback).
- The scan functions (`scan`, `scan_file`) tolerate truncation of
  `total_length as usize` because the subsequent bounds check catches it.
- The hash-while-encoding pipeline
  (`PipelineConfig.compute_hash = true` plus the streaming encoder's
  inline-hash path) verifies its CBOR-length invariant *before* writing
  any bytes and surfaces a `TensogramError::Framing` if a
  variable-length hash algorithm is ever configured — the caller's
  sink is never left in a partial-write state on that specific failure
  mode.  Internal debug assertions guard against non-deterministic CBOR
  serialisation during development.
