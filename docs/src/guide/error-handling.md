# Error Handling

Tensogram uses typed errors across all language bindings. Every fallible
operation returns a `Result` (Rust), raises an exception (Python/C++), or
returns an error code (C). No library code panics.

## Error Categories

| Category | Trigger | Rust | Python | C++ | C Code |
|----------|---------|------|--------|-----|--------|
| **Framing** | Invalid magic bytes, truncated message, bad terminator | `TensogramError::Framing` | `ValueError` | `framing_error` | `TGM_ERROR_FRAMING (1)` |
| **Metadata** | CBOR parse failure, missing required field, schema violation | `TensogramError::Metadata` | `ValueError` | `metadata_error` | `TGM_ERROR_METADATA (2)` |
| **Encoding** | Encoding pipeline failure (e.g. NaN in simple\_packing) | `TensogramError::Encoding` | `ValueError` | `encoding_error` | `TGM_ERROR_ENCODING (3)` |
| **Compression** | Decompression failure, unknown codec | `TensogramError::Compression` | `ValueError` | `compression_error` | `TGM_ERROR_COMPRESSION (4)` |
| **Object** | Invalid descriptor, object index out of range | `TensogramError::Object` | `ValueError` | `object_error` | `TGM_ERROR_OBJECT (5)` |
| **I/O** | File not found, permission denied, disk full | `TensogramError::Io` | `OSError` | `io_error` | `TGM_ERROR_IO (6)` |
| **Hash Mismatch** | Payload integrity check fails on `verify_hash=True` | `TensogramError::HashMismatch` | `RuntimeError` | `hash_mismatch_error` | `TGM_ERROR_HASH_MISMATCH (7)` |
| **Invalid Arg** | NULL pointer or invalid argument in C/C++ FFI | — | — | `invalid_arg_error` | `TGM_ERROR_INVALID_ARG (8)` |

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
  └─ I/O write failure ───────────► I/O error
```

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

### Converter Operations (convert-grib / convert-netcdf)

Both converter crates (`tensogram-grib`, `tensogram-netcdf`) use typed
error enums and never panic on invalid or exotic input. Anything the
converter can't represent cleanly is either surfaced as a typed error
or skipped with a `warning: …` line on stderr so the operator can see
what was dropped.

```
tensogram-netcdf errors (crates/tensogram-netcdf/src/error.rs)
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
  │         unreachable; if it fires the converter is buggy)
  │
  ├─ NetcdfError::Encode(String)
  │     tensogram-core rejected the pipeline. Common cause:
  │     szip on raw f64 (bits_per_sample=64 exceeds libaec's
  │     32-bit cap). Fix: add --filter shuffle or --encoding
  │     simple_packing first.
  │
  └─ NetcdfError::Io(std::io::Error)
        Reserved for future use — the current converter reads
        through libnetcdf and writes through the CLI wrapper, so
        stdlib I/O errors don't currently reach this variant.
```

Soft warnings (stderr, exit 0):

```
warning: {file}: sub-groups found; only root-group variables are converted
warning: skipping variable '{name}': Char variables are not supported
warning: skipping variable '{name}': complex type Compound(_) is not supported
warning: skipping simple_packing for variable '{name}' (Float32 is not float64)
warning: skipping simple_packing for variable '{name}': NaN value encountered at index N
warning: variable '{name}': failed to read attribute '{attr}': {cause}
warning: failed to read global attribute '{name}': {cause}
```

The last two are rare — they only fire on corrupt attribute values
or unsupported upstream AttributeValue variants — but they surface
instead of dropping data silently so operators can trace unexpected
missing metadata.

```
tensogram-grib errors (crates/tensogram-grib/src/error.rs)
  │
  ├─ GribError::Eccodes(String) — ecCodes C library error
  ├─ GribError::NoMessages — empty GRIB file
  ├─ GribError::MissingKey — required MARS key absent
  ├─ GribError::InvalidShape — grid dimension mismatch
  └─ GribError::Encode — tensogram-core encode failure
```

## Language-Specific Patterns

### Rust

```rust
use tensogram_core::{decode, DecodeOptions, TensogramError};

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
    # File I/O errors
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

## Common Error Scenarios

### Garbage or Truncated Input

Any non-Tensogram bytes passed to `decode()` produce a **Framing error**.
The decoder looks for the 8-byte magic `TENSOGRM` and a matching terminator.

### Hash Mismatch After Corruption

When `verify_hash=True` (Python) or `TGM_DECODE_VERIFY_HASH` (C), the
decoder recomputes the xxh3-64 hash of each payload and compares it to the
stored hash. A mismatch produces a **HashMismatch error** with both the
expected and actual hash values.

Messages encoded without a hash (`hash=None`) silently pass verification —
there is nothing to check.

### Object Index Out of Range

Accessing `decode_object(buf, index=N)` where N ≥ number of objects
produces an **Object error** (Rust/C/C++) or **ValueError** (Python).
File indexing `file[N]` raises **IndexError** for out-of-range N.

### NaN in Simple Packing

`compute_packing_params()` rejects NaN values with a **ValueError** that
includes the index of the first NaN. Inf values are accepted but produce
extreme scale factors — filter them before packing.

### File Not Found / Permission Denied

`TensogramFile.open()` raises **OSError** (Python), **io\_error** (C++),
or returns `TGM_ERROR_IO` (C) for any file system failure.

### NetCDF Converter — `--split-by=record` on Files Without Unlimited Dim

`tensogram convert-netcdf --split-by record foo.nc` where `foo.nc` has
no unlimited dimension hard-errors with
`NetcdfError::NoUnlimitedDimension { file }` (exit code 1). The error
message includes the path so the caller can identify which file in a
multi-input batch triggered it.

### NetCDF Converter — `simple_packing` on Mixed-dtype Files

`--encoding simple_packing` is f64-only by design. Mixed files (a
typical CF temperature file has `f32` lat/lon coordinates alongside
`f64` data) are handled gracefully: non-f64 variables emit a stderr
warning and pass through with `encoding="none"`, and the conversion
overall succeeds. The same fallback fires when a specific f64
variable contains NaN values (common with unpacked fill values) —
`simple_packing::compute_params` rejects NaN, so that variable
falls back to `encoding="none"` with a warning.

### NetCDF Converter — Unknown Codec Name

`--encoding foo`, `--filter bar`, `--compression baz` all hard-error
with `NetcdfError::InvalidData` listing the expected values. The
pre-validation fires inside `apply_pipeline` so the error surfaces
immediately, before any data is read from disk.

### NetCDF Converter — szip on Raw f64

libaec szip caps at 32 bits per sample, but raw `f64` gives
`bits_per_sample = 64`, so `--compression szip` on unencoded f64
produces a low-level `aec_encode_init failed` error from
`tensogram-core` wrapped in `NetcdfError::Encode`. Fix:

- Combine with `--encoding simple_packing --bits N` (N ≤ 32), or
- Combine with `--filter shuffle` (which makes the element size 8 bits).

### Unknown Hash Algorithm (Forward Compatibility)

When the decoder encounters a hash algorithm string it doesn't recognize
(e.g. a future `"sha256"` hash), it logs a warning via `tracing::warn!`
and **skips** verification rather than failing. This ensures forward
compatibility: older decoders can still read messages produced by newer
encoders that use new hash algorithms.

## No-Panic Guarantee

All Rust library code in `tensogram-core`, `tensogram-encodings`, and
`tensogram-ffi` is free from `panic!()`, `unwrap()`, `expect()`, `todo!()`,
and `unimplemented!()` in non-test code paths. The library guarantees:

- All fallible operations return `Result<T, TensogramError>`.
- Integer arithmetic uses checked operations (`checked_mul`, `try_from`)
  to prevent overflow and truncation.
- `u64 → usize` conversions use `usize::try_from()` to prevent truncation
  on 32-bit platforms.
- Array indexing is guarded by prior bounds checks.
- FFI boundary code returns error codes instead of panicking, and uses
  `unwrap_or_default()` only for `CString::new()` (interior null fallback).
- The scan functions (`scan`, `scan_file`) tolerate truncation of
  `total_length as usize` because the subsequent bounds check catches it.
