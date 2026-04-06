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
  ├─ NaN in simple_packing ───────► Encoding error
  ├─ Compression codec failure ───► Compression error
  └─ File I/O failure ────────────► I/O error
```

### Decoding

```
Raw bytes
  │
  ├─ No magic bytes / truncated ──► Framing error
  ├─ Bad frame type codes ────────► Framing error
  ├─ CBOR parse failure ──────────► Metadata error
  ├─ Object index out of range ───► Object error
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
