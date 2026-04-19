# Strict-Finite Encode Checks

Tensogram's default encode path is permissive: NaN and Inf values pass
through `encoding="none"` pipelines byte-for-byte, and `simple_packing`
rejects NaN but not Inf (see [Edge Cases](../edge-cases.md)). For
workflows where a non-finite value in a produced message is a
production incident — operational forecasting, financial time series,
anything with a strict contract — Tensogram provides **opt-in,
pipeline-independent strict-finite checks**.

Two orthogonal encode options:

- `reject_nan` — on the first NaN in a float payload, encode fails
  with an `EncodingError` carrying the element index and dtype.
- `reject_inf` — same, for `+Inf` / `-Inf`. The error message
  includes the sign.

Both default to `false` (backwards-compatible). Integer and bitmask
dtypes skip the scan entirely — zero cost regardless of flags.

## Why two flags?

NaN and Inf have different severity profiles:

- **NaN is usually a missing-value marker** — common in CF-convention
  data unpacked from NetCDF, for example. Many workflows treat it as
  information, not a defect.
- **Inf is almost always a bug** — it points at a divide-by-zero,
  overflow, or uninitialised buffer upstream.

Splitting the flags lets you reject Inf (catches bugs) while
tolerating NaN (acceptable missingness).

## Pipeline-independent guarantee

The scan runs **before** any encoding/filter/compression stage. That
means the same contract holds whether you pick:

- `encoding="none"` (byte-level passthrough) — without the flag, NaN
  just round-trips silently.
- `encoding="simple_packing"` — without the flag, Inf silently
  produces numerically-useless parameters and decodes to NaN
  everywhere (the §3.1 gotcha in `plans/RESEARCH_NAN_HANDLING.md`).
- `compression="zfp"` / `"sz3"` — without the flag, behaviour is
  whatever the upstream library does with NaN/Inf; with the flag,
  the problem is caught at encode time.

Pre-encoded inputs (`encode_pre_encoded`) are opaque to the library,
so the flags intentionally do **not** apply there. Callers of the
pre-encoded API accepted their own contract by choosing that path.

## Using the flags

### Rust

```rust
use tensogram::{encode, EncodeOptions, GlobalMetadata};

let opts = EncodeOptions {
    reject_nan: true,
    reject_inf: true,
    ..Default::default()
};

match encode(&meta, &descriptors, &opts) {
    Ok(bytes) => write_to_disk(bytes),
    Err(tensogram::TensogramError::Encoding(msg)) => {
        // msg like: "strict-NaN check: NaN at element 42 of float32 array"
        eprintln!("refusing to ship non-finite payload: {msg}");
    }
    Err(e) => return Err(e),
}
```

### Python

```python
import numpy as np
import tensogram

data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
meta = {"version": 2}
desc = {
    "type": "ntensor", "shape": [3], "dtype": "float32",
    "byte_order": "little", "encoding": "none", "filter": "none",
    "compression": "none",
}

try:
    tensogram.encode(meta, [(desc, data)], reject_nan=True)
except ValueError as e:
    print(f"rejected: {e}")
    # rejected: EncodingError: strict-NaN check: NaN at element 1 of float32 array
```

Both `tensogram.encode()`, `TensogramFile.append()`, and
`StreamingEncoder(...)` accept `reject_nan=False` and
`reject_inf=False` kwargs.

### TypeScript

```typescript
import { encode, EncodingError } from '@ecmwf/tensogram';

const data = new Float32Array([1.0, NaN, 3.0]);
try {
  encode(meta, [{ descriptor, data }], {
    rejectNan: true,
    rejectInf: true,
  });
} catch (e) {
  if (e instanceof EncodingError) {
    console.error('refusing to ship non-finite payload:', e.message);
  }
}
```

### C++

```cpp
#include <tensogram.hpp>

tensogram::encode_options opts;
opts.reject_nan = true;
opts.reject_inf = true;

try {
    auto bytes = tensogram::encode(metadata_json, objects, opts);
} catch (const tensogram::encoding_error& e) {
    std::cerr << "refusing to ship non-finite payload: " << e.what() << '\n';
}
```

### C FFI

```c
#include "tensogram.h"

tgm_bytes_t out;
tgm_error e = tgm_encode(metadata_json, data_ptrs, data_lens, 1,
                         NULL /*hash*/, 0 /*threads*/,
                         /*reject_nan=*/true, /*reject_inf=*/true,
                         &out);
if (e == TGM_ERROR_ENCODING) {
    fprintf(stderr, "non-finite detected: %s\n", tgm_last_error());
}
```

### CLI

The flags are global — they apply to every encoding-capable subcommand
(`merge`, `split`, `reshuffle`, `convert-grib`, `convert-netcdf`). They
also accept the `TENSOGRAM_REJECT_NAN` / `TENSOGRAM_REJECT_INF`
environment variables for ops rollouts.

```bash
# Flag form
tensogram --reject-nan --reject-inf \
          convert-netcdf input.nc -o out.tgm

# Env-var form
TENSOGRAM_REJECT_INF=1 tensogram merge a.tgm b.tgm -o out.tgm
```

Exit code is non-zero on rejection. stderr carries the element index,
dtype, and an "NaN" or "Inf" indicator. Pipelines can grep for those
tokens to distinguish strict-finite failures from other errors.

## Parallel scans

When `threads > 0` and the payload exceeds the 64 KiB parallel
threshold, the scan is split across rayon workers. Each worker
short-circuits on the first NaN / Inf it sees in its own chunk, so
the reported element index is **not necessarily the globally first
occurrence** — it's the first one the winning worker handled.

This is documented as part of `EncodeOptions` and matches the
existing semantics of `simple_packing`'s parallel scan. If you need
exact first-index reporting (for debugging a specific NaN source),
pass `threads = 0` (sequential).

## What's not covered

- **Pre-encoded bytes** (`encode_pre_encoded`,
  `write_object_pre_encoded`). Opaque to the library. Setting either
  strict flag on these APIs **returns an error** rather than silently
  discarding the flag:
    - Rust: `TensogramError::Encoding("reject_nan / reject_inf do not apply to encode_pre_encoded…")`
    - C++: `tensogram::encoding_error` with the same message
    - Python: `TypeError` — the kwargs are not exposed on
      `encode_pre_encoded` at all
    - C FFI: `tgm_encode_pre_encoded` does not take the parameters
  This mismatch is by design — pre-encoded payloads have already
  committed to their representation and the caller accepted that
  contract. Cross-language uniformity is achieved by refusing the
  misuse rather than silently dropping it.
- **Metadata NaN** — NaN values in CBOR metadata (e.g. a
  scalar attribute) are unrelated. The strict flags only scan data
  payloads. See [CBOR Metadata Schema](../format/cbor-metadata.md)
  for metadata semantics.
- **Decode-side NaN detection**. For that, use
  `tensogram validate --full` or the library-level
  `ValidateOptions::Fidelity` level. The strict flags are an encode
  gate, not a decode gate — both complement each other.

## `simple_packing` params safety net (always on)

Independent of the strict-finite flags above, `simple_packing::encode`
and `simple_packing::encode_with_threads` now validate their
[`SimplePackingParams`] input against the values
that would produce silently-wrong output:

- `reference_value` must be finite (`NaN` / `±Inf` rejected).
- `|binary_scale_factor|` must be ≤ `256` (the constant
  [`MAX_REASONABLE_BINARY_SCALE`] is exposed as a public constant).
  This threshold catches the `i32::MAX`-saturation fingerprint that
  results from feeding `Inf` through `compute_params`'s range
  arithmetic, while leaving ample headroom for real-world scientific
  data (typical range: `[-60, 30]`).

Validation runs unconditionally at the top of `encode_with_threads`;
the high-level `tensogram::encode()` path hits the same check through
delegation.  Errors surface as `PackingError::InvalidParams { field,
reason }` at the Rust core and propagate unchanged through every
language binding:

```rust
// Rust
match encode(&values, &params) {
    Err(PackingError::InvalidParams { field, reason }) => {
        eprintln!("bad {field}: {reason}");
    }
    Ok(bytes) => /* ... */,
    Err(other) => /* ... */,
}
```

```python
# Python
import tensogram
desc = {..., "binary_scale_factor": 2**31 - 1, ...}
try:
    tensogram.encode(meta, [(desc, data)])
except ValueError as e:
    # "binary_scale_factor: value 2147483647 is outside the reasonable range ±256; ..."
    print(e)
```

**Legitimate edge cases that the safety net does NOT reject:**

- `bits_per_value = 0` — a valid constant-field encoding; the
  decoder reconstructs every value from `reference_value` alone and
  the packed-int output is empty.  Saves bytes when a whole tensor
  collapses to one value.

## Design notes

`plans/RESEARCH_NAN_HANDLING.md` §3.1 enumerates the user-visible
corners this feature closes, and §4.1 describes the design trade-offs
that led to two orthogonal flags instead of a single combined flag.
Short version:

- `reject_nan` and `reject_inf` are independently useful, so combining
  them into a single flag would either force users to accept Inf
  along with NaN or vice versa.
- `reject_nan=true, reject_inf=true` is the "domain contract" mode
  for callers who want a strong guarantee that no non-finite value
  reaches their consumers.

The scan runs pre-pipeline rather than being folded into each codec
because:

1. `simple_packing` already has NaN-only rejection baked in, but its
   Inf path is a silent-corruption vector — catching it upstream is
   the safest fix.
2. `zfp` / `sz3` have undefined NaN behaviour that varies across
   upstream versions. A single gate upstream is more robust than
   per-codec defences that might drift.
3. One audit point is easier to reason about than six.
