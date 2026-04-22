# NaN / Inf handling

By default the Tensogram encoder **rejects** any NaN or ±Inf in
float / complex payloads.  The encode call fails with
`TensogramError::Encoding` (C FFI: `TgmError::Encoding`; Python:
`EncodingError`; TypeScript: `EncodingError`; C++: `tensogram::encoding_error`)
and names the element index, dtype, and a hint that points at the
opt-in flags described below.

This chapter walks through the three policies available on
encode:

1. **Reject** (default) — any non-finite input fails the call.
   Use this when your pipeline guarantees finite values and any
   NaN / Inf is a bug you want to surface loudly.
2. **Allow NaN** — NaN values are substituted with `0.0` on the wire
   and their positions are recorded in a compressed bitmask stored
   alongside the payload.  Decode restores canonical NaN at those
   positions by default.
3. **Allow ±Inf** — same as `allow_nan` but for `+∞` and `−∞`
   together (the flag covers both signs; two per-sign bitmasks are
   written when both kinds appear in the payload).

The mask companion is part of the *`NTensorFrame`* — wire-format
type 9.  For the byte-level specification, see
[the wire-format reference](../format/wire-format.md#ntensorframe-type-9)
and the normative spec in
[plans/WIRE_FORMAT.md §6.5](https://github.com/ecmwf/tensogram/blob/main/plans/WIRE_FORMAT.md).

## When to use which policy

| Situation | Flag to set |
|---|---|
| Finite data only, want hard failure on contamination | default (both off) |
| NetCDF `_FillValue` → NaN, Zarr missing data, sensor gaps | `allow_nan=true` |
| Propagating numerical overflow as `±Inf` | `allow_inf=true` |
| Mixed missing-value / overflow data | both `true` |

**Don't** pre-process to a sentinel value when `allow_nan` /
`allow_inf` does the job — the bitmask is designed to compress
aggressively (hybrid Roaring containers by default) and keeps the
missing-data semantics visible to the decoder.  Sentinel values
throw that information away.

## Cross-language opt-in

### Rust

```rust
use tensogram::{encode, EncodeOptions, GlobalMetadata, DataObjectDescriptor};

let options = EncodeOptions {
    allow_nan: true,
    allow_inf: true,
    ..Default::default()
};
let msg = encode(&meta, &[(&desc, payload_bytes)], &options)?;
```

### Python

```python
import numpy as np
import tensogram

data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
msg = tensogram.encode(
    {},
    [(desc, data)],
    allow_nan=True,
)
decoded = tensogram.decode(msg)
# decoded.objects[0].data() → [1.0, nan, 3.0]
```

### TypeScript

```ts
import { encode, decode } from '@ecmwf.int/tensogram';

const msg = encode(
    { version: 3 },
    [{ descriptor, data: new Float64Array([1, NaN, 3]) }],
    { allowNan: true },
);
const decoded = decode(msg);
```

### C++

```cpp
tensogram::encode_options opts;
opts.allow_nan = true;
auto msg = tensogram::encode(metadata_json, objects, opts);
```

### CLI

```bash
$ tensogram --allow-nan reshuffle -o out.tgm input.tgm
$ TENSOGRAM_ALLOW_NAN=1 tensogram convert-netcdf data.nc -o data.tgm
```

## Decode-side reconstruction

By default every decode path restores the canonical quiet-NaN / ±Inf
bit pattern at every masked position.  Opt out (e.g. to inspect
the on-disk zero-substituted representation) by passing
`restore_non_finite=false`:

```python
# Get the 0.0-substituted payload without the NaN bits.
raw = tensogram.decode(msg, restore_non_finite=False)
# raw.objects[0].data() → [1.0, 0.0, 3.0]
```

The advanced `decode_with_masks` API (Rust + Python) returns both
the zero-substituted payload AND the raw decompressed
per-kind `Vec<bool>` masks, so callers can build custom
missing-value representations without materialising canonical NaN
bytes.

## Lossy reconstruction — read this carefully

The masked encode path **does not preserve** the original NaN
payload bits.  On decode every masked NaN is restored with the
canonical quiet-NaN pattern:

- `f32::NAN` bits = `0x7FC00000`
- `f64::NAN` bits = `0x7FF8000000000000`
- Float16 / bfloat16 use their dtype-native quiet-NaN patterns
- Complex64 / complex128 restore the canonical pattern to **both**
  real and imag components

Signalling NaNs, custom payload bits, and mixed real / imag
kinds for complex dtypes are therefore flattened to the canonical
form through a mask round-trip.  If you need bit-exact NaN
preservation, pre-encode your payload and use
`encode_pre_encoded` to bypass the substitute-and-mask stage
entirely.  See [plans/WIRE_FORMAT.md §6.5.4](https://github.com/ecmwf/tensogram/blob/main/plans/WIRE_FORMAT.md)
for the normative spec.

## Mask compression methods

Six methods are available per-kind:

| Method | Best for | Feature |
|---|---|---|
| `roaring` (default) | any mask shape | pure Rust, works on WASM |
| `rle` | highly clustered masks (land / sea, swath gaps) | pure Rust |
| `blosc2` | dense dtype-aligned masks | `blosc2` feature |
| `zstd` | generic good-ratio | `zstd` feature |
| `lz4` | decode-speed priority | `lz4` feature |
| `none` | tiny masks (auto-fallback) | always available |

Small masks (uncompressed bit-packed byte count ≤ 128 by default)
automatically fall back to `none` regardless of the requested
method — compressing a few bytes costs more than it saves.  Set
`small_mask_threshold_bytes = 0` to disable the auto-fallback.

Set per-kind methods via the matching options:

```python
msg = tensogram.encode(
    meta, [(desc, data)],
    allow_nan=True, allow_inf=True,
    nan_mask_method='rle',
    pos_inf_mask_method='roaring',
    neg_inf_mask_method='roaring',
    small_mask_threshold_bytes=0,
)
```

## Validation

`tensogram validate --full` cross-checks every NaN / ±Inf in the
decoded payload against the frame's mask companion: masked
positions are *expected* and pass; any NaN / Inf at a non-masked
position is reported as `NanDetected` / `InfDetected`
(see [the validator reference](../reference/validate-issue-codes.md)).

Files without a mask companion keep the pre-0.17 semantics — any
non-finite value in the decoded output is an error.

## Migration from pre-0.17

Prior to 0.17 the `reject_nan` / `reject_inf` opt-in flags upgraded
the NaN check to be pipeline-independent.  **These flags are
removed** in 0.17 (breaking change).  Rejection is now always on by
default; opt in to masked substitution with the replacement flags:

| Pre-0.17 | 0.17+ |
|---|---|
| `reject_nan=False` (default, pass-through) | `allow_nan=True` (substitute + mask) |
| `reject_nan=True` (opt-in reject) | default (always reject) |
| `reject_inf=False` / `True` | same split, `allow_inf` |

See [CHANGELOG.md](../../CHANGELOG.md) for the full breaking-change
list and upgrade notes.
