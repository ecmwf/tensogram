# Metadata Value Types

All Tensogram metadata — whether in [`GlobalMetadata`](cbor-metadata.md#globalmetadata), the `base` / `_reserved_` / `_extra_` sections, or per-object `params` — is stored as CBOR. This page describes which value types are valid, which are forbidden, and why.

## Allowed Types

Use only the subset of CBOR types that have direct JSON equivalents:

| CBOR type | Rust / Python equivalent | Example |
|-----------|--------------------------|---------|
| **text string** | `String` / `str` | `"imaging"`, `"2026-01-12"` |
| **integer** | `i64` / `int` | `850`, `-1`, `0` |
| **float** | `f64` / `float` | `3.14`, `-273.15` |
| **boolean** | `bool` / `bool` | `true`, `false` |
| **null** | `None` / `None` | (absence of a value) |
| **array** | `Vec<Value>` / `list` | `[1440, 721]`, `["t2", "flair"]` |
| **map** | `BTreeMap<String, Value>` / `dict` | `{"device": "mri", "sequence": "t2_flair"}` |

Map keys **must** be text strings. Nested arrays and maps are allowed and encoded recursively.

## Forbidden Types

The following CBOR types are **not** allowed in Tensogram metadata:

| Type | Reason |
|------|--------|
| **byte strings** | Opaque blobs break cross-language interoperability; use base64 text instead |
| **CBOR tags** | Tags (`#6.<n>`) are not parsed by most CBOR libraries and can change value semantics |
| **undefined** | Only valid in streaming CBOR; never appears in map values |
| **half-precision floats** (f16) | Not supported by many JSON bridges; use `f64` |
| **non-string map keys** | Integer or binary keys are non-canonical and not searchable |

## The `base` Section

The `base` section of `GlobalMetadata` is a CBOR **array of maps** — one entry
per data object. Each entry holds ALL structured metadata for that object
independently. The encoder auto-populates `_reserved_.tensor` (with ndim, shape,
strides, dtype) in each entry when you call `encode()` or
`StreamingEncoder::finish()`. Any other keys the application placed in a base
entry before encoding (e.g. a per-object vocabulary namespace) are preserved.
The example below uses the MARS vocabulary; any application namespace works the
same way:

```json
{
  "version": 3,
  "base": [
    {
      "mars": { "class": "od", "type": "fc", "grid": "O1280", "param": "2t", "levtype": "sfc" },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float64" }
      }
    },
    {
      "mars": { "class": "od", "type": "fc", "grid": "O1280", "param": "lnsp", "levtype": "ml" },
      "_reserved_": {
        "tensor": { "ndim": 1, "shape": [137], "strides": [1], "dtype": "float64" }
      }
    }
  ]
}
```

Each entry is fully self-contained — all keys for that object appear in its
entry. There is no separate "common" section for shared keys. If you need to
extract commonalities (e.g. for display), use the `compute_common()` utility in
software after decoding.

> **Note:** `base` describes the *collection* of objects at the message level.
> Individual tensor encoding details (encoding pipeline, hash) remain in each
> object's own `DataObjectDescriptor`. The `DataObjectDescriptor.params` field
> is reserved for encoding parameters only — it does not carry application
> metadata.

## Practical Guidance

- Prefer **integers** for numeric identifiers (`paramId`, `date`, `run_id`).
- Use **text strings** for classification codes even if they happen to be
  numeric-looking — consistency with your chosen vocabulary is more important
  than type optimisation.
- Use **nested maps** for namespaced keys (e.g., `"mars": {...}`, `"bids": {...}`,
  `"dicom": {...}`).
- Keep individual values small. Avoid storing large arrays (e.g., grid
  coordinates) in metadata — they belong in data objects.

## See Also

- [CBOR Metadata Schema](cbor-metadata.md) — field-level reference for all CBOR structures
- [Metadata Concepts](../concepts/metadata.md) — how global and per-object metadata relate
- [Vocabularies](../guide/vocabularies.md) — example application-layer vocabularies used with Tensogram
- [GRIB MARS Key Mapping](../grib/metadata-mapping.md) — how GRIB keys are mapped during import
