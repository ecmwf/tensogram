# Metadata Value Types

All Tensogram metadata — whether in [`GlobalMetadata`](cbor-metadata.md#globalmetadata), the `common` / `payload` / `reserved` sections, or per-object `params` — is stored as CBOR. This page describes which value types are valid, which are forbidden, and why.

## Allowed Types

Use only the subset of CBOR types that have direct JSON equivalents:

| CBOR type | Rust / Python equivalent | Example |
|-----------|--------------------------|---------|
| **text string** | `String` / `str` | `"ecmwf"`, `"20260404"` |
| **integer** | `i64` / `int` | `850`, `-1`, `0` |
| **float** | `f64` / `float` | `3.14`, `-273.15` |
| **boolean** | `bool` / `bool` | `true`, `false` |
| **null** | `None` / `None` | (absence of a value) |
| **array** | `Vec<Value>` / `list` | `[1440, 721]`, `["2t", "10u"]` |
| **map** | `BTreeMap<String, Value>` / `dict` | `{"class": "od", "step": 6}` |

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

## The `payload` Section

The `payload` section of `GlobalMetadata` is a CBOR **array of maps** — one entry per data object. The encoder automatically populates each entry with `ndim`, `shape`, `strides`, and `dtype` when you call `encode()` or `StreamingEncoder::finish()`. Any keys the application placed in a payload entry before encoding (e.g. per-object MARS keys) are preserved:

```json
{
  "version": 2,
  "common": {
    "mars": { "class": "od", "type": "fc", "grid": "O1280" }
  },
  "payload": [
    {
      "ndim": 2,
      "shape": [721, 1440],
      "strides": [1440, 1],
      "dtype": "float64",
      "mars": { "param": "2t", "levtype": "sfc" }
    },
    {
      "ndim": 1,
      "shape": [137],
      "strides": [1],
      "dtype": "float64",
      "mars": { "param": "lnsp", "levtype": "ml" }
    }
  ]
}
```

Each entry mirrors the corresponding `DataObjectDescriptor` shape fields. MARS keys that are **shared** across all objects (e.g. `class`, `type`, `grid`) live under `common["mars"]`, while MARS keys that **vary** per object (e.g. `param`, `levtype`) live under each `payload[i]["mars"]`. The GRIB key `gridType` is stored as `"grid"` in the mars namespace.

> **Note:** `payload` describes the *collection* of objects. Individual tensor encoding details (encoding pipeline, hash) remain in each object's own `DataObjectDescriptor`. The `DataObjectDescriptor.params` field is reserved for encoding parameters only — it no longer carries MARS keys.

## Practical Guidance

- Prefer **integers** for numeric identifiers (`paramId`, `date`, `step`).
- Use **text strings** for classification codes (`class`, `type`, `stream`) even if they happen to be numeric-looking — consistency with MARS vocabulary is more important than type optimisation.
- Use **nested maps** for namespaced keys (e.g., `"mars": {"class": "od", ...}`).
- Keep individual values small. Avoid storing large arrays (e.g., grid coordinates) in metadata — they belong in data objects.

## See Also

- [CBOR Metadata Schema](cbor-metadata.md) — field-level reference for all CBOR structures
- [Metadata Concepts](../concepts/metadata.md) — how global and per-object metadata relate
- [GRIB MARS Key Mapping](../grib/metadata-mapping.md) — how GRIB keys are mapped to Tensogram metadata
