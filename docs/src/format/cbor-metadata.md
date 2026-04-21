# CBOR Metadata Schema

Tensogram v2 uses CBOR (Concise Binary Object Representation) for all structured metadata. There are four kinds of CBOR structures in a message, each living in its own frame:

1. **GlobalMetadata** — in header or footer metadata frames
2. **DataObjectDescriptor** — inside each data object frame
3. **IndexFrame** — in header or footer index frames
4. **HashFrame** — in header or footer hash frames

All CBOR maps use **deterministic encoding** with canonical key ordering per RFC 8949 section 4.2. Keys are sorted by the byte representation of their CBOR-encoded key, applied recursively to nested maps. This means the same metadata always produces the same bytes — important if you hash messages or compare them by digest.

## GlobalMetadata

The global metadata frame contains a single CBOR map. The only required key is `version`; everything else is optional.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `version` | uint | Yes | Format version. Currently `2` |
| `base` | array of maps | No | Per-object metadata — one entry per data object, each entry holds ALL metadata for that object independently |
| `_reserved_` | map | No | Library internals (provenance: encoder, time, uuid). Client code MUST NOT write to this. |
| `_extra_` | map | No | Client-writable catch-all for ad-hoc message-level annotations |
| *any unknown key* | any | No | Silently ignored on decode (forward compatibility) |

Each data object is self-describing via its own per-frame descriptor (see below). The `base` array provides per-object metadata at the message level so readers can discover object metadata from the global frame alone, without opening each data object frame.

### The `base` Array

The `base` array is one entry per data object. Each entry is a CBOR map holding ALL structured metadata for that object. The encoder auto-populates `_reserved_.tensor` (containing `ndim`, `shape`, `strides`, `dtype`) in each entry. Application keys (e.g. `"mars"`) are preserved:

```json
{
  "base": [
    {
      "mars": { "class": "od", "stream": "oper", "param": "2t", "date": "20260404" },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float32" }
      }
    },
    {
      "mars": { "class": "od", "stream": "oper", "param": "10u", "date": "20260404" },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float32" }
      }
    }
  ]
}
```

Each entry corresponds to one data object in order. Entries are **independent** — there is no tracking of which keys are common across objects. If you need to extract commonalities (e.g. for display or merge operations), use the `compute_common()` utility in software after decoding.

> **Key difference from earlier versions:** There is no `common`/`payload` split. Every `base[i]` entry is self-contained. MARS keys that are shared across all objects (e.g. `class`, `stream`, `date`) are simply repeated in each entry.

### The `_reserved_` Section

The `_reserved_` section at the message level holds library-managed provenance information. Client code can read these values but **must not write to `_reserved_`** — the encoder validates this and rejects messages where client code has written to it.

```json
{
  "_reserved_": {
    "encoder": { "name": "tensogram", "version": "0.1.0" },
    "time": "2026-04-06T12:00:00Z",
    "uuid": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

> **Note:** `_reserved_.encoder.version` is set to the library's crate version at compile time via `env!("CARGO_PKG_VERSION")` — the value above reflects the tensogram version in use.

Within each `base[i]` entry, the encoder also auto-populates `_reserved_.tensor`:

```json
{
  "_reserved_": {
    "tensor": {
      "ndim": 2,
      "shape": [721, 1440],
      "strides": [1440, 1],
      "dtype": "float32"
    }
  }
}
```

### The `_extra_` Section

The `_extra_` section is a client-writable catch-all for ad-hoc message-level annotations:

```json
{
  "_extra_": {
    "source": "ifs-cycle49r2",
    "experiment_tag": "alpha-run-003"
  }
}
```

### Example GlobalMetadata

A complete example with two data objects (temperature and wind fields):

```json
{
  "version": 2,
  "base": [
    {
      "mars": {
        "class": "od", "stream": "oper", "expver": "0001",
        "date": "20260404", "time": "0000", "step": "0",
        "levtype": "sfc", "grid": "regular_ll", "param": "2t"
      },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float32" }
      }
    },
    {
      "mars": {
        "class": "od", "stream": "oper", "expver": "0001",
        "date": "20260404", "time": "0000", "step": "0",
        "levtype": "sfc", "grid": "regular_ll", "param": "10u"
      },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float32" }
      }
    }
  ],
  "_reserved_": {
    "encoder": { "name": "tensogram", "version": "0.6.0" },
    "time": "2026-04-06T12:00:00Z",
    "uuid": "550e8400-e29b-41d4-a716-446655440000"
  },
  "_extra_": {
    "source": "ifs-cycle49r2"
  }
}
```

Each `base[i]` entry is fully self-contained. The only key that varies between the two entries above is `param`. All other MARS keys are repeated — this is by design. Commonalities can be computed in software via `compute_common()` when needed.

### Optional: Full GRIB Namespace Keys

When the GRIB importer runs with `preserve_all_keys` (CLI: `--all-keys`), all non-mars ecCodes namespace keys are stored under a `"grib"` sub-object within each `base[i]` entry:

```json
{
  "base": [
    {
      "mars": { "class": "od", "grid": "regular_ll", "param": "2t", "..." : "..." },
      "grib": {
        "geography": { "Ni": 1440, "Nj": 721, "gridType": "regular_ll" },
        "time":      { "dataDate": 20260404, "dataTime": 0 },
        "ls":        { "edition": 2, "centre": "ecmf", "packingType": "grid_ccsds" },
        "parameter":  { "paramId": 167, "shortName": "2t", "units": "K" },
        "statistics": { "max": 311.03, "min": 212.84, "avg": 277.6 }
      },
      "_reserved_": {
        "tensor": { "ndim": 2, "shape": [721, 1440], "strides": [1440, 1], "dtype": "float64" }
      }
    }
  ]
}
```

The namespaces captured are: `ls`, `geography`, `time`, `vertical`, `parameter`, `statistics`. Keys may overlap between namespaces (e.g. `gridType` appears in both `ls` and `geography`); each namespace stores its own copy. Empty namespaces are omitted.

## DataObjectDescriptor

Each data object frame contains its own CBOR descriptor. This descriptor fully describes how to decode the payload — its type, shape, encoding pipeline, and optional per-object metadata. It lives **inside** the data object frame (not in a central metadata block).

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `type` | text | Yes | Object type, e.g. `"ntensor"` (Rust field: `obj_type`) |
| `ndim` | uint | Yes | Number of dimensions |
| `shape` | array of uint | Yes | Size of each dimension |
| `strides` | array of uint | Yes | Element stride per dimension |
| `dtype` | text | Yes | Data type string (see [Data Types](dtypes.md)) |
| `byte_order` | text | Yes | `"big"` or `"little"` |
| `encoding` | text | Yes | `"none"` or `"simple_packing"` |
| `filter` | text | Yes | `"none"` or `"shuffle"` |
| `compression` | text | Yes | `"none"`, `"szip"`, `"zstd"`, `"lz4"`, `"blosc2"`, `"zfp"`, or `"sz3"` |
| `hash` | map | No | Integrity hash of the payload (see below) |
| `masks` | map | No | NaN / Inf bitmask companion descriptors (see below) |
| *encoding params* | various | Conditional | Required when `encoding != "none"` |
| *filter params* | various | Conditional | Required when `filter != "none"` |
| *compression params* | various | Conditional | Required when `compression != "none"` |
| *any other key* | any | No | Per-object encoding parameters |

### Example: Temperature Field Descriptor

Here is what a descriptor might look like for a global temperature field at 0.25-degree resolution, compressed with zstd:

```json
{
  "type": "ntensor",
  "ndim": 2,
  "shape": [721, 1440],
  "strides": [1440, 1],
  "dtype": "float32",
  "byte_order": "little",
  "encoding": "simple_packing",
  "reference_value": 193.72,
  "binary_scale_factor": -16,
  "decimal_scale_factor": 0,
  "bits_per_value": 16,
  "filter": "none",
  "compression": "zstd",
  "zstd_level": 3,
  "hash": {
    "type": "xxh3",
    "value": "a1b2c3d4e5f60718"
  }
}
```

The `params` field in `DataObjectDescriptor` is for encoding parameters only (e.g. `reference_value`, `bits_per_value`). MARS keys and other application metadata are stored in the global metadata `base[i]["mars"]`.

### Encoding Parameters (simple_packing)

| Key | Type | Description |
|-----|------|-------------|
| `reference_value` | float | Minimum value in the original data |
| `binary_scale_factor` | int | Power-of-2 scaling factor |
| `decimal_scale_factor` | int | Power-of-10 scaling factor |
| `bits_per_value` | uint | Number of bits per packed value (1-64) |

### Filter Parameters (shuffle)

| Key | Type | Description |
|-----|------|-------------|
| `shuffle_element_size` | uint | Byte width of each element (e.g., 4 for float32) |

### Compression Parameters

**szip:**

| Key | Type | Description |
|-----|------|-------------|
| `szip_rsi` | uint | Reference sample interval |
| `szip_block_size` | uint | Block size (typically 8 or 16) |
| `szip_flags` | uint | AEC encoding flags |
| `szip_block_offsets` | array of uint | Bit offsets of RSI block boundaries (computed by the library or provided via `encode_pre_encoded`, see [Pre-encoded Payloads](../guide/encode-pre-encoded.md)) |

**zstd:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `zstd_level` | int | 3 | Compression level (1-22) |

**lz4:** No additional parameters required.

**blosc2:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `blosc2_codec` | text | `"lz4"` | Internal codec: `blosclz`, `lz4`, `lz4hc`, `zlib`, `zstd` |
| `blosc2_clevel` | int | 5 | Compression level (0-9) |
| `blosc2_typesize` | uint | (auto) | Element byte width for shuffle optimization |

**zfp:**

| Key | Type | Description |
|-----|------|-------------|
| `zfp_mode` | text | `"fixed_rate"`, `"fixed_precision"`, or `"fixed_accuracy"` |
| `zfp_rate` | float | Bits per value (only for `fixed_rate`) |
| `zfp_precision` | uint | Bit planes to keep (only for `fixed_precision`) |
| `zfp_tolerance` | float | Max absolute error (only for `fixed_accuracy`) |

**sz3:**

| Key | Type | Description |
|-----|------|-------------|
| `sz3_error_bound_mode` | text | `"abs"`, `"rel"`, or `"psnr"` |
| `sz3_error_bound` | float | Error bound value |

### Hash Descriptor

The optional `hash` field records an integrity digest of the raw payload bytes.

| Key | Type | Description |
|-----|------|-------------|
| `type` | text | `"xxh3"` |
| `value` | text | Hex-encoded digest |

### NaN / Inf mask companion (`masks`)

When the object was encoded with `allow_nan=true` and/or
`allow_inf=true` AND the payload actually contained at least one
matching non-finite value, the descriptor carries a `masks`
sub-map.  Each kind (`nan`, `inf+`, `inf-`) is independently
optional — only the kinds that appeared are present.

```cbor
{
  ... standard DataObjectDescriptor fields ...,
  "masks": {
    "nan": {
      "method": "roaring",
      "offset": 40,
      "length": 12
    },
    "inf+": {
      "method": "rle",
      "offset": 52,
      "length": 3
    }
  }
}
```

Each entry:

| Key | Type | Description |
|-----|------|-------------|
| `method` | text | `"none"` \| `"rle"` \| `"roaring"` \| `"blosc2"` \| `"zstd"` \| `"lz4"` — compression method actually used (may differ from the requested method due to the small-mask auto-fallback) |
| `offset` | uint | Byte offset of the mask blob from the **start of the payload region** (= first byte after the 16-byte frame header) |
| `length` | uint | Byte length of the mask blob on disk |
| `params` | map | Optional method-specific parameters (e.g. `{"level": 3}` for zstd, `{"codec": "lz4", "level": 5}` for blosc2) |

Canonical key order for `masks` is the byte-lex sort `inf+` < `inf-` < `nan`.
The encoder writes mask blobs between the payload and the CBOR
descriptor in the same canonical order.  See
[NaN / Inf Handling](../guide/nan-inf-handling.md) for the encode
/ decode semantics.

## IndexFrame

Index frames (header or footer) contain a CBOR map that lets readers
jump directly to any data object without scanning.

| Key       | Type           | Description                                            |
|-----------|----------------|--------------------------------------------------------|
| `offsets` | array of uint  | Byte offset of each data object frame from message start |
| `lengths` | array of uint  | Byte length of each data object frame                  |

Object count is derived from `offsets.len()`; `lengths.len()` must
equal `offsets.len()` or the decoder emits a `MetadataError`.

### Example IndexFrame

```json
{
  "offsets": [256, 1048832, 2097408],
  "lengths": [1048576, 1048576, 524288]
}
```

The `offsets` array gives O(1) random access to any object — seek
to `offsets[i]` and read `lengths[i]` bytes.

## HashFrame

Hash frames (header or footer) mirror the per-object inline hash
slots of each data-object frame's footer (see
[wire-format.md §2.4](./wire-format.md#24-frame-hash-scope)), so
readers can inspect the aggregate without walking every frame.

| Key         | Type           | Description                                        |
|-------------|----------------|----------------------------------------------------|
| `algorithm` | text           | Hash algorithm name.  `"xxh3"` is the only value a v3 encoder emits. |
| `hashes`    | array of text  | Hex-encoded digest for each object, in emission order. |

Object count is derived from `hashes.len()`.  An unknown
`algorithm` value triggers an `UnknownHashAlgorithm` warning at
validate time; the inline slots remain the authoritative check.

### Example HashFrame

```json
{
  "algorithm": "xxh3",
  "hashes": [
    "a1b2c3d4e5f60718",
    "b2c3d4e5f6071829",
    "c3d4e5f60718293a"
  ]
}
```

## Canonical Encoding

All CBOR maps are encoded with keys sorted by the byte representation of their CBOR-encoded key (RFC 8949 section 4.2). This sorting is applied recursively — nested maps are also sorted.

For short string keys (the common case), this is equivalent to sorting by the key string itself. For long keys or non-string keys, the CBOR byte encoding determines the order.

> **Why does this matter?** If you hash an entire message or compare messages by digest, deterministic encoding ensures that logically identical messages produce identical bytes even if the keys were inserted in different order during construction.
