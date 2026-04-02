# CBOR Metadata Schema

The metadata section of every Tensogram message is encoded as a CBOR map. This page documents the required and optional fields.

## Top-Level Map

| Key | Type | Required | Description |
|---|---|---|---|
| `version` | uint | Yes | Format version. Currently `1` |
| `objects` | array | Yes | One entry per tensor object |
| `payload` | array | Yes | One entry per tensor object |
| *any other key* | any | No | Application metadata (e.g. `"mars"`) |

## Object Descriptor (`objects[i]`)

| Key | Type | Required | Description |
|---|---|---|---|
| `type` | text | Yes | Always `"ntensor"` |
| `ndim` | uint | Yes | Number of dimensions |
| `shape` | array of uint | Yes | Size of each dimension |
| `strides` | array of uint | Yes | Elements to skip per dimension step |
| `dtype` | text | Yes | Data type string (see [Data Types](dtypes.md)) |
| *any other key* | any | No | Per-object application metadata |

### Example Object Descriptor

```json
{
  "type": "ntensor",
  "ndim": 3,
  "shape": [721, 1440, 30],
  "strides": [43200, 30, 1],
  "dtype": "float32",
  "mars": {
    "param": "wave_spectra",
    "levtype": "sfc"
  }
}
```

## Payload Descriptor (`payload[i]`)

| Key | Type | Required | Description |
|---|---|---|---|
| `byte_order` | text | Yes | `"big"` or `"little"` |
| `encoding` | text | Yes | `"none"` or `"simple_packing"` |
| `filter` | text | Yes | `"none"` or `"shuffle"` |
| `compression` | text | Yes | `"none"` or `"szip"` |
| `hash` | map | No | Integrity hash (see below) |
| *encoding params* | various | Conditional | Required when `encoding != "none"` |
| *filter params* | various | Conditional | Required when `filter != "none"` |

### Encoding Parameters (simple_packing)

| Key | Type | Description |
|---|---|---|
| `reference_value` | float | Minimum value in the original data |
| `binary_scale_factor` | int | Power-of-2 scaling factor |
| `decimal_scale_factor` | int | Power-of-10 scaling factor |
| `bits_per_value` | uint | Number of bits per packed value (1–64) |

### Filter Parameters (shuffle)

| Key | Type | Description |
|---|---|---|
| `shuffle_element_size` | uint | Byte width of each element (e.g. 4 for float32) |

### Hash Descriptor

| Key | Type | Description |
|---|---|---|
| `type` | text | `"xxh3"`, `"sha1"`, or `"md5"` |
| `value` | text | Hex-encoded digest |

## Canonical Encoding

All map keys are sorted by the byte representation of their CBOR-encoded key (RFC 8949 §4.2). This is done recursively — nested maps are also sorted. The goal is deterministic output: the same metadata always produces the same bytes.

For short string keys (the common case), this is equivalent to sorting by the key string itself. For long keys or non-string keys the CBOR byte encoding determines the order.

> **Why does this matter?** If you hash the entire message or compare messages by hash, deterministic encoding ensures that logically identical messages produce identical hashes even if the keys were inserted in different order.
