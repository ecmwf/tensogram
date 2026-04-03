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
| `compression` | text | Yes | `"none"`, `"szip"`, `"zstd"`, `"lz4"`, `"blosc2"`, `"zfp"`, or `"sz3"` |
| `hash` | map | No | Integrity hash (see below) |
| *encoding params* | various | Conditional | Required when `encoding != "none"` |
| *filter params* | various | Conditional | Required when `filter != "none"` |
| *compression params* | various | Conditional | Required when `compression != "none"` |

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

### Compression Parameters

**szip:**

| Key | Type | Description |
|---|---|---|
| `szip_rsi` | uint | Reference sample interval |
| `szip_block_size` | uint | Block size (typically 8 or 16) |
| `szip_flags` | uint | AEC encoding flags |
| `szip_block_offsets` | array of uint | Bit offsets of RSI block boundaries (computed during encoding) |

**zstd:**

| Key | Type | Default | Description |
|---|---|---|---|
| `zstd_level` | int | 3 | Compression level (1–22) |

**lz4:** No parameters required.

**blosc2:**

| Key | Type | Default | Description |
|---|---|---|---|
| `blosc2_codec` | text | `"lz4"` | Internal codec: `blosclz`, `lz4`, `lz4hc`, `zlib`, `zstd` |
| `blosc2_clevel` | int | 5 | Compression level (0–9) |
| `blosc2_typesize` | uint | (auto) | Element byte width for shuffle optimization |

**zfp:**

| Key | Type | Description |
|---|---|---|
| `zfp_mode` | text | `"fixed_rate"`, `"fixed_precision"`, or `"fixed_accuracy"` |
| `zfp_rate` | float | Bits per value (only for `fixed_rate`) |
| `zfp_precision` | uint | Bit planes to keep (only for `fixed_precision`) |
| `zfp_tolerance` | float | Max absolute error (only for `fixed_accuracy`) |

**sz3:**

| Key | Type | Description |
|---|---|---|
| `sz3_error_bound_mode` | text | `"abs"`, `"rel"`, or `"psnr"` |
| `sz3_error_bound` | float | Error bound value |

### Hash Descriptor

| Key | Type | Description |
|---|---|---|
| `type` | text | `"xxh3"`, `"sha1"`, or `"md5"` |
| `value` | text | Hex-encoded digest |

## Canonical Encoding

All map keys are sorted by the byte representation of their CBOR-encoded key (RFC 8949 §4.2). This is done recursively — nested maps are also sorted. The goal is deterministic output: the same metadata always produces the same bytes.

For short string keys (the common case), this is equivalent to sorting by the key string itself. For long keys or non-string keys the CBOR byte encoding determines the order.

> **Why does this matter?** If you hash the entire message or compare messages by hash, deterministic encoding ensures that logically identical messages produce identical hashes even if the keys were inserted in different order.
