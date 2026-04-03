# CBOR Metadata Schema

Tensogram v2 uses CBOR (Concise Binary Object Representation) for all structured metadata. There are four kinds of CBOR structures in a message, each living in its own frame:

1. **GlobalMetadata** — in header or footer metadata frames
2. **DataObjectDescriptor** — inside each data object frame
3. **IndexFrame** — in header or footer index frames
4. **HashFrame** — in header or footer hash frames

All CBOR maps use **deterministic encoding** with canonical key ordering per RFC 8949 section 4.2. Keys are sorted by the byte representation of their CBOR-encoded key, applied recursively to nested maps. This means the same metadata always produces the same bytes — important if you hash messages or compare them by digest.

## GlobalMetadata

The global metadata frame contains a single CBOR map. The only required key is `version`; everything else is application-defined.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `version` | uint | Yes | Format version. Currently `2` |
| *any other key* | any | No | Application metadata (e.g., `"mars"`, `"source"`, `"timestamp"`) |

Unlike v1, the global metadata does **not** contain an `objects` or `payload` array. Each data object is self-describing via its own descriptor (see below).

### Example GlobalMetadata

```json
{
  "version": 2,
  "mars": {
    "class": "od",
    "stream": "oper",
    "expver": "0001",
    "date": "20260404",
    "time": "0000",
    "step": "0"
  },
  "source": "ifs-cycle49r2"
}
```

## DataObjectDescriptor

Each data object frame contains its own CBOR descriptor. This descriptor fully describes how to decode the payload — its type, shape, encoding pipeline, and optional per-object metadata. It lives **inside** the data object frame (not in a central metadata block).

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `obj_type` | text | Yes | Object type, e.g. `"ntensor"` |
| `ndim` | uint | Yes | Number of dimensions |
| `shape` | array of uint | Yes | Size of each dimension |
| `strides` | array of uint | Yes | Element stride per dimension |
| `dtype` | text | Yes | Data type string (see [Data Types](dtypes.md)) |
| `byte_order` | text | Yes | `"big"` or `"little"` |
| `encoding` | text | Yes | `"none"` or `"simple_packing"` |
| `filter` | text | Yes | `"none"` or `"shuffle"` |
| `compression` | text | Yes | `"none"`, `"szip"`, `"zstd"`, `"lz4"`, `"blosc2"`, `"zfp"`, or `"sz3"` |
| `hash` | map | No | Integrity hash of the payload (see below) |
| *encoding params* | various | Conditional | Required when `encoding != "none"` |
| *filter params* | various | Conditional | Required when `filter != "none"` |
| *compression params* | various | Conditional | Required when `compression != "none"` |
| *any other key* | any | No | Per-object application metadata |

### Example: Temperature Field with MARS Metadata

Here is what a descriptor might look like for a global temperature field at 0.25-degree resolution, compressed with zstd:

```json
{
  "obj_type": "ntensor",
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
  },
  "mars": {
    "param": "temperature",
    "levtype": "pl",
    "levelist": "500"
  }
}
```

Notice how MARS metadata sits alongside the encoding parameters — the descriptor is a flat map with well-known keys for encoding and free-form keys for application use.

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
| `szip_block_offsets` | array of uint | Bit offsets of RSI block boundaries (computed during encoding) |

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
| `type` | text | `"xxh3"`, `"sha1"`, or `"md5"` |
| `value` | text | Hex-encoded digest |

## IndexFrame

Index frames (header or footer) contain a CBOR map that lets readers jump directly to any data object without scanning.

| Key | Type | Description |
|-----|------|-------------|
| `object_count` | uint | Number of data objects in the message |
| `offsets` | array of uint | Byte offset of each data object frame from message start |
| `lengths` | array of uint | Byte length of each data object frame |

### Example IndexFrame

```json
{
  "object_count": 3,
  "offsets": [256, 1048832, 2097408],
  "lengths": [1048576, 1048576, 524288]
}
```

The `offsets` array gives O(1) random access to any object — seek to `offsets[i]` and read `lengths[i]` bytes.

## HashFrame

Hash frames (header or footer) store per-object integrity hashes, allowing verification without reading the individual descriptors.

| Key | Type | Description |
|-----|------|-------------|
| `object_count` | uint | Number of data objects |
| `hash_type` | text | Hash algorithm: `"xxh3"`, `"sha1"`, or `"md5"` |
| `hashes` | array of text | Hex-encoded digest for each object, in order |

### Example HashFrame

```json
{
  "object_count": 3,
  "hash_type": "xxh3",
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
