# Pre-Encoded Data API (Advanced)

## When to use this API

The `encode_pre_encoded` API is for **advanced callers** whose data is already
encoded by an external pipeline (e.g., a GPU kernel that emits packed bytes,
or a streaming receiver passing payloads through). It bypasses Tensogram's
internal encoding pipeline and uses the supplied bytes verbatim.

Do NOT use this API for ordinary encoding. Use `encode()` instead.

## ⚠️ The bit-vs-byte trap

> **WARNING**: When using `compression="szip"`, the `szip_block_offsets` parameter
> contains **bit offsets**, not byte offsets. The first offset must be 0 and
> every offset must satisfy `offset <= encoded_bytes_len * 8`. This matches
> the libaec/szip wire format. See [cbor-metadata.md](../format/cbor-metadata.md#szip-block-offsets)
> for the format reference.
>
> Getting this wrong is the #1 caller mistake. Tensogram validates the offsets
> structurally (monotonicity, bounds) but cannot detect a byte-instead-of-bit
> mistake until decode_range fails.

## API surface

### Rust
```rust
pub fn encode_pre_encoded(
    metadata: &GlobalMetadata,
    descriptors_and_data: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
) -> Result<Vec<u8>, TensogramError>
```

### Python
```python
import tensogram

msg: bytes = tensogram.encode_pre_encoded(
    global_meta={"name": "demo"},
    descriptors_and_data=[(descriptor_dict, raw_bytes)],
    hash="xxh3",
)
```

### C
```c
tgm_error tgm_encode_pre_encoded(
    const char *metadata_json,
    const uint8_t *const *data_ptrs,
    const size_t *data_lens,
    size_t num_objects,
    const char *hash_algo,
    tgm_bytes_t *out
);
```

### C++
```cpp
std::vector<uint8_t> tensogram::encode_pre_encoded(
    const GlobalMetadata &meta,
    const std::vector<std::pair<DataObjectDescriptor, std::span<const uint8_t>>> &pairs,
    const EncodeOptions &opts
);
```

## Hash semantics

The library **always recomputes** the hash of the pre-encoded bytes using
the algorithm specified in `EncodeOptions.hash` (default `xxh3`). Any hash
the caller stored on the descriptor is silently overwritten. This guarantees
the wire format invariant `descriptor.hash == hash_algo(bytes)` always holds.

## Provenance semantics

The encoded message is byte-format-indistinguishable from one produced by
`encode()`. The decoder cannot tell which API produced it. The provenance
fields `_reserved_.encoder.name`, `_reserved_.time`, and `_reserved_.uuid`
are populated identically.

## Self-consistency checks

Before encoding, the library validates:
1. Caller has not set `EncodeOptions.emit_preceders` (rejected).
2. Caller has not put `_reserved_` in their metadata (rejected).
3. Each descriptor passes the standard `validate_object` checks.
4. If `compression="szip"` and `szip_block_offsets` is supplied:
   - It's a CBOR Array of u64.
   - First offset is 0.
   - Strictly monotonically increasing.
   - All bit offsets `<= bytes_len * 8`.
5. If `szip_block_offsets` is supplied but `compression != "szip"`, rejected.

These are **structural** checks only. The library does NOT trial-decode the
bytes to verify they actually decode correctly.

## Worked example: simple_packing + szip with decode_range

```rust
use tensogram_core::{encode_pre_encoded, decode_range, GlobalMetadata, EncodeOptions};
use std::collections::BTreeMap;

// Pre-encoded bytes from a GPU kernel + szip block offsets in BITS
let pre_encoded_bytes: Vec<u8> = /* from GPU */;
let szip_offsets_bits: Vec<u64> = vec![0, 8192, 16384, /* ... */];

let mut params: BTreeMap<String, ciborium::Value> = BTreeMap::new();
params.insert("bits_per_value".into(), 24u64.into());
params.insert("reference_value".into(), 0.0_f64.into());
params.insert("scale".into(), 0.001_f64.into());
params.insert("szip_block_offsets".into(),
    ciborium::Value::Array(szip_offsets_bits.into_iter()
        .map(|o| ciborium::Value::Integer(o.into()))
        .collect()));

let desc = DataObjectDescriptor {
    name: "demo".into(),
    encoding: "simple_packing".into(),
    compression: "szip".into(),
    shape: vec![1024, 1024],
    dtype: "f32".into(),
    params,
    ..Default::default()
};

let msg = encode_pre_encoded(
    &GlobalMetadata::default(),
    &[(&desc, &pre_encoded_bytes)],
    &EncodeOptions::default(),
)?;

// Now decode_range works because szip_block_offsets is present:
let partial = decode_range(&msg, 0, /* range */ 0..512)?;
```

## How it works (mermaid)

```mermaid
flowchart LR
    A[Caller bytes] -->|encode_pre_encoded| B[validate_object]
    B --> C[validate_szip_block_offsets]
    C --> D[Recompute hash]
    D --> E[Wrap in CBOR framing]
    E --> F[Wire message]
    
    G[Caller bytes] -.->|encode| H[Run pipeline]
    H -.-> D
```

The pre-encoded path skips the pipeline entirely. The wire format is identical.

## Cross-references

- [Encoding](encoding.md) — the normal `encode()` API
- [Decoding](decoding.md) — `decode_range` requirements for partial reads
- [Compression](../encodings/compression.md) — szip details
- [CBOR metadata](../format/cbor-metadata.md) — wire format reference
