# Encoding Data

This page covers the `encode()` function and `EncodeOptions` in detail.

## Function Signature

```rust
pub fn encode(
    global_metadata: &GlobalMetadata,
    descriptors: &[(&DataObjectDescriptor, &[u8])],
    options: &EncodeOptions,
) -> Result<Vec<u8>>
```

- `global_metadata` — reference to message-level metadata (version, base entries, `_extra_` fields)
- `descriptors` — a slice of `(descriptor, data)` pairs, one per object
- `options` — controls hash algorithm and compression backend selection (the `emit_preceders` field is reserved for future buffered-mode support; preceders are currently only emitted via `StreamingEncoder::write_preceder`)

Returns a complete, self-contained message as a `Vec<u8>`.

## EncodeOptions

```rust
pub struct EncodeOptions {
    /// Hash algorithm to use. None disables hashing entirely.
    pub hash_algorithm: Option<HashAlgorithm>,
    /// Reserved — buffered `encode()` rejects `true`. Use
    /// `StreamingEncoder::write_preceder()` instead.
    pub emit_preceders: bool,
    /// Which backend to use for szip / zstd when both FFI and pure-Rust
    /// implementations are compiled in.
    pub compression_backend: CompressionBackend,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
            emit_preceders: false,
            compression_backend: CompressionBackend::default(),
        }
    }
}
```

The default applies xxh3 hashing to every object payload. Use `None` to skip hashing:

```rust
let options = EncodeOptions {
    hash_algorithm: None,
    ..Default::default()
};
```

## What Encode Does

For each object, in order:

1. **Validate** — checks that each pair has a descriptor and corresponding data
2. **Run the encoding pipeline** — applies encoding, filter, compression from the object's `DataObjectDescriptor`
3. **Hash** — if `hash_algorithm` is set, computes and stores the hash in the descriptor
4. **Serialize CBOR** — encodes the `GlobalMetadata` and all `DataObjectDescriptor`s to canonical CBOR
5. **Frame** — assembles preamble, header frames (metadata/index/hash), data object frames, and postamble

## Encoding with Simple Packing

To use simple_packing, you need to compute the quantization parameters first, then put them in the `DataObjectDescriptor`:

```rust
use tensogram_encodings::simple_packing;
use ciborium::Value;

// Your original values as f64 (simple_packing always works on f64).
// source_data might be a temperature grid, pressure field, intensity
// image, or any other bounded-range scalar field.
let values: Vec<f64> = source_data.iter().map(|&x| x as f64).collect();

// Compute quantization parameters for 16 bits per value
let params = simple_packing::compute_params(&values, 16, 0)?;

// Put the parameters into the descriptor
let mut packing_params = BTreeMap::new();
packing_params.insert("reference_value".into(),
    Value::Float(params.reference_value));
packing_params.insert("binary_scale_factor".into(),
    Value::Integer((params.binary_scale_factor as i64).into()));
packing_params.insert("decimal_scale_factor".into(),
    Value::Integer((params.decimal_scale_factor as i64).into()));
packing_params.insert("bits_per_value".into(),
    Value::Integer((params.bits_per_value as i64).into()));

let desc = DataObjectDescriptor {
    obj_type: "ntensor".to_string(),
    ndim: 2,
    shape: vec![100, 200],
    strides: vec![200, 1],
    dtype: Dtype::Float64,
    byte_order: ByteOrder::Big,
    encoding: "simple_packing".to_string(),
    filter: "none".to_string(),
    compression: "none".to_string(),
    masks: None,
    params: packing_params,
};
```

Then encode as normal, passing the original raw bytes (as f64 bytes):

```rust
let raw: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

let global = GlobalMetadata::default();
let message = encode(&global, &[(&desc, &raw)], &EncodeOptions::default())?;
```

The encoder applies simple_packing internally. The payload stored in the message is the packed bits, not the original f64 bytes.

## Encoding Multiple Objects

Pass multiple `(descriptor, data)` pairs:

```rust
let global = GlobalMetadata::default();

let message = encode(
    &global,
    &[(&spectrum_desc, &spectrum_data), (&mask_desc, &land_mask_data)],
    &EncodeOptions::default(),
)?;
```

Each descriptor independently specifies its own encoding, compression, dtype, and byte order. The encoder processes each pair in sequence.

## Error Conditions

| Error | Cause |
|---|---|
| `Encoding` | NaN in data when using simple_packing |
| `Encoding` | bits_per_value out of range (0–64) |
| `Compression` | Compressor-specific error (invalid params, unsupported dtype) |
| `Metadata` | CBOR serialization failed |
