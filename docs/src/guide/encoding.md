# Encoding Data

This page covers the `encode()` function and `EncodeOptions` in detail.

## Function Signature

```rust
pub fn encode(
    metadata: &Metadata,
    data: &[&[u8]],
    options: &EncodeOptions,
) -> Result<Vec<u8>>
```

- `metadata` — describes all objects and their payload configuration
- `data` — a slice of raw byte slices, one per object
- `options` — controls hash algorithm

Returns a complete, self-contained message as a `Vec<u8>`.

## EncodeOptions

```rust
pub struct EncodeOptions {
    /// Hash algorithm to use. None disables hashing entirely.
    pub hash_algorithm: Option<HashAlgorithm>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            hash_algorithm: Some(HashAlgorithm::Xxh3),
        }
    }
}
```

The default applies xxh3 hashing to every object payload. Use `None` to skip hashing:

```rust
let options = EncodeOptions {
    hash_algorithm: None,
};
```

## What Encode Does

For each object, in order:

1. **Validate** — checks that `objects.len() == payload.len() == data.len()`
2. **Run the encoding pipeline** — applies encoding, filter, compression from the payload descriptor
3. **Hash** — if `hash_algorithm` is set, computes and stores the hash in the payload descriptor
4. **Serialize CBOR** — encodes the (now complete) metadata to canonical CBOR
5. **Frame** — assembles magic, header, CBOR block, OBJS/payload/OBJE blocks, terminator

## Encoding with Simple Packing

To use simple_packing, you need to compute the quantization parameters first, then put them in the payload descriptor:

```rust
use tensogram_encodings::simple_packing;
use ciborium::Value;

// Your original values as f64 (simple_packing always works on f64)
let values: Vec<f64> = temperature_data.iter().map(|&x| x as f64).collect();

// Compute quantization parameters for 16 bits per value
let params = simple_packing::compute_params(&values, 16, 0)?;

// Put the parameters into the payload descriptor
let mut packing_params = BTreeMap::new();
packing_params.insert("reference_value".into(),
    Value::Float(params.reference_value));
packing_params.insert("binary_scale_factor".into(),
    Value::Integer((params.binary_scale_factor as i64).into()));
packing_params.insert("decimal_scale_factor".into(),
    Value::Integer((params.decimal_scale_factor as i64).into()));
packing_params.insert("bits_per_value".into(),
    Value::Integer((params.bits_per_value as i64).into()));

let payload = PayloadDescriptor {
    byte_order: ByteOrder::Big,
    encoding: "simple_packing".to_string(),
    filter: "none".to_string(),
    compression: "none".to_string(),
    params: packing_params,
    hash: None,
};
```

Then encode as normal, passing the original raw bytes (as f64 bytes):

```rust
let raw: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
let message = encode(&metadata, &[&raw], &EncodeOptions::default())?;
```

The encoder applies simple_packing internally. The payload stored in the message is the packed bits, not the original f64 bytes.

## Encoding Multiple Objects

Pass multiple data slices, one per object:

```rust
let message = encode(
    &metadata,
    &[&spectrum_data, &land_mask_data],
    &EncodeOptions::default(),
)?;
```

`metadata.objects`, `metadata.payload`, and the data slice must all have the same length. The encoder checks this and returns `TensogramError::Object` if they differ.

## Error Conditions

| Error | Cause |
|---|---|
| `Object` | `objects.len()` ≠ `payload.len()` ≠ `data.len()` |
| `Encoding` | NaN in data when using simple_packing |
| `Encoding` | bits_per_value out of range (0–64) |
| `Compression` | Compressor-specific error (invalid params, unsupported dtype) |
| `Metadata` | CBOR serialization failed |
