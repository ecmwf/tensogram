# Decoding Data

Tensogram provides four decode functions for different use cases. Choose the one that does the least work for your situation — they are all zero-copy on the metadata path.

## Four Decode Functions

### `decode` — full message

```rust
pub fn decode(
    message: &[u8],
    options: &DecodeOptions,
) -> Result<(Metadata, Vec<Vec<u8>>)>
```

Decodes all objects. Returns the metadata and a vector of raw byte buffers (one per object, in the logical dtype after de-quantization).

```rust
let (meta, objects) = decode(&message, &DecodeOptions::default())?;
// objects[0] is Vec<u8> — the raw float32/float64/etc bytes
```

### `decode_metadata` — metadata only

```rust
pub fn decode_metadata(message: &[u8]) -> Result<Metadata>
```

Reads only the CBOR section. Does not touch any payload bytes. Use this for filtering and listing.

```rust
let meta = decode_metadata(&message)?;
println!("shape: {:?}", meta.objects[0].shape);
```

### `decode_object` — single object by index

```rust
pub fn decode_object(
    message: &[u8],
    index: usize,
    options: &DecodeOptions,
) -> Result<(ObjectDescriptor, Vec<u8>)>
```

Decodes one object without reading the others. Uses the binary header's offset table to seek directly to the right payload. O(1) seek regardless of how many objects the message contains.

```rust
// Decode only the second object (index 1)
let (descriptor, payload) = decode_object(&message, 1, &DecodeOptions::default())?;
```

> **Edge case:** If `index >= num_objects`, returns `TensogramError::Object("index out of range")`.

### `decode_range` — partial sub-tensor (uncompressed only)

```rust
pub fn decode_range(
    message: &[u8],
    object_index: usize,
    ranges: &[(usize, usize)],  // (start, count) per flattened dimension
    options: &DecodeOptions,
) -> Result<Vec<u8>>
```

Decodes a contiguous slice of elements from an uncompressed, unencoded object. Useful when you have a large object and only need a subset of it.

```rust
// Elements 100 through 149 of object 0
let partial = decode_range(&message, 0, &[(100, 50)], &DecodeOptions::default())?;
```

> **Edge case:** `decode_range` only works for `encoding: "none"` and `compression: "none"`. It returns an error for simple_packing or any compressed object because those formats require decoding from the beginning.

## DecodeOptions

```rust
pub struct DecodeOptions {
    /// If true, verify the hash of each decoded payload. Returns an error on mismatch.
    pub verify_hash: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self { verify_hash: false }
    }
}
```

Hash verification is opt-in. Enable it when data integrity is critical:

```rust
let options = DecodeOptions { verify_hash: true };
let result = decode(&message, &options);
// Returns Err(TensogramError::HashMismatch { expected, actual }) if corrupted
```

> **Edge case:** If the payload descriptor has no hash (i.e. the message was encoded with `hash_algorithm: None`), `verify_hash: true` silently skips verification for that object. No error is returned.

## Working with the Decoded Bytes

The decoded bytes are in the **logical dtype** of the object. For example, a float32 object returns 4 bytes per element in native endianness after decoding. Cast them the same way you would any raw buffer:

```rust
// object[0] is a float32 tensor, decoded
let floats: Vec<f32> = objects[0]
    .chunks_exact(4)
    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
    .collect();
```

For simple_packing decoded data, the output is always **f64** bytes (8 bytes per element), regardless of the original dtype stored in the descriptor:

```rust
// simple_packing always decodes to f64
let values: Vec<f64> = objects[0]
    .chunks_exact(8)
    .map(|c| f64::from_ne_bytes(c.try_into().unwrap()))
    .collect();
```

## Scanning for Messages First

If you're working with a buffer that might contain multiple messages (e.g. a `.tgm` file loaded into memory), scan it first to get message boundaries:

```rust
let offsets = scan(&big_buffer); // Vec<(usize, usize)> = (start, length)

for (start, len) in offsets {
    let msg = &big_buffer[start..start + len];
    let meta = decode_metadata(msg)?;
    // ...
}
```

The `scan` function is tolerant of corruption — it skips invalid regions and continues looking for the next valid `TENSOGRM` marker.
