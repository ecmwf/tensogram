# Decoding Data

Tensogram provides four decode functions for different use cases. Choose the one that does the least work for your situation — they are all zero-copy on the metadata path.

## The DecodedObject Type

Before diving in, it helps to know the common return type:

```rust
type DecodedObject = (DataObjectDescriptor, Vec<u8>);
```

A `DecodedObject` is a tuple of the object's descriptor (shape, dtype, encoding info, etc.) and the decoded raw bytes. You will see this pattern throughout the decode API.

## Four Decode Functions

### `decode` — full message

```rust
pub fn decode(
    message: &[u8],
    options: &DecodeOptions,
) -> Result<(GlobalMetadata, Vec<(DataObjectDescriptor, Vec<u8>)>)>
```

Decodes all objects. Returns the global metadata and a vector of `DecodedObject` tuples — one per object, with raw bytes in the logical dtype after de-quantization.

```rust
let (meta, objects) = decode(&message, &DecodeOptions::default())?;

// Each element is (DataObjectDescriptor, Vec<u8>)
let (ref desc, ref data) = objects[0];
println!("shape: {:?}, dtype: {}, bytes: {}", desc.shape, desc.dtype, data.len());
```

### `decode_metadata` — metadata only

```rust
pub fn decode_metadata(message: &[u8]) -> Result<GlobalMetadata>
```

Reads only the CBOR section. Does not touch any payload bytes. Use this for filtering and listing.

```rust
let meta = decode_metadata(&message)?;
println!("version: {}", meta.version);
```

### `decode_object` — single object by index

```rust
pub fn decode_object(
    message: &[u8],
    index: usize,
    options: &DecodeOptions,
) -> Result<(GlobalMetadata, DataObjectDescriptor, Vec<u8>)>
```

Decodes one object without reading the others. Uses the binary header's offset table to seek directly to the right payload. O(1) seek regardless of how many objects the message contains.

Returns the global metadata, the object's descriptor, and the decoded bytes as a three-element tuple.

```rust
// Decode only the second object (index 1)
let (meta, descriptor, payload) = decode_object(&message, 1, &DecodeOptions::default())?;
println!("shape: {:?}, dtype: {}", descriptor.shape, descriptor.dtype);
```

> **Edge case:** If `index >= num_objects`, returns `TensogramError::Object("index out of range")`.

### `decode_range` — partial sub-tensor

```rust
pub fn decode_range(
    message: &[u8],
    object_index: usize,
    ranges: &[(usize, usize)],  // (offset, count) per dimension
    options: &DecodeOptions,
) -> Result<Vec<Vec<u8>>>
```

Decodes one or more contiguous slices of elements from an object. Each `(offset, count)` pair in `ranges` selects a span of elements along the flattened dimension; the function returns **one byte vector per range** by default. This split-result design avoids an unnecessary copy when the caller needs the ranges individually (e.g. to feed separate array slices).

#### Rust — split results (default)

```rust
// Two separate ranges from object 0
let parts: Vec<Vec<u8>> = decode_range(
    &message, 0,
    &[(100, 50), (300, 25)],
    &DecodeOptions::default(),
)?;
assert_eq!(parts.len(), 2);           // one Vec<u8> per range
println!("first  range bytes: {}", parts[0].len());
println!("second range bytes: {}", parts[1].len());
```

#### Rust — joined result

If you prefer a single contiguous buffer, flatten the results:

```rust
let joined: Vec<u8> = parts.into_iter().flatten().collect();
```

#### Python — split results (default, `join=False`)

```python
import tensogram

parts = tensogram.decode_range(buf, object_index=0, ranges=[(100, 50), (300, 25)])
# parts is a list of numpy arrays, one per range
print(len(parts))        # 2
print(parts[0].shape)    # (50,)
```

#### Python — joined result (`join=True`)

```python
arr = tensogram.decode_range(buf, object_index=0, ranges=[(100, 50), (300, 25)], join=True)
# arr is a single flat numpy array with all ranges concatenated
print(arr.shape)          # (75,)
```

> **N-dimensional slicing:** The xarray backend maps N-dimensional slice notation
> (e.g. `ds["temperature"].sel(lat=slice(10, 20), lon=slice(30, 40))`) into the
> `(offset, count)` pairs that `decode_range` expects, so you rarely need to
> compute flattened offsets by hand when working through xarray.

> **Edge case:** `decode_range` works with all encoding+compression combinations that support random access: uncompressed data, `simple_packing` (bit extraction), `szip` (RSI block seeking), `blosc2` (chunk access), and `zfp` fixed-rate mode. It returns an error for the `shuffle` filter (byte rearrangement breaks contiguous sample ranges) and for stream compressors (`zstd`, `lz4`, `sz3`) that don't support partial decode.

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

> **Edge case:** If the descriptor has no hash (i.e. the message was encoded with `hash_algorithm: None`), `verify_hash: true` silently skips verification for that object. No error is returned.

## Working with the Decoded Bytes

The decoded bytes are in the **logical dtype** of the object. For example, a float32 object returns 4 bytes per element in native endianness after decoding. Cast them the same way you would any raw buffer:

```rust
// objects[0] is a (DataObjectDescriptor, Vec<u8>) for a float32 tensor
let (ref desc, ref data) = objects[0];
let floats: Vec<f32> = data
    .chunks_exact(4)
    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
    .collect();
```

For simple_packing decoded data, the output is always **f64** bytes (8 bytes per element), regardless of the original dtype stored in the descriptor:

```rust
// simple_packing always decodes to f64
let (ref _desc, ref data) = objects[0];
let values: Vec<f64> = data
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
    println!("version: {}", meta.version);
}
```

The `scan` function is tolerant of corruption — it skips invalid regions and continues looking for the next valid `TENSOGRM` marker.
