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
    ranges: &[(usize, usize)],  // (offset, count) in flattened element order
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

> **Pre-encoded messages:** Messages produced via `encode_pre_encoded` only support `decode_range` if the caller provided the necessary bit-precise `szip_block_offsets` (see [Pre-encoded Payloads](./encode-pre-encoded.md)).

> **Edge case:** `decode_range` works with all encoding+compression combinations that support random access: uncompressed data, `simple_packing` (bit extraction), `szip` (RSI block seeking), `blosc2` (chunk access), and `zfp` fixed-rate mode. It returns an error for the `shuffle` filter (byte rearrangement breaks contiguous sample ranges) and for stream compressors (`zstd`, `lz4`, `sz3`) that don't support partial decode.

## DecodeOptions

```rust
pub struct DecodeOptions {
    /// If true, verify the hash of each decoded payload.
    pub verify_hash: bool,
    /// When true (the default), decoded payloads are converted to the
    /// caller's native byte order. Set to false to receive bytes in the
    /// message's declared wire byte order.
    pub native_byte_order: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self { verify_hash: false, native_byte_order: true }
    }
}
```

### Native byte order (default)

By default, all decoded data is returned in the caller's **native byte order** — the library handles any necessary byte-swapping automatically. You never need to check `byte_order` or call `.byteswap()`:

```rust
let (_, objects) = decode(&message, &DecodeOptions::default())?;
let floats: Vec<f32> = objects[0].1
    .chunks_exact(4)
    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
    .collect();
```

In Python, numpy arrays are always directly usable:

```python
_, objects = tensogram.decode(msg)
arr = objects[0][1]   # numpy array — values are correct, no byteswap needed
```

This applies to all decode functions (`decode`, `decode_object`, `decode_range`), all encodings (`none`, `simple_packing`), all compression codecs, and all language bindings (Rust, Python, C, C++).

### Wire byte order (opt-in)

Set `native_byte_order: false` to receive the raw bytes in the message's declared wire byte order. This is useful for zero-copy forwarding or when you need the exact on-wire representation:

```rust
let opts = DecodeOptions { native_byte_order: false, ..Default::default() };
let (_, objects) = decode(&message, &opts)?;
// objects[0].1 is in the descriptor's declared byte_order (e.g. big-endian)
```

### Hash verification

Hash verification is opt-in. Enable it when data integrity is critical:

```rust
let options = DecodeOptions { verify_hash: true, ..Default::default() };
let result = decode(&message, &options);
// Returns Err(TensogramError::HashMismatch { expected, actual }) if corrupted
```

> **Edge case:** If the descriptor has no hash (i.e. the message was encoded with `hash_algorithm: None`), `verify_hash: true` silently skips verification for that object. No error is returned.

## Working with the Decoded Bytes

Decoded bytes are in native byte order (with the default `DecodeOptions`). Cast them as native:

```rust
// float32 object → use from_ne_bytes
let floats: Vec<f32> = data
    .chunks_exact(4)
    .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
    .collect();
```

For simple_packing decoded data, the output is always **f64** bytes (8 bytes per element), regardless of the original dtype stored in the descriptor:

```rust
// simple_packing always decodes to f64, in native byte order
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
