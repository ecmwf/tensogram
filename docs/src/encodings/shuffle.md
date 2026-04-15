# Byte Shuffle Filter

The shuffle filter rearranges the bytes of a multi-byte array to improve compression. It is the same algorithm used by HDF5 and NetCDF4.

## Why Shuffle Helps

For float32 data, each value occupies 4 bytes. The bytes within a float are not independent — nearby values tend to share their most-significant bytes (exponent + high mantissa) while the least-significant bytes are more random.

Without shuffle, the bytes are interleaved:

```
[B0 B1 B2 B3][B0 B1 B2 B3][B0 B1 B2 B3]...
```

A compressor sees `B0 B1 B2 B3 B0 B1 B2 B3 B0 B1 B2 B3 ...` — not very compressible because the predictable (B0, B1) bytes are mixed with the random (B3) bytes.

After shuffle, all byte-0s come first, then all byte-1s, etc.:

```
[B0 B0 B0 ...][B1 B1 B1 ...][B2 B2 B2 ...][B3 B3 B3 ...]
```

Now the B0 run and B1 run are highly compressible (long runs of similar values). The B3 run is still noisy, but it's isolated. Overall compression improves significantly.

## API

### shuffle

```rust
pub fn shuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>, ShuffleError>
```

Rearranges bytes. `element_size` is the byte width of each element (e.g. 4 for float32, 8 for float64).

```rust
let floats: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let raw: Vec<u8> = floats.iter().flat_map(|f| f.to_ne_bytes()).collect();
let shuffled = shuffle(&raw, 4).expect("aligned to 4 bytes");
// shuffled is ready for compression
```

### unshuffle

```rust
pub fn unshuffle(data: &[u8], element_size: usize) -> Result<Vec<u8>, ShuffleError>
```

Reverses the shuffle. Applied automatically by the decode pipeline.

## Using Shuffle in a Message

Set `filter: "shuffle"` in the payload descriptor and provide `shuffle_element_size`:

```rust
use ciborium::Value;

let mut params = BTreeMap::new();
params.insert(
    "shuffle_element_size".to_string(),
    Value::Integer(4.into()), // 4 bytes per float32
);

let desc = DataObjectDescriptor {
    obj_type: "ntensor".to_string(),
    ndim: 1,
    shape: vec![100],
    strides: vec![1],
    dtype: Dtype::Float32,
    byte_order: ByteOrder::Big,
    encoding: "none".to_string(),
    filter: "shuffle".to_string(),
    compression: "none".to_string(),
    params,
    hash: None,
};
```

## Edge Cases

### Element Size Must Divide the Buffer

The shuffle operation requires `data.len() % element_size == 0`. If this is not true, the function returns `Err(ShuffleError::Misaligned)`. Ensure your data buffer is a whole number of elements.

### Shuffle Alone Does Not Compress

Shuffle rearranges bytes but does not reduce the total byte count. It only helps when followed by a compression stage (e.g. szip, zstd, lz4, blosc2). Set `compression` in the descriptor to apply compression after the shuffle step.

### Combining with simple_packing

When using both `encoding: "simple_packing"` and `filter: "shuffle"`, the pipeline applies them in order: encode first, then shuffle. The simple_packing output is 1-byte-per-packed-chunk (MSB-first bits), so `shuffle_element_size` should be 1 in this case (no benefit from shuffling already-packed data). In practice, the combination is unusual — either use simple_packing alone (for weather data) or shuffle alone (before a lossless compressor).
