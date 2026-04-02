# Compression

Compression is the third stage of the encoding pipeline. It reduces the total byte count of the already-encoded and filtered payload.

## Current Status

| Compressor | Status | Notes |
|---|---|---|
| `none` | Stable | Pass-through — no compression |
| `szip` | Stub | Returns an error — libaec bindings not yet implemented |

Setting `compression: "szip"` in your payload descriptor will cause encoding to fail with `TensogramError::Compression("szip not available")` until the bindings are implemented.

## The Compressor Trait

The compression stage is designed for extensibility. New compressors can be added by implementing the `Compressor` trait:

```rust
pub trait Compressor {
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;
    fn decompress(&self, data: &[u8], original_len: usize) -> Result<Vec<u8>>;
}
```

## Szip / libaec

Szip (implemented by the libaec C library) is a lossless compressor designed specifically for scientific floating-point data. It exploits the block structure of packed data. Key parameters once the bindings are added:

| Parameter key | Type | Description |
|---|---|---|
| `szip_rsi` | uint | Reference sample interval (pixels per block) |
| `szip_block_size` | uint | Block size (typically 8 or 16) |
| `szip_flags` | uint | Szip flags (NN/EC mode flags) |
| `szip_block_offsets` | array of uint | Per-block byte offsets for random access |

The `szip_block_offsets` array enables seeking to a specific block for partial decode, which is needed for range decoding of compressed data.

## What to Use Today

Until szip is implemented, the best options are:

1. **simple_packing alone** — 4-8x size reduction for typical float data, no compression needed
2. **shuffle + future szip** — best lossless option (shuffle dramatically improves szip's ratio)
3. **none + none** — exact copy, maximum decode speed

For most weather data use cases, `simple_packing` at 16 bits gives adequate compression with essentially no precision loss.
