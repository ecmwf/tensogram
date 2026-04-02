# Data Types

The `dtype` field in an object descriptor names the element type of the tensor. It is stored as a lowercase text string in CBOR.

## Type Table

| CBOR string | Rust variant | Bytes per element | Notes |
|---|---|---|---|
| `float16` | `Dtype::Float16` | 2 | IEEE 754 half-precision |
| `bfloat16` | `Dtype::Bfloat16` | 2 | Brain float — same exponent range as float32, less mantissa precision |
| `float32` | `Dtype::Float32` | 4 | IEEE 754 single-precision |
| `float64` | `Dtype::Float64` | 8 | IEEE 754 double-precision |
| `complex64` | `Dtype::Complex64` | 8 | Pair of float32 (real, imaginary) |
| `complex128` | `Dtype::Complex128` | 16 | Pair of float64 (real, imaginary) |
| `int8` | `Dtype::Int8` | 1 | Signed |
| `int16` | `Dtype::Int16` | 2 | Signed |
| `int32` | `Dtype::Int32` | 4 | Signed |
| `int64` | `Dtype::Int64` | 8 | Signed |
| `uint8` | `Dtype::Uint8` | 1 | Unsigned |
| `uint16` | `Dtype::Uint16` | 2 | Unsigned |
| `uint32` | `Dtype::Uint32` | 4 | Unsigned |
| `uint64` | `Dtype::Uint64` | 8 | Unsigned |
| `bitmask` | `Dtype::Bitmask` | 0* | Packed bits |

*`bitmask` returns `0` from `byte_width()` — see the edge case note below.

## Byte Order

The `byte_order` field in the payload descriptor specifies whether multi-byte elements are stored in big-endian (`"big"`) or little-endian (`"little"`) order. This applies to the **stored payload bytes** after encoding.

Single-byte types (`int8`, `uint8`, `bitmask`) are unaffected by byte order.

## Bitmask Edge Case

`Dtype::Bitmask` is for packing boolean or categorical data sub-byte. The payload size is `ceil(num_elements / 8)` bytes. The `byte_width()` method returns `0` as a sentinel; callers that need the actual payload size must compute it:

```rust
let payload_bytes = if dtype == Dtype::Bitmask {
    (num_elements + 7) / 8
} else {
    num_elements * dtype.byte_width()
};
```

## Choosing a dtype

| Situation | Recommended dtype |
|---|---|
| Temperature, wind speed, pressure (weather) | `float32` |
| High-precision scientific analysis | `float64` |
| ML model weights | `bfloat16` or `float16` |
| Integer indices, counts | `int32` or `int64` |
| Land-sea masks, validity flags | `uint8` or `bitmask` |
| Complex wave spectra | `complex64` |
