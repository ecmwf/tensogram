# Edge Cases

A collection of non-obvious situations and how the library handles them.

## Corrupted Messages

**What happens:** The scanner (`scan()`) reads `total_length` from the header and checks that `message[total_length - 8..]` equals `39277777`. If it does not, the message is considered corrupted.

**Recovery:** The scanner skips one byte and resumes searching from the next position. This means a single corrupted message in a multi-message file does not prevent reading the others.

```rust
let offsets = scan(&file_bytes);
// offsets only contains valid (start, length) pairs
// Corrupted regions are silently skipped
```

**Edge case within edge case:** If a random byte sequence inside a valid payload happens to match `TENSOGRM`, the scanner might try to parse a "message" starting mid-payload. The `total_length` cross-check catches this: the false start's total_length will point to a position without a `39277777` terminator.

## NaN in Simple Packing

Simple packing cannot represent NaN. The quantization formula maps the range `[min, max]` onto integers, and NaN has no defined place in this range.

**What happens:** `compute_params()` returns `TensogramError::Encoding("NaN values not supported in simple_packing")` if any value is NaN.

**Solution:** Replace NaN values with a sentinel (e.g. the minimum representable value, or a separate bitmask object) before encoding.

## Objects/Payload/Data Length Mismatch

`encode()` requires that `metadata.objects.len()`, `metadata.payload.len()`, and `data.len()` are all equal.

**What happens:** `TensogramError::Object("objects, payload, and data slices must have the same length")` is returned immediately, before any encoding work is done.

## Decode Range on Compressed Data

`decode_range()` can only return a sub-slice of an uncompressed, unencoded payload. Compressed payloads must be fully decompressed before any element can be accessed.

**What happens:** `TensogramError::Encoding("decode_range not supported for encoded/compressed objects")` is returned.

**Workaround:** Decode the full object with `decode_object()` and slice the result in memory.

## Bitmask Byte Width

`Dtype::Bitmask` returns `0` from `byte_width()`. This is a sentinel, not a real byte width.

**Why:** A bitmask of N elements occupies `ceil(N / 8)` bytes. The library cannot infer N from the byte width alone, so the "element size" concept doesn't apply. Callers that need the payload size must compute it from the element count.

```rust
let payload_bytes = if descriptor.dtype == Dtype::Bitmask {
    let num_elements: usize = descriptor.shape.iter().product::<u64>() as usize;
    (num_elements + 7) / 8
} else {
    let num_elements: usize = descriptor.shape.iter().product::<u64>() as usize;
    num_elements * descriptor.dtype.byte_width()
};
```

## verify_hash on Messages Without Hashes

If a message was encoded with `hash_algorithm: None` (no hash), and you decode it with `verify_hash: true`, the decoder silently skips hash verification for that object. No error is returned.

**Rationale:** The absence of a hash is not an error. The decoder cannot verify what was never stored. If you need to enforce that all messages have hashes, check `payload[i].hash.is_some()` after decoding metadata.

## Constant-Value Fields with simple_packing

If all values in a field are identical (range = 0), `compute_params()` sets `binary_scale_factor` such that all packed integers are 0, and the full value is recovered from `reference_value` alone. This is correct and handled without special cases.

## Very Short Buffers

Passing a buffer shorter than 40 bytes to any decode function returns `TensogramError::Framing("buffer too short for header")`. No panic.

## Object Index Out of Range

`decode_object(&message, 99, &options)` when the message has fewer than 100 objects returns `TensogramError::Object("index out of range")`.

## Empty Files

`TensogramFile::message_count()` returns `0`. `read_message(0)` returns `TensogramError::Object("message index out of range")`.

## CBOR Key Ordering

The library uses canonical CBOR key ordering (RFC 8949 §4.2). If you construct a `Metadata` struct with keys in one order and then check the CBOR bytes, the bytes may not match your insertion order. This is intentional and correct — it ensures deterministic output.

If you need to compare metadata across languages or implementations, always compare the decoded values, not the raw CBOR bytes from different encoders.
