# Edge Cases

A collection of non-obvious situations and how the library handles them.

## Corrupted Messages

**What happens:** The scanner (`scan()`) searches for `TENSOGRM` magic bytes and validates the postamble (last 8 bytes should be `39277777`). If `total_length` is set, the scanner checks for the end magic at the expected position.

**Recovery:** If a message fails validation, the scanner skips one byte and resumes searching. A single corrupted message in a multi-message file does not prevent reading the others.

```rust
let offsets = scan(&file_bytes);
// offsets only contains valid (start, length) pairs
// Corrupted regions are silently skipped
```

**Edge case within edge case:** If a random byte sequence inside a valid payload happens to match `TENSOGRM`, the scanner might try to parse a "message" starting mid-payload. The postamble cross-check catches this: the false start's postamble won't contain the expected `39277777` end magic.

## NaN in Simple Packing

Simple packing cannot represent NaN. The quantization formula maps the range `[min, max]` onto integers, and NaN has no defined place in this range.

**What happens:** `compute_params()` returns `PackingError::NanValue(index)` if any value is NaN. The `encode()` function also rejects NaN inputs before packing.

**Solution:** Replace NaN values with a sentinel (e.g. the minimum representable value, or a separate bitmask object) before encoding.

## Decode Range on Compressed Data

`decode_range()` supports partial range decode for compressors that have random access capability: szip (via RSI block offsets), blosc2 (via chunk-based access), and zfp fixed-rate mode. Stream compressors (zstd, lz4, sz3) return `CompressionError::RangeNotSupported`.

**Workaround for stream compressors:** Decode the full object with `decode_object()` and slice the result in memory.

## Bitmask Byte Width

`Dtype::Bitmask` returns `0` from `byte_width()`. This is a sentinel, not a real byte width.

**Why:** A bitmask of N elements occupies `ceil(N / 8)` bytes. The library cannot infer N from the byte width alone, so the "element size" concept doesn't apply. Callers that need the payload size must compute it from the element count.

```rust
let num_elements: u64 = descriptor.shape.iter().product();
let payload_bytes = if descriptor.dtype == Dtype::Bitmask {
    let n = usize::try_from(num_elements)?;
    (n + 7) / 8
} else {
    let n = usize::try_from(num_elements)?;
    n * descriptor.dtype.byte_width()
};
```

## verify_hash on Messages Without Hashes

If a message was encoded with `hash_algorithm: None` (no hash), and you decode it with `verify_hash: true`, the decoder silently skips hash verification for that object. No error is returned.

**Rationale:** The absence of a hash is not an error. The decoder cannot verify what was never stored. If you need to enforce that all messages have hashes, check `descriptor.hash.is_some()` after decoding.

## Constant-Value Fields with simple_packing

If all values in a field are identical (range = 0), `compute_params()` sets `binary_scale_factor` such that all packed integers are 0, and the full value is recovered from `reference_value` alone. This is correct and handled without special cases.

## Very Short Buffers

Passing a buffer shorter than the preamble size (24 bytes) to any decode function returns `TensogramError::Framing("buffer too short ...")`. No panic.

## Object Index Out of Range

`decode_object(&message, 99, &options)` when the message has fewer than 100 objects returns `TensogramError::Object("object index N out of range")`.

## Empty Files

`TensogramFile::message_count()` returns `0`. `read_message(0)` returns an error.

## CBOR Key Ordering

The library uses canonical CBOR key ordering (RFC 8949 §4.2). If you construct a `GlobalMetadata` struct with keys in one order and then check the CBOR bytes, the bytes may not match your insertion order. This is intentional and correct — it ensures deterministic output.

If you need to compare metadata across languages or implementations, always compare the decoded values, not the raw CBOR bytes from different encoders.

You can verify that any CBOR output is canonical using the `verify_canonical_cbor()` utility:

```rust
use tensogram_core::verify_canonical_cbor;

let cbor_bytes = /* ... */;
verify_canonical_cbor(&cbor_bytes)?; // Returns Ok(()) if canonical, Err if not
```

## Frame Ordering Violations

The decoder validates that frames appear in the expected order: header frames first, then data object frames, then footer frames. A message with frames out of order (e.g. a header metadata frame appearing after a data object frame) is rejected with `TensogramError::Framing`.

This catches malformed or tampered messages. Valid messages produced by the encoder always have correct ordering.

## Streaming Mode (total_length = 0)

When encoding for a non-seekable output (e.g. TCP socket), the preamble's `total_length` is set to 0. In this mode:

- Header index and header hash frames are omitted (the encoder doesn't know the data object count or offsets upfront).
- The footer must contain at least the metadata frame.
- The `first_footer_offset` in the postamble points to the first footer frame.

Decoders that encounter `total_length = 0` should read from the postamble backward to find the footer frames, then use the footer index (if present) for random access to data objects.

## first_footer_offset is Never Zero

The postamble's `first_footer_offset` field always points to a valid position:
- If footer frames exist: it points to the start of the first footer frame.
- If no footer frames exist: it points to the start of the postamble itself.

This invariant means decoders can always seek to `first_footer_offset` and determine whether they've landed on a footer frame or the postamble.

## Inter-Frame Padding

The encoder may insert padding bytes between frames for memory alignment (e.g. 64-bit alignment). Padding appears between the `ENDF` marker of one frame and the `FR` marker of the next. Decoders should scan for the `FR` marker rather than assuming frames are contiguous.

## Zero-Element Tensors

Shapes containing zero dimensions are valid: `shape: [0]`, `shape: [3, 0, 5]`. This matches numpy and PyTorch semantics where zero-element tensors are legitimate objects (e.g. an empty batch). The encoded payload for a zero-element tensor is zero bytes.

## Scalar Tensors

`shape: []` (empty shape, `ndim: 0`) represents a scalar tensor containing exactly one element. The payload size equals `dtype.byte_width()` bytes.

## Metadata-Only Messages

A message with zero data objects is valid. This can be used to transmit metadata without any tensor data (e.g. coordination signals, timestamps, provenance records). Both `encode()` with an empty descriptors slice and `StreamingEncoder` with no `write_object()` calls produce valid messages.

## Mixed Dtypes in One Message

Multiple data objects in the same message may have different dtypes. For example, a `Float32` tensor paired with a `Bitmask` object used as a missing-data mask. Each object's pipeline (encoding, filter, compression) is configured independently.

## Bitmask with Encoding/Compression

Bitmask data is internally packed into `uint8` bytes. Any encoding or compression pipeline that supports `uint8` should work with bitmask data. The total bit count must be stored separately (in the shape) since the byte count `ceil(N / 8)` may not equal `N` exactly.

## Strides Validation

Strides are validated for length: `strides.len()` must match `shape.len()`. Non-contiguous strides (e.g. `shape: [4, 4], strides: [8, 1]`) are accepted — they indicate a view into a larger array and are semantically valid.

## Version Constraints

- `version: 0` and `version: 1` are deprecated and must be rejected by the decoder.
- `version: 2` is the current version.
- Higher versions (3+) are reserved for future use and will be valid once defined.

## NaN/Infinity in Simple Packing Parameters

If `reference_value` is NaN or Infinity, encoding fails immediately with a clear error. This value is used in the quantization formula and would produce corrupt output. (`binary_scale_factor` and `decimal_scale_factor` are integers and cannot be NaN/Infinity.)

## Duplicate CBOR Keys

Duplicate keys at the same level in a CBOR map are never accepted. The library uses canonical CBOR (RFC 8949 §4.2) which inherently rejects duplicate keys. Same-name keys at different nesting levels are acceptable: `common["foo"]` and `reserved["foo"]` are distinct keys.

## Unknown Hash Algorithm on Decode

If a message contains a hash with an algorithm the decoder doesn't recognize (e.g. `"sha256"` when only `xxh3` is implemented), `verify_hash: true` issues a warning and skips verification rather than returning an error. This ensures forward compatibility when new hash algorithms are added.

## decode_range with Empty Ranges

Calling `decode_range()` with an empty `ranges` slice (`&[]`) returns an empty `Vec<u8>`. This is not an error.

## File Concatenation

Tensogram is a message format, not a file format. Multiple `.tgm` files can be concatenated:

```bash
cat 1.tgm 2.tgm > all.tgm
```

The resulting file is valid. `scan()` and `TensogramFile` will find all messages from both source files.
