# Internals

This page explains implementation decisions that are not obvious from the public API. Useful if you're contributing to the library or implementing a compatible reader in another language.

## Deterministic CBOR Canonicalization

The library encodes all CBOR structures (global metadata, data object descriptors, index frames, hash frames) using a three-step process:

1. **Serialize** the struct to a `ciborium::Value` tree using serde.
2. **Recursively sort** all map keys by their CBOR byte encoding.
3. **Write** the sorted Value tree to bytes.

Standard serde serialization into ciborium does not guarantee key order (it depends on the HashMap/BTreeMap iteration order of the struct). Even though the library uses `BTreeMap` throughout (which gives alphabetical iteration order for string keys), relying on that would be fragile. The explicit canonicalization step ensures the output matches RFC 8949 §4.2 regardless of how the keys were stored.

```text
GlobalMetadata / DataObjectDescriptor struct
    ↓ serde serialization
ciborium::Value::Map (arbitrary key order)
    ↓ canonicalize() — sort all maps recursively by CBOR-encoded key bytes
ciborium::Value::Map (canonical order)
    ↓ write to bytes
CBOR bytes (deterministic)
```

Note: `canonicalize()` returns `Result<()>` and propagates errors rather than panicking.

## BTreeMap Throughout

The `extra` (serialized as `_extra_`), `reserved` (serialized as `_reserved_`), and `base` entry fields in `GlobalMetadata`, as well as the `params` field in `DataObjectDescriptor`, are `BTreeMap<String, ciborium::Value>`. This:

- Gives alphabetical iteration order for string keys (which matches CBOR canonical order for short strings).
- Avoids the non-determinism of `HashMap`.
- Makes it easy to read and write keys without worrying about order.

## Frame-Based Wire Format (v2)

The v2 wire format uses a frame-based structure instead of the v1 monolithic binary header.

### Preamble (24 bytes)

```text
MAGIC "TENSOGRM" (8) + version u16 (2) + flags u16 (2) + reserved u32 (4) + total_length u64 (8)
```

The preamble flags indicate which optional frames are present (header/footer metadata, index, hashes). `total_length = 0` signals streaming mode.

### Frame Header (16 bytes)

Every frame (metadata, index, hash, data object) starts with:

```text
"FR" (2) + frame_type u16 (2) + version u16 (2) + flags u16 (2) + total_length u64 (8)
```

And ends with `"ENDF"` (4 bytes). Frame versions are independent of message version.

### Data Object Frame Layout

Each data object is a self-contained frame:

```text
Frame header (16B) + [CBOR descriptor] + payload bytes + [CBOR descriptor] + cbor_offset u64 (8B) + "ENDF" (4B)
```

The `cbor_offset` is the byte offset from the frame start to the CBOR descriptor. A flag bit controls whether the CBOR descriptor appears before or after the payload (default: after, since encoding parameters like hash are only known after encoding completes).

### Postamble (16 bytes)

```text
first_footer_offset u64 (8) + END_MAGIC "39277777" (8)
```

`first_footer_offset` is never zero. It points to the first footer frame, or to the postamble itself when no footer frames are present.

### Two-Pass Index Construction

When encoding a non-streaming message, the index frame contains byte offsets of each data object. But the index frame's own size affects those offsets (circular dependency). The encoder solves this with a two-pass approach:

1. First pass: compute index CBOR with placeholder offsets to determine the index frame size.
2. Second pass: compute final offsets using the known index frame size, re-encode the index CBOR.

If the re-encoded CBOR changes size (edge case), the encoder returns an error rather than silently producing incorrect offsets.

### Encoder Structure

The `encode_message()` function delegates to five focused helpers:

- `build_hash_frame_cbor()` — collects hashes from objects and serializes the HashFrame
- `build_index_frame()` — runs the two-pass index construction described above
- `compute_object_offsets()` — calculates byte offsets with 8-byte alignment
- `compute_message_flags()` — sets preamble flags from optional frame presence
- `assemble_message()` — writes preamble, frames, and postamble into the final buffer

## simple_packing Bit Layout

Values are packed MSB-first (most significant bit first), matching the GRIB 2 simple packing specification:

```text
Element 0: bits [0 .. B-1]
Element 1: bits [B .. 2B-1]
Element 2: bits [2B .. 3B-1]
...
```

The last byte is zero-padded on the right if `N × B` is not a multiple of 8.

The decode formula is:

```text
V[i] = R + (packed[i] × 2^E) / 10^D
```

Where:
- `R` = reference_value (minimum of original data)
- `E` = binary_scale_factor
- `D` = decimal_scale_factor
- `packed[i]` = the integer read from the packed bits

## Lazy File Scanning

`TensogramFile::open()` does not read the file. The first call that needs the message list (e.g. `message_count()`, `read_message()`) triggers a streaming scan using `scan_file()`. The scanner reads only preamble-sized chunks and seeks forward, so it never loads the entire file into memory. After that, the list of `(offset, length)` pairs is cached in memory for the lifetime of the `TensogramFile` object.

```rust,ignore
// No I/O here
let mut file = TensogramFile::open("huge.tgm")?;

// Streaming scan happens here (once) — reads preamble chunks, seeks forward
let count = file.message_count()?;

// O(1) seek + read
let msg = file.read_message(999)?;
```

## Error Hierarchy

```text
TensogramError
├── Framing     — invalid magic, truncated preamble, bad frame markers, missing postamble
├── Metadata    — CBOR serialization/deserialization failure
├── Encoding    — invalid encoding params, NaN in simple_packing
├── Compression — compressor error (szip, zstd, lz4, blosc2, zfp, sz3)
├── Object      — index out of range
├── Io          — filesystem errors (wraps std::io::Error)
└── HashMismatch { expected, actual } — payload integrity failure
```

All public functions return `Result<T>` where the error is `TensogramError`. The `Io` variant wraps `std::io::Error` via the `From` impl, so `?` on any `std::io::Result` produces a `TensogramError::Io` automatically.

## Memory-Mapped I/O (`mmap` feature)

The `mmap` feature gate enables memory-mapped file access via `memmap2`. When you open a file with `TensogramFile::open_mmap()`, the file is mapped into virtual memory and the existing `scan()` function runs directly on the mapped buffer. Subsequent `read_message()` calls return copies from the mapped region without additional seeks.

```rust,ignore
// Requires: cargo build --features mmap
let mut file = TensogramFile::open_mmap("huge.tgm")?;
let count = file.message_count()?; // already scanned during open_mmap
let msg = file.read_message(42)?;  // copies from mmap, no seek
```

The regular `open()` path still works without the feature and uses streaming seek-based scanning.

## Async I/O (`async` feature)

The `async` feature gate adds tokio-based async variants: `open_async()`, `read_message_async()`, and `decode_message_async()`. All CPU-intensive work (scanning, decoding, FFI calls to libaec/zfp/blosc2) runs via `spawn_blocking` to avoid blocking the async runtime.

```rust,ignore
// Requires: cargo build --features async
let mut file = TensogramFile::open_async("forecast.tgm").await?;
let (meta, objects) = file.decode_message_async(0, &opts).await?;
```

## Frame Ordering Validation

The decoder enforces that frames appear in the expected order within a message: header frames first, then data object frames, then footer frames. A `DecodePhase` state machine tracks the current phase and returns `TensogramError::Framing` if a frame type appears out of order.

This catches malformed messages where, for example, a header metadata frame appears after a data object frame.

## Canonical CBOR Verification

The library provides `verify_canonical_cbor()` to check that a CBOR byte slice is in RFC 8949 §4.2.1 canonical form. This is used internally by tests to verify that all CBOR output (metadata, descriptors, index frames, hash frames) is deterministic. It can also be used by external tools that need to validate Tensogram CBOR output against the spec.
