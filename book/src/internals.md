# Internals

This page explains implementation decisions that are not obvious from the public API. Useful if you're contributing to the library or implementing a compatible reader in another language.

## Deterministic CBOR Canonicalization

The library encodes metadata using a two-step process:

1. **Serialize** the `Metadata` struct to a `ciborium::Value` tree using serde.
2. **Recursively sort** all map keys by their CBOR byte encoding.
3. **Write** the sorted Value tree to bytes.

Standard serde serialization into ciborium does not guarantee key order (it depends on the HashMap/BTreeMap iteration order of the struct). Even though the library uses `BTreeMap` throughout (which gives alphabetical iteration order for string keys), relying on that would be fragile. The explicit canonicalization step ensures the output matches RFC 8949 §4.2 regardless of how the keys were stored.

```
Metadata struct
    ↓ serde serialization
ciborium::Value::Map (arbitrary key order)
    ↓ canonicalize() — sort all maps recursively by CBOR-encoded key bytes
ciborium::Value::Map (canonical order)
    ↓ write to bytes
CBOR bytes (deterministic)
```

## BTreeMap Throughout

All `extra` fields in `Metadata`, `ObjectDescriptor`, and `PayloadDescriptor` are `BTreeMap<String, ciborium::Value>`. This:

- Gives alphabetical iteration order for string keys (which matches CBOR canonical order for short strings).
- Avoids the non-determinism of `HashMap`.
- Makes it easy to read and write keys without worrying about order.

## Binary Header Size

The fixed header is exactly 40 bytes:

```
MAGIC (8) + total_length (8) + metadata_offset (8) + metadata_length (8) + num_objects (8)
```

Each object offset is an additional 8 bytes. So the full header is `40 + 8 × N` bytes, where N is the number of objects.

The `metadata_offset` field in the header is always equal to `40 + 8 × N`. It is stored explicitly so a decoder can locate the CBOR section without knowing N (by reading `num_objects` first).

## OBJS/OBJE Markers

Each payload is wrapped in 4-byte ASCII markers:

- `OBJS` = `4F 42 4A 53` — Object Start
- `OBJE` = `4F 42 4A 45` — Object End

These markers serve as a secondary corruption check. If a decoder finds the right byte count of payload but the bytes after it are not `OBJE`, the message is corrupted.

## simple_packing Bit Layout

Values are packed MSB-first (most significant bit first), matching the GRIB 2 simple packing specification:

```
Element 0: bits [0 .. B-1]
Element 1: bits [B .. 2B-1]
Element 2: bits [2B .. 3B-1]
...
```

The last byte is zero-padded on the right if `N × B` is not a multiple of 8.

The decode formula is:

```
V[i] = R + (packed[i] × 2^E) / 10^D
```

Where:
- `R` = reference_value (minimum of original data)
- `E` = binary_scale_factor
- `D` = decimal_scale_factor
- `packed[i]` = the integer read from the packed bits

## Lazy File Scanning

`TensogramFile::open()` does not read the file. The first call that needs the message list (e.g. `message_count()`, `read_message()`, `messages()`) triggers a full scan. After that, the list of `(offset, length)` pairs is cached in memory for the lifetime of the `TensogramFile` object.

```rust
// No I/O here
let mut file = TensogramFile::open("huge.tgm")?;

// Scan happens here (once)
let count = file.message_count()?;

// O(1) seek + read
let msg = file.read_message(999)?;
```

## Error Hierarchy

```
TensogramError
├── Framing     — invalid magic, truncated header, bad terminator
├── Metadata    — CBOR serialization/deserialization failure
├── Encoding    — invalid encoding params, NaN in simple_packing
├── Compression — szip not available
├── Object      — index out of range, objects/payload length mismatch
├── Io          — filesystem errors (wraps std::io::Error)
└── HashMismatch { expected, actual } — payload integrity failure
```

All public functions return `Result<T>` where the error is `TensogramError`. The `Io` variant wraps `std::io::Error` via the `From` impl, so `?` on any `std::io::Result` produces a `TensogramError::Io` automatically.

## Why Not mmap?

An `mmap` feature gate is planned. The current implementation reads messages into `Vec<u8>` buffers. With mmap, the file bytes are mapped directly into virtual memory, and the decoder can work on slices of the mapping without any copies. This is the zero-copy path. For now, the API is designed to accept `&[u8]` slices so the mmap implementation can slot in without changing the decode functions.
