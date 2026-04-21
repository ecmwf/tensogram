# Wire Format (v3)

This page describes the exact byte layout of a Tensogram v3
message — the format shipped in 0.17.0.  You need this if you are
implementing a reader in another language, debugging a corrupted
file, or just want to understand what is happening under the hood.
For the normative specification, see
[`plans/WIRE_FORMAT.md`](https://github.com/ecmwf/tensogram/blob/main/plans/WIRE_FORMAT.md).

All integer fields are **big-endian** (network byte order).

## Overview

A Tensogram message is built from three sections: a **header**
(preamble + optional frames), one or more **data object frames**,
and a **footer** (optional frames + postamble).

```
┌────────────────────────────────────────────────────────────────────┐
│  PREAMBLE                  magic, version, flags, length  (24 B)   │
├────────────────────────────────────────────────────────────────────┤
│  HEADER METADATA FRAME     CBOR global metadata      (optional)    │
├────────────────────────────────────────────────────────────────────┤
│  HEADER INDEX FRAME        CBOR object offsets       (optional)    │
├────────────────────────────────────────────────────────────────────┤
│  HEADER HASH FRAME         CBOR object hashes        (optional)    │
├────────────────────────────────────────────────────────────────────┤
│  PRECEDER METADATA FRAME   per-object metadata       (optional)    │
│  DATA OBJECT FRAME 0       header + payload + descriptor           │
│  PRECEDER METADATA FRAME   per-object metadata       (optional)    │
│  DATA OBJECT FRAME 1       ...                                     │
│  DATA OBJECT FRAME 2       (no preceder)                           │
│  ...                       (any number of objects)                 │
├────────────────────────────────────────────────────────────────────┤
│  FOOTER HASH FRAME         CBOR object hashes        (optional)    │
├────────────────────────────────────────────────────────────────────┤
│  FOOTER INDEX FRAME        CBOR object offsets       (optional)    │
├────────────────────────────────────────────────────────────────────┤
│  FOOTER METADATA FRAME     CBOR global metadata      (optional)    │
├────────────────────────────────────────────────────────────────────┤
│  POSTAMBLE   first_footer_offset, total_length, end_magic  (24 B)  │
└────────────────────────────────────────────────────────────────────┘
```

At least one metadata frame (header or footer) must be present —
messages cannot exist without metadata. Index and hash frames are
optional but highly encouraged. By default, the encoder places
them in the header when writing to a buffer, or in the footer when
streaming.

**Frame ordering:** The decoder enforces that frames appear in
order: header frames, then data object frames, then footer frames.
A header frame appearing after a data object frame, or a data
object frame appearing after a footer frame, is rejected as
malformed.

## Preamble (24 bytes)

The preamble is the fixed-size start of every message.

```
Offset  Size    Field
──────  ──────  ─────────────────────────────────
0       8       Magic: "TENSOGRM" (ASCII)
8       2       Version (uint16 BE) — must be 3 in v3
10      2       Flags (uint16 BE)
12      4       Reserved (uint32 BE) — set to zero
16      8       Total length (uint64 BE)
```

**Total length** is the byte count of the entire message from the
first byte of the preamble to the last byte of the postamble. A
value of **zero** means the encoder is in **streaming mode** — the
total length was not known when the preamble was written.

**Version compatibility.** v3 decoders reject any preamble whose
version field is not exactly `3`.  Older v1/v2 messages must be
re-encoded.

### Preamble flags

The flags field is a bitmask indicating which optional frames are
present and, new in v3, whether inline per-frame hash slots are
populated:

| Bit | Flag             | Meaning                                            |
|-----|------------------|----------------------------------------------------|
| 0   | `HEADER_METADATA`| A HeaderMetadata frame is present.                 |
| 1   | `FOOTER_METADATA`| A FooterMetadata frame is present.                 |
| 2   | `HEADER_INDEX`   | A HeaderIndex frame is present.                    |
| 3   | `FOOTER_INDEX`   | A FooterIndex frame is present.                    |
| 4   | `HEADER_HASHES`  | A HeaderHash aggregate frame is present.           |
| 5   | `FOOTER_HASHES`  | A FooterHash aggregate frame is present.           |
| 6   | `PRECEDER_METADATA` | At least one PrecederMetadata frame is present. |
| 7   | `HASHES_PRESENT` | Every frame's inline hash slot is populated with a non-zero xxh3-64 digest (new in v3). |

Unused flag bits must be set to zero.

## Frames

Every frame (header, footer, and data object) shares a common
16-byte frame header and ends with a type-specific footer whose
last 12 bytes are always `[hash u64][ENDF 4]` (new in v3).

### Frame header (16 bytes)

```
Offset  Size    Field
──────  ──────  ─────────────────────────────────
0       2       Start marker: "FR" (ASCII)
2       2       Frame type (uint16 BE)
4       2       Frame version (uint16 BE)
6       2       Reserved flags (uint16 BE)
8       8       Frame length — offset to end of frame (uint64 BE)
```

Frame versions are independent from the message version and from
each other.

### Frame common footer (12 bytes)

Every frame ends with this fixed-size tail:

```
Offset (from frame end)  Size    Field
───────────────────────  ──────  ─────────────────────────────────
-12                      8       hash (uint64 BE) — xxh3-64 digest of the frame body, or 0x0000000000000000 when HASHES_PRESENT = 0
-4                       4       End marker: "ENDF" (ASCII)
```

Data-object frames (type 9) have a larger 20-byte footer that
adds an 8-byte `cbor_offset` field before the common tail.

### Frame types

| Type | Name | Contents |
|------|------|----------|
| 1 | Header Metadata | CBOR global metadata map |
| 2 | Header Index | CBOR index of data object offsets |
| 3 | Header Hash | CBOR aggregate of per-object hashes |
| 4 | *(reserved)* | Occupied by the obsolete v2 `NTensorFrame`; any v3 decoder errors on read |
| 5 | Footer Hash | CBOR aggregate of per-object hashes |
| 6 | Footer Index | CBOR index of data object offsets |
| 7 | Footer Metadata | CBOR global metadata map |
| 8 | Preceder Metadata | Per-object CBOR metadata (see below) |
| 9 | `NTensorFrame` | Descriptor + payload + optional NaN / Inf bitmask companion sections (see [NaN / Inf Handling](../guide/nan-inf-handling.md)) |

The body phase of a v3 message carries one or more
**data-object frames**.  In v3 only `NTensorFrame` (type 9) is
defined; future types can slot in at fresh unused numbers without
bumping the wire version.

### Padding between frames

It is valid to have padding bytes between a frame's `ENDF` marker
and the next frame's `FR` marker. This allows encoders to align
frame starts to 8-byte (64-bit) boundaries for memory-mapped
access.

## Data Object Frames

A data object frame wraps one tensor's payload together with its
CBOR descriptor.  v3 defines exactly one concrete data-object
type, `NTensorFrame` (type 9).  The descriptor can go either
**before** or **after** the payload — flag bit 0 in the frame
header controls this.  The default is **after**, because when
encoding the descriptor is sometimes only fully known once the
payload has been written (e.g. after computing a hash or
determining compressed size).

### `NTensorFrame` (type 9) — v3 canonical layout

```
┌──────────────────────────────────────────────────────────────┐
│  FRAME HEADER       "FR" + type(9) + ver + flags + len (16 B)│
├──────────────────────────────────────────────────────────────┤
│  DATA PAYLOAD       raw or compressed bytes, NaN/Inf         │
│                     positions substituted with 0.0           │
├──────────────────────────────────────────────────────────────┤
│  mask_nan blob      OPTIONAL — compressed NaN position mask  │
├──────────────────────────────────────────────────────────────┤
│  mask_inf+ blob     OPTIONAL — compressed +Inf position mask │
├──────────────────────────────────────────────────────────────┤
│  mask_inf- blob     OPTIONAL — compressed -Inf position mask │
├──────────────────────────────────────────────────────────────┤
│  CBOR DESCRIPTOR    carries a top-level "masks" sub-map      │
│                     when any mask is present (see below)     │
├──────────────────────────────────────────────────────────────┤
│  cbor_offset (uint64 BE, 8 B)                                │
│  hash        (uint64 BE, 8 B)   xxh3-64 of body              │
│  "ENDF"      (4 B)                                           │
└──────────────────────────────────────────────────────────────┘
```

The **data-object footer** is 20 bytes: `[cbor_offset u64]
[hash u64][ENDF 4]`.  The `cbor_offset` field points at the CBOR
descriptor's start relative to the frame's first byte.  The inline
`hash` slot carries the xxh3-64 of the frame *body* (everything
between the 16-byte header and this 20-byte footer) when the
message's `HASHES_PRESENT` preamble flag is set; otherwise it is
`0x0000000000000000`.

Hash scope includes payload + masks + CBOR.  It does NOT include
the header, the `cbor_offset` field, the hash slot itself, or
`ENDF`.

The CBOR descriptor fully describes the data object: its type,
shape, strides, data type, byte order, encoding pipeline, and
optional per-object metadata.  See the
[CBOR Metadata](cbor-metadata.md) page for the schema.

See [NaN / Inf Handling](../guide/nan-inf-handling.md) for the
mask encode / decode semantics and the documented
lossy-reconstruction caveat.

## Preceder Metadata Frame

A Preceder Metadata Frame (type 8) optionally appears immediately before a Data Object Frame. It carries per-object metadata for the following data object, using the same GlobalMetadata CBOR format but with a single-entry `base` array.

**Use case:** Streaming producers that do not know ahead of time when the message will end can emit per-object metadata early via preceders, rather than waiting for the footer.

**Ordering rules:**
- Must appear in the data objects phase (after headers, before footers).
- Must be followed by exactly one Data Object Frame.
- Two consecutive preceders without an intervening DataObject are invalid.
- A dangling preceder (not followed by a DataObject) is invalid.
- Preceders are optional per-object.

**CBOR structure:**
```cbor
{
  "version": 2,
  "base": [{"mars": {"param": "2t"}, "units": "K"}]
}
```

**Merge on decode:** Preceder keys override footer `base[i]` keys on conflict. Footer-only keys (e.g., auto-populated `_reserved_.tensor` with ndim, shape, strides, dtype) are preserved. The consumer sees a unified `GlobalMetadata.base` — the preceder/footer distinction is transparent.

## Postamble (16 bytes)

The postamble sits at the very end of every message.

```
Offset  Size    Field
──────  ──────  ─────────────────────────────────
0       8       first_footer_offset (uint64 BE)
8       8       End magic: "39277777" (ASCII)
```

**`first_footer_offset`** is the byte offset (from the start of the message) to the first footer frame. This is **never zero**:

- If footer frames exist, it points to the start of the first one (e.g., the Footer Hash Frame).
- If no footer frames exist, it points to the postamble itself.

This guarantee means a reader can always distinguish "no footer frames" from "footer at offset 0" without ambiguity.

The end magic `39277777` was chosen because it is unlikely to appear naturally in floating-point or integer data, making it useful as a corruption boundary detector.

## Random Access Patterns

### With a header index (most common)

When a message was written in non-streaming mode, the index is in
the header. This is the fastest path — no seeking to the end
required.

```
1. Read preamble (24 B) → check flags
2. Read header metadata frame → global context
3. Read header index frame → offsets[], lengths[]
4. Seek to offsets[N], read data object frame → decode
```

### With a footer index only (streaming mode)

When a message was written in streaming mode, the encoder did not
know the object count or offsets up front. The index lives in the
footer.

```
1. Seek to end − 24, read postamble → first_footer_offset
2. Seek to first_footer_offset, scan footer frames → find index
3. Read footer index frame → offsets[], lengths[]
4. Seek to offsets[N], read data object frame → decode
```

Both paths give **O(1) access** to any data object by index.  The
object count is derived from `offsets.len()`.

## Scanning a Multi-Message File

Multiple messages can be concatenated into a single `.tgm` file. To find message boundaries:

1. Scan forward for the `TENSOGRM` magic (8 bytes).
2. Read `total_length` from the preamble.
   - If `total_length` is non-zero, advance by that many bytes to reach the next message.
   - If `total_length` is zero (streaming mode), use the header index frame length if present.
3. If neither total length nor header index is available, walk frame-by-frame — each frame header contains a length field — until the next `TENSOGRM` magic or EOF.
4. Verify the `39277777` end magic at the expected position to confirm message integrity.

```mermaid
flowchart TD
    A[Start of file] --> B{Find TENSOGRM?}
    B -- No --> Z[End of scan]
    B -- Yes --> C[Read total_length at +16]
    C --> D{total_length > 0?}
    D -- Yes --> E[Advance to offset + total_length]
    D -- No --> F[Walk frame-by-frame to next magic]
    E --> G[Verify 39277777 end magic]
    F --> G
    G -- Valid --> H[Record message]
    H --> B
    G -- Invalid --> I[Skip 1 byte, resume scan]
    I --> B
```

If the end magic does not match, the message is likely corrupt. The scanner skips one byte and resumes searching — this is the **corruption recovery** path.

## A Note on CBOR

Frames that contain CBOR data (metadata, index, hash) use length-prefixed CBOR encoding — there are no explicit start/end markers within the CBOR stream itself. The CBOR decoder reads the first byte to determine the data type and item count, then consumes exactly that many bytes. The frame boundaries (`FR`...`ENDF`) provide the outer containment.

All CBOR maps use deterministic encoding with canonical key ordering (RFC 8949 section 4.2). See [CBOR Metadata](cbor-metadata.md) for details.
