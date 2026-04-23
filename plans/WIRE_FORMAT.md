# Wire Format (v3)

> **Status.** This document is the canonical specification of the
> Tensogram wire format.  It describes version **3**, the only version
> currently supported.  Decoders hard-fail when they read a preamble
> with `version != 3`.  For the user-facing companion-mask usage
> guide, see [`docs/src/guide/nan-inf-handling.md`](../docs/src/guide/nan-inf-handling.md).

---

## 1. Overview

A Tensogram message is a self-contained binary blob composed of a
**HEADER** (fixed preamble + optional frames), a **BODY** (one or
more data-object frames, each optionally preceded by a per-object
preceder metadata frame), and a **FOOTER** (optional frames + fixed
postamble).  Multiple messages may be appended to a file.

Every integer field is **big-endian** unless stated otherwise.  All
frame payloads except the data payload of data-object frames are
serialised as **canonical CBOR** (RFC 8949 §4.2).  Unknown CBOR keys
at any level are silently ignored on decode (forward-compatibility
rule).

**Generic data objects.** The body phase carries one or more
*data-object frames*.  The wire format is designed to accommodate
multiple kinds of data object in the future (e.g. observation
tables, non-tensor products).  In v3 the registry defines exactly
one concrete data-object type — [`NTensorFrame`](#65-ntensorframe-type-9),
the N-dimensional tensor frame (type 9).  New data-object types,
when added, slot in alongside `NTensorFrame` at fresh frame-type
numbers without a wire-format version bump, as long as they
follow the common frame structure defined in §2.

Top-level layout:

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER PREAMBLE            24 B  magic, version, flags, tl  │
├──────────────────────────────────────────────────────────────┤
│  Header frames (optional, in this order)                     │
│    HEADER METADATA FRAME      type 1                         │
│    HEADER INDEX FRAME         type 2                         │
│    HEADER HASH FRAME          type 3                         │
├──────────────────────────────────────────────────────────────┤
│  Body — repeated per data object:                            │
│    PRECEDER METADATA FRAME    type 8   (optional, pre-object)│
│    NTENSORFRAME               type 9   (currently the only   │
│    …                                    data-object type)    │
├──────────────────────────────────────────────────────────────┤
│  Footer frames (optional, in this order)                     │
│    FOOTER HASH FRAME          type 5                         │
│    FOOTER INDEX FRAME         type 6                         │
│    FOOTER METADATA FRAME      type 7                         │
├──────────────────────────────────────────────────────────────┤
│  FOOTER POSTAMBLE           24 B  first_footer_offset,       │
│                                   total_length, end_magic    │
└──────────────────────────────────────────────────────────────┘
```

**Invariant.** Either a header metadata frame or a footer metadata
frame must be present.  Messages cannot be emitted without metadata.
Index and hash frames are optional individually, but the encoder
strongly encourages at least one of each per message and controls
them through `EncodeOptions.create_header_hashes` /
`create_footer_hashes` and the corresponding index flags.

---

## 2. Common frame structure

Every frame has the same 16-byte **frame header** and a
**frame-type-specific footer** whose size is fixed per type.
The footer always ends with the two-field tail `[hash u64][ENDF]`
(12 bytes), but may contain additional fixed-size fields *before*
that tail.  Between header and footer, the per-frame payload is
either CBOR-encoded (most frame types) or a custom layout
(data-object frames).  Eight-byte alignment padding may follow a
frame's `ENDF` — see §2.3.

### 2.1 Frame header (16 bytes)

```
Offset  Size  Field
──────  ────  ───────────────────────────────────────
0       2     Magic: ASCII "FR"
2       2     Frame type   (uint16 BE)  — see §4
4       2     Version      (uint16 BE)  — frame-type-specific
6       2     Frame flags  (uint16 BE)  — frame-type-specific
8       8     total_length (uint64 BE)  — incl. header, payload,
                                           footer, excl. padding
```

The `total_length` is the number of bytes from the first byte of the
frame header through the last byte of `ENDF`, not counting any
alignment padding that follows.

### 2.2 Frame footer (type-specific, always ends with `[hash][ENDF]`)

**New in v3.** Every frame ends with a fixed-size footer whose
exact size depends on the frame type, but whose **last 12 bytes
are always `[hash u64][ENDF 4]`**.  Any additional footer fields
specific to a frame type (e.g. the `cbor_offset` field of
data-object frames) appear *immediately before* the common tail.

```
Offset (from frame_end)  Size  Field
───────────────────────  ────  ───────────────────────────────────
 ⋮                        ⋮    type-specific footer fields
-12                      8     hash (uint64 BE) — xxh3-64 digest,
                                or 0x0000000000000000 when the
                                preamble HASHES_PRESENT flag is 0
-4                       4     End marker: ASCII "ENDF"
```

The hash slot lives at exactly `frame_end − 12` for every frame
type, so a validator can locate it without knowing anything about
the frame's payload structure.  The total footer size varies by
type — see §2.4 and the per-frame sections (§6).

Footer sizes in v3:

| Frame type(s)                     | Footer size | Layout                      |
|-----------------------------------|:----------:|------------------------------|
| `HeaderMetadata` / `FooterMetadata` (1, 7) | 12 B | `[hash][ENDF]` |
| `HeaderIndex` / `FooterIndex`     (2, 6)   | 12 B | `[hash][ENDF]` |
| `HeaderHash` / `FooterHash`       (3, 5)   | 12 B | `[hash][ENDF]` |
| `PrecederMetadata`                (8)      | 12 B | `[hash][ENDF]` |
| `NTensorFrame`                    (9)      | 20 B | `[cbor_offset][hash][ENDF]` |

### 2.3 Alignment padding

`0–7` zero bytes may follow a frame's `ENDF` to align the next frame
header to an 8-byte boundary.  This padding is *not* part of the
frame — it is not covered by the hash, not included in
`total_length`, and is simply skipped by the frame scanner.

### 2.4 Frame hash scope

**Rule.** The hash covers the **frame body** — i.e. the bytes
between the frame header and the frame's type-specific footer.
Neither the header nor any byte of the footer is ever covered by
the hash.

Formally, given a frame with `frame_start = s` and
`frame_end = s + total_length`:

```
hash_scope = frame_bytes[s + 16 .. frame_end - footer_size(type))
```

where `footer_size(type)` is the per-type size from the table in
§2.2.

Examples:

- **Metadata / Index / Hash / Preceder frames** (footer = 12 B) —
  hash covers `bytes[16 .. total_length - 12)`, i.e. the CBOR
  payload in full.
- **`NTensorFrame`** (footer = 20 B) — hash covers
  `bytes[16 .. total_length - 20)`, i.e. the encoded tensor
  payload + any mask blobs + the CBOR descriptor.  The
  `cbor_offset` field is part of the footer and is **not** hashed.

Excluded from the hash in every case: the 16-byte frame header, the
4-byte `ENDF` marker, the 8-byte hash slot itself, every other
fixed-size footer field (e.g. `cbor_offset`), and any alignment
padding after `ENDF`.

The hash slot is **8 bytes of raw xxh3-64 digest in big-endian
order** — no algorithm identifier byte, no length prefix.  The
algorithm name (always `"xxh3"` in v3) lives in the message-level
hash frames (§6.3) and is therefore extensible.

When `HASHES_PRESENT = 0`, every frame's hash slot is written as
`0x0000000000000000` and validators skip hash verification.

---

## 3. Preamble (24 bytes)

```
Offset  Size  Field
──────  ────  ─────────────────────────────────────────────
0       8     Magic: ASCII "TENSOGRM"
8       2     Version (uint16 BE)            — must be 3 in v3
10      2     Flags   (uint16 BE)            — see §3.1
12      4     Reserved flags (uint32 BE)     — set to 0
16      8     total_length (uint64 BE)       — see §3.2
```

A v3 decoder **rejects** any preamble where `version != 3` with a
`FramingError`.  The version field is the sole source of truth for
wire compatibility; changes that require new parsing logic always
bump it.

### 3.1 Preamble flags (bit positions)

```
Bit  Flag                    Meaning
───  ──────────────────────  ─────────────────────────────────
0    HEADER_METADATA         A HeaderMetadata frame is present.
1    FOOTER_METADATA         A FooterMetadata frame is present.
2    HEADER_INDEX            A HeaderIndex frame is present.
3    FOOTER_INDEX            A FooterIndex frame is present.
4    HEADER_HASHES           A HeaderHash frame is present.
5    FOOTER_HASHES           A FooterHash frame is present.
6    PRECEDER_METADATA       At least one PrecederMetadata frame
                             appears in the body.
7    HASHES_PRESENT          Per-frame hash slots are populated
                             (non-zero).  When 0, every frame's
                             hash slot is 0x00…00 and readers
                             skip hash verification.
8–15                         Reserved; set to 0.
```

### 3.2 `total_length`

Total byte length of this message from the `TENSOGRM` magic (byte 0
of the preamble) through the `39277777` magic (last 8 bytes of the
postamble), inclusive.  Alignment padding between frames is counted;
padding after the postamble is not (the postamble's last byte is
the last byte of the message).

In **streaming encode mode**, when the encoder does not know the
total message size at the time it emits the preamble, this field is
written as `0`.  If the output sink is seekable, the encoder
back-fills the real value at `finish()` time (see §9.2).  A zero
here tells readers to fall back to forward scanning.

---

## 4. Frame type registry

| Type | Name | Phase | Status in v3 |
|-----:|---|---|---|
| 1    | `HeaderMetadata`     | header | active |
| 2    | `HeaderIndex`        | header | active |
| 3    | `HeaderHash`         | header | active |
| 4    | *(reserved, removed)*| body   | **error** — occupied by the obsolete v2 `NTensorFrame`; any v3 decoder that reads it emits `FramingError` |
| 5    | `FooterHash`         | footer | active |
| 6    | `FooterIndex`        | footer | active |
| 7    | `FooterMetadata`     | footer | active |
| 8    | `PrecederMetadata`   | body   | active |
| 9    | `NTensorFrame`       | body   | active — currently the only concrete data-object type |

Types ≥ 10 and types `0`, `4` are invalid.  Decoders reject them
with a `FramingError("unknown frame type {n}")` or
`FramingError("reserved frame type 4 (obsolete v2 NTensorFrame) not supported in v3")`
respectively.

The body phase is designed to hold multiple kinds of data object;
new data-object types slot in at fresh unused frame-type numbers
(alongside `NTensorFrame` at type 9) without a version bump, as
long as they follow the common frame structure of §2.

---

## 5. Phase ordering

Frames must appear in phase order: **header → body → footer**.
Within a phase, the order above in §1 and §4 is enforced.  Two
consecutive `PrecederMetadata` frames without an intervening
data-object frame are invalid.  A `PrecederMetadata` frame not
followed by a data-object frame (including at end-of-body) is
invalid.  In v3, "data-object frame" means exclusively
`NTensorFrame` (type 9); future data-object types will join that
set without otherwise changing the ordering rules.

---

## 6. Frame types

### 6.1 `HeaderMetadata` / `FooterMetadata` (types 1, 7)

CBOR payload: the `GlobalMetadata` structure.  The CBOR metadata
frame is **fully free-form** — the library interprets only three
top-level keys (`base`, `_reserved_`, `_extra_`).  Any other key
the caller supplies — including a stray legacy `"version"` — is
preserved as a free-form annotation and routed into `_extra_` on
decode.  The wire-format version lives **exclusively in the
preamble** (see §3); it is never duplicated in CBOR.

```cbor
{
  "base": [                              ; per-object metadata, one map per object
    { "mars": { ... }, ... },
    { "mars": { ... }, ... }
  ],
  "_reserved_": {                        ; library-managed, see below
    "encoder": { "name": "tensogram", "version": "0.17.0" },
    "time":    "2026-04-20T10:00:00Z",
    "uuid":    "550e8400-e29b-41d4-a716-446655440000"
  },
  "_extra_": {                           ; client-writable catch-all
    ...
  }
}
```

- **There are no required top-level keys.** An encoder MAY emit
  an empty map (`{}`); a decoder MUST accept arbitrary top-level
  keys.  The wire-format version is fixed by the preamble and MUST
  NOT be written to CBOR by an encoder.  A stray `"version"` key
  from a legacy pre-0.17 producer is tolerated and routed into
  `_extra_` so the data round-trips cleanly.
- `base[i]` holds ALL structured metadata for object `i`
  independently.  The encoder auto-populates `_reserved_.tensor`
  (with `ndim`, `shape`, `strides`, `dtype`) in each entry.
- `_reserved_` is library-managed — client code may read but MUST
  NOT write; the encoder validates this and rejects messages where
  client code has set keys inside `_reserved_`.
- `_extra_` is a client-writable catch-all for ad-hoc message-level
  annotations.  Every unrecognised top-level CBOR key is routed
  here on decode, so round-trips preserve unknown input.

Exactly one of `HeaderMetadata` / `FooterMetadata` **must** be
present.  If both are present, the decoder prefers the header frame
(the most common case); any divergence between the two is a
`MetadataError`.

### 6.2 `HeaderIndex` / `FooterIndex` (types 2, 6)

CBOR payload: a map listing the byte offset and byte length of each
data-object frame, in emission order.

```cbor
{
  "offsets": [u64, u64, ...],            ; offset from message start
                                         ; (= from the TENSOGRM magic byte)
  "lengths": [u64, u64, ...]             ; frame total_length, excl. padding
}
```

- `offsets.len() == lengths.len()` — else `MetadataError`.
- `offsets[i]` is the byte offset of the i-th data-object frame from
  the start of the message (i.e. from `TENSOGRM` magic at byte 0),
  **not** an absolute file offset.
- `lengths[i]` mirrors the frame header's `total_length` for object
  `i`, so readers can know the exact size of each frame without
  reading its header first.
- `PrecederMetadata` frames and alignment padding are NOT included
  in the index; only data-object frames are indexed.  In v3 that
  means only `NTensorFrame` frames; future data-object types will
  also be indexed.
- Object count is derived from `offsets.len()` — no separate
  `object_count` key.

Either a `HeaderIndex` or a `FooterIndex` (or both) is encouraged
but not required.  If both are present, the decoder prefers the
header frame.

**Placement.** In buffered encoding the encoder typically emits a
`HeaderIndex` (default).  In streaming encoding, where frame
offsets are not known in advance, the encoder emits a `FooterIndex`
instead (default), and sets the postamble `first_footer_offset`
field to the byte offset of the footer frame.

Unknown CBOR keys are ignored, so future additions (e.g. a
`frame_types` array, or per-object flags) can be carried
forward-compatibly without a version bump.

### 6.3 `HeaderHash` / `FooterHash` (types 3, 5)

CBOR payload: a map listing the per-object payload hashes.

```cbor
{
  "algorithm": "xxh3",                   ; hash algorithm name
  "hashes":    ["abcdef0123456789",      ; hex string per object,
                "1122334455667788",      ; in emission order
                ...]
}
```

- `hashes[i]` is a lowercase hex string equal to the xxh3-64 digest
  stored in the inline hash slot of the i-th data-object frame —
  the same 8-byte value, just rendered as 16 hex characters for
  CBOR-level readability.
- Object count is derived from `hashes.len()` — no separate
  `object_count` key.
- `algorithm` is free-form CBOR text.  In v3 the only value used by
  the encoder is `"xxh3"`.  Unknown values cause hash verification
  to be skipped with a warning; they are never a hard error.
- Hex strings (rather than `bytes`) are retained to leave room for
  future longer digests (e.g. xxh3-128, BLAKE3-256) without a
  schema change.  Use `Vec<u8>` of fixed width 8 on the Rust side;
  convert to hex on serialise.

**Placement.** By default:

- **Buffered encoding** emits a `HeaderHash` frame (all hashes
  known up-front).
- **Streaming encoding** emits a `FooterHash` frame (hashes only
  known after every object is written).

The user controls this via `EncodeOptions.create_header_hashes`
and `create_footer_hashes`.  In streaming mode,
`create_header_hashes = true` is an `EncodingError` at encoder
construction time.  Both flags may be set simultaneously in
buffered mode; the same hash list is written to both frames.

**Forward-compatibility.** The map is intentionally minimal so
future fields (per-hash flags, per-hash algorithm overrides,
signatures) can be added as new keys without breaking readers.

### 6.4 `PrecederMetadata` (type 8)

A `PrecederMetadata` frame optionally precedes a single
data-object frame, carrying per-object metadata for the
immediately following data object.  Primary use: streaming
producers that want to associate per-object metadata early.

CBOR payload: the same `GlobalMetadata` structure as §6.1, with:

- `base` = a single-entry array containing one metadata map for
  the next data object.
- `_reserved_` must be empty; the encoder strips any `_reserved_`
  key from preceder frames to avoid colliding with the encoder's
  auto-populated `_reserved_.tensor`.
- `_extra_` must be empty.

**Ordering rules.**

- A `PrecederMetadata` frame lives in the body phase.
- It must be followed by exactly one data-object frame.  In v3
  that means an `NTensorFrame`.  Two consecutive preceders
  without an intervening data-object frame are invalid.
- A preceder at end-of-body (with no following data-object frame)
  is invalid.
- Preceders are optional per object.  Some objects may have a
  preceder; others may not.

**Merge semantics on decode.**  When both a preceder and a
header/footer metadata frame carry `base[i]` entries for the same
object index:

- Keys from the preceder override keys from the
  header/footer on conflict (preceder wins).
- Keys present only in the header/footer (e.g. auto-populated
  `_reserved_.tensor`) are preserved.
- The decoder presents a unified `GlobalMetadata.base[i]` to the
  consumer; the preceder / footer distinction is transparent.

**Flag.** Preamble bit 6 (`PRECEDER_METADATA`) is set iff at least
one preceder frame is present.  In streaming mode the flag is
always set because the encoder emits preceders per-object by
default.

### 6.5 `NTensorFrame` (type 9)

The N-dimensional tensor data-object frame — one tensor per frame,
optionally with compressed bitmask companions identifying positions
of non-finite values (NaN / +Inf / −Inf).  The mask design is
specified inline below (see §6.5 *Masks sub-map* and §8 *Bitmask
dtype and compression codecs*).  In v3 this is the only concrete
data-object type defined; future versions may introduce other
data-object frame types at fresh type numbers.

**Layout.** Two variants, selected by the `CBOR_AFTER_PAYLOAD`
frame-flag bit (bit 0 of the frame header's `flags` field).
Default is CBOR after payload.

```
┌─────────────────────────────────────────────────────────────┐
│ Frame header                                          16 B  │
│  magic "FR" | type=9 | version=1 | flags | total_length     │
├─────────────────────────────────────────────────────────────┤
│ Payload region:                                             │
│  ┌ encoded tensor payload                  (variable) ────┐ │
│  ├ mask blob: nan                         (optional) ─────┤ │
│  ├ mask blob: inf+                        (optional) ─────┤ │
│  ├ mask blob: inf-                        (optional) ─────┤ │
│  └─────────────────────────────────────────────────────────┘│
│ CBOR descriptor                              (variable)     │
├─────────────────────────────────────────────────────────────┤
│ Frame footer (type-specific, 20 B)                          │
│  cbor_offset (uint64 BE)                               8 B  │
│  hash        (uint64 BE)                               8 B  │
│  "ENDF"                                                4 B  │
└─────────────────────────────────────────────────────────────┘
```

When `CBOR_AFTER_PAYLOAD = 0` (CBOR-before-payload), the CBOR
descriptor precedes the payload region; the two sections are
swapped but every other offset is unchanged.

**Fixed offsets** (relative to `frame_end`, the byte after `ENDF`):

- `-4  ..  0`  `"ENDF"` marker
- `-12 ..  -4`  hash (uint64 BE; see §2.2)
- `-20 ..  -12` `cbor_offset` (uint64 BE) — byte offset within
  the frame from the first byte of the frame header to the first
  byte of the CBOR descriptor

A reader that has `total_length` from the frame header can compute
both slots directly from `frame_start + total_length − 12` (hash)
and `frame_start + total_length − 20` (`cbor_offset`) without
parsing any CBOR.

**`cbor_offset`** must fall within `[16, total_length − 20]`.  Out
of range → `FramingError`.

**Hash scope.** Per §2.4, the hash covers
`frame_bytes[16 .. frame_end − 20)` — i.e. the payload region
(including any mask blobs) + the CBOR descriptor only.  The
`cbor_offset` field is part of the footer and is **not** covered
by the hash, nor are the hash slot itself, the `ENDF` marker, or
the frame header.  Writing the hash is therefore strictly after
the rest of the frame body is known.

**CBOR descriptor** — the `DataObjectDescriptor`:

```cbor
{
  ; Tensor metadata
  "type":       "ntensor",               ; object type tag
  "ndim":       u64,
  "shape":      [u64, ...],
  "strides":    [u64, ...],
  "dtype":      "float64" | "float32" | … | "bitmask",

  ; Encoding pipeline
  "byte_order": "big" | "little",           ; optional — absent ⇒ native
  "encoding":   "none" | "simple_packing",
  "filter":     "none" | "shuffle",
  "compression":"none" | "szip" | "zstd" | "lz4" | "blosc2"
             |  "zfp"  | "sz3"
             |  "rle"  | "roaring",       ; rle/roaring: bitmask dtype only

  ; Encoding-specific parameters (flattened into the map)
  "reference_value":       ...,
  "binary_scale_factor":   ...,
  "decimal_scale_factor":  ...,
  "bits_per_value":        ...,
  "szip_block_offsets":    [...],
  ...

  ; Mask metadata (optional — absent when no non-finite masks)
  "masks": {
    "nan":  { "method": "roaring", "offset": 800000, "length": 512 },
    "inf+": { "method": "rle",     "offset": 800512, "length":  64 },
    "inf-": { "method": "rle",     "offset": 800576, "length":  32 }
  }
}
```

The `hash` field on the descriptor (present pre-v3) is **removed**.
The inline hash slot in the frame's common footer is the sole
source of truth for per-object integrity.

**Optional `byte_order`.** The key may be omitted from the CBOR
descriptor; decoders treat a missing key as the native byte order of
the platform doing the deserialisation.  Encoders produced by this
reference implementation always write the field explicitly, so the
absent form only arises when a CBOR descriptor is constructed by an
external tool or when the JSON convenience surface (C FFI / CLI)
accepts a descriptor whose JSON omits the key.  The on-wire absence
rule is strictly lenient: explicit `"big"` / `"little"` values round-
trip unchanged.

**Dtype-restricted codecs.** In v3, `compression = "rle"` and
`compression = "roaring"` require `dtype = "bitmask"`.  The
encoder rejects any other combination at pipeline-build time with
an `EncodingError`.  See §8.

#### 6.5.1 `masks` sub-map — per-mask entry schema

Each of the three optional keys (`nan`, `inf+`, `inf-`) inside the
`masks` sub-map describes one compressed bitmask blob:

```cbor
{
  "method": "roaring",            ; one of "rle" | "roaring" |
                                  ; "blosc2" | "zstd" | "lz4" | "none"
  "offset": 800000,               ; byte offset of the mask blob,
                                  ; measured from the start of the
                                  ; payload region (= first byte
                                  ; after the 16-byte frame header)
  "length": 512,                  ; byte length of the (compressed)
                                  ; mask blob on disk
  "params": { ... }               ; optional method-specific params
}
```

`params` by method:

- `rle` — no params.
- `roaring` — no params (format embeds its own metadata).
- `blosc2` — `{ "codec": "lz4" | "zstd", "level": int }`.  Defaults
  `codec="lz4"`, `level=5`.
- `zstd` — `{ "level": int }` (optional, default 3).
- `lz4` — no params.
- `none` — no params (raw packed bytes).

Canonical CBOR sort order for `masks` keys: `inf+` < `inf-` <
`nan` (byte-lex).

#### 6.5.2 Bit packing layout for masks

Raw bit layout (before compression):

- MSB-first, matching the existing `Dtype::Bitmask` convention.
- `ceil(N / 8)` bytes for `N` elements.
- Trailing bits in the last byte are **zero-filled** for
  determinism (required for stable hashing).
- Bit `i` (element index, 0-based) lives at byte `i / 8`, bit
  position `7 - (i % 8)`.
- `1` = the element at that index was the specific non-finite kind
  being masked.  `0` = the element is finite (or a different
  non-finite kind).

**Priority for simultaneous classifications.**  A single element
cannot simultaneously be "NaN" and "Inf".  For complex dtypes
(`complex64` / `complex128`), where real and imag are independent:

1. **NaN** wins over any Inf (either component being NaN → nan mask).
2. **+Inf** wins over −Inf (either component being +Inf while
   neither is NaN → inf+ mask).
3. **−Inf** otherwise (only non-finiteness is −Inf).

After substitution (both components set to `0.0 + 0.0i`), decode
restores with the canonical bit pattern of the mask's kind
(`f64::NAN` / `f64::INFINITY` / `f64::NEG_INFINITY`) in **both**
real and imag components.

#### 6.5.3 Small-mask fallback

When the uncompressed mask byte-count is
`≤ small_mask_threshold_bytes` (default 128, configurable via
`EncodeOptions.small_mask_threshold_bytes` — single threshold
across all three masks), the encoder writes the mask as `"none"`
regardless of the user-requested method.  The descriptor's
`method` field reflects what was actually written.

#### 6.5.4 Lossy NaN-payload reconstruction

The encoder replaces the original float bits (any NaN / Inf value)
with `0.0` and records only the position.  The decoder restores
using the **canonical** bit pattern of the kind:

- NaN → `f64::NAN` bits (`0x7FF8000000000000`; quiet NaN).
- +Inf → `f64::INFINITY` bits (`0x7FF0000000000000`).
- −Inf → `f64::NEG_INFINITY` bits (`0xFFF0000000000000`).

For `f16` / `bf16` / `f32` / `complex64` / `complex128`, the
dtype-specific canonical patterns are used.

**Implication**: specific NaN payloads (signalling NaN, custom
payload bits) are **not preserved** through an `allow_nan=true`
encode.  Callers needing bit-exact NaN preservation must
pre-process the payload and encode with `allow_nan=false`, then
ship the semantics in their own data layer.

---

## 7. Postamble (24 bytes, NEW in v3)

```
Offset  Size  Field
──────  ────  ──────────────────────────────────────────────
0       8     first_footer_offset (uint64 BE)
8       8     total_length       (uint64 BE)  — NEW in v3
16      8     Magic: ASCII "39277777"
```

The postamble grew from 16 bytes (v2) to 24 bytes (v3) with the
addition of the mirrored `total_length` field.

- **`first_footer_offset`** — byte offset from the message start
  (the `TENSOGRM` byte) to the first byte of the first footer
  frame, or to the postamble itself if no footer frames exist.
  Same semantics as v2.
- **`total_length`** — same value as the preamble's `total_length`,
  mirrored here so backward-scanning readers can locate the
  message start without consulting the preamble first.  Zero in
  the streaming-non-seekable-sink case (see §9.2), in which case
  readers fall back to forward scanning.
- **End magic** — ASCII `"39277777"` at the last 8 bytes of the
  message.  Its position is invariant: the last 8 bytes of any
  message are always the end magic.

**Backward locatability.**  A reader that has *any* byte position
inside a message can locate that message's start as follows:

1. Search backward byte-by-byte for the 8-byte pattern
   `"39277777"`, bounded by the reader's configured max-message-size
   (default 4 GiB — configurable via `ReaderOptions`).
2. The byte immediately after the found magic is the start of the
   next message (or EOF).
3. Read the 8 bytes at `end_magic_pos − 8`: that's
   `total_length`.  If non-zero, subtract it from
   `end_magic_pos + 8` to get the start of the current message
   (the `TENSOGRM` byte).
4. Validate by reading `TENSOGRM` at that offset.  If absent,
   fall back to forward scanning from file start.

This enables bidirectional `scan_file()` (§9.1).

---

## 8. Bitmask dtype and compression codecs

`Dtype::Bitmask` represents packed 1-bit values, MSB-first, with
the byte count = `ceil(shape_product / 8)`.  Trailing bits in the
last byte are zero-filled for hash determinism.

Compression codecs split into two tiers:

- **Dtype-agnostic codecs** — work on any dtype including
  `bitmask`:
  `none`, `szip`, `zstd`, `lz4`, `blosc2`, `zfp`, `sz3`.
- **Bitmask-only codecs** — error at encode time if used with any
  other dtype:
  `rle`, `roaring`.

The guard lives at pipeline-build time.  Attempting
`compression = "rle"` or `"roaring"` on a non-bitmask dtype returns
an `EncodingError("codec {codec} only supports dtype=bitmask, got
dtype={dtype}")`.

**RLE on-wire format.**

```
[u8 start_bit] [varint run_1] [varint run_2] ... [varint run_k]
```

- `start_bit`: `0x00` or `0x01` — the value of the first run.
- Each `run_i`: unsigned LEB128 (ULEB128) — count of consecutive bits
  of the alternating value.  Minimum run length 1.  Sum of runs must
  equal exactly `N` (descriptor's total element count); decoder
  errors otherwise.
- Total length of the serialised form is naturally self-delimiting
  via the `length` field in the CBOR descriptor.

**Roaring on-wire format.**  Uses the standard
[Roaring Portable Serialization Format](https://github.com/RoaringBitmap/RoaringFormatSpec).
The `roaring` Rust crate's `serialize_into` produces this format by
default.  Bytes are stored as-is — no Tensogram-specific framing.

`decompress_range` is not supported — both codecs return
`CompressionError::RangeNotSupported` to match `zstd` / `lz4`.

---

## 9. Scanning procedures

### 9.1 Forward scan (primary path)

```
pos = 0
while pos + PREAMBLE_SIZE + POSTAMBLE_SIZE ≤ buf.len():
  if buf[pos..pos+8] != "TENSOGRM":
    pos += 1; continue
  preamble = read_preamble(buf[pos..])
  if preamble.version != 3: fail
  if preamble.total_length > 0:
    # Happy path — deterministic jump.
    end_magic_pos = pos + preamble.total_length - 8
    if buf[end_magic_pos..end_magic_pos+8] == "39277777":
      emit (pos, preamble.total_length)
      pos += preamble.total_length
      continue
  else:
    # Streaming message — scan forward for end magic.
    end = find_next_end_magic(buf, pos + 24)
    emit (pos, end + 8 - pos)
    pos = end + 8
```

### 9.2 Backward scan (new in v3)

Given an `end_pos` (end-of-file for a file, end-of-buffer for an
in-memory scan), walk backward:

```
pos = end_pos
while pos >= PREAMBLE_SIZE + POSTAMBLE_SIZE:
  magic_pos = pos - 8
  if buf[magic_pos..pos] != "39277777":
    pos -= 1; continue
  # We are at a postamble end magic.
  total_length = read_u64_be(buf, magic_pos - 8)
  if total_length == 0:
    # Streaming message with non-seekable producer — bail back
    # to forward scan from the current position.
    switch to forward scan
    break
  msg_start = magic_pos + 8 - total_length
  if buf[msg_start..msg_start+8] != "TENSOGRM":
    # Stray postamble-looking bytes inside payload.  Keep walking.
    pos -= 1; continue
  emit (msg_start, total_length)
  pos = msg_start
```

### 9.3 Bidirectional scan

For large files with both ends addressable, `scan_file()` spawns
two walkers (forward from offset 0, backward from EOF) and meets
in the middle.  Expected ~2× reduction in hop count.  Falls back
to pure forward scanning if any backward hop hits a zero
`total_length` (streaming non-seekable message).

### 9.4 Streaming producers and seekable sinks

A streaming encoder writes `total_length = 0` in the preamble
when it starts, because the total length is not yet known.  At
`finish()`:

- If the sink is **seekable** (file, seekable byte-stream): the
  encoder seeks back and back-fills both the preamble's
  `total_length` and the postamble's `total_length`.
- If the sink is **not seekable** (pipe, socket): both values
  remain `0`.  Readers fall back to forward scan (§9.1).

---

## 10. Random-access reading

### 10.1 With header or footer index frame (fast path)

```
1. Read preamble; check version == 3.
2. Read HeaderMetadata → global metadata.
3. If HEADER_INDEX flag is set:
     Read HeaderIndex → offsets[], lengths[]
   else if FOOTER_INDEX flag is set:
     Seek to postamble; read first_footer_offset.
     Seek to first_footer_offset; read footer frames until
       FooterIndex → offsets[], lengths[]
4. For any object i, seek to offsets[i] and read one frame of
   size lengths[i].
```

Both paths give `O(1)` access to any object.

### 10.2 Without index (slow path)

When neither a header nor a footer index is present, random
access requires a full forward scan of the body.  Reader
implementations may choose to materialise an index lazily on first
random-access call.

### 10.3 Partial range decode

See `docs/src/guide/decoding.md` for the `decode_range` API.  The
partial-decode wire contract is unchanged from v2; frame-body
hashing (see §2.4) lives strictly at the frame boundary and does
not interact with partial-range slicing inside a single data
object.

---

## 11. Integrity verification

### 11.1 Fast integrity scan (`tensogram validate --checksum`)

When `HASHES_PRESENT = 1`, every frame's inline hash slot can be
verified without parsing CBOR.  `--checksum` recomputes each
frame's xxh3-64 over the hash scope defined in §2.4 and compares
it bit-for-bit to the stored slot value; any mismatch is a fatal
validation error.

```
for each frame F in the message:
  footer_size = footer_size_for(F.frame_type)   # 12 B or 20 B
  scope_end   = F.total_length - footer_size
  computed    = xxh3(F.bytes[16 .. scope_end])
  stored      = read_u64_be(F.bytes, F.total_length - 12)
  if computed != stored:
    fail F with HashMismatch { computed, stored }
```

This is the intended path for `tensogram validate --checksum`.
It needs no codec and no CBOR parser, and hashes exactly the
frame-body bytes on disk — the fastest possible integrity path.

When `HASHES_PRESENT = 0`, `validate --checksum` fails with
`ValidationError("message has no inline hashes — cannot run
checksum validation; re-encode with hash_algorithm = Some(Xxh3)")`.

### 11.2 Full validation (`tensogram validate --full`)

Includes the above plus:
- CBOR canonical ordering check
- Decode-pipeline round-trip
- NaN / Inf scan that is **mask-aware**: at each bit set in the NaN /
  Inf± masks, NaN / ±Inf is expected and does not fail validation.
  At any other position, NaN / Inf still fails as `NanDetected` /
  `InfDetected`.

### 11.3 Hash frame consistency

If either a `HeaderHash` or `FooterHash` frame is present, the
validator cross-checks each `hashes[i]` entry against the hex
rendering of the inline hash slot of the i-th data-object frame.
A mismatch is a fatal `ValidationError::HashFrameMismatch`.

---

## 12. Deterministic encoding

All CBOR output is **canonical** per RFC 8949 §4.2:

- Map keys sorted by encoded byte representation (lex, length-first).
- Integers encoded in the shortest form.
- No indefinite-length items.

Any encoder that produces canonical CBOR from the same logical
message will produce byte-identical output regardless of key
insertion order.  Non-canonical CBOR in an incoming message is
accepted on decode but produces a `ValidationCode::NonCanonicalCbor`
warning from `--canonical` mode.

---

## 13. Change log vs. v2

| Aspect | v2 | v3 |
|---|---|---|
| Preamble size | 24 B | 24 B (same) |
| Preamble `version` | 2 | 3 |
| Preamble flags | bits 0–6 | bits 0–7 (`HASHES_PRESENT` added) |
| Postamble size | 16 B | 24 B |
| Postamble fields | `first_footer_offset`, end magic | + `total_length` |
| Frame common tail | `ENDF` only (4 B) | `[hash u64][ENDF]` (12 B), prepended by any type-specific footer fields |
| Frame footer size (most types) | 4 B (`ENDF` only) | 12 B (`[hash][ENDF]`) |
| Data-object frame | types 4 (`NTensorFrame`) and 9 (`NTensorMaskedFrame`) both emitted; type 4 legacy-read on decode | type 4 reserved (obsolete, `FramingError` on read); type 9 renamed `NTensorFrame` and is the only concrete data-object type |
| Data-object footer size | 12 B (`[cbor_offset][ENDF]`) | 20 B (`[cbor_offset][hash][ENDF]`) |
| Per-object hash | `hash` field in CBOR descriptor (optional) | Inline slot in frame footer (populated iff `HASHES_PRESENT=1`); CBOR `hash` field removed |
| Hash scope | just the encoded payload bytes | frame body only: `bytes[16 .. end − footer_size)`; header and footer (including `cbor_offset`) are never covered |
| Hash flagging | per-descriptor, optional | message-wide via preamble bit 7 `HASHES_PRESENT`; no per-frame flag |
| Hash frame CBOR key | `hash_type` | `algorithm` |
| Hash frame CBOR key | `object_count` (separate) | removed (derived from `hashes.len()`) |
| Index frame CBOR key | `object_count` (separate) | removed (derived from `offsets.len()`) |
| Compression codecs | `none`, `szip`, `zstd`, `lz4`, `blosc2`, `zfp`, `sz3` | + `rle`, `roaring` (bitmask-dtype-only) |
| Scan direction | forward only | forward / backward / bidirectional |
| Streaming total_length in postamble | (field did not exist) | mirrored when sink is seekable; 0 otherwise |
| Generic data-object concept | implicit (only NTensorFrame existed) | documented — body phase holds data-object frames; new types slot in at fresh type numbers without a version bump |
| CBOR metadata `version` key | required; cross-check against preamble | **removed** — CBOR metadata frame is free-form; wire-format version lives in the preamble alone.  Decoders route any legacy `"version"` top-level key into `_extra_` for forward-compatibility. |

v2 messages are rejected at preamble read.  No migration path is
provided.
