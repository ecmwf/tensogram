# Frame Introspection

Every tensogram message is a chain of **frames** wrapped in a fixed
envelope. This page is the single, language-neutral contract for reading
that structure — which frames a message holds, where each one starts, how
long it is, what its content bytes are, and whether its hash slot is
populated — **without decoding a payload or a byte of CBOR**. The same two
capabilities are available in **Rust, C, C++, Python, TypeScript, and
Fortran**, with matching semantics.

## The model

Two calls, each taking the bytes of **one** message:

- **The message header** — the 24-byte preamble as typed values: the wire
  `version`, the whole-message `total_length`, and eight `has_*`
  predicates naming the optional frames. It reads 24 bytes and walks
  nothing, so it is the cheapest way to tell a *random-access* message
  (metadata / index / hashes in the **header**) from a *streaming* one
  (in the **footer**).
- **The frame walk** — one record per frame, in wire order, carrying the
  frame's type, its frame-header `version` and raw `flags`, its `offset`
  and `length` within the message, its content bytes, and a `has_hash`
  predicate.

| Language | Frame walk | Message header |
|---|---|---|
| Rust | `frames(message) -> Result<FrameIter>` | `message_header(message) -> Result<MessageHeader>` |
| C | `tgm_frame_iter_create` / `_next` / `_free` | `tgm_message_header` |
| C++ | `frames(msg, len) -> frame_range` | `read_message_header(msg, len)` |
| Python | `frames(buf) -> FrameIter` | `message_header(buf)` |
| TypeScript | `frames(buf) -> Frame[]` | `messageHeader(buf)` |
| Fortran | `tensogram_frames(buffer, iterator, err)` | `tensogram_message_header_read(buffer, header, err)` |

### What counts as a frame

The walk yields the message's frames and nothing else — types **1–3 and
5–9**:

| Code | Name | Phase | Carries |
|---|---|---|---|
| 1 | `HeaderMetadata` | header | CBOR global metadata |
| 2 | `HeaderIndex` | header | CBOR index of data-object offsets |
| 3 | `HeaderHash` | header | CBOR aggregate of the per-object hashes |
| 5 | `FooterHash` | footer | CBOR aggregate of the per-object hashes |
| 6 | `FooterIndex` | footer | CBOR index of data-object offsets |
| 7 | `FooterMetadata` | footer | CBOR global metadata |
| 8 | `PrecederMetadata` | body | per-object metadata for the next data object |
| 9 | `NTensorFrame` | body | one data object: payload + masks + descriptor |

These numbers are the wire's frame-type field, not a binding invention
(see [Message Layout](../format/wire-format.md#frame-types)). **Type 4 is
reserved** — it held the obsolete v2 data-object layout — which is why
the sequence skips from 3 to 5; a message that contains one is reported
as malformed. Only the spelling of the names differs per language:
`FrameType::NTensorFrame` (Rust), `TGM_FRAME_TYPE_NTENSOR` (C, Fortran),
`frame_type::ntensor` (C++), `"NTensorFrame"` (Python, TypeScript).

> **The preamble and postamble are not frames.** They are the envelope,
> they are never yielded by the walk, and everything they hold is
> available from the message header instead. Inter-frame alignment
> padding is likewise part of no frame: the walk steps over it.

### One message per call

Both calls describe exactly **one** message and must be handed bytes that
start at its `TENSOGRM` preamble magic. A `.tgm` file — and any
multi-message buffer — is a plain concatenation of messages, so find the
boundaries with `scan()` first and slice:

```text
scan(buf) → [(offset, length), …]     one entry per message
            └── frames(buf[offset .. offset + length])
```

Frame offsets are then relative to **that slice**, not to the file; add
the message offset back when you want a file-absolute position. Passing
an unsliced multi-message buffer is not an error, but the answer only
describes the first message.

### Offsets, spans, and payload boundaries

- **`offset`** — byte position of the frame's 16-byte frame header,
  relative to the start of the message that was passed in. It is
  **0-based in every binding except Fortran**, which reports a 1-based
  index (matching `tensogram_scan`), so the frame there occupies
  `buffer(offset : offset + length - 1)`.
- **`length`** — the whole-frame span: frame header through the closing
  `ENDF` marker, excluding any alignment padding that follows.
- **`payload`** — the frame's **content**, with the 16-byte frame header
  *and* the type-specific footer stripped. The footer is **20 bytes** for
  the data-object frame type (`[cbor_offset][hash][ENDF]`) and **12
  bytes** for every other type (`[hash][ENDF]`).

```text
      offset                                        offset + length
      │                                             │
      ▼                                             ▼
      ┌──────────────┬───────────────────┬──────────┐
      │ frame header │ payload (content) │  footer  │ (padding)
      │     16 B     │                   │ 12 / 20 B│
      └──────────────┴───────────────────┴──────────┘
```

So `payload` is `length - 16 - footer` bytes long. For a data-object
frame it is the encoded tensor payload, any NaN / Inf mask blobs, and the
trailing CBOR descriptor; for every other frame type it is the CBOR body.
Use `offset` / `length` when you want the whole frame — including its
`FR` header and `ENDF` marker — rather than just the content.

### Lifetime: who owns the payload bytes

| Binding | Payload | Contract |
|---|---|---|
| Rust | **borrowed** | `FrameInfo::payload` is a `&[u8]` view into the message slice; the compiler enforces the lifetime. |
| C | **borrowed** | `TgmFrame::payload` points *into* the `msg` buffer you passed to `tgm_frame_iter_create`. Never freed. It stays valid for as long as `msg` lives — later `_next` calls and `tgm_frame_iter_free` do **not** invalidate it. The cursor borrows `msg` for its whole lifetime, so `msg` must outlive the iterator and must not be moved, reallocated, or mutated meanwhile. |
| C++ | **borrowed** | `frame::payload()` / `payload_data()` view the same buffer, with the same rule: the buffer must outlive the `frame_range`, and the views survive the range's destruction. |
| Python | copied | `Frame.payload` materialises `bytes` on access from the source buffer the iterator retains, so the walk itself copies nothing and the frames stay valid independently of the caller's object. |
| TypeScript | copied | `Frame.payload` is a `Uint8Array` on the JS heap, not a view into WASM linear memory — safe to retain across later WASM calls and safe to mutate. |
| Fortran | copied | `frame%payload()` returns an independent `integer(c_int8_t)` array. `tensogram_frames` also keeps a **private copy of the message**, so the caller's array may be modified, deallocated, or go out of scope while the walk continues. |

In the borrowing bindings, copy the bytes out if you need them to outlive
the message buffer.

### Header flags: exact when buffered, advisory when streaming

The eight predicates are the preamble's structural flags decoded into
named booleans:

| Predicate | Meaning |
|---|---|
| `has_header_metadata` / `has_footer_metadata` | a metadata frame is present in the header / footer |
| `has_header_index` / `has_footer_index` | an object-index frame is present in the header / footer |
| `has_header_hashes` / `has_footer_hashes` | an aggregate-hash frame is present in the header / footer |
| `has_preceder_metadata` | at least one `PrecederMetadata` frame appears in the body |
| `has_hashes_present` | advisory: every frame has its per-frame `HASH_PRESENT` bit set |

For a **buffered** message (one produced by `encode` / `append`) the
encoder knows the whole message before it writes the preamble, so every
flag is an exact statement about the frames present, and `total_length`
is the real byte count.

For a **streaming** message the preamble is written before the first
object exists, so the flags are *advisory*. Only one direction is
guaranteed:

> **frame present ⇒ flag set.** The converse does not hold. In
> particular the streaming encoder sets the `PRECEDER_METADATA` flag
> unconditionally — it cannot know at preamble-write time whether any
> preceder will follow — so `has_preceder_metadata` may be `true` for a
> message whose body holds no `PrecederMetadata` frame. Walk the frames
> if you need certainty.
>
> Note that the flag table in `plans/WIRE_FORMAT.md` §3.1 (and its
> rendering under [Message Layout](../format/wire-format.md#preamble-flags))
> states the flag definitively; the behaviour described here is what the
> encoder actually writes.

`total_length` is `0` in a streaming message whose length was never
back-filled (`finish()` rather than the back-filling variant). That is
not an error — it means "unknown at write time" — and the walk still
stops cleanly at the postamble.

Two more asymmetries worth knowing:

- A streaming message carries a `HeaderMetadata` frame *as well as* the
  footer metadata / index, so `has_header_metadata` alone does not
  identify a random-access layout. Use `has_header_index`.
- `has_hashes_present` is a coarse message-wide summary. For any single
  frame the per-frame `has_hash` predicate is **authoritative**.

### End of walk vs malformed chain

A walk stops for two very different reasons, and every binding keeps them
distinguishable. Frames that parsed before the damage are always yielded
first — corruption is never silently rendered as "fewer frames".

| Binding | Clean end | Malformed chain | Unreadable preamble |
|---|---|---|---|
| Rust | iterator returns `None` | one `Err` item, then the iterator stops | `frames()` / `message_header()` return `Err` |
| C | `tgm_frame_iter_next` returns `false` **and** `tgm_last_error()` is `NULL` | `_next` returns `false` with the reason in `tgm_last_error()`; iteration stays stopped | `tgm_frame_iter_create` returns `NULL`; `tgm_message_header` returns an error code |
| C++ | the range-for loop ends | `framing_error` thrown from the increment that finds it | `framing_error` from `frames()` / `read_message_header()` (`invalid_arg_error` for a null pointer) |
| Python | `StopIteration` | `ValueError` from `next()`, in position | `ValueError` from `frames()` / `message_header()` |
| TypeScript | the returned array ends | `FramingError` from `frames()` (no short array) | `FramingError`; `InvalidArgumentError` for a non-`Uint8Array` |
| Fortran | `found = .false.` with `err == TGM_ERROR_OK` | `found = .false.` with `err == TGM_ERROR_FRAMING`; `tensogram_last_error()` says why | `err` from `tensogram_frames` / `tensogram_message_header_read` |

Laziness follows the host idiom: Rust, C, C++, and Fortran pull one frame
per step; Python performs the (header-only) structural walk up front and
materialises one `Frame` per `next()`; TypeScript returns the whole array,
because the underlying WASM call already materialises it.

## By language

All examples answer the same questions about one message: *is this a
random-access or a streaming layout?*, *which frames does it hold, and
where?*, *how many data objects are there?*

### Rust

```rust
use tensogram::{frames, message_header};

// message: &[u8] — ONE message (e.g. a slice located with `scan`)
// `version` / `total_length` / `flags` are plain fields on MessageHeader;
// the eight structural predicates are methods.
let header = message_header(message)?;
if header.has_header_index() {
    // metadata + index are in the header: random access is cheap
    println!("random-access, {} bytes", header.total_length);
}

let mut data_objects = 0;
for frame in frames(message)? {
    let f = frame?;                       // one Err, then the walk stops
    if f.frame_type.is_data_object() {
        data_objects += 1;
        let _content = f.payload;         // &[u8] borrowed from `message`
    }
    let _whole_frame = &message[f.offset..f.offset + f.length];
    let _hashed = f.has_hash();           // per-frame, authoritative
}
```

### C

The cursor borrows `msg`; `msg` must outlive it. `TgmFrame::payload`
points into `msg` and is never freed.

```c
TgmMessageHeader h;
if (tgm_message_header(msg, msg_len, &h) != TGM_ERROR_OK) {
    fprintf(stderr, "not a message: %s\n", tgm_last_error());
    return 1;
}
printf("v%u, %llu bytes, header index=%d\n", (unsigned)h.version,
       (unsigned long long)h.total_length, (int)h.has_header_index);

tgm_frame_iter_t *it = tgm_frame_iter_create(msg, msg_len);
if (it == NULL) { /* tgm_last_error() says why */ return 1; }

TgmFrame f;
size_t data_objects = 0;
while (tgm_frame_iter_next(it, &f)) {
    if (f.frame_type == TGM_FRAME_TYPE_NTENSOR) data_objects++;
    printf("type=%d offset=%zu length=%zu payload=%zu hash=%d\n",
           (int)f.frame_type, f.offset, f.length, f.payload_len,
           (int)tgm_frame_has_hash(&f));
}
/* false means either a clean end or a broken chain: */
const char *err = tgm_last_error();       /* NULL => clean end */
tgm_frame_iter_free(it);                  /* payload pointers stay valid */
```

### C++

```cpp
const auto h = tensogram::read_message_header(msg.data(), msg.size());
if (h.has_footer_index()) { /* streaming layout */ }

std::size_t data_objects = 0;
for (const auto& f : tensogram::frames(msg.data(), msg.size())) {
    if (f.is_data_object()) ++data_objects;
    std::string_view content = f.payload();     // borrows `msg`
    (void)content;
}
// A broken chain throws tensogram::framing_error from the increment
// that discovers it, after the intact frames have been yielded.
```

`frame_range` is move-only and single-pass: it owns the C cursor (freed
in its destructor) and a second `begin()` resumes where the previous
iterator stopped.

### Python

```python
header = tensogram.message_header(msg)
header.has_header_index      # True → random-access layout
header.total_length          # 0 for a non-back-filled streaming message
header.flags                 # raw bits, if you really want them

for frame in tensogram.frames(msg):
    frame.frame_type         # "NTensorFrame", "HeaderMetadata", …
    frame.frame_type_code    # 9, 1, … (the wire number)
    frame.offset             # start of the frame within `msg`
    frame.length             # whole-frame span, through ENDF
    frame.payload            # bytes; header + footer already stripped
    frame.has_hash, frame.is_data_object

# one message at a time
for offset, length in tensogram.scan(buf):
    message = buf[offset:offset + length]
    n = sum(1 for f in tensogram.frames(message) if f.is_data_object)
```

`FrameIter` also implements `len()` (frames left to yield) and `repr()`,
so it is comfortable at a REPL.

### TypeScript

```ts
import { frames, messageHeader, scan } from '@ecmwf.int/tensogram';

const header = messageHeader(msg);
header.hasHeaderIndex;       // random-access layout
header.totalLength;          // 0 for a non-back-filled streaming message

for (const f of frames(msg)) {
  f.frameType;               // 'NTensorFrame' | 'HeaderMetadata' | …
  f.frameTypeCode;           // 9 | 1 | …
  f.offset;                  // start of the frame within `msg`
  f.length;                  // whole-frame span, through `ENDF`
  f.payload;                 // Uint8Array copy on the JS heap
  f.hasHash;
}

// one message at a time
for (const { offset, length } of scan(fileBytes)) {
  const chain = frames(fileBytes.subarray(offset, offset + length));
  const objects = chain.filter((f) => f.frameType === 'NTensorFrame');
  console.log(offset, chain.length, objects.length);
}
```

### Fortran

Offsets are **1-based**; the iterator is a non-copyable handle that keeps
its own copy of the message, and every payload is copied out.

```fortran
type(tensogram_message_header) :: hdr
type(tensogram_frame_iterator) :: it
type(tensogram_frame)          :: fr
integer(c_int8_t), allocatable :: payload(:)
integer(c_int) :: err
integer :: data_objects
logical :: found

call tensogram_message_header_read(wire, hdr, err)
call tensogram_check(err, 'message_header_read')
print *, hdr%version(), hdr%total_length(), hdr%has_header_index()

call tensogram_frames(wire, it, err)      ! takes its own copy of `wire`
call tensogram_check(err, 'frames')
data_objects = 0
do
   call it%next(fr, found, err)
   if (.not. found) exit
   if (fr%frame_type() == TGM_FRAME_TYPE_NTENSOR) data_objects = data_objects + 1
   payload = fr%payload()                 ! an independent copy
   print *, fr%frame_type(), fr%offset(), fr%length(), size(payload), &
            fr%has_hash()
end do
if (err /= TGM_ERROR_OK) print *, 'malformed chain: ', tensogram_last_error()
call it%free()
```

## Runnable examples

| Language | Example |
|---|---|
| C++ | [`examples/cpp/26_frame_walker.cpp`](https://github.com/ecmwf/tensogram/blob/main/examples/cpp/26_frame_walker.cpp) |
| Python | [`examples/python/21_frames_and_message_header.py`](https://github.com/ecmwf/tensogram/blob/main/examples/python/21_frames_and_message_header.py) |
| TypeScript | [`examples/typescript/21_frame_walker.ts`](https://github.com/ecmwf/tensogram/blob/main/examples/typescript/21_frame_walker.ts) |
| Fortran | [`examples/fortran/frame_walker.f90`](https://github.com/ecmwf/tensogram/blob/main/examples/fortran/frame_walker.f90) |

## See also

- [What is a Message?](../concepts/messages.md) — why the format is
  frame-based in the first place.
- [Message Layout](../format/wire-format.md) — the byte-level spec for
  the preamble, frame header, footers, and every frame type.
- [Reading Metadata](metadata.md) — the companion contract for reading a
  frame's *contents* rather than its structure.
- [Iterators](iterators.md) — walking *messages* and *objects*, one level
  up from frames.
