# Wire Format

A tensor message is composed by HEADER (Preamble + Frames), DATA OBJECT FRAMES and FOOTER (Frames + Postamble).

The HEADER and FOOTER each have a fixed part (Preamble and Postamble) contain optional FRAMES.

Whenever Reserved flags are unused, they shall be set to to zero.

# Frames

Each frame is always identified by a start marker ASCII 'FR' and a uint16 that defines the type. It finishes with an end marker ASCII ENDF. uint16 types assigned are:
1 - HEADER METADATA FRAME
2 - HEADER INDEX FRAME
3 - HEADER HASH FRAME
4 - DATA OBJECT FRAME
5 - FOOTER HASH FRAME
6 - FOOTER INDEX FRAME
7 - FOOTER METADATA FRAME

```
FRAME (16 bytes)
Offset  Size    Field
──────  ──────  ─────────────────────────────────
0       4       Magic: "FR" + uint16 (Frame type)
4       2       Version (uint16 BE)
6       2       Reserved Flags (uint16 BE)
8       8       Total Lenght as Offset to end of object (uint64 BE)
```
Frame versions are independent from version of message and other frame types.
HEADER PREAMBLE and FOOTER POSTAMBLE are not considered Frames.

It is always possible to have padding between Frames, from the ENDF to next FR+int. This exists to allow for memory bounds alignment to 64bits, which is a option of the encoder.

Some Frame types will contain structures encoded in CBOR.

### CBOR

Note that natively, a standard CBOR byte stream does not use universal "start" and "end" markers** in the way that JSON uses `{` and `}` or XML uses tags.
Instead, CBOR is fundamentally a **length-prefixed** format (or "self-describing").
Here is exactly how CBOR handles boundaries at the byte level: when a CBOR encoder writes an object (like a Map/dictionary, an Array, or a String), the very first byte acts as a "header" that declares the data type and how many items (or bytes) follow. Because the decoder knows exactly how many bytes or items to expect right from the start, **there is no end marker.** The decoder simply counts the bytes or items as it reads them and stops when it reaches the declared length.

## Message

Here is the overall message format:

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER PREAMBLE    magic, version, flags, length (fixed sz) │
├──────────────────────────────────────────────────────────────┤
│  HEADER METADATA FRAME  CBOR metadata (optional)             │
├──────────────────────────────────────────────────────────────┤
│  HEADER INDEX FRAME CBOR [count, object offsets] (optional)  │
├──────────────────────────────────────────────────────────────┤
│  HEADER HASH FRAME CBOR [count, hash_type, hashes] (optional)│
├──────────────────────────────────────────────────────────────┤
│  DATA OBJECT FRAME  header + payload + footer  (object 0)    │
│  DATA OBJECT FRAME  header + payload + footer  (object 1)    │
│  ...                (any number of objects)                  │
├──────────────────────────────────────────────────────────────┤
│  FOOTER HASH FRAME CBOR [count, hash_type, hashes] (optional)│
├──────────────────────────────────────────────────────────────┤
│  FOOTER INDEX FRAME CBOR [count, object offsets] (optional)  │
├──────────────────────────────────────────────────────────────┤
│  FOOTER METADATA FRAME  CBOR metadata (optional)             │
├──────────────────────────────────────────────────────────────┤
│  FOOTER POSTAMBLE first_footer_offset | end_magic (fixed sz) │
└──────────────────────────────────────────────────────────────┘
```

## Header

The HEADER starts with magic, version and reserved flags.

```
Tensogram message HEADER PREAMBLE (24 bytes)
Offset  Size    Field
──────  ──────  ─────────────────────────────────
0       8       Magic: "TENSOGRM"
8       2       Version (uint16 BE)
10      2       Flags (uint16 BE)
12      4       Reserved Flags (uint32 BE)
16      8       Total Lenght to end of message (may be zero if stream)
```

The flags in the message header preamble will indicate if the optional frames exist:
- Header Metadata Frame
- Footer Metadata Frame
- Header Index Frame
- Footer Index Frame
- Header Hashes Frame
- Footer Hashes Frame

Either a header or a footer metadata frame must always be present, ie Messages cannot be without metadata. Indexes and hashes, in header and footer are optional, but highly encouraged to always have 1 of them. Default is to add them, in the header encoding in a single buffer or default in the footer if encoding while streaming.

## Metadata Frame

Irrespective of position, Hearder or Footer, the metadata frame for uniquely identifies the message.
Each metadata CBOR contains a mandatory sub-objects:
 - 'common' holds metadata that is common to all data objects in the message. this may contain internal namespaces for managing different vocabularies.
 - 'payload' holds a list of metadata per data object. the index of the list matches the order in the message, can be used to assert the data obj count. this may contain internal namespaces that should match 'common'.
 - 'reserved' is set aside for internals of the message handling and support future features. Inside it contains:
    - 'encoder' describes the library that encoded the message. contains:
        - 'name', 'tensogram' for this one
        - 'version', software version
    - 'time' date-time in UTC zulu of time of encoding
    - 'uuid' UUID RFC 4122 generated at time of encoding, useful for provenance and tracking.

## Footer

```
Footer Postamble (16 bytes)
Offset  Size    Field
──────  ──────  ─────────────────────────────────
0       8       Offset to start of first footer Frame object (uint64 BE)
8       8       Magic: "39277777"
```

## Data Object Frames

We assume that a single data object frame can always be encoded in a single buffer, hence it is possible to always encode its CBOR encoding information together with the data payload. 1 flag is dedicated to identify if CBOR object is before (0) or after (1) of the data payload. Default is to encode AFTER since it is a single append to record the CBOR, sometimes only fully known after all the encoding is finished.

The CBOR information completely describes the data object encoding. For an N-Tensor, it also describes the shape of the layout once decoded: ndim, shape, strides, dtype.
 This header has a preamble followed by a CBOR block. Preamble is 16 bytes (see below). Then a CBOR structure describing the full encoding information, including the offset to the start of data because it is possible to have padding between the end of CBOR and the start of the data (for memory alignment).

```
Data Object Preamble (16 bytes)
Offset  Size    Field
──────  ──────  ─────────────────────────────────
0       4       Magic: "FR" + uint16 (Frame type)
4       2       Version (uint16 BE)
6       2       Reserved Flags (uint16 BE)
8       8       Total Lenght as Offset to end of object (uint64 BE)
```

```
┌──────────────────────────────────────────────────────────────┐
│  DATA OBJ PREAMBLE  magic, version, flags, lenght (fixed sz) │
├──────────────────────────────────────────────────────────────┤
│  DATA OBJ ENCODING    CBOR descritption (before)             │
├──────────────────────────────────────────────────────────────┤
│  DATA BYTESTREAM PAYLOAD                                     │
├──────────────────────────────────────────────────────────────┤
│  DATA OBJ ENCODING CBOR descritption (after - default)       │
├──────────────────────────────────────────────────────────────┤
│  DATA OBJ FOOTER CBOR offset (uint64) + end_magic 'ENDF'     │
└──────────────────────────────────────────────────────────────┘
```

## Operations
### Stored-file reader (seekable, random access)

**With header index:**
This is expected to be the most common case, when a message was encoded not in streaming mode.

```
1. Read header preamble → flags
2. Read metadata frame → global context
3. Read header index frame → offsets[], object_count
4. Seek to offsets[N], read object frame → decode
```

**Without header index, ie footer index only:**
This happens when a message was encoded in streaming mode, ie it was not possible to know ahead of time how many data objects were being sent, and the index is placed at the end.

```
1. Seek to end - 16, read footer → first_footer_offset
2. Seek to first_footer, read footer index → offsets[], object_count
3. Seek to offsets[N], read object frame → decode
```

Both paths give O(1) access to any object.

Multiple messages can be concatenated in a `.tgm` file. To find message boundaries:
1. Scan for `TENSOGRM` magic (8 bytes)
2. If `total_length` not zero, use to advance to the next message otherwise use header index if present.
3. If no header index is present: scan forward frame-by-frame (each frame is has header with length) until the next magic or EOF