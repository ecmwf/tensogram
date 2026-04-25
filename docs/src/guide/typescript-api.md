# TypeScript API

Tensogram ships `@ecmwf.int/tensogram`, a TypeScript package that wraps the
WebAssembly build with typed, idiomatic helpers. Use it in any modern
browser or Node ≥ 20.

The package exposes typed encode / decode / scan, dtype-aware payload
views, metadata helpers, progressive streaming decode, the
`TensogramFile` file / URL helper, the `validate` wrapper,
`encodePreEncoded`, and first-class `float16` / `bfloat16` /
`complex*` view classes.

## Installation

The package is not yet published to npm. Build it locally:

```bash
# First, build the WebAssembly blob from the Rust source
cd typescript
npm install
npm run build:wasm   # runs wasm-pack build -t web -d typescript/wasm
npm run build        # runs wasm-pack + tsc
```

Or use the top-level `Makefile`:

```bash
make ts-build        # build WASM + tsc
make ts-test         # vitest
make ts-typecheck    # strict tsc --noEmit on src + tests
```

## Quick start

```ts
import {
  init, encode, decode,
  type DataObjectDescriptor,
  type GlobalMetadata,
} from '@ecmwf.int/tensogram';

// One-time WASM initialisation (idempotent)
await init();

// ── Encode ────────────────────────────────────────────────────────────
const temps = new Float32Array(100 * 200);
for (let i = 0; i < temps.length; i++) temps[i] = 273.15 + i / 100;

const meta: GlobalMetadata = { version: 3 };
const descriptor: DataObjectDescriptor = {
  type: 'ntensor',
  ndim: 2,
  shape: [100, 200],
  strides: [200, 1],
  dtype: 'float32',
  byte_order: 'little',
  encoding: 'none',
  filter: 'none',
  compression: 'none',
};

const msg: Uint8Array = encode(meta, [{ descriptor, data: temps }]);

// ── Decode ────────────────────────────────────────────────────────────
const { metadata, objects } = decode(msg);
const arr = objects[0].data();  // Float32Array (inferred from dtype)
console.log(arr.length);        // 20000
```

## API surface

### `init(opts?)`

Loads and instantiates the WASM blob. Must be awaited before any
other function is called. Safe to call multiple times — subsequent
calls reuse the same instance.

```ts
await init();                                              // defaults
await init({ wasmInput: new URL('...', import.meta.url) });  // custom location
```

### `encode(metadata, objects, opts?)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `metadata` | `GlobalMetadata` | Free-form metadata; only `base`, `_reserved_`, `_extra_` are library-interpreted. An empty `{}` is valid. The wire-format version lives in the preamble — see [`WIRE_VERSION`](#wire_version). |
| `objects` | `Array<{ descriptor, data }>` | Each `data` is a `TypedArray` or `Uint8Array` |
| `opts.hash` | `'xxh3' \| false` | Hash algorithm. Default `'xxh3'`. Pass `false` to disable. |

Returns: `Uint8Array` containing the complete wire-format message.

### `decode(buf, opts?)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `Uint8Array` | Raw message bytes |
| `opts.verifyHash` | `boolean` | Default `false`. If `true`, throws `HashMismatchError` on corruption. |

Returns: `{ metadata: GlobalMetadata, objects: DecodedObject[], close() }`.

### `decodeMetadata(buf)`

Returns only the metadata; does not touch any payload bytes.

### `decodeObject(buf, index, opts?)`

O(1) seek to object `index`, decoding only that object.

### `scan(buf)`

Returns `Array<{ offset: number; length: number }>` for each
Tensogram message found in a (potentially multi-message) buffer.
Garbage between messages is silently skipped.

### `DecodedObject` / `DecodedFrame`

```ts
interface DecodedObject {
  readonly descriptor: DataObjectDescriptor;
  /** Copy into the JS heap.  Safe across WASM memory growth. */
  data(): TypedArray;
  /** Zero-copy view.  Invalidated if WASM memory grows. */
  dataView(): TypedArray;
  readonly byteLength: number;
}

interface DecodedFrame extends /* structurally */ DecodedObject {
  /** The matching `base[i]` entry from the containing message. */
  readonly baseEntry: BaseEntry | null;
  close(): void;
}
```

The returned array type is picked from `descriptor.dtype`:

| `dtype` | Returned TypedArray |
|---------|---------------------|
| `float32` | `Float32Array` |
| `float64` | `Float64Array` |
| `int8` | `Int8Array` |
| `int16` | `Int16Array` |
| `int32` | `Int32Array` |
| `int64` | `BigInt64Array` |
| `uint8` | `Uint8Array` |
| `uint16` | `Uint16Array` |
| `uint32` | `Uint32Array` |
| `uint64` | `BigUint64Array` |
| `float16` / `bfloat16` | `Uint16Array` (no native half-precision in JS) |
| `complex64` | `Float32Array` (interleaved real, imag) |
| `complex128` | `Float64Array` (interleaved real, imag) |
| `bitmask` | `Uint8Array` (packed bits) |

### `getMetaKey(meta, path)`

Dot-path lookup matching the Rust / Python / CLI first-match-across-base
semantics: searches `base[0]`, `base[1]`, …, skipping the `_reserved_`
key in each, then falls back to `_extra_`.

```ts
getMetaKey(meta, 'mars.param')      // 'base[0].mars.param' first match
getMetaKey(meta, '_extra_.source')  // explicit _extra_ prefix
```

Returns `undefined` if the key is missing (never throws).

### `computeCommon(meta)`

Mirror of `tensogram::compute_common`. Returns a
`Record<string, CborValue>` of keys that are present with identical
values in every entry of `meta.base`. Useful for display and merge
operations.

## Error classes

All errors thrown from this package are instances of the abstract
`TensogramError` class. Eight concrete subclasses match the Rust
`TensogramError` variants plus the TS-layer `InvalidArgumentError`
and `StreamingLimitError`:

```ts
import {
  TensogramError,
  FramingError,
  MetadataError,
  EncodingError,
  CompressionError,
  ObjectError,
  IoError,
  RemoteError,
  HashMismatchError,
  InvalidArgumentError,
  StreamingLimitError,
} from '@ecmwf.int/tensogram';

try {
  decode(corruptBuffer);
} catch (err) {
  if (err instanceof FramingError) {
    console.error('bad wire format:', err.message);
  } else if (err instanceof HashMismatchError) {
    console.error('integrity failure:', err.expected, err.actual);
  } else {
    throw err;
  }
}
```

## Memory model

- **Safe-copy by default.** `object.data()` / `frame.data()` always
  allocate a new `TypedArray` on the JS heap. It remains valid even
  after the underlying `DecodedMessage` / `DecodedFrame` is freed or
  WASM memory grows.
- **Zero-copy opt-in.** `object.dataView()` / `frame.dataView()` return
  a view directly into WASM linear memory. It is **invalidated the
  next time any WASM call grows linear memory** — which can happen on
  the next `encode()` / `decode()`. Read the view immediately or copy
  it.
- **Explicit cleanup.** `DecodedMessage`, `DecodedFrame`, and
  `TensogramFile` all expose `.close()` to release WASM-side memory.
  A `FinalizationRegistry` also calls `.free()` on the underlying
  WASM handle when the wrapper is garbage-collected, but explicit
  `.close()` is strongly recommended for deterministic cleanup.

## Streaming decode

Use `decodeStream(readable, opts?)` to progressively decode a
`ReadableStream<Uint8Array>`. Works against any stream source —
`fetch().body`, a Node `Readable.toWeb()`, a `Blob.stream()`, or a
hand-rolled `ReadableStream`.

```ts
import { decodeStream } from '@ecmwf.int/tensogram';

const res = await fetch('/data.tgm');
for await (const frame of decodeStream(res.body!)) {
  render(frame.descriptor.shape, frame.data());
  frame.close();
}
```

Options:

| Option | Type | Description |
|---|---|---|
| `signal` | `AbortSignal` | Cancels the iteration. The underlying reader is cancelled and the decoder is freed cleanly. |
| `maxBufferBytes` | `number` | Max size of the internal staging buffer. Default: 256 MiB. Exceeding this throws `StreamingLimitError`. |
| `onError` | `(err: StreamDecodeError) => void` | Called whenever a corrupt message is skipped. The iterator does **not** throw on skips — it keeps going. |

Key behaviours:

- **Chunk-boundary tolerant.** A message can be split across any
  number of chunks. The decoder accumulates until a complete message
  is seen, then emits every object as a separate frame.
- **Corruption resilient.** A single bad message is skipped; the
  iterator keeps going with subsequent messages. Pass `onError` to
  observe the skips.
- **Early break is safe.** Breaking out of the `for await` loop runs
  the generator's `finally` block, which releases the stream reader
  and frees the decoder.
- **AbortSignal cancels cleanly.** Firing the signal cancels the
  underlying reader; the generator throws whatever error the signal
  carries.

## File API

`TensogramFile` gives you random-access reads over a `.tgm` file,
whether it lives on the local file system, behind an HTTPS URL, or
already in memory.

```ts
import { TensogramFile } from '@ecmwf.int/tensogram';

// Node: from the local file system
const file = await TensogramFile.open('/data/input.tgm');

// Browser or Node: over HTTPS
const file = await TensogramFile.fromUrl('https://example.com/input.tgm');

// Any runtime: from pre-loaded bytes
const file = TensogramFile.fromBytes(uint8ArrayFromSomewhere);
```

All three factories produce an identical object:

```ts
interface TensogramFile extends AsyncIterable<DecodedMessage> {
  readonly messageCount: number;
  readonly byteLength: number;
  readonly source: 'local' | 'remote' | 'buffer';

  // Whole-message access (unchanged since Scope B).
  message(index: number, opts?: DecodeOptions): Promise<DecodedMessage>;
  messageMetadata(index: number): Promise<GlobalMetadata>;
  rawMessage(index: number): Promise<Uint8Array>;

  // Layout-aware per-object access (new — cheap over HTTP Range).
  messageDescriptors(index: number): Promise<{
    metadata: GlobalMetadata;
    descriptors: DataObjectDescriptor[];
  }>;
  messageObject(msgIndex: number, objectIndex: number,
    opts?: DecodeOptions): Promise<DecodedMessage>;
  messageObjectRange(msgIndex: number, objectIndex: number,
    ranges: readonly RangePair[],
    opts?: DecodeRangeOptions): Promise<DecodeRangeResult>;

  // Bounded-concurrency fan-out.
  messageObjectBatch(msgIndices: readonly number[], objectIndex: number,
    opts?: DecodeOptions & { concurrency?: number }): Promise<DecodedMessage[]>;
  messageObjectRangeBatch(msgIndices: readonly number[], objectIndex: number,
    ranges: readonly RangePair[],
    opts?: DecodeRangeOptions & { concurrency?: number }): Promise<DecodeRangeResult[]>;
  prefetchLayouts(msgIndices: readonly number[],
    opts?: { concurrency?: number }): Promise<void>;

  [Symbol.asyncIterator](): AsyncIterator<DecodedMessage>;
  close(): void;
}
```

> **Numeric limit.**  All TensogramFile file positions are JavaScript
> `number` values, capped at `Number.MAX_SAFE_INTEGER` (2<sup>53</sup> − 1,
> ≈ 9 PB).  Files larger than that must be processed via the Rust or
> Python bindings; the TS wrapper throws
> `InvalidArgumentError` when a WASM-returned `u64` exceeds the safe
> range.

Usage:

```ts
const file = await TensogramFile.open('/data/input.tgm');
try {
  console.log(`${file.messageCount} messages, ${file.byteLength} bytes`);

  // Random access
  const first = await file.message(0);
  console.log(first.objects[0].descriptor.shape);
  first.close();

  // Async iteration
  for await (const msg of file) {
    // ...
    msg.close();
  }
} finally {
  file.close();
}
```

### `TensogramFile.open(path, opts?)` (Node only)

Loads the file via `node:fs/promises`. The `node:fs/promises` import
is dynamic so browser bundlers can tree-shake this code path.

| Option | Type | Description |
|---|---|---|
| `signal` | `AbortSignal` | Cancels the initial read. |

### `TensogramFile.fromUrl(url, opts?)` (any fetch-capable runtime)

Downloads the file over HTTPS using the ambient `globalThis.fetch`.

| Option | Type | Description |
|---|---|---|
| `fetch` | `typeof fetch` | Override the fetch implementation (useful for tests and for browsers with a polyfill, and for AWS-authenticated S3 via [`createAwsSigV4Fetch`](#aws-signed-s3-compatible-access)). |
| `headers` | `HeadersInit` | Extra request headers (auth, etc.). |
| `signal` | `AbortSignal` | Cancels the download. |
| `concurrency` | `number` | Per-host concurrency cap for fan-out operations (`messageObjectBatch`, `prefetchLayouts`, descriptor prefix fetches).  Defaults to `6`, matching typical browser per-host connection limits. |
| `bidirectional` | `boolean` | Enable the bidirectional remote-scan walker on open.  See [Bidirectional scan](#bidirectional-scan) below.  Default `false`. |
| `debug` | `boolean` | Emit `console.debug` events on every walker state transition.  Default `false`. |

### `TensogramFile.fromBytes(bytes)`

Wraps an already-loaded `Uint8Array`. The buffer is defensively
copied, so later mutation of the caller's buffer is invisible to
the `TensogramFile`.

### Range-based lazy access

`TensogramFile.fromUrl` automatically probes the server for HTTP
Range support.  When the `HEAD` response advertises `Accept-Ranges:
bytes` and a finite `Content-Length`, the file switches to a **lazy
backend**:

- The initial open issues a small `HEAD` + one 24-byte Range read per
  message preamble to build the boundary index.  **No payload data is
  downloaded.**
- `rawMessage(i)` / `message(i)` fetch the full message via a
  `Range: bytes=offset-(offset+length-1)` GET and cache it in a
  32-entry LRU.
- `messageMetadata(i)` fetches at most a 256 KB header chunk (or
  256 KB footer suffix for footer-indexed messages) and caches
  the decoded `GlobalMetadata` in a per-message `MessageLayout`
  entry.  Subsequent metadata reads are free.
- `messageDescriptors(i)` uses the cached index frame and the
  descriptor-prefix optimisation (header + footer + CBOR region for
  large frames, full frame for small ones) so a 10-object message
  with 100 MB frames fetches only a few KB per descriptor.
- `messageObject(i, j)` and `messageObjectRange(i, j, ranges)` each
  issue exactly one Range GET for the target object's frame bytes.
- `messageObjectBatch`, `messageObjectRangeBatch`, and
  `prefetchLayouts` fan out with bounded concurrency (default 6,
  configurable via `FromUrlOptions.concurrency` or per-call
  `opts.concurrency`).

When the server omits `Accept-Ranges`, returns non-`200` on HEAD, or
the file uses streaming-mode messages (`total_length=0` — the writer
did not know the final length up front), the open falls back to a
single eager GET.  Behaviour is indistinguishable to callers except in
memory use and timing.

Browser callers using `fromUrl` directly need CORS to expose the
`Accept-Ranges`, `Content-Range`, and `Content-Length` headers.

### Bidirectional scan

`fromUrl` accepts an opt-in `bidirectional: boolean` flag that
enables a two-cursor walker.  The lazy backend issues paired
forward-preamble and backward-postamble Range fetches per scan round,
alternating with forward-only steps whenever backward yields (format
error, streaming preamble, gap-below-min, overlap, exceeds-bound).
On well-formed header-indexed files this roughly halves the number
of `GET` requests needed for tail / full-scan access.

```typescript
import { init, TensogramFile } from '@ecmwf.int/tensogram';

await init();
const file = await TensogramFile.fromUrl('https://example.com/data.tgm', {
    bidirectional: true,
});
console.log('messageCount:', file.messageCount);
console.log('messageLayouts:', file.messageLayouts);
file.close();
```

Forward-only and bidirectional opens produce identical
`messageLayouts` on well-formed files; the parity harness asserts
this on every fixture.  Default `false` until benchmarks confirm the
win across every workload tier — same default across Rust, Python,
and TypeScript.

`bidirectional: true` requires `concurrency >= 2` (the default of
`6` is fine).  Passing `concurrency: 1` alongside `bidirectional:
true` rejects the `fromUrl` promise with `InvalidArgumentError`
before any HTTP probe is issued, since the paired round needs two
parallel fetches to be useful.

`debug: true` emits `console.debug` events on every state transition
— `tensogram:scan:mode`, `tensogram:scan:fallback`,
`tensogram:scan:fwd-terminated`, `tensogram:scan:gap-closed`,
`tensogram:scan:hop`, `tensogram:scan:footer-eager` — same vocabulary
as the Rust `tracing` events at `target = "tensogram::remote_scan"`.

#### Eager footer-indexed backward discovery

When the bidirectional walker discovers a footer-indexed message via
its postamble, the dispatcher folds an eager footer-region fetch into
the same paired round as the candidate-preamble validation.  The
parsed footer's `metadata` and `index` frames land in the cached
layout inline, so a subsequent `messageMetadata(idx)` /
`messageDescriptors(idx)` short-circuits without issuing a separate
footer-region GET.

The fetch is best-effort: if the footer Range request fails or the
chunk fails to parse, the layout still commits via the validated
preamble alone, and the lazy populate path picks up footer discovery
on first metadata access.  Header-indexed messages on backward keep
the lazy path (the eager-footer code path is gated on the
`FOOTER_METADATA + FOOTER_INDEX` flag combination).

Behaviour is symmetric across the Rust sync / async dispatchers and
the TypeScript walker — the same wire-format outcome enum
(`BackwardOutcome.NeedPreambleValidation`) carries the postamble's
`first_footer_offset` so both sides decide identically.

### AWS-signed (S3-compatible) access

For authenticated reads against S3 or any S3-compatible HTTPS
endpoint, wrap `fetch` with `createAwsSigV4Fetch` and pass it through
`FromUrlOptions.fetch`:

```ts
import { createAwsSigV4Fetch, TensogramFile } from '@ecmwf.int/tensogram';

const signedFetch = createAwsSigV4Fetch({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  region: 'eu-west-1',
  // Optional: `sessionToken` for STS credentials; `service` for non-S3
  // S3-compatible endpoints (default 's3').
});

const file = await TensogramFile.fromUrl(
  'https://my-bucket.s3.eu-west-1.amazonaws.com/data.tgm',
  { fetch: signedFetch },
);
```

The pure signer `signAwsV4Request` is also exported for callers that
want to manage the request lifecycle themselves.  Both are covered by
byte-for-byte AWS `sig-v4-test-suite` parity tests, including
query-string canonicalisation, header value trim, session-token
handling, and pre-encoded path round-tripping.

Azure Blob and Google Cloud Storage are not yet supported by a bundled
helper — generate a presigned URL in your control plane and pass it as
a plain HTTPS URL instead.

### Append (Node local file system)

`TensogramFile#append(meta, objects, opts?)` encodes the new message
in-memory, appends it to the on-disk file, refreshes the position
index, and makes the new message reachable via `message(i)` on the
same handle.  Only supported when the file was opened via
`TensogramFile.open(path)` — `fromBytes`- and `fromUrl`-backed files
throw `InvalidArgumentError`, matching the contract in the other
language bindings.

```ts
const file = await TensogramFile.open('/data/forecast.tgm');
try {
  await file.append({ version: 3 }, [{ descriptor, data }]);
  console.log(`now has ${file.messageCount} messages`);
} finally {
  file.close();
}
```

## Scope-C API additions

Scope C brought the TypeScript wrapper to full API parity with Rust /
Python / FFI / C++.  The surface additions are:

| Function / class | What it does |
|---|---|
| `decodeRange(buf, objIndex, ranges, opts?)` | Partial sub-tensor decode.  `ranges` is an array of `[offset, count]` pairs in element units; each returned `parts[i]` is a dtype-typed view.  Option `join: true` concatenates every range into a single view. |
| `computeHash(bytes, algo?)` | Standalone `xxh3` hash — matches the digest stamped by `encode()` on the same bytes. |
| `simplePackingComputeParams(values, bits, decScale?)` | GRIB-style simple-packing parameter computation.  Return shape uses snake-case keys so the result spreads directly into a descriptor. |
| `validate(buf, opts?)` | Report-only validation (never throws on bad input).  Modes: `quick`, `default`, `checksum`, `full`. |
| `validateBuffer(buf, opts?)` | Multi-message buffer: reports file-level gaps / trailing garbage plus per-message reports. |
| `validateFile(path, opts?)` | Node-only helper: reads the file via `node:fs/promises` then delegates to `validateBuffer`. |
| `encodePreEncoded(meta, objects, opts?)` | Wrap already-encoded bytes verbatim into a wire-format message.  The library still validates descriptor structure and stamps a fresh hash. |
| `StreamingEncoder` | Frame-at-a-time construction.  Two modes: **buffered** (default, `finish()` returns the complete `Uint8Array`) or **streaming** via `opts.onBytes` callback (bytes flow through the callback as they're produced; `finish()` returns an empty `Uint8Array`). |
| `TensogramFile#append` | Append a new message to a file opened via `TensogramFile.open(path)`.  Node-only. |

## Streaming `StreamingEncoder` (no full-message buffering)

For browser uploads, WebSocket pushes, or any sink that needs bytes as
soon as they are produced, pass an `onBytes` callback to the
`StreamingEncoder` constructor:

```ts
const enc = new StreamingEncoder({ version: 3 }, {
  onBytes: (chunk) => uploadSocket.send(chunk),   // e.g. WebSocket.send
});
enc.writeObject(descriptor, new Float32Array([1, 2, 3]));
enc.finish();    // flushes footer; returns empty Uint8Array in streaming mode
enc.close();
```

Semantics:

- The callback is invoked during construction (preamble + header
  metadata frame), during each `writeObject` / `writeObjectPreEncoded`
  (one data-object frame's bytes, potentially across multiple
  invocations), and during `finish()` (footer frames + postamble).
- Concatenating every chunk the callback sees (in order) yields a
  message byte-for-byte identical to what buffered mode would
  return.  Tested via round-trip with `decode()`.
- The callback **must be synchronous** — `Promise` return values are
  silently discarded because the Rust/WASM writer contract is
  synchronous.  Buffer internally first if you need async work.
- Each `chunk` is JS-owned and fresh per invocation.  Copy
  (`new Uint8Array(chunk)` or `chunk.slice()`) if you need to keep it
  past the next `writeObject` — the underlying `ArrayBuffer` is
  invalidated when WASM memory grows.
- If the callback throws, the exception surfaces as an `IoError` on
  the next `writeObject` / `finish`.  The encoder state is undefined
  after an error — call `close()` and start over.
- `enc.streaming` (getter) reports whether an `onBytes` sink was
  supplied — useful for code that needs to branch on mode.

Parity note: the Rust core `StreamingEncoder<W: Write>` has always
supported arbitrary sinks; the WASM/TS surface now exposes this
capability to JS code.  Python / FFI / C++ bindings remain
buffered-only; extending them would follow the same `JsCallbackWriter`
pattern with a language-specific sink abstraction.

## First-class half-precision and complex dtypes

Scope C also upgraded the dtype dispatch in {@link typedArrayFor}.
`obj.data()` now returns a **first-class view** for dtypes JS does not
have a native TypedArray for:

| Dtype | `data()` return type |
|---|---|
| `float16` | `Float16Array` (native when available) or `Float16Polyfill` (TC39-accurate) |
| `bfloat16` | `Bfloat16Array` — 1-8-7 layout, truncating-with-round-to-nearest-even narrow |
| `complex64` / `complex128` | `ComplexArray` — `.real(i)`, `.imag(i)`, `.get(i) → {re, im}`, iteration |

All three classes expose `.bits` / `.data` for zero-copy access to the
underlying raw storage if you need it.

```ts
const m = decode(buf);
const f16 = m.objects[0].data();           // Float16Array or polyfill
const asFloat32 = f16.toFloat32Array();    // widened copy
const bits = f16.bits;                      // raw binary16

const cplx = m.objects[1].data() as ComplexArray;
for (let i = 0; i < cplx.length; i++) {
  console.log(cplx.real(i), cplx.imag(i));
}
```

The polyfill is used automatically when the host runtime does not
ship `globalThis.Float16Array`.  `hasNativeFloat16Array()` and
`getFloat16ArrayCtor()` expose the detection machinery for callers
that want direct control.

> **Breaking change from Scope B:** Before Scope C, `obj.data()` on
> `float16` / `bfloat16` returned a raw `Uint16Array` of bits, and
> complex dtypes returned an interleaved `Float32Array` /
> `Float64Array`.  Consumers that relied on that shape can reach the
> same bytes via `.bits` (for f16/bf16) or `.data` (for complex).

The low-level bit-conversion helpers (`halfBitsToFloat`,
`floatToHalfBits`, `bfloat16BitsToFloat`, `floatToBfloat16Bits`) and
the `isComplexDtype` type-guard are **internal** and are not re-exported
from `@ecmwf.int/tensogram`.  Callers that need bit-level manipulation
should grab the raw storage from a view's `.bits` / `.data` accessor
and do the conversion themselves, or import directly from
`@ecmwf.int/tensogram/float16`, `…/bfloat16`, `…/complex` with the
understanding that these module paths are not part of the stable API.

## Examples

See `examples/typescript/` in the repository for runnable scripts:

- `01_encode_decode.ts` — basic round-trip
- `02_mars_metadata.ts` — per-object metadata using the MARS vocabulary
- `02b_generic_metadata.ts` — per-object metadata using a generic application namespace
- `03_multi_object.ts` — multiple dtypes in one message
- `04_decode_range.ts` — partial sub-tensor decode
- `05_streaming_fetch.ts` — progressive decode over a `ReadableStream`
- `06_file_api.ts` — `TensogramFile` over Node fs, fetch, and in-memory bytes
- `07_hash_and_errors.ts` — hash verification and typed errors
- `08_validate.ts` — `validate(buf)` + `validateFile(path)`
- `11_encode_pre_encoded.ts` — wrap already-encoded bytes
- `12_streaming_encoder.ts` — frame-at-a-time encoder with preceders
- `13_range_access.ts` — lazy `TensogramFile.fromUrl` over HTTP Range
- `14_streaming_callback.ts` — `StreamingEncoder` with `onBytes` callback sink

Run them with:

```bash
cd examples/typescript
npm install
npx tsx 01_encode_decode.ts     # or any other file
```

## Cross-language parity

This TypeScript package decodes the **same golden `.tgm` files** used
by the Rust, Python, and C++ test suites. The committed files at
`rust/tensogram/tests/golden/*.tgm` are decoded by each language's
test runner; any drift in wire-format semantics fails all four suites.

Specifically, `typescript/tests/golden.test.ts` decodes:

- `simple_f32.tgm` — single-object Float32 round-trip
- `multi_object.tgm` — mixed-dtype message (f32 / i64 / u8)
- `mars_metadata.tgm` — MARS keys under `base[0].mars`
- `multi_message.tgm` — two concatenated messages (via `scan()`)
- `hash_xxh3.tgm` — verifyHash success + tamper detection

`typescript/tests/property.test.ts` and the Scope-C dtype suites add
`fast-check` property tests pinning:

- `mapTensogramError` never throws for any finite-string input and
  always returns a `TensogramError` subclass;
- `encode → decode` is bit-exact for random Float32 shapes across
  random application metadata;
- `decode` on random byte input either succeeds with a structurally
  valid message or throws a typed `TensogramError` — never panics;
- `float32 → float16 → float32` round-trip stays within half-precision
  ulp for any random value in a reasonable magnitude band;
- `float32 → bfloat16 → float32` round-trip stays within bfloat16 ulp;
- `complex64` encode → decode preserves `real(i)` / `imag(i)`
  byte-for-byte across random shapes and values.

The CI `typescript` job rebuilds and runs every TS test on every PR.
