# TypeScript API

Tensogram ships `@ecmwf/tensogram`, a TypeScript package that wraps the
WebAssembly build with typed, idiomatic helpers. Use it in any modern
browser or Node ≥ 20.

> **Status:** Scope B is complete. Typed encode / decode / scan, dtype
> dispatch, metadata helpers, **progressive streaming decode**, and the
> **`TensogramFile` file / URL helper** are all available. Scope C
> follow-ups (validate, encode_pre_encoded, float16 / bfloat16 /
> complex first-class support, npm publish) are tracked in
> `plans/TYPESCRIPT_WRAPPER.md`.

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
} from '@ecmwf/tensogram';

// One-time WASM initialisation (idempotent)
await init();

// ── Encode ────────────────────────────────────────────────────────────
const temps = new Float32Array(100 * 200);
for (let i = 0; i < temps.length; i++) temps[i] = 273.15 + i / 100;

const meta: GlobalMetadata = { version: 2 };
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
| `metadata` | `GlobalMetadata` | Wire-format metadata; `version: 2` is required |
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

Mirror of `tensogram_core::compute_common`. Returns a
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
} from '@ecmwf/tensogram';

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
import { decodeStream } from '@ecmwf/tensogram';

const res = await fetch('/forecast.tgm');
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
import { TensogramFile } from '@ecmwf/tensogram';

// Node: from the local file system
const file = await TensogramFile.open('/data/forecast.tgm');

// Browser or Node: over HTTPS
const file = await TensogramFile.fromUrl('https://example.com/forecast.tgm');

// Any runtime: from pre-loaded bytes
const file = TensogramFile.fromBytes(uint8ArrayFromSomewhere);
```

All three factories produce an identical object:

```ts
interface TensogramFile extends AsyncIterable<DecodedMessage> {
  readonly messageCount: number;
  readonly byteLength: number;
  readonly source: 'local' | 'remote' | 'buffer';

  message(index: number, opts?: DecodeOptions): Promise<DecodedMessage>;
  messageMetadata(index: number): Promise<GlobalMetadata>;
  rawMessage(index: number): Uint8Array;

  [Symbol.asyncIterator](): AsyncIterator<DecodedMessage>;
  close(): void;
}
```

Usage:

```ts
const file = await TensogramFile.open('/data/forecast.tgm');
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
| `fetch` | `typeof fetch` | Override the fetch implementation (useful for tests and for browsers with a polyfill). |
| `headers` | `HeadersInit` | Extra request headers (auth, etc.). |
| `signal` | `AbortSignal` | Cancels the download. |

### `TensogramFile.fromBytes(bytes)`

Wraps an already-loaded `Uint8Array`. The buffer is defensively
copied, so later mutation of the caller's buffer is invisible to
the `TensogramFile`.

### Range-based lazy access (Scope C)

This Scope-B implementation downloads the whole file before building
its index. For very large `.tgm` files (several GB) where that's too
expensive, a future release will add a lazy backend that reads only
the footer index + the requested message via HTTP `Range` requests.
The public interface above will not change. See
`plans/TYPESCRIPT_WRAPPER.md` for details.

## Examples

See `examples/typescript/` in the repository for runnable scripts:

- `01_encode_decode.ts` — basic round-trip
- `02_mars_metadata.ts` — MARS keys on per-object `base[i]` entries
- `03_multi_object.ts` — multiple dtypes in one message
- `05_streaming_fetch.ts` — progressive decode over a `ReadableStream` (Phase 3)
- `06_file_api.ts` — `TensogramFile` over Node fs, fetch, and in-memory bytes (Phase 4)
- `07_hash_and_errors.ts` — hash verification and typed errors

Run them with:

```bash
cd examples/typescript
npm install
npx tsx 01_encode_decode.ts     # or 02, 03, 05, 06, 07
```

## Design notes

See `plans/TYPESCRIPT_WRAPPER.md` for the full design document covering
architecture, phases, test strategy, memory model, and open follow-ups.

## Cross-language parity

This TypeScript package decodes the same wire format used by the Rust,
Python, and C++ test suites. The TS test suite exercises encode / decode
round-trips and streaming decode against synthesised messages, and the
CI `typescript` job rebuilds and runs the full TS surface on every PR —
making TS the fourth language on the cross-language parity matrix.
