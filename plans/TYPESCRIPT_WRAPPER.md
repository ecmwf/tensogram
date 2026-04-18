# TypeScript Wrapper

> Design and implementation plan for `@ecmwf/tensogram` — a TypeScript
> package that provides an ergonomic, typed API on top of the existing
> `tensogram-wasm` bindings.
>
> For why Tensogram exists, see `MOTIVATION.md`.
> For the wire format, see `WIRE_FORMAT.md`.
> For what already exists, see `DONE.md` (the `tensogram-wasm` section).

## Purpose

The `tensogram-wasm` crate already exposes a complete encode / decode /
scan / streaming API through `wasm-bindgen`. Generated bindings are
functional but low-level: returned metadata is typed as `any`, payload
access requires dtype-specific method names (`data_f32`, `data_f64`, ...),
there is no file / stream helper, and naming is snake_case. This
package closes that gap.

The goal is a **thin, typed, idiomatic TypeScript layer**:
- Zero new encoding or decoding logic — all heavy lifting stays in WASM.
- Strong types for `GlobalMetadata`, `DataObjectDescriptor`, and dtypes.
- Dtype-aware payload dispatch returning the correct `TypedArray`.
- Async init wrapping `wasm-pack --target web` output.
- Node + browser file helpers using the platform's native APIs.
- Streaming decode via the Web Streams API.

## Scope

**Scope B — typed surface + file + stream ergonomics** (see README.md
git history / earlier design discussion for the original A/B/C scoping).

Included:
- Typed `encode`, `decode`, `decodeMetadata`, `decodeObject`, `scan`.
- `DecodedMessage` / `DecodedFrame` wrappers with typed `data()` dispatch.
- `decodeStream(readable)` for Web Streams API input.
- `TensogramFile.open(path)` (Node, via `node:fs/promises`).
- `TensogramFile.fromUrl(url)` (browser / fetch with Range requests).
- `getMetaKey(meta, "mars.param")` — first-match across `base[*]` then
  `_extra_`, same semantics as Rust / Python / CLI.
- `computeCommon(meta)` — TS mirror of `tensogram_core::compute_common`.
- Error class hierarchy mirroring the seven `TensogramError` variants.
- Examples in `examples/typescript/`.

Not included (explicitly):
- Reimplementing compression codecs in TS — WASM handles everything.
- Native remote object-store backends — browser `fetch` with Range
  headers is sufficient and works against any HTTPS-served `.tgm` file.
- `validate` / `encode_pre_encoded` / Zarr.js integration — deferred
  to a follow-up (Scope C).
- `float16` / `bfloat16` / `complex*` specialised TypedArray handling —
  exposed as `Uint16Array` / interleaved `Float32Array`, with a clear
  note; full support is a Scope-C extension.
- Publishing to npm — build locally only for now. A publish pipeline
  will follow once the API has stabilised.

## Architecture

```
                  ┌──────────────────────────────────┐
                  │ TS Application (browser / Node)  │
                  └────────────────┬─────────────────┘
                                   │
                  ┌────────────────▼─────────────────┐
                  │   @ecmwf/tensogram (this pkg)    │
                  │                                  │
                  │  typed API · dtype dispatch      │
                  │  file helpers · stream helpers   │
                  │  error classes · metadata utils  │
                  └────────────────┬─────────────────┘
                                   │
                  ┌────────────────▼─────────────────┐
                  │  generated wasm-pack output      │
                  │  (tensogram-wasm, unchanged)     │
                  └────────────────┬─────────────────┘
                                   │
                  ┌────────────────▼─────────────────┐
                  │  tensogram-core (Rust) via WASM  │
                  └──────────────────────────────────┘
```

The TS layer is a pure translation / ergonomic layer. It never re-parses
wire-format bytes or re-implements decoding.

### Rust-side adjustments needed

Only one change is required in `tensogram-wasm`:
- `src/convert.rs::to_js` currently uses `serde_wasm_bindgen::to_value`
  with the default serializer, which emits ES `Map` for Rust maps. TS
  developers expect plain objects. Switch to
  `Serializer::json_compatible()` so that maps come across as
  `{ [key: string]: value }` objects and safe-range u64 values come
  across as `number` rather than `BigInt`.

This is a backwards-compatible change for all existing
`wasm-bindgen-test` tests (they round-trip via `from_value`, which
accepts both). A dedicated test in the TS layer pins the new shape.

## Public API sketch

```ts
import { init, encode, decode, scan,
         decodeMetadata, decodeObject,
         decodeStream,
         TensogramFile,
         getMetaKey, computeCommon,
         FramingError, HashMismatchError, /* ... */ } from '@ecmwf/tensogram';

await init();  // idempotent, loads + instantiates the WASM blob

// Encode
const msg: Uint8Array = encode(meta, [{ descriptor, data }]);

// Decode
const { metadata, objects } = decode(msg);
const arr = objects[0].data();   // Float32Array (inferred from dtype)

// Scan
const pairs = scan(fileBytes);   // [{ offset, length }, ...]

// Stream (browser)
const res = await fetch('/forecast.tgm');
for await (const frame of decodeStream(res.body!)) {
  render(frame.descriptor.shape, frame.data());
}

// File (Node)
const file = await TensogramFile.open('forecast.tgm');
const { metadata, objects } = await file.message(0);

// File (browser)
const file = await TensogramFile.fromUrl('https://example.com/a.tgm');
const { metadata, objects } = await file.message(0);  // Range request
```

## Package layout

```
typescript/                             ← new top-level dir
├── package.json                        ← "@ecmwf/tensogram", ESM-only
├── tsconfig.json                       ← strict, ES2022, node+dom libs
├── vitest.config.ts
├── .gitignore                          ← dist/, wasm/, node_modules/
├── README.md
├── src/
│   ├── index.ts                        ← barrel export
│   ├── init.ts                         ← awaits WASM instantiation
│   ├── types.ts                        ← Metadata, Descriptor, Dtype, ByteOrder
│   ├── errors.ts                       ← typed error hierarchy
│   ├── dtype.ts                        ← dtype → TypedArray + byte widths
│   ├── encode.ts                       ← encode() wrapper
│   ├── decode.ts                       ← decode() / scan() / decodeObject()
│   ├── metadata.ts                     ← getMetaKey(), computeCommon()
│   ├── streaming.ts                    ← decodeStream() (phase 3)
│   └── file.ts                         ← TensogramFile (phase 4)
├── tests/
│   ├── smoke.test.ts
│   ├── encode.test.ts
│   ├── decode.test.ts
│   ├── dtype.test.ts
│   ├── metadata.test.ts
│   ├── errors.test.ts
│   ├── streaming.test.ts               ← phase 3
│   ├── file.node.test.ts               ← phase 4
│   └── golden.test.ts                  ← rust/tensogram-core/tests/golden/* parity
├── wasm/                               ← wasm-pack output, gitignored
└── dist/                               ← tsc output, gitignored

examples/typescript/                    ← new dir (mirrors examples/{rust,python,cpp}/)
├── README.md
├── 01_encode_decode.ts
├── 02_mars_metadata.ts
├── 02b_generic_metadata.ts
├── 03_simple_packing.ts
├── 04_multi_object.ts
├── 05_streaming_fetch.ts
├── 06_file_api.ts
└── 07_hash_and_errors.ts
```

## Phases (TDD, behaviour-driven)

Each phase ends with green tests. The test suite is the definition of
done. For each phase I list the primary **Given / When / Then**
behaviours under test.

### Phase 0 — scaffold + WASM integration
- Fix `to_js` in `rust/tensogram-wasm/src/convert.rs` to use
  `Serializer::json_compatible()`.
- Rebuild the WASM package with `wasm-pack build --target web`.
- Create `typescript/` with `package.json`, `tsconfig.json`,
  `vitest.config.ts`, `.gitignore`, `README.md`.
- Minimal `init()` + smoke test.
- **Given** a built WASM package, **when** `init()` is called in Node,
  **then** `decode(encode(...))` round-trips a `Float32Array`.

### Phase 1 — typed encode / decode / scan
- `types.ts`: `Dtype`, `ByteOrder`, `Encoding`, `Filter`, `Compression`,
  `DataObjectDescriptor`, `GlobalMetadata`, `HashDescriptor`, `CborValue`.
- `encode.ts`: typed wrapper that accepts `TypedArray | Uint8Array`
  payloads.
- `decode.ts`: `decode`, `decodeMetadata`, `decodeObject`, `scan`.
- `errors.ts`: `TensogramError` (base) + 7 named subclasses
  (`FramingError`, `MetadataError`, `EncodingError`, `CompressionError`,
  `ObjectError`, `IoError`, `HashMismatchError`). Factory that parses
  the `JsError` message prefix produced by the Rust side.
- Behaviours:
  - **Given** a 100×200 f32 array, **when** encoded and decoded,
    **then** the round-trip is bit-exact.
  - **Given** an encoded message, **when** `decodeMetadata` is called,
    **then** no payload bytes are touched (verified by timing / size).
  - **Given** a corrupt buffer, **when** `decode` is called,
    **then** a `FramingError` is thrown with the expected message.
  - **Given** a message with `hash="xxh3"` and `verifyHash: true`,
    **when** a single byte is flipped, **then** a
    `HashMismatchError` is thrown with `expected` and `actual` fields.

### Phase 2 — dtype dispatch + metadata helpers
- `dtype.ts`: `DTYPE_BYTE_WIDTH`, `dtypeTypedArrayCtor`, `dtypeToTypedArray(desc, bytes)`.
- Safe-copy default, zero-copy opt-in:
  - `object.data()` → new `TypedArray` copied onto the JS heap.
  - `object.dataView()` → zero-copy view (WASM-memory-backed).
- `metadata.ts`: `getMetaKey`, `computeCommon`.
- Behaviours:
  - **Given** a descriptor with `dtype: "float64"`, **when** `object.data()`
    is called, **then** the result is a `Float64Array` with the expected
    element count.
  - **Given** a message with `base[0].mars.param = "2t"` and
    `_extra_.source = "ifs"`, **when** `getMetaKey(meta, "mars.param")`
    is called, **then** `"2t"` is returned (first-match, `_reserved_`
    hidden).
  - **Given** two `base[i]` entries sharing `{class: "od"}` and
    differing on `param`, **when** `computeCommon` is called, **then**
    `{class: "od"}` is returned.
  - **Given** a memory-growth simulation, **when** `data()` (copy) is
    accessed afterwards, **then** its bytes are still valid; **but**
    `dataView()` is invalidated.

### Phase 3 — streaming via Web Streams API
- `streaming.ts`: `decodeStream(readable: ReadableStream<Uint8Array>, opts?)`
  → `AsyncIterable<DecodedFrame>`. Uses the existing WASM
  `StreamingDecoder` class.
- Options: `signal?: AbortSignal`, `maxBufferBytes?: number`.
- Behaviours:
  - **Given** a chunked `.tgm` stream, **when** decoded via
    `decodeStream`, **then** every frame matches a direct `decode` of
    the concatenated bytes.
  - **Given** a corrupted mid-stream message, **when** decoded,
    **then** subsequent messages still decode and the error is exposed
    via `{ lastError, skippedCount }` once iteration ends.
  - **Given** an `AbortSignal`, **when** aborted mid-stream, **then**
    the iteration terminates cleanly and the decoder is `free()`d.
  - **Given** `maxBufferBytes` is exceeded, **when** `feed()` errors,
    **then** the async iterator throws with a clear message.

### Phase 4 — TensogramFile (Node + Browser)
- `file.ts`:
  - `TensogramFile.open(path: string | URL)` — Node only; uses
    `node:fs/promises` via dynamic import so browser bundlers don't
    pull it in.
  - `TensogramFile.fromUrl(url: string, opts?: { fetch?, headers? })` —
    uses global `fetch` + Range requests; probes once for content
    length + Range support, then reads specific messages via
    `Range: bytes=offset-end`.
- Shared interface:
  - `file.messageCount()` — number
  - `file.message(i)` — decode a single message
  - `file.rawMessage(i)` — `Uint8Array` of the raw bytes
  - `file[Symbol.asyncIterator]()` — iterate all messages
  - `file.close()` — releases handles / aborts pending fetches
- Behaviours:
  - **Given** a local `.tgm` with 10 messages, **when**
    `TensogramFile.open` is called and `message(5)` read, **then** the
    result matches a buffered `decode`.
  - **Given** a `fetch` mock, **when** `fromUrl` reads message 5,
    **then** exactly two Range requests are sent: one small request
    covering the scan header, one covering the message frame.
  - **Given** a server that rejects Range, **when** `fromUrl` is
    called, **then** a clear `UnsupportedServerError` is thrown.
  - **Given** empty file, **when** `messageCount()` called, **then**
    returns `0` and `message(0)` throws `ObjectError`.

### Phase 5 — examples, docs, CI, Makefile
- `examples/typescript/01_encode_decode.ts` … `07_hash_and_errors.ts`,
  aligned with the existing Python / C++ example numbering.
- `docs/src/guide/typescript-api.md` is the primary user-facing doc;
  it already exists as a stub and is populated as each phase lands.
- `.github/workflows/ci.yml` adds a `typescript` job running
  `wasm-pack build` → `npm ci` → `tsc --noEmit` → `vitest run` on
  Node 20 and 22.
- Top-level `Makefile` gets `ts-build`, `ts-test`, `ts-lint`,
  `ts-fmt` targets.
- `CLAUDE.md` and `CONTRIBUTING.md` updated with TS setup instructions.

## Typing strategy

Hand-written types, loose where necessary:

- All known enum-like string fields (`dtype`, `byte_order`, `encoding`,
  `filter`, `compression`) are string literal unions.
- `params` on `DataObjectDescriptor` is intersected with
  `Record<string, unknown>` to permit encoding-specific keys
  (`reference_value`, `zstd_level`, `szip_block_offsets`, ...) without
  enumerating them at the top level.
- `GlobalMetadata.base` entries are `Record<string, CborValue>`.
  Application-layer shapes (`mars`, `grib`, `cf`, `netcdf`, `bids`,
  `dicom`, or any custom vocabulary) are **not** typed statically —
  consumers can cast with `as` if they want strict vocabulary typing in
  their own code.
- `CborValue` is the recursive union
  `string | number | boolean | null | CborValue[] | { [k: string]: CborValue }`.

No auto-generation from Rust. `tsify` / `typeshare` both struggle with
`ciborium::Value`, and hand-written types cost less maintenance than
chasing upstream tooling regressions.

## Memory model

- **Safe-copy default.** `object.data()` on a `DecodedMessage` or
  `DecodedFrame` always returns a `TypedArray` copied onto the JS
  heap. The copy survives WASM memory growth and outlives the
  underlying `DecodedMessage` / `DecodedFrame`.
- **Zero-copy opt-in.** `object.dataView()` returns a view directly
  into WASM linear memory. Documented as "invalidated if any further
  WASM call grows memory; do not retain past the next call".
- **`free()` is called for you.** Wrapping classes hold the raw
  `DecodedMessage` / `DecodedFrame` handle as a private field; calling
  `.close()` on the wrapper releases WASM memory. A `FinalizationRegistry`
  fallback triggers `free()` if the wrapper is GC'd without explicit
  `close()`, but explicit close is recommended.

## Testing strategy

| Layer | Tool | Purpose |
|-------|------|---------|
| TS unit tests | vitest | Types, dispatch, metadata helpers, error mapping |
| Round-trip tests | vitest + fast-check | Property-based round-trips across dtypes / pipelines |
| Cross-language | golden `.tgm` files | Verify TS decodes Rust-produced bytes byte-for-byte |
| WASM contract | (existing) wasm-bindgen-test | 134 tests in `rust/tensogram-wasm` — unchanged |
| Type correctness | `tsc --strict --noEmit` | No `any` leakage, all discriminated unions exhaustive |
| Bundle size | `size-limit` (follow-up) | Track WASM + JS glue growth over time |

The golden files in `rust/tensogram-core/tests/golden/` (e.g.
`simple_f32.tgm`, `multi_object.tgm`, `mars_metadata.tgm`,
`multi_message.tgm`, `hash_xxh3.tgm`) are byte-for-byte deterministic
and already used by the Rust, Python, and C++ suites for cross-language
verification. The TS suite will decode these same files and assert
expected metadata + payload bytes, making TS the **fourth** language
on the cross-language parity matrix.

## Verification & validation

- **tsc in strict mode** must pass with zero errors and zero `any` in
  the published surface.
- **vitest** must pass on Node 20 and 22.
- **Golden-file parity** tests must pass byte-for-byte against
  `rust/tensogram-core/tests/golden/*.tgm`.
- **Property-based round-trip** must survive 1,000 generated shape /
  dtype combinations per dtype.
- **CI** runs all of the above on every PR that touches
  `rust/tensogram-wasm/**` or `typescript/**`.

## Cross-language parity matrix

Scopes B + C.1 closed the read-path and write-path API gaps; C.2
closed the half-precision / complex dtype ergonomics gap.  The
remaining entries on the list — npm publishing, browser CI, bundle
budgets, Zarr.js — are distribution and ecosystem tasks tracked under
Scope C.3 / C.4 in `plans/TODO.md`.

| Concept | Rust | Python | FFI | C++ | TypeScript |
|---|---|---|---|---|---|
| `encode` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `decode` / `decode_metadata` / `decode_object` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `scan` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `streaming_decoder` | ✓ | ✓ | ✓ | ✓ | ✓ (`decodeStream`) |
| File open / message / iter | ✓ | ✓ | ✓ | ✓ | ✓ (`TensogramFile`) |
| Remote fetch | ✓ | ✓ | — | — | ✓ (`.fromUrl` + HTTP Range backend) |
| `metadata_get_key` (dotted path) | — | — | ✓ (typed: `_get_string/_int/_float`) | ✓ (typed: `get_string/get_int/get_float`) | ✓ (`getMetaKey`) |
| `compute_common` | ✓ | — | — | — | ✓ |
| `decode_range` | ✓ | ✓ | ✓ | ✓ | ✓ (`decodeRange`) |
| `streaming_encoder` | ✓ | ✓ | ✓ | ✓ | ✓ (`StreamingEncoder`) |
| `file_append` | ✓ | ✓ | ✓ | ✓ | ✓ (Node local-path only) |
| `validate_buffer` / `validate_file` | ✓ | ✓ | ✓ | ✓ | ✓ (`validate` / `validateBuffer` / `validateFile`) |
| `compute_hash` | ✓ | ✓ (`compute_hash`) | ✓ | ✓ | ✓ (`computeHash`) |
| `encode_pre_encoded` | ✓ | ✓ | ✓ | ✓ | ✓ (`encodePreEncoded`) |
| `simple_packing_params` | ✓ | ✓ (`compute_packing_params`) | ✓ | ✓ | ✓ (`simplePackingComputeParams`) |
| First-class `float16` / `bfloat16` / `complex*` | ✓ | ✓ (numpy dtypes) | — (opaque bytes) | — (opaque bytes) | ✓ (Scope C.2 view classes) |

Notes on `metadata_get_key` and `compute_common`:

- The Rust crate and Python package deliberately do **not** expose a
  dotted-path key helper — callers use nested access on `meta.base[i]`,
  `meta.extra`, and (Python only) `meta["ns"]["field"]` instead. The CLI,
  C FFI, C++ wrapper, and TypeScript package all accept a full dotted
  path with first-match-across-`base[i]` + `_extra_`-fallback semantics.
  See [`docs/src/guide/vocabularies.md`](../../docs/src/guide/vocabularies.md)
  for cross-binding examples.
- `compute_common` is currently Rust + TypeScript only. The Python
  package does not re-export it; callers needing a common-map can walk
  `meta.base` manually or shell out to the CLI. Adding a Python binding
  is a mechanical follow-up if the need arises.

## Scope-C.1 wire details

WASM-side additions live in `rust/tensogram-wasm`:

- `encoder.rs` — `StreamingEncoder` class backed by `Vec<u8>`.
- `extras.rs` — `decode_range`, `encode_pre_encoded`, `compute_hash`,
  `simple_packing_compute_params`, `validate_buffer`.
- `convert.rs` gains `typed_array_or_u8_to_bytes` covering every
  `ArrayBufferView` + `DataView`, used by the new write paths.

TS-side surface (`typescript/src/`):

- `range.ts`, `hash.ts`, `simplePacking.ts`, `validate.ts`,
  `encodePreEncoded.ts`, `streamingEncoder.ts` — one module per
  concept, each with its own dedicated test file.
- `file.ts` rewritten to support two new paths: `append` (Node local
  file) and a lazy Range-based `fromUrl` backend with transparent
  eager fallback.

### `rawMessage` is now async

`TensogramFile#rawMessage(index)` returns `Promise<Uint8Array>` (was
sync in Scope B).  The signature change lets the lazy HTTP backend
issue the `Range` GET on first access; existing callers add `await`.

## Scope-C.2 wire details

- `typescript/src/float16.ts` — `Float16Polyfill` with TC39-accurate
  semantics (round-ties-to-even narrow, NaN / ±Inf / subnormal
  preservation), plus `halfBitsToFloat` / `floatToHalfBits` /
  `hasNativeFloat16Array` / `getFloat16ArrayCtor` / `float16FromBytes`
  zero-copy factory.
- `typescript/src/bfloat16.ts` — matching `Bfloat16Array` view with
  the 1-8-7 layout used by ML frameworks.
- `typescript/src/complex.ts` — `ComplexArray` view over interleaved
  Float32 / Float64 storage, with `.real(i)`, `.imag(i)`, `.get(i)`,
  iteration.
- `typescript/src/dtype.ts` — `typedArrayFor` updated to route these
  dtypes through the view classes.  Callers who still want raw bits
  / interleaved storage reach them via `.bits` / `.data`.

## Open items & follow-ups

- **Scope C.3** — npm publish pipeline, browser CI via
  Vitest-browser + Playwright, `size-limit` bundle budget, vitest
  bench on hot paths.  See `plans/TODO.md`.
- **Scope C.4** — `@ecmwf/tensogram-zarr` mirror of the Python
  `tensogram-zarr` package.
- Dual `--target web` + `--target nodejs` wasm-pack build for tighter
  Node compatibility (currently Node ≥ 20 required).
- Integration with `earthkit-data` if a JS loader is ever needed.
- Callback-per-frame sinks in Python / C FFI / C++ — the WASM/TS
  surface gained this via `onBytes` in Pass 6, but other bindings
  remain buffered-only.  Extension pattern: a per-language sink
  abstraction (Python file-like / `io.BufferedWriter`, FFI
  function-pointer, C++ `std::ostream&`) wrapping the Rust core's
  `StreamingEncoder<W: Write>` generic just like `JsCallbackWriter`
  does for WASM.  Would close cross-language parity for the
  streaming-sink capability.

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Duplicating the 134 wasm-bindgen-test assertions at the TS layer | TS tests focus on ergonomics (types, dispatch, file/stream helpers), not re-testing decode logic |
| `FinalizationRegistry` unreliable across JS runtimes | Document explicit `.close()` as required; registry is best-effort cleanup only |
| `wasm-pack --target web` emits ESM with `import.meta.url` and `fetch()` — may not work with every bundler out of the box | Document supported bundlers (Vite, esbuild, Rollup with `@rollup/plugin-wasm`, webpack 5); mark older bundler support as out-of-scope |
| Range requests blocked by CORS on remote fetches | Documented; `fromUrl` surfaces a clear error and instructs users to enable `Accept-Ranges`, `Content-Range`, and `Content-Length` in their CORS responses |
| Metadata with u64 > 2^53 would overflow JS `number` | `Serializer::json_compatible()` emits `BigInt` for out-of-range values; documented on the type for `base[i]` |
