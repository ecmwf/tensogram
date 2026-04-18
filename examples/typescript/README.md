# Tensogram — TypeScript examples

Runnable TypeScript examples that mirror the `examples/python/` and
`examples/rust/` sets. They demonstrate the idiomatic way to encode,
decode, and inspect `.tgm` messages from TypeScript.

## Prerequisites

- Node ≥ 20
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/) and a
  Rust toolchain with the `wasm32-unknown-unknown` target
- `npm` (bundled with Node)

## Setup

Build the TypeScript package from the repo root:

```bash
cd typescript
npm install
npm run build      # wasm-pack build + tsc
cd ..
```

Then install the example dependencies (just `tsx` to run `.ts` files
directly without a separate build step):

```bash
cd examples/typescript
npm install
```

## Running an example

```bash
# From examples/typescript/
npx tsx 01_encode_decode.ts
```

## Available examples

| File | Topic |
|---|---|
| [`01_encode_decode.ts`](01_encode_decode.ts) | Basic round-trip encode / decode of a 2-D `Float32Array` |
| [`02_mars_metadata.ts`](02_mars_metadata.ts) | Attach MARS keys via `base[i]` and use `getMetaKey` to read them |
| [`03_multi_object.ts`](03_multi_object.ts) | Multiple objects with different dtypes in one message |
| [`04_decode_range.ts`](04_decode_range.ts) | Partial sub-tensor decode in split and join modes |
| [`05_streaming_fetch.ts`](05_streaming_fetch.ts) | Progressive decode over a `ReadableStream` |
| [`06_file_api.ts`](06_file_api.ts) | `TensogramFile` over Node fs, fetch, and in-memory bytes |
| [`07_hash_and_errors.ts`](07_hash_and_errors.ts) | Hash verification and the typed error hierarchy |
| [`08_validate.ts`](08_validate.ts) | `validate(buf)` and `validateFile(path)` — structural + integrity |
| [`11_encode_pre_encoded.ts`](11_encode_pre_encoded.ts) | Wrap already-encoded bytes without re-running the pipeline |
| [`12_streaming_encoder.ts`](12_streaming_encoder.ts) | Frame-at-a-time `StreamingEncoder` with per-object preceders |
| [`13_range_access.ts`](13_range_access.ts) | Lazy `TensogramFile.fromUrl` over HTTP Range requests |
| [`14_streaming_callback.ts`](14_streaming_callback.ts) | `StreamingEncoder` with `onBytes` callback — no full-message buffering |

Numbers 09 / 10 / 15+ are reserved to stay in step with the
`examples/python/` numbering.  See `plans/TYPESCRIPT_WRAPPER.md` for
follow-up examples (xarray/Zarr cross-runtime demos, first-class
half-precision, etc.).
